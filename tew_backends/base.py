"""
backends/base.py

Heavyタガー各バックエンド(ONNX Runtime / timm等)共通の基底クラスと、
ComfyUIプロセス内で共有されるモデルロード状態(LRUレジストリ)を提供する。

設計上の要点(実装指示書 Phase 1 準拠):
- モデルキャッシュはモジュールレベルのグローバル辞書として保持する。
  ComfyUIはノードオブジェクトを実行のたびに再インスタンス化することがあるため、
  インスタンス変数でキャッシュを持つと毎回ロードし直してVRAMを浪費する。
  このモジュール(base.py)がプロセス全体で単一の状態を持つ「事実上のシングルトン」になる。
- VRAM見積もり(VRAM_TABLE)は起動時は概算値だが、初回ロード時の実測値で上書きされる。
  実測値は utils/vram_manager.record_actual_vram() 経由で models.json に永続化する。
"""

from __future__ import annotations

import abc
import gc
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("ComfyUI_Tagger_Ensemble_Worker")

try:
    import torch
except ImportError:  # ComfyUI本体のvenv外で単体テストする場合に備える
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# VRAM概算テーブル (GB単位)
# 初期値はブループリントの概算。record_actual_vram() が呼ばれるたびに実測値へ
# 上書きされ、models.json に永続化される (vram_manager.py が担当)。
# ---------------------------------------------------------------------------
VRAM_TABLE: Dict[str, float] = {
    "cl_v2": 1.6,
    "cl_v1": 1.1,
    "dtq_l16": 1.2,
    "dtq_b16": 0.6,
    "oppai_v11": 1.0,
    "wd_eva02_l": 1.2,
    "at_eva02": 1.5,
    "at_convnext_huge": 2.8,
}

# at_convnext_huge は仕様上 MAX_LOADED_MODELS=1 固定 (他モデルは強制アンロード)。
FORCE_EXCLUSIVE_MODELS = {"at_convnext_huge"}

# モデルごとのアイドルタイムアウト(秒)。未指定はデフォルト値(呼び出し側で管理)を使う。
IDLE_TIMEOUT_OVERRIDES = {
    "at_convnext_huge": 60,  # Hugeモードは推論後速やかにアンロードする
}


@dataclass
class LoadedModelEntry:
    """ロード済みモデル1件分の状態。"""

    model_id: str
    backend: "ModelBase"
    vram_gb: float
    last_used_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_used_at = time.time()


# プロセス全体で共有するロード済みモデルのレジストリ (LRU管理の対象)。
# key: model_id, value: LoadedModelEntry
_loaded_models: Dict[str, LoadedModelEntry] = {}


def get_loaded_models() -> Dict[str, LoadedModelEntry]:
    """現在ロードされている全モデルのレジストリを返す(読み取り専用の想定)。"""
    return _loaded_models


def get_idle_timeout(model_id: str, default_timeout: int) -> int:
    return IDLE_TIMEOUT_OVERRIDES.get(model_id, default_timeout)


class ModelBase(abc.ABC):
    """
    各バックエンド(ONNX Runtime / timm等)が実装する共通インターフェース。

    サブクラスは以下を実装する:
      - _do_load(): 実際のモデルロード処理(セッション/state_dict構築など)
      - infer(image): 1枚(またはバッチ)の画像に対する推論
      - _do_unload(): バックエンド固有の解放処理(必要な場合のみoverride)
    """

    # 指示書5: VRAM測定手段はバックエンド種別によって信頼できる指標が異なるため、
    # サブクラスが自己申告する。
    #   "torch"               : torch.cuda.memory_allocated() の差分を実測値として使える
    #                            (TorchBackend。torchのキャッシュアロケータ経由で確保するため)
    #   "device_free_estimate": onnxruntime CUDA EP はtorchのアロケータを経由しないため、
    #                            torch_allocatedの差分はVRAM使用量を反映しない。
    #                            device上の空きVRAM減少量を "推定値" として使う(実測とは呼ばない)。
    vram_measurement_kind: str = "torch"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._is_loaded = False

    # -- 公開API -----------------------------------------------------------

    def load(self) -> None:
        """
        モデルをロードする。VRAM容量の確保(ensure_capacity)は呼び出し元
        (node_heavy.py 等)が vram_manager.ensure_capacity() を通じて事前に
        行っている前提だが、二重呼び出しでも安全なようにここでも軽くチェックする。

        指示書 9 (rollback) / 10 (成功判定強化) に基づき、トランザクション的に扱う:
          begin -> _do_load() -> 妥当性検証(dummy推論含む) -> commit
        途中で失敗した場合は、backendリソースの解放とレジストリからの除外を行い、
        _is_loaded=False のまま(=ロードされていない状態)で例外を送出する。
        """
        if self._is_loaded:
            self._touch()
            return

        vram_before = _get_vram_snapshot()
        logger.info("[TEW][LOAD] begin: model_id=%s backend=%s", self.model_id, type(self).__name__)

        try:
            self._do_load()
            self._validate_after_load()
        except Exception as exc:
            logger.exception(
                "モデルロードに失敗しました。ロールバックします: model_id=%s", self.model_id
            )
            self._rollback_failed_load()
            self._record_registry_status(status="FAILED", detail=str(exc))
            raise

        vram_after = _get_vram_snapshot()
        actual_vram, vram_kind = _diff_vram_snapshot(vram_before, vram_after, self)

        self._is_loaded = True

        # 実測できた場合のみ VRAM_TABLE を更新する(取れない場合は概算値を維持)。
        if actual_vram is not None and actual_vram > 0:
            record_actual_vram(self.model_id, actual_vram)

        entry = LoadedModelEntry(
            model_id=self.model_id,
            backend=self,
            vram_gb=VRAM_TABLE.get(self.model_id, actual_vram if actual_vram is not None else 1.0),
        )
        _loaded_models[self.model_id] = entry

        logger.info(
            "[TEW][LOAD] commit: model_id=%s vram_gb(%s)=%s",
            self.model_id,
            vram_kind,
            f"{actual_vram:.2f}" if actual_vram is not None else "unknown",
        )
        logger.info(
            "[TEW][VRAM] model_id=%s\n"
            "  device_free_before=%s\n"
            "  device_free_after=%s\n"
            "  backend_allocated=%s(%s)\n"
            "  estimated_vram=%s",
            self.model_id,
            f"{vram_before['device_free_gb']:.2f}GB" if vram_before["device_free_gb"] is not None else "unknown",
            f"{vram_after['device_free_gb']:.2f}GB" if vram_after["device_free_gb"] is not None else "unknown",
            f"{actual_vram:.2f}GB" if actual_vram is not None else "unknown",
            vram_kind,
            f"{VRAM_TABLE.get(self.model_id):.2f}GB" if self.model_id in VRAM_TABLE else "unknown",
        )

        # 指示書15: 実際にロード+ダミー推論まで成功して初めてVALIDATEDとして記録する
        # (「登録済み」を「利用可能」と誤表示しないため)。provider情報はサブクラスが提供する。
        self._record_registry_status(status="VALIDATED", detail="load + dummy infer succeeded")

        # at_convnext_huge 等の排他モデルは、ロード直後に他モデルを全アンロードする。
        if self.model_id in FORCE_EXCLUSIVE_MODELS:
            unload_all_except(self.model_id)

    def _record_registry_status(self, status: str, detail: str) -> None:
        """
        model_registry(models.json)へ状態機械(指示書15)を反映する。
        レジストリ更新自体が失敗してもロード処理そのものは継続させたいので、例外は握りつぶしログのみ。
        """
        try:
            from tew_utils import model_registry
            provider_status = self._get_provider_status()
            model_registry.set_model_status(
                self.model_id, status, detail=detail, provider_status=provider_status,
            )
        except Exception:
            logger.debug(
                "model_registryへの状態記録に失敗しましたが、ロード処理自体は継続します: model_id=%s",
                self.model_id, exc_info=True,
            )

    def _get_provider_status(self) -> Optional[str]:
        """
        指示書15: CPU_FALLBACK/CUDA_READYの判定。デフォルトはNone(=判定なし)。
        GPU実行が意味を持つバックエンド(OnnxBackend/TorchBackend)がoverrideする。
        """
        return None

    def _rollback_failed_load(self) -> None:
        """
        指示書9: ロード失敗時、session/model/cache/VRAM registry/_is_loaded の状態が
        途中まで進んだまま残らないようにする。
        """
        self._is_loaded = False
        _loaded_models.pop(self.model_id, None)
        try:
            self._do_unload()
        except Exception:
            logger.debug(
                "ロールバック中の_do_unload()でも例外が発生しましたが無視して続行します: model_id=%s",
                self.model_id, exc_info=True,
            )
        _release_vram()
        logger.info("[TEW][LOAD] rollback complete: model_id=%s is_loaded=False", self.model_id)

    def _validate_after_load(self) -> None:
        """
        指示書10: 「ロード完了」を名乗る前に最低限のダミー推論まで確認する。
        1x1の単色ダミー画像で infer() を実行し、
          - 例外が出ないこと
          - 空でない{tag: prob}辞書が返ること
          - 出力次元とタグ数が(ある程度)対応していること(0件は不可)
        をチェックする。ここで失敗した場合は load() 側でロールバックされる。
        """
        from PIL import Image as _Image

        dummy = _Image.new("RGB", (64, 64), color=(128, 128, 128))
        probs = self.infer(dummy)
        if not probs:
            raise RuntimeError(
                f"model_id={self.model_id}: ロード後のダミー推論結果が空でした"
                f"(output次元とtag数の対応、ONNX inputs等を確認してください)"
            )

        # 指示書10 (output/tag数の一致) を「warningで見逃す」のではなく明示的に検証する。
        # infer()は不一致時に先頭の一致範囲だけを返すため、その場合でも_last_raw_output_dimで
        # 生の出力次元とタグ数を突き合わせ、ロード完了の可否を判定する。
        raw_dim = getattr(self, "_last_raw_output_dim", None)
        tag_count = len(getattr(self, "tags", []) or [])
        if raw_dim is not None and tag_count and raw_dim != tag_count:
            raise RuntimeError(
                f"model_id={self.model_id}: モデル出力次元({raw_dim})とタグ数({tag_count})が"
                f"一致しません。tags_path/タグパーサーの形式、またはtimm_model_name/num_classesの"
                f"対応関係を確認してください。"
            )

        logger.info(
            "[TEW][LOAD] validate: model_id=%s dummy_infer_ok=True returned_tags=%d raw_output_dim=%s tag_count=%d",
            self.model_id, len(probs), raw_dim, tag_count,
        )

    def unload(self) -> None:
        if not self._is_loaded:
            return
        released = True
        try:
            self._do_unload()
        except Exception:
            released = False
            logger.exception("model_id=%s: _do_unload()中に例外が発生しました", self.model_id)
        finally:
            self._is_loaded = False
            _loaded_models.pop(self.model_id, None)
            _release_vram()
            logger.info("[TEW][UNLOAD] model_id=%s released=%s", self.model_id, released)

    def _touch(self) -> None:
        entry = _loaded_models.get(self.model_id)
        if entry is not None:
            entry.touch()

    @abc.abstractmethod
    def infer(self, image: Any) -> Dict[str, float]:
        """
        画像1枚(またはバッチ)を推論し、{tag: probability} の辞書(またはバッチの場合はリスト)を返す。
        呼び出し前に load() 済みであることを前提とする。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _do_load(self) -> None:
        raise NotImplementedError

    def _do_unload(self) -> None:
        """
        デフォルト実装。サブクラス固有のリソース(セッションハンドル等)がある場合は
        override して del してから super()._do_unload() を呼ぶこと。
        """
        return

    @property
    def vram_weight(self) -> float:
        """現在の(実測 or 概算)VRAM見積もり(GB)。"""
        return VRAM_TABLE.get(self.model_id, 1.0)


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------

def _get_vram_snapshot() -> Dict[str, Optional[float]]:
    """
    指示書5: 「実測」というラベルを実際に測定していない値に使わないため、
    測定できる複数の指標を区別して保持するスナップショット。
    """
    snapshot: Dict[str, Optional[float]] = {
        "torch_allocated_gb": None,
        "torch_reserved_gb": None,
        "device_free_gb": None,
    }
    if torch is not None and torch.cuda.is_available():
        snapshot["torch_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
        snapshot["torch_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            snapshot["device_free_gb"] = free_bytes / (1024 ** 3)
        except Exception:
            logger.debug("torch.cuda.mem_get_info() に失敗しました", exc_info=True)
    return snapshot


def _diff_vram_snapshot(
    before: Dict[str, Optional[float]],
    after: Dict[str, Optional[float]],
    backend: "ModelBase",
):
    """
    backend.vram_measurement_kind に応じて、信頼できる指標のみを使って差分を返す。
    戻り値: (差分GB または None, ラベル文字列)
    """
    kind = getattr(backend, "vram_measurement_kind", "torch")

    if kind == "torch":
        if before["torch_allocated_gb"] is None or after["torch_allocated_gb"] is None:
            return None, "unknown(CUDA未検出)"
        diff = max(0.0, after["torch_allocated_gb"] - before["torch_allocated_gb"])
        return diff, "torch_allocated_vram(実測)"

    if kind == "device_free_estimate":
        if before["device_free_gb"] is None or after["device_free_gb"] is None:
            return None, "unknown(CUDA未検出)"
        diff = max(0.0, before["device_free_gb"] - after["device_free_gb"])
        return diff, "estimated_vram_gb(device空きVRAM差分。onnxruntimeはtorchアロケータ非経由のため実測不可)"

    return None, "unknown"


def _release_vram() -> None:
    """
    アンロード共通処理。実装指示書 Phase 2 (backends/onnx_backend.py 等) の
    unload_model() と同一方針: gc + ipc_collect(失敗しても続行) + empty_cache
    + ComfyUI本体のキャッシュ解放 (soft_empty_cache)。
    """
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.ipc_collect()
        except Exception:
            logger.debug("torch.cuda.ipc_collect() に失敗しましたが続行します", exc_info=True)
        torch.cuda.empty_cache()

    try:
        import comfy.model_management as mm  # ComfyUI本体プロセス内でのみ利用可能
        mm.soft_empty_cache()
    except ImportError:
        # ComfyUI外(単体テスト等)では無視する。
        pass


def unload_all_except(keep_model_id: str) -> None:
    """keep_model_id 以外の全ロード済みモデルをアンロードする(排他モデル用)。"""
    for model_id in list(_loaded_models.keys()):
        if model_id == keep_model_id:
            continue
        entry = _loaded_models.get(model_id)
        if entry is not None:
            entry.backend.unload()


def record_actual_vram(model_id: str, actual_gb: float) -> None:
    """
    実測VRAM値でVRAM_TABLEを更新する。永続化(models.json書き込み)は
    vram_manager.persist_vram_table() が担当するため、ここではメモリ上の
    テーブル更新と、永続化フックの呼び出しのみ行う。
    """
    VRAM_TABLE[model_id] = round(actual_gb, 3)
    try:
        from tew_utils.vram_manager import persist_vram_table
        persist_vram_table(VRAM_TABLE)
    except ImportError:
        logger.debug("vram_manager.persist_vram_table が見つかりません(Phase順序の都合上、単体テスト時は無視して構いません)")
