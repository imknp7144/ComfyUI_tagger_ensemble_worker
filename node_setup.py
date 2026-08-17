"""
node_setup.py

ComfyUIカスタムノード: TaggerWorkerSetup。
- MAX_VRAM_GB のユーザー設定(ハードウェアごとに可変)
- 各モデルのセットアップ状態確認/非gatedモデルの自動ダウンロード
- GPLモデルの既定無効化+明示的な有効化スイッチ
- 現在のVRAM使用状況の表示

【方針変更 (ユーザー指示)】
連絡先情報の共有や利用規約への同意が必要な「gated」モデル(cl_v2, dtq_l16, dtq_b16,
at_eva02, at_convnext_huge)は、全自動ダウンロード(hf_token + snapshot_download)を行わない。
予期せぬトラブル(意図しない同意送信、規約違反等)を防ぐため、ユーザーが配布元から手動で
ダウンロード・配置する方式に統一する。このノードは「配置されているか確認し、あれば登録する」
役割のみを担う。hf_token入力フィールド自体を廃止する(リスク評価#5 hf_token漏洩の懸念も
この方針変更により解消される)。

非gatedモデル(cl_v1, oppai_v11, wd_eva02_l)は huggingface_hub 経由で引き続き自動ダウンロードする。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from tew_utils import model_registry, vram_manager
from tew_backends.base import get_loaded_models

logger = logging.getLogger("ComfyUI_Tagger_Ensemble_Worker")


def _get_models_dir() -> str:
    """
    モデルファイルの配置先ディレクトリ。

    【方針】オリジナルの _MyEXT_ComfyUI_Tagger_Worker(NPU Worker)の慣習に合わせ、
    ComfyUI本体の共有 models/ ディレクトリではなく、本拡張機能フォルダ直下の
    models/ を使う(例: <拡張機能フォルダ>/models/cl_v2/model.onnx )。
    こうすることで拡張機能単体でモデル資産が完結し、ComfyUI本体のアップデートや
    複数バージョン共存時にも影響を受けにくくなる。
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(project_root, "models")
    os.makedirs(base, exist_ok=True)
    return base


@dataclass
class ModelCatalogEntry:
    model_id: str
    backend: str  # "onnx" | "torch"
    gated: bool
    license: str
    source_url: str
    expected_model_filename: str
    expected_tags_filename: str
    timm_name: Optional[str] = None
    apply_sigmoid: bool = True
    best_threshold: float = 0.35
    repo_id: Optional[str] = None  # 非gatedモデルの自動ダウンロード用(huggingface_hub repo id)
    repo_model_filename: Optional[str] = None  # repo内の実際のパス(サブフォルダ含む)。未指定時はexpected_model_filenameのbasenameを使う
    repo_tags_filename: Optional[str] = None  # 同上、タグファイル用
    expected_tag_category_filename: Optional[str] = None  # タグ名とカテゴリ情報が別ファイルのモデル用(dtq系)


# --------------------------------------------------------------------------
# モデルカタログ
#
# 2026-08時点でユーザーから提供・確認された配布元に基づく。cl_v1のみ配布元未確認のため
# 「要確認」のまま残している(README/このファイルの当該箇所を参照)。
# ファイル配置は _get_models_dir() 配下に「<model_id>/<ファイル名>」の形で行うこと。
# --------------------------------------------------------------------------
MODEL_CATALOG: List[ModelCatalogEntry] = [
    ModelCatalogEntry(
        model_id="cl_v2", backend="onnx", gated=True,
        license="CL Tagger v2 Model License v1.0(独自ライセンス。再配布禁止・自己使用/条件付きサーブのみ許諾。詳細は配布元のLICENSE.md参照)",
        source_url="https://huggingface.co/cella110n/cl_tagger_v2",
        expected_model_filename="cl_v2/model.onnx",  # model.onnx.data も同じフォルダに配置すること(ONNX外部データ)
        expected_tags_filename="cl_v2/model_vocabulary.json",
        best_threshold=0.55,  # 配布元READMEの推奨しきい値
    ),
    ModelCatalogEntry(
        model_id="cl_v1", backend="onnx", gated=False, license="Apache-2.0",
        source_url="https://huggingface.co/cella110n/cl_tagger (cl_tagger_1_02/ フォルダ、最新版)",
        repo_id="cella110n/cl_tagger",
        repo_model_filename="cl_tagger_1_02/model.onnx",
        repo_tags_filename="cl_tagger_1_02/tag_mapping.json",
        expected_model_filename="cl_v1/model.onnx",
        expected_tags_filename="cl_v1/tag_mapping.json",
        best_threshold=0.35,  # 【要検証】配布元READMEに推奨しきい値の明記なし。cl_v2の値を暫定流用
    ),
    ModelCatalogEntry(
        model_id="dtq_l16", backend="onnx", gated=True,
        license="DINOv3 License(Meta社の独自ライセンス。バックボーンがDINOv3のため配布元HFで利用規約への同意が必要)",
        source_url="https://huggingface.co/realphongha/danbooru-tag-query"
                    " (models/DanbooruTagQuery_l16_448x448/ フォルダ)",
        expected_model_filename="dtq_l16/model.onnx",
        expected_tags_filename="dtq_l16/tag_to_id.json",
        expected_tag_category_filename="dtq_l16/tag_category.json",
        best_threshold=0.20,  # 配布元READMEのベンチマーク表記載のベストしきい値
    ),
    ModelCatalogEntry(
        model_id="dtq_b16", backend="onnx", gated=True,
        license="DINOv3 License(Meta社の独自ライセンス。バックボーンがDINOv3のため配布元HFで利用規約への同意が必要)",
        source_url="https://huggingface.co/realphongha/danbooru-tag-query"
                    " (models/DanbooruTagQuery_b16_448x448/ フォルダ)",
        expected_model_filename="dtq_b16/model.onnx",
        expected_tags_filename="dtq_b16/tag_to_id.json",
        expected_tag_category_filename="dtq_b16/tag_category.json",
        best_threshold=0.20,
    ),
    ModelCatalogEntry(
        model_id="oppai_v11", backend="onnx", gated=False, license="Apache-2.0",
        source_url="https://huggingface.co/Grio43/OppaiOracle (V1.1_onnx/ フォルダ)",
        repo_id="Grio43/OppaiOracle",
        repo_model_filename="V1.1_onnx/model.onnx",
        repo_tags_filename="V1.1_onnx/selected_tags.csv",
        expected_model_filename="oppai_v11/model.onnx",
        expected_tags_filename="oppai_v11/selected_tags.csv",
        best_threshold=0.753,  # 配布元READMEのmacro P=Rしきい値(pr_thresholds.json記載)
        # 【実機で確認・修正】apply_sigmoid=True(デフォルト)のままだと、oppai_v11のONNXグラフが
        # 既に内部でsigmoidを適用済みの確率(0〜1)を出力しているため、後段でさらにsigmoidを
        # 二重適用してしまい、出力レンジが sigmoid(0)〜sigmoid(1)≈0.5〜0.731 に圧縮されていた。
        # best_threshold=0.753 はこの圧縮後レンジの最大値(0.731)を超えており、構造的に
        # どんなタグも閾値を超えられずtag出力が常に空になる不具合が実機で確認された
        # (raw_scores_jsonの最高スコアが0.731付近で頭打ちになる症状として観測)。
        # モデル自身の出力を直接プロパティ値として使うよう apply_sigmoid=False に修正。
        apply_sigmoid=False,
    ),
    ModelCatalogEntry(
        model_id="wd_eva02_l", backend="onnx", gated=False, license="Apache-2.0",
        source_url="https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3",
        repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
        repo_model_filename="model.onnx",
        repo_tags_filename="selected_tags.csv",
        expected_model_filename="wd_eva02_l/model.onnx",
        expected_tags_filename="wd_eva02_l/selected_tags.csv",
        best_threshold=0.5296,  # 配布元README記載のP=Rしきい値
    ),
    ModelCatalogEntry(
        model_id="at_eva02", backend="torch", gated=True, license="GPL-3.0",
        source_url="https://huggingface.co/animetimm/eva02_large_patch14_448.dbv4-full",
        timm_name="eva02_large_patch14_448",
        expected_model_filename="at_eva02/model.safetensors",
        expected_tags_filename="at_eva02/selected_tags.csv",
        best_threshold=0.39,  # 配布元README記載のgeneralカテゴリしきい値
    ),
    ModelCatalogEntry(
        model_id="at_convnext_huge", backend="torch", gated=True, license="GPL-3.0",
        source_url="https://huggingface.co/animetimm/convnextv2_huge.dbv4-full",
        timm_name="convnextv2_huge",
        expected_model_filename="at_convnext_huge/model.safetensors",
        expected_tags_filename="at_convnext_huge/selected_tags.csv",
        best_threshold=0.38,  # 配布元README記載のgeneralカテゴリしきい値
    ),
]

_CATALOG_BY_ID = {e.model_id: e for e in MODEL_CATALOG}


class TaggerWorkerSetup:
    """NPU/Heavy一括管理ノード。MAX_VRAM_GB設定、モデルセットアップ状態確認、VRAM表示を行う。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_vram_gb": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 128.0, "step": 0.1}),
                "enable_gpl_models": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_report",)
    FUNCTION = "run"
    CATEGORY = "Tagger Ensemble Worker/Setup"
    OUTPUT_NODE = True

    def run(self, max_vram_gb: float, enable_gpl_models: bool):
        vram_manager.set_max_vram_gb(max_vram_gb)

        models_dir = _get_models_dir()
        lines: List[str] = [f"# Tagger Ensemble Worker セットアップ状況 (MAX_VRAM_GB={max_vram_gb:.1f})", ""]

        for entry in MODEL_CATALOG:
            lines.append(self._ensure_model(entry, models_dir, enable_gpl_models))

        lines.append("")
        lines.extend(self._model_status_lines())
        lines.append("")
        lines.extend(self._vram_status_lines())

        report = "\n".join(lines)
        logger.info("Setup実行結果:\n%s", report)
        return (report,)

    # -- 内部ユーティリティ ---------------------------------------------

    @staticmethod
    def _ensure_model(entry: ModelCatalogEntry, models_dir: str, enable_gpl_models: bool) -> str:
        if entry.license == "GPL-3.0" and not enable_gpl_models:
            return (
                f"[SKIP] {entry.model_id}: GPL-3.0ライセンスのモデルは既定で無効です。"
                f"有効化するには enable_gpl_models をONにしてください"
                f"(配布物にGPLが伝播する可能性があることに同意した上で行ってください)。"
            )

        model_path = os.path.join(models_dir, entry.expected_model_filename)
        tags_path = os.path.join(models_dir, entry.expected_tags_filename)
        category_path = (
            os.path.join(models_dir, entry.expected_tag_category_filename)
            if entry.expected_tag_category_filename else None
        )
        required_paths = [model_path, tags_path] + ([category_path] if category_path else [])

        if entry.gated:
            if all(os.path.exists(p) for p in required_paths):
                state = TaggerWorkerSetup._register(entry, model_path, tags_path, category_path)
                return f"[OK] {entry.model_id}: 手動配置済みのファイルを検出し、登録しました (state={state})"
            missing_desc = ", ".join(required_paths)
            model_registry.set_model_status(entry.model_id, "NOT_INSTALLED", detail="gatedモデル、手動配置待ち")
            return (
                f"[ACTION REQUIRED] {entry.model_id}: 同意が必要な配布元のため自動ダウンロードは行いません。"
                f"README記載の配布元 ({entry.source_url}) から手動でダウンロードし、"
                f"次のパスに配置してください: {missing_desc} (state=NOT_INSTALLED)"
            )

        # 非gatedモデル: 既に配置済みでなければ自動ダウンロードを試みる
        if all(os.path.exists(p) for p in required_paths):
            state = TaggerWorkerSetup._register(entry, model_path, tags_path, category_path)
            return f"[OK] {entry.model_id}: 既にダウンロード済みのファイルを検出し、登録しました (state={state})"

        try:
            TaggerWorkerSetup._download(entry, model_path, tags_path)
            state = TaggerWorkerSetup._register(entry, model_path, tags_path, category_path)
            return f"[OK] {entry.model_id}: 自動ダウンロードが完了し、登録しました (state={state})"
        except Exception as exc:  # noqa: BLE001 — ノードのステータス表示に理由を出すため意図的に広く捕捉
            logger.exception("model_id=%s の自動ダウンロードに失敗しました", entry.model_id)
            model_registry.set_model_status(entry.model_id, "NOT_INSTALLED", detail=f"ダウンロード失敗: {exc}")
            return (
                f"[ERROR] {entry.model_id}: 自動ダウンロードに失敗しました ({exc})。"
                f"README記載の配布元 ({entry.source_url}) から手動配置することもできます: {model_path} "
                f"(state=NOT_INSTALLED)"
            )

    @staticmethod
    def _register(entry: ModelCatalogEntry, model_path: str, tags_path: str, category_path: Optional[str] = None) -> str:
        """
        モデル設定をmodels.jsonへ登録する。「ファイルを発見した」だけで「利用可能」と
        表示しないよう(指示書15)、登録直後にget_model_config()での必須フィールド検証も
        行い、通過すればLOADABLE、そうでなければ理由付きでREGISTEREDに留める。
        実際に推論可能かどうか(VALIDATED/FAILED)は node_heavy.py 側の初回ロード時に
        tew_backends.base.ModelBase.load() が判定・記録する。
        戻り値は最終的なstate文字列(ログ/status_report表示用)。
        """
        model_registry.set_model_config(
            entry.model_id,
            backend=entry.backend,
            model_path=model_path,
            tags_path=tags_path,
            tag_category_path=category_path,
            timm_name=entry.timm_name,
            apply_sigmoid=entry.apply_sigmoid,
            best_threshold=entry.best_threshold,
            license=entry.license,
            gated=entry.gated,
            source_url=entry.source_url,
        )
        try:
            model_registry.get_model_config(entry.model_id)  # 必須フィールド検証のみ目的
            model_registry.set_model_status(entry.model_id, "LOADABLE", detail="ファイル配置・登録・必須フィールド検証OK")
            return "LOADABLE"
        except (KeyError, ValueError) as exc:
            model_registry.set_model_status(entry.model_id, "REGISTERED", detail=f"必須フィールド検証失敗: {exc}")
            return f"REGISTERED(検証失敗: {exc})"

    @staticmethod
    def _download(entry: ModelCatalogEntry, model_path: str, tags_path: str, max_retries: int = 3) -> None:
        """
        非gatedモデルのみを対象とした自動ダウンロード。
        huggingface_hub の hf_hub_download を使用し、tqdm進捗表示(hf_hub_download標準)+
        3回リトライ(ブループリント方針)を行う。repo_id が未確定("要確認")の場合は
        ダウンロードを試みず、明示的にエラーとして扱う(誤ったURLへの接続を避けるため)。
        """
        if not entry.repo_id or entry.repo_id == "要確認":
            raise RuntimeError(
                f"repo_id が未設定です。README記載の配布元を確認し、"
                f"node_setup.py の MODEL_CATALOG に正しい repo_id を設定してください"
            )
        if not entry.repo_model_filename or not entry.repo_tags_filename:
            raise RuntimeError(
                f"repo_model_filename / repo_tags_filename が未設定です(repo内のサブフォルダ込みパスが必要)。"
                f"node_setup.py の MODEL_CATALOG に設定してください"
            )

        from huggingface_hub import hf_hub_download

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(tags_path), exist_ok=True)

        # 指示書16: モデル+タグファイルのどちらかだけ取得成功する中間状態を避けるため、
        # 両方を一時ファイル(*.tmp<pid>)へコピーしてから、両方揃って初めて最終パスへ
        # アトミックにリネームする(os.replace)。片方だけ失敗した場合は最終パスに
        # 何も残らない(中途半端な取得失敗として次回再試行される)。
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            model_tmp = f"{model_path}.tmp{os.getpid()}"
            tags_tmp = f"{tags_path}.tmp{os.getpid()}"
            try:
                downloaded_model = hf_hub_download(
                    repo_id=entry.repo_id,
                    filename=entry.repo_model_filename,
                )
                downloaded_tags = hf_hub_download(
                    repo_id=entry.repo_id,
                    filename=entry.repo_tags_filename,
                )
                import shutil
                shutil.copy2(downloaded_model, model_tmp)
                shutil.copy2(downloaded_tags, tags_tmp)

                if not os.path.exists(model_tmp) or not os.path.exists(tags_tmp):
                    raise RuntimeError("一時ファイルへのコピー後にファイルが見つかりません(検証失敗)")

                os.replace(model_tmp, model_path)
                os.replace(tags_tmp, tags_path)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                for tmp in (model_tmp, tags_tmp):
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except OSError:
                        pass
                logger.warning(
                    "model_id=%s のダウンロードに失敗(試行%d/%d): %s",
                    entry.model_id, attempt, max_retries, exc,
                )
                time.sleep(min(2 ** attempt, 10))

        raise RuntimeError(f"{max_retries}回のリトライ後も失敗しました: {last_error}")

    @staticmethod
    def _model_status_lines() -> List[str]:
        """
        指示書15: 「登録済み」と「実際に推論可能」を混同しないよう、
        各モデルの状態機械(NOT_INSTALLED〜VALIDATED/FAILED)とprovider情報を一覧表示する。
        VALIDATED/FAILED/provider_status(CUDA_READY/CPU_FALLBACK)は、実際に
        TaggerWorkerHeavyでロードを試みるまでは反映されない(=Setup時点ではLOADABLE止まり)。
        """
        lines = ["## モデル状態"]
        for entry in MODEL_CATALOG:
            status = model_registry.get_model_status(entry.model_id)
            provider_suffix = (
                f", provider={status['provider_status']}" if status["provider_status"] != "UNKNOWN" else ""
            )
            lines.append(f"- {entry.model_id}: status={status['status']}{provider_suffix}")
        return lines

    @staticmethod
    def _vram_status_lines() -> List[str]:
        lines = ["## VRAM使用状況"]
        free_gb = vram_manager.get_free_vram_gb()
        if free_gb is not None:
            lines.append(f"- 実デバイス空きVRAM: {free_gb:.2f} GB (MAX_VRAM_GB設定: {vram_manager.get_max_vram_gb():.2f} GB)")
        else:
            lines.append("- CUDAデバイスが検出できないため、実デバイス空きVRAMは表示できません")

        loaded = get_loaded_models()
        if not loaded:
            lines.append("- 現在ロード中のモデル: なし")
        else:
            lines.append("- 現在ロード中のモデル:")
            for model_id, entry in loaded.items():
                idle_sec = time.time() - entry.last_used_at
                lines.append(f"  - {model_id}: {entry.vram_gb:.2f} GB (最終使用から{idle_sec:.0f}秒)")
        return lines


NODE_CLASS_MAPPINGS = {
    "TaggerWorkerSetup": TaggerWorkerSetup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TaggerWorkerSetup": "Tagger Ensemble Worker Setup",
}
