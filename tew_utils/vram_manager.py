"""
utils/vram_manager.py

Heavyタガーのロード可否判断(空きVRAM確認)とVRAM容量ベースのLRUエビクションを担当する。

実装指示書 リスク評価 #1, #2 の対策をここに集約する:
  #1 VRAM見積もりと実測の乖離
     -> ロード前に torch.cuda.mem_get_info() で実際の空き容量を確認してから判断する。
        固定テーブルの概算値だけで「足りるはず」と決め打ちしない。
  #2 ComfyUI本体のモデル管理との競合
     -> 自前のアンロードだけでなく comfy.model_management.soft_empty_cache() を
        併用し、ComfyUI側が保持しているキャッシュも解放を試みる。

MAX_VRAM_GB はハードコードせず、node_setup.py (Phase 4) のノード入力から
set_max_vram_gb() 経由で変更できるようにする(ユーザー環境ごとにVRAM容量が
異なるため、8GB機と32GB機の双方に対応できるようにする)。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

from tew_utils.file_lock import file_lock

logger = logging.getLogger("ComfyUI_Tagger_Ensemble_Worker")

try:
    import torch
except ImportError:
    torch = None  # type: ignore

# models.json はプロジェクトルート直下に置く想定。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_JSON_PATH = os.path.join(_PROJECT_ROOT, "models.json")

# デフォルト値。ユーザー環境(Anima Sampler使用時に残りVRAM約4GB)を初期値とするが、
# node_setup.py から動的に上書きされることを前提とする。
_MAX_VRAM_GB: float = 4.0

# mem_get_info() 実測が MAX_VRAM_GB より少ない場合に備えた安全マージン(GB)。
# この分は常に他プロセス(Anima Sampler等)用に残し、Heavyタガーには使わせない。
_SAFETY_MARGIN_GB: float = 0.3


def set_max_vram_gb(value: float) -> None:
    """node_setup.py のFLOAT入力から呼び出される想定。"""
    global _MAX_VRAM_GB
    if value <= 0:
        raise ValueError("MAX_VRAM_GB は正の値である必要があります")
    _MAX_VRAM_GB = value
    logger.info("MAX_VRAM_GB を更新しました: %.2f GB", value)


def get_max_vram_gb() -> float:
    return _MAX_VRAM_GB


def get_free_vram_gb() -> Optional[float]:
    """
    torch.cuda.mem_get_info() のラッパー。GPU非搭載/torch未導入環境では None を返す。
    戻り値は (デバイス上の実際の空きVRAM, GB)。
    """
    if torch is None or not torch.cuda.is_available():
        return None
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    return free_bytes / (1024 ** 3)


def ensure_capacity(required_gb: float) -> None:
    """
    required_gb 分のVRAMを確保できるようにする。

    手順:
      1. 現在ロード中のモデル群のVRAM合計 + required_gb が MAX_VRAM_GB を超える場合、
         LRU順(最終使用時刻が古い順)にアンロードして予算内に収める。
      2. 実デバイスの空きVRAM(mem_get_info)が不足している場合も同様にLRUアンロード。
      3. それでも不足する場合は soft_empty_cache() を呼んでから再チェック。
      4. 最終的に確保できなければ RuntimeError を送出する
         (呼び出し側 node_heavy.py はこれを捕捉してComfyUI上にエラー表示する)。

    実測VRAM(record_actual_vram経由)がある場合はそちらを優先し、無ければ概算値を使う。
    """
    from tew_backends.base import VRAM_TABLE, get_loaded_models  # 遅延import(循環回避)

    loaded = get_loaded_models()

    def _current_total_gb() -> float:
        return sum(entry.vram_gb for entry in loaded.values())

    def _evict_lru_until(budget_gb: float) -> None:
        # 最終使用時刻が古い順にソートしてアンロード。
        while _current_total_gb() + required_gb > budget_gb and loaded:
            oldest_id = min(loaded, key=lambda mid: loaded[mid].last_used_at)
            logger.info(
                "VRAM予算超過のためLRUアンロード: model_id=%s (budget=%.2fGB, required=%.2fGB)",
                oldest_id, budget_gb, required_gb,
            )
            loaded[oldest_id].backend.unload()

    # --- ステップ1: 論理予算(MAX_VRAM_GB)ベースのチェック ---
    _evict_lru_until(_MAX_VRAM_GB)

    # --- ステップ2: 実デバイスの空きVRAMチェック ---
    free_gb = get_free_vram_gb()
    if free_gb is not None:
        needed_with_margin = required_gb + _SAFETY_MARGIN_GB
        while free_gb < needed_with_margin and loaded:
            oldest_id = min(loaded, key=lambda mid: loaded[mid].last_used_at)
            logger.warning(
                "実デバイスの空きVRAM不足のためLRUアンロード: model_id=%s (free=%.2fGB, needed=%.2fGB)",
                oldest_id, free_gb, needed_with_margin,
            )
            loaded[oldest_id].backend.unload()
            free_gb = get_free_vram_gb()

        # --- ステップ3: それでも不足していればComfyUI側キャッシュの解放を試みる ---
        if free_gb is not None and free_gb < needed_with_margin:
            _soft_empty_cache()
            free_gb = get_free_vram_gb()

        # --- ステップ4: 最終判定 ---
        if free_gb is not None and free_gb < needed_with_margin:
            raise RuntimeError(
                f"VRAM不足のためモデルをロードできません: "
                f"必要={required_gb:.2f}GB(+安全マージン{_SAFETY_MARGIN_GB:.2f}GB), "
                f"空き={free_gb:.2f}GB。ComfyUI/Animaの他の処理を終了するか、"
                f"MAX_VRAM_GBの設定やロードするモデルを見直してください。"
            )


def _soft_empty_cache() -> None:
    import gc
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except ImportError:
        pass


def persist_vram_table(vram_table: Dict[str, float]) -> None:
    """
    実測VRAM値を models.json に永続化する。
    models.json は Phase 6 で閾値・mAP・ライセンス情報も持つ想定のため、
    既存内容を壊さないよう "vram_gb" キーのみをマージ更新する。

    指示書17: model_registry.set_model_config() と同じ models.json を
    read-modify-writeするため、file_lock() で保護する。
    """
    with file_lock(_MODELS_JSON_PATH):
        data: Dict[str, dict] = {}
        if os.path.exists(_MODELS_JSON_PATH):
            try:
                with open(_MODELS_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("models.json の読み込みに失敗したため、新規作成します", exc_info=True)
                data = {}

        for model_id, vram_gb in vram_table.items():
            entry = data.setdefault(model_id, {})
            entry["vram_gb"] = vram_gb

        tmp_path = _MODELS_JSON_PATH + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp_path, _MODELS_JSON_PATH)  # アトミックな書き換え
        except OSError:
            logger.exception("models.json への書き込みに失敗しました")
