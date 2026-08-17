"""
utils/model_registry.py

各Heavyタガーモデルの設定(バックエンド種別・モデルパス・タグリストパス・閾値・
mAP・ライセンス等)を models.json から読み書きする。

models.json は複数の書き込み元を持つ共有ファイル:
  - Phase 1 (vram_manager.persist_vram_table) が "vram_gb" キーを更新
  - Phase 4 (node_setup.py, モデルダウンロード完了時) がパス・閾値等を書き込む
  - ユーザーが手動で編集することも想定する(ダウンロード前に手動でパスを指定するケース等)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from tew_utils.file_lock import file_lock

logger = logging.getLogger("ComfyUI_Tagger_Ensemble_Worker")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_JSON_PATH = os.path.join(_PROJECT_ROOT, "models.json")

# backend種別ごとに models.json のエントリに必須のフィールド。
REQUIRED_FIELDS_BY_BACKEND = {
    "onnx": ("model_path", "tags_path"),
    "torch": ("model_path", "tags_path", "timm_name"),
}

# 指示書15: 「ファイルを発見した(登録済み)」と「実際に推論可能」を混同しないための状態機械。
#   NOT_INSTALLED : ファイル未配置(gatedモデルの手動配置待ち等)
#   INSTALLED     : ファイルは配置されているが、まだmodels.jsonへ登録されていない
#   REGISTERED    : models.jsonへ登録されたが、必須フィールドの検証はまだ行っていない
#   LOADABLE      : 登録内容が get_model_config() の必須フィールド検証を通過した
#                    (=ロードを試みる資格がある状態。実際にロード成功するとは限らない)
#   VALIDATED     : 実際にロード+ダミー推論まで成功した(tew_backends.base.ModelBase.load()が設定)
#   FAILED        : ロード試行が失敗した(tew_backends.base.ModelBase.load()が設定)
# CPU_FALLBACK/CUDA_READY は上記stateとは直交する実行時プロバイダ情報として
# 別フィールド "provider_status" に記録する(状態機械を汚さないため)。
MODEL_STATES = (
    "NOT_INSTALLED",
    "INSTALLED",
    "REGISTERED",
    "LOADABLE",
    "VALIDATED",
    "FAILED",
)
PROVIDER_STATES = ("CPU_FALLBACK", "CUDA_READY", "DIRECTML_READY", "UNKNOWN")


def _load_all() -> Dict[str, dict]:
    if not os.path.exists(_MODELS_JSON_PATH):
        return {}
    try:
        with open(_MODELS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("models.json の読み込みに失敗しました。空の設定として扱います", exc_info=True)
        return {}


def _save_all(data: Dict[str, dict]) -> None:
    tmp_path = _MODELS_JSON_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, _MODELS_JSON_PATH)


def get_model_config(model_id: str) -> Dict[str, Any]:
    """
    model_id の設定を取得する。backend種別に応じた必須フィールドの欠落もここで検査し、
    問題があれば原因が分かるメッセージ付きで例外を送出する
    (ComfyUI上でノード実行時にそのままエラー表示される)。
    """
    data = _load_all()
    entry = data.get(model_id)
    if not entry:
        raise KeyError(
            f"models.json に '{model_id}' の設定がありません。"
            f"Setupノード(node_setup.py)でモデルをセットアップするか、"
            f"models.json に手動でエントリを追加してください。"
        )

    backend = entry.get("backend")
    if backend not in REQUIRED_FIELDS_BY_BACKEND:
        raise ValueError(
            f"model_id={model_id}: backend フィールドが不正です"
            f"('onnx' または 'torch' を指定してください): {backend!r}"
        )

    missing = [f for f in REQUIRED_FIELDS_BY_BACKEND[backend] if not entry.get(f)]
    if missing:
        raise KeyError(f"model_id={model_id}: models.json に必須フィールドが不足しています: {missing}")

    return entry


def set_model_config(model_id: str, **fields: Any) -> None:
    """
    model_id の設定を部分更新する(既存フィールドは維持し、指定分のみ上書き)。
    node_setup.py のダウンロード完了処理から呼び出される想定。

    指示書17: vram_manager.persist_vram_table() 等、他の書き込み元と同じ models.json を
    read-modify-write するため、file_lock() でロックしてから読み直して更新する
    (ロック取得前に読んでいた古いデータで上書きするとlost updateになるため、
    ロック内で再読み込みしてからmergeする)。
    """
    with file_lock(_MODELS_JSON_PATH):
        data = _load_all()
        entry = data.setdefault(model_id, {})
        entry.update(fields)
        _save_all(data)


def available_model_ids() -> List[str]:
    """models.json に設定済みのmodel_id一覧(COMBOのUI選択肢に使う)。"""
    return sorted(_load_all().keys())


def set_model_status(model_id: str, status: str, detail: str = "", provider_status: Optional[str] = None) -> None:
    """
    指示書15: モデルの状態機械を更新する。status は MODEL_STATES のいずれかである必要がある。
    provider_status を渡した場合、PROVIDER_STATES(CPU_FALLBACK/CUDA_READY/DIRECTML_READY/UNKNOWN)として
    別フィールドに記録する(「登録済み」を「利用可能」と誤表示しないため)。
    """
    if status not in MODEL_STATES:
        raise ValueError(f"未知のモデル状態です: {status!r} (許容値: {MODEL_STATES})")
    if provider_status is not None and provider_status not in PROVIDER_STATES:
        raise ValueError(f"未知のprovider状態です: {provider_status!r} (許容値: {PROVIDER_STATES})")

    import time

    fields: Dict[str, Any] = {
        "status": status,
        "status_detail": detail,
        "status_updated_at": time.time(),
    }
    if provider_status is not None:
        fields["provider_status"] = provider_status

    set_model_config(model_id, **fields)


def get_model_status(model_id: str) -> Dict[str, Any]:
    """現在のstatus/provider_statusを取得する。未登録の場合はNOT_INSTALLED扱い。"""
    data = _load_all()
    entry = data.get(model_id, {})
    return {
        "status": entry.get("status", "NOT_INSTALLED"),
        "status_detail": entry.get("status_detail", ""),
        "provider_status": entry.get("provider_status", "UNKNOWN"),
        "status_updated_at": entry.get("status_updated_at"),
    }
