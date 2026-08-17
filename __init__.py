r"""
__init__.py

ComfyUIカスタムノードパッケージのエントリポイント。
ComfyUI本体はこのファイルの `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` を
読み取ってノードを登録する。

【重要 / 過去の不具合と対策の経緯】
v1: 各モジュールが "from utils import ..." のような汎用的な名前でのトップレベル絶対import
    を使っており、ComfyUI本体自身の "utils" パッケージや他拡張機能の同名パッケージと
    sys.modules 上で衝突し、ノード登録が失敗する不具合があった。

v2: 上記を避けるため相対import("from .backends.base import ...")に切り替えたが、
    ComfyUI環境(特にStabilityMatrix同梱版で確認)によっては、カスタムノードの
    __init__.py をロードする際に `__name__` / `__package__` にモジュール名ではなく
    ファイルパスそのものを設定するローダー実装があり、相対importの解決に失敗する
    (`ModuleNotFoundError: No module named 'C:\...\<フォルダ名>.backends'` のような
    エラーになる)ことが実機で確認された。つまり相対importはComfyUIのローダー実装に
    依存してしまい、確実に動く保証がない。

v3(現在): 上記どちらの問題も避けるため、以下の方式に統一する。
    1. サブパッケージ名を汎用的な "backends"/"utils" ではなく、衝突しない一意な名前
       "tew_backends" / "tew_utils" にリネームする(TEW = Tagger Ensemble Worker)。
    2. このファイル自身のディレクトリを sys.path に追加した上で、
       それらを「トップレベルの絶対import」として読み込む(相対importは使わない)。
    この組み合わせにより、ComfyUIのローダー実装がどうであれ(__name__に何が入ろうが)、
    かつ他の拡張機能やComfyUI本体と同名のパッケージが無い限り、確実に動作する。
"""

from __future__ import annotations

import logging
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

logger = logging.getLogger("ComfyUI_Tagger_Ensemble_Worker")

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

# ノードを実装しているモジュール名の一覧。1つずつimportし、失敗したモジュールがあっても
# 他のノードの登録は継続する(例: 依存パッケージが一部未導入でも、動くノードだけは使えるようにする)。
_NODE_MODULES = [
    "node_heavy",
    "node_setup",
]

for _module_name in _NODE_MODULES:
    try:
        _module = __import__(_module_name)
        NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception:  # noqa: BLE001 — 1モジュールの読み込み失敗で拡張機能全体を落とさないため意図的に広く捕捉
        logger.exception(
            "ComfyUI_Tagger_Ensemble_Worker: モジュール '%s' の読み込みに失敗しました。"
            "このモジュールが提供するノードは利用できません(他のノードは影響を受けません)",
            _module_name,
        )

logger.info(
    "ComfyUI_Tagger_Ensemble_Worker: %d個のノードを登録しました: %s",
    len(NODE_CLASS_MAPPINGS), list(NODE_CLASS_MAPPINGS.keys()),
)

# カスタムJS等は本バージョンでは同梱していない(実装指示書 Phase 3 リスク評価#7参照)。
WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
