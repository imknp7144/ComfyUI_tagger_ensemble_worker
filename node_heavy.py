"""
node_heavy.py

ComfyUIカスタムノード: 汎用Heavyタガー (TaggerWorkerHeavy)。

役割は「選択した1モデルで画像を推論し、閾値適用済みのタグ文字列を出力すること」に限定する。
複数モデルの重み付け合議・タグ正規化は行わない
(タグの統合は既存カスタムノード ComfyUI-Danbooru-Prompt-Formatter の combine_mode に委譲する。
 出力される `tags` はそのまま同ノードの `wd_tags` 入力に接続できる)。
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image

from tew_backends.base import ModelBase, VRAM_TABLE, get_loaded_models
from tew_backends.onnx_backend import OnnxBackend
from tew_backends.torch_backend import TorchBackend
from tew_backends.preprocess import load_tag_categories
from tew_utils import vram_manager, model_registry

logger = logging.getLogger("ComfyUI_Tagger_Ensemble_Worker")

# tagcomplete/Danbooru互換のカテゴリ値 (preprocess.load_tag_categories() 参照)。
_CATEGORY_CHARACTER = 4
_CATEGORY_COPYRIGHT = 3


class TaggerWorkerHeavy:
    """汎用Heavyタガーノード。model_id COMBOで選択したモデル1件を推論する。"""

    @classmethod
    def INPUT_TYPES(cls):
        model_ids = model_registry.available_model_ids() or list(VRAM_TABLE.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "model_id": (model_ids,),
                "use_compile": ("BOOLEAN", {"default": False}),
                "use_best_threshold": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "threshold_general": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "threshold_character": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01}),
                "threshold_copyright": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_n_raw_scores": ("INT", {"default": 30, "min": 0, "max": 500, "step": 1}),
                # 【注意】新規ウィジェットは必ずここ(既存optional列の末尾)へ追加すること。
                # ComfyUIは保存済みワークフローのwidgets_valuesを位置(インデックス)ベースで
                # 復元するため、requiredへ追加したり既存項目の手前に挿入したりすると、
                # 既に配置済みのノードの値がずれて誤表示される
                # (実機で確認済みの不具合。device欄がdml_device_idの値にずれ込んで数字表示された)。
                "device": (["AUTO", "GPU", "CPU"], {"default": "AUTO"}),
                # cl_v2(語彙数106536)のような巨大語彙モデルでは、閾値を統計的に超えるタグが
                # 数百〜数千件単位で発生しうる(語彙が大きいほど、閾値超えの絶対数が増えるため)。
                # tags出力・ログ表示が際限なく膨らまないよう上限を設ける。0=無制限。
                "max_tags": ("INT", {"default": 200, "min": 0, "max": 5000, "step": 10}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tags", "raw_scores_json")
    FUNCTION = "run"
    CATEGORY = "Tagger Ensemble Worker/Heavy"

    def run(
        self,
        image,
        model_id: str,
        use_compile: bool,
        use_best_threshold: bool,
        device: str = "AUTO",
        threshold_general: float = 0.35,
        threshold_character: float = 0.60,
        threshold_copyright: float = 0.50,
        top_n_raw_scores: int = 30,
        max_tags: int = 200,
    ):
        pil_image = self._tensor_to_pil(image)

        config = model_registry.get_model_config(model_id)
        backend = self._get_or_create_backend(model_id, config, use_compile, device)

        vram_manager.ensure_capacity(backend.vram_weight)
        backend.load()

        probs: Dict[str, float] = backend.infer(pil_image)

        threshold_for = self._build_threshold_fn(
            config=config,
            use_best_threshold=use_best_threshold,
            threshold_general=threshold_general,
            threshold_character=threshold_character,
            threshold_copyright=threshold_copyright,
            tags_path=config["tags_path"],
            expected_tag_count=len(probs),
        )

        kept = [(tag, p) for tag, p in probs.items() if p >= threshold_for(tag)]
        kept.sort(key=lambda kv: kv[1], reverse=True)

        # 常時ログ(件数のみ、タグ内容は出さない)。「ログが大量のタグで埋まる」症状が
        # 実際にはtags出力自体の肥大化なのか、閾値ロジックの不具合なのかを、タグ本文を
        # 出さずに即座に切り分けられるようにする。
        logger.info(
            "[TEW][THRESHOLD] model_id=%s kept=%d/%d use_best_threshold=%s",
            model_id, len(kept), len(probs), use_best_threshold,
        )

        truncated = False
        if max_tags > 0 and len(kept) > max_tags:
            truncated = True
            logger.warning(
                "[TEW][THRESHOLD] model_id=%s: 閾値通過タグ数(%d)がmax_tags(%d)を超えたため、"
                "スコア上位%d件のみを出力します。cl_v2のような巨大語彙モデルでは、閾値に対して"
                "統計的に多数のタグが超過することがあります。use_best_threshold=Trueを試すか、"
                "threshold_generalを上げることを検討してください。",
                model_id, len(kept), max_tags, max_tags,
            )
            kept = kept[:max_tags]

        tags_str = ", ".join(tag for tag, _p in kept)

        if not kept and probs:
            # 「raw scoreは出るがtag出力が空」というケース(oppai_v11で実際に報告された症状)を
            # ログだけで即座に診断できるようにする。best_threshold自体が実際のスコア分布に対して
            # 到達不可能な値になっていないか(apply_sigmoidの二重適用等)を切り分けるための情報。
            max_tag, max_score = max(probs.items(), key=lambda kv: kv[1])
            applied_threshold = threshold_for(max_tag)
            logger.warning(
                "[TEW][THRESHOLD] model_id=%s: 閾値を超えたタグが0件でした。"
                "最高スコア=%.4f(tag=%s) に対して適用閾値=%.4f。"
                "最高スコアが閾値に対して構造的に低すぎる場合、apply_sigmoidの二重適用や"
                "best_thresholdの値がこのモデルの出力スケールと合っていない可能性があります"
                "(config: apply_sigmoid=%s, best_threshold=%s)。",
                model_id, max_score, max_tag, applied_threshold,
                config.get("apply_sigmoid"), config.get("best_threshold"),
            )

        raw_scores_json = self._build_raw_scores_json(probs, top_n_raw_scores)

        return (tags_str, raw_scores_json)

    # -- 内部ユーティリティ ---------------------------------------------

    @staticmethod
    def _tensor_to_pil(image) -> Image.Image:
        """
        ComfyUIのIMAGE型(torch.Tensor, shape=[B,H,W,C], 値域0-1)をPIL.Imageに変換する。

        【MVP実装の制約】バッチの先頭1枚のみを処理する。ComfyUIの標準ノードはバッチ全件を
        並列処理して複数出力を返すのが通例だが、STRING出力を安全にバッチ化するには
        ComfyUIの LIST 出力機構との整合を別途設計する必要があるため、初期実装では
        「1画像=1回のノード実行」を前提とし、バッチ対応は次フェーズ以降の課題とする。
        """
        if hasattr(image, "shape") and len(image.shape) == 4 and image.shape[0] > 1:
            logger.warning(
                "バッチサイズ%dが入力されましたが、現バージョンは先頭の1枚のみを処理します",
                image.shape[0],
            )
        first = image[0]
        arr = first.detach().cpu().numpy() if hasattr(first, "detach") else np.asarray(first)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _get_or_create_backend(
        model_id: str, config: dict, use_compile: bool, device: str,
    ) -> ModelBase:
        """
        リスク評価#4対策: ノードが再インスタンス化されても、共有レジストリ
        (backends.base._loaded_models) に既存のバックエンドがあればそれを再利用し、
        毎回ロードし直すことを避ける。
        """
        device_pref = (device or "AUTO").strip().lower()

        loaded = get_loaded_models()
        existing = loaded.get(model_id)
        if existing is not None:
            # 指示書12 (use_compile cache) と同じ方針(Option A): compile設定・device設定は
            # backendインスタンスの属性として固定し、ロード済みモデルへの以後の変更は無視する。
            # VRAM面を考慮した単純な実装とするため、cache keyへの追加(Option B)は採らない。
            existing_use_compile = getattr(existing.backend, "use_compile", None)
            if existing_use_compile is not None and existing_use_compile != use_compile:
                logger.warning(
                    "model_id=%s は既に use_compile=%s でロード済みのため、"
                    "今回指定された use_compile=%s は無視されます"
                    "(設定を変えたい場合は先にモデルをアンロードしてください)",
                    model_id, existing_use_compile, use_compile,
                )
            existing_device_pref = getattr(existing.backend, "device_pref", None)
            if existing_device_pref is not None and existing_device_pref != device_pref:
                logger.warning(
                    "model_id=%s は既に device=%s でロード済みのため、"
                    "今回指定された device=%s は無視されます"
                    "(設定を変えたい場合は先にモデルをアンロードしてください)",
                    model_id, existing_device_pref, device_pref,
                )
            return existing.backend

        backend_type = config["backend"]
        if backend_type == "onnx":
            return OnnxBackend(
                model_id=model_id,
                model_path=config["model_path"],
                tags_path=config["tags_path"],
                apply_sigmoid=config.get("apply_sigmoid", True),
                device=device_pref,
                # dml_device_id: DirectML明示利用時のみ意味を持つが、CUDA/DirectMLは同一venvで
                # 共存できず現状は使われていない(ノードのUIウィジェットとしても廃止済み)。
                # OnnxBackend側のデフォルト(0)をそのまま使う。将来DirectMLを使う際は
                # OnnxBackendのコンストラクタ引数として直接渡すか、ウィジェットを再度追加すること。
            )
        if backend_type == "torch":
            return TorchBackend(
                model_id=model_id,
                timm_model_name=config["timm_name"],
                weights_path=config["model_path"],
                tags_path=config["tags_path"],
                use_compile=use_compile,
                device_pref=device_pref,
            )
        raise ValueError(f"model_id={model_id}: 未知のbackend種別です: {backend_type!r}")

    @staticmethod
    def _build_threshold_fn(
        config: dict,
        use_best_threshold: bool,
        threshold_general: float,
        threshold_character: float,
        threshold_copyright: float,
        tags_path: str,
        expected_tag_count: Optional[int] = None,
    ) -> Callable[[str], float]:
        """
        タグごとの適用閾値を返す関数を構築する。

        - use_best_threshold=True: models.json記載の best_threshold (モデル全体で単一値) を使う。
        - use_best_threshold=False: タグのカテゴリ(character/copyright/その他=general)に応じて
          ノード入力のスライダー値を使い分ける。カテゴリ情報が無いタグリスト(.txt形式等)の場合は
          全て general 扱いにフォールバックする。
        """
        if use_best_threshold:
            flat_threshold = float(config.get("best_threshold", 0.35))
            return lambda _tag: flat_threshold

        categories = load_tag_categories(
            tags_path,
            category_path=config.get("tag_category_path"),
            expected_tag_count=expected_tag_count,
        )

        def _threshold_for(tag: str) -> float:
            category = categories.get(tag)
            if category == _CATEGORY_CHARACTER:
                return threshold_character
            if category == _CATEGORY_COPYRIGHT:
                return threshold_copyright
            return threshold_general

        return _threshold_for

    @staticmethod
    def _build_raw_scores_json(probs: Dict[str, float], top_n: int) -> str:
        """
        上位N件を JSON Lines形式({"tag":...,"score":...}を1行ずつ)で出力する。
        top_n<=0 の場合は空文字列を返す(出力を使わない場合にUIを軽くするため)。

        全件sortedしてから先頭N件を切り出す実装は、cl_v2のような10万語彙級のモデルで
        無駄なO(n log n)コストになる(レビュー指摘、妥当なため採用)。heapq.nlargestは
        top_nが全体よりずっと小さい場合に効率的。
        """
        top_n = max(0, int(top_n))
        if top_n == 0:
            return ""
        import heapq
        ranked = heapq.nlargest(top_n, probs.items(), key=lambda kv: kv[1])
        lines = [json.dumps({"tag": tag, "score": round(score, 4)}, ensure_ascii=False) for tag, score in ranked]
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "TaggerWorkerHeavy": TaggerWorkerHeavy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TaggerWorkerHeavy": "Tagger Worker (Heavy)",
}
