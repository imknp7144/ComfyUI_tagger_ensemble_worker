"""
backends/preprocess.py

モデルごとの前処理仕様(入力サイズ・正規化パラメータ)をレジストリとして一元管理し、
共通の preprocess() 関数で分岐する(ブループリント 2.1 準拠)。

onnx_backend.py / torch_backend.py の双方から呼ばれる。ここではモデル固有の
「学習時と同じ前処理を再現する」ことだけに責務を絞り、バックエンド固有の
テンソル変換(torch.Tensor化 or numpy ndarrayのまま渡すか)は呼び出し側で行う。

【精度に関する注記】
各モデルの前処理値(mean/std/パディング色/BGR有無等)は、各配布元のREADME記載の
公式推論コードを参照して設定しているが、本サンドボックス環境からは実重み・実画像で
検証できていない(huggingface.co へのネットワークアクセスがない)。
特に at_eva02 / at_convnext_huge の公式前処理は
"PadToSize(白背景で指定サイズまでパディング) -> Resize(bicubic) -> CenterCrop" という
dghs-imgutils 固有のパイプラインだが、ここでは「正方形パディング+直接リサイズ」で
近似している。実運用前に実画像で出力タグの妥当性を確認することを強く推奨する。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("ComfyUI_Tagger_Ensemble_Worker")


@dataclass
class PreprocessSpec:
    input_size: int
    mean: List[float]
    std: List[float]
    rgb: bool = True  # Falseの場合BGR順で並べ替える(WD系タガー対応)
    interpolation: str = "bilinear"  # "bilinear" | "lanczos" | "bicubic"
    dino_v3_norm: bool = False  # DINOv3系(dtqシリーズ)は正規化定数が異なるため明示フラグ化
    pad_to_square: bool = True  # WD/Danbooru系タガーは白背景パディングでアスペクト比を保持するのが通例
    pad_color: Tuple[int, int, int] = (255, 255, 255)  # 一部モデル(oppai_v11等)はレターボックス色がグレー
    rescale_0_1: bool = True  # Falseの場合 0-255 レンジのまま(WD系ONNXグラフは内部で正規化を持つため)


# モデルごとの前処理仕様。実際のモデル追加時はここに1エントリ追加するだけでよい。
#
# 出典 (2026-08 時点でユーザーから提供・確認された配布元READMEに基づく):
#   cl_v2      : https://huggingface.co/cella110n/cl_tagger_v2 (公式推論例に準拠, bicubic, パディングなし)
#   dtq_l16/b16: https://huggingface.co/realphongha/danbooru-tag-query (DINOv3標準ImageNet正規化)
#   oppai_v11  : https://huggingface.co/Grio43/OppaiOracle (V1.1_onnx, letterbox pad=[114,114,114])
#   wd_eva02_l : https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3 (WD系: BGR, 0-255そのまま)
#   at_eva02   : https://huggingface.co/animetimm/eva02_large_patch14_448.dbv4-full (近似値, 上記の注記参照)
#   at_convnext_huge: https://huggingface.co/animetimm/convnextv2_huge.dbv4-full (近似値, 上記の注記参照。入力512)
#   cl_v1      : 配布元READMEに記載なしのため要検証だったが、実機エラー(ONNX Add nodeの
#                broadcast失敗: "730 by 1025")から入力サイズを逆算して特定した。
#                730=27*27+1(patch14, 入力384pxで発生する誤ったパッチ数+CLS)、
#                1025=32*32+1(patch14, 入力448pxで一致するパッチ数+CLS=モデルの位置埋め込み
#                テーブルの実サイズ)。よって正しいinput_sizeは448(cl_v2の384とは異なる)。
MODEL_REGISTRY: Dict[str, PreprocessSpec] = {
    "cl_v2": PreprocessSpec(
        input_size=384, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
        interpolation="bicubic", pad_to_square=False,
    ),
    "cl_v1": PreprocessSpec(  # 【実機エラーから特定】input_size=448(384ではない。上記コメント参照)
        input_size=448, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
        interpolation="bicubic", pad_to_square=False,
    ),
    "dtq_l16": PreprocessSpec(
        input_size=448, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        dino_v3_norm=True, interpolation="bicubic",
    ),
    "dtq_b16": PreprocessSpec(
        input_size=448, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        dino_v3_norm=True, interpolation="bicubic",
    ),
    "oppai_v11": PreprocessSpec(
        input_size=448, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
        pad_color=(114, 114, 114), interpolation="bilinear",
    ),
    "wd_eva02_l": PreprocessSpec(
        input_size=448, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],
        rgb=False, rescale_0_1=False, interpolation="bilinear",  # BGR順・0-255のまま(WD系ONNXグラフの慣例)
    ),
    "at_eva02": PreprocessSpec(  # 近似(本来はPadToSize(512)->Resize(448,bicubic)->CenterCrop)
        input_size=448, mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758],
        interpolation="bicubic",
    ),
    "at_convnext_huge": PreprocessSpec(  # 近似(本来はPadToSize(512)->Resize(512,bicubic)->CenterCrop)。入力512
        input_size=512, mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250],
        interpolation="bicubic",
    ),
}

_RESAMPLE_MAP = {
    "bilinear": Image.BILINEAR,
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
}


def _pad_to_square(img: Image.Image, color: Tuple[int, int, int]) -> Image.Image:
    """指定色の背景で正方形にパディングする(WD/Danbooru系タガー標準の前処理)。"""
    w, h = img.size
    size = max(w, h)
    canvas = Image.new("RGB", (size, size), color)
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    return canvas


def log_preprocess_spec(model_id: str) -> None:
    """
    指示書20: [TEW][PREPROCESS] ログ。モデルロード時に1回、そのモデルへ適用される
    前処理仕様(input_size/dtype/layout/normalization)を出す。
    (推論のたびに出すと冗長なため、_do_load()から1回だけ呼ぶ想定)
    """
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        logger.warning("[TEW][PREPROCESS] model_id=%s: MODEL_REGISTRYに前処理仕様が登録されていません", model_id)
        return
    logger.info(
        "[TEW][PREPROCESS] model_id=%s\n"
        "  input_size=%s\n"
        "  dtype=float32\n"
        "  layout=%s\n"
        "  normalization=mean=%s std=%s rescale_0_1=%s dino_v3_norm=%s\n"
        "  pad_to_square=%s pad_color=%s interpolation=%s",
        model_id, spec.input_size, "RGB" if spec.rgb else "BGR",
        spec.mean, spec.std, spec.rescale_0_1, spec.dino_v3_norm,
        spec.pad_to_square, spec.pad_color, spec.interpolation,
    )


def preprocess(image: Image.Image, model_id: str) -> np.ndarray:
    """
    PIL.Image を受け取り、MODEL_REGISTRY[model_id] の仕様に従って正規化した
    numpy配列 (H, W, C), dtype=float32 を返す(NHWCのままバックエンド側でtranspose有無を判断する)。
    """
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"未登録のモデルIDです: {model_id} (backends/preprocess.py の MODEL_REGISTRY に追加してください)")
    spec = MODEL_REGISTRY[model_id]

    img = image.convert("RGB")
    if spec.pad_to_square:
        img = _pad_to_square(img, spec.pad_color)

    resample = _RESAMPLE_MAP.get(spec.interpolation, Image.BILINEAR)
    img = img.resize((spec.input_size, spec.input_size), resample=resample)

    arr = np.asarray(img, dtype=np.float32)  # (H, W, 3) RGB, 0-255レンジ
    if not spec.rgb:
        arr = arr[:, :, ::-1]  # RGB -> BGR

    if spec.rescale_0_1:
        arr = arr / 255.0

    mean = np.asarray(spec.mean, dtype=np.float32)
    std = np.asarray(spec.std, dtype=np.float32)
    arr = (arr - mean) / std

    return arr


def _read_tag_csv(tags_path: str):
    """
    tagcomplete/WD/DeepGHS系のCSVを読み込む。
    ヘッダに "name"/"tag_name"/"tag" のいずれかがあればその列をタグ名として使う
    (実際のWD/idolsankaku系CSVは "tag_id,name,category,count" の順でid列が先頭に来るため、
    単純に先頭列をタグ名とみなすと誤動作する)。
    ヘッダが無い(=既知の列名が見つからない)場合は、後方互換のため先頭列をタグ名とみなす。

    戻り値: (data_rows, name_idx, category_idx)
    """
    import csv

    with open(tags_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return [], 0, None

    header = [c.strip().lower() for c in rows[0]]
    known_name_keys = {"name", "tag_name", "tag"}
    matched = [k for k in known_name_keys if k in header]

    if matched:
        name_idx = header.index(matched[0])
        category_idx = header.index("category") if "category" in header else None
        data_rows = rows[1:]
    else:
        # ヘッダ行が見つからない単純なリスト(1列目=タグ名)として扱う
        name_idx = 0
        category_idx = None
        data_rows = rows

    return data_rows, name_idx, category_idx


def _load_json_vocab(tags_path: str) -> Any:
    with open(tags_path, "r", encoding="utf-8") as f:
        return json.load(f)


# メタ情報dict(スキーマD)の中からインデックスとして使えそうなキーを探すための候補名。
# 大文字小文字混在に備えて小文字化して照合する。
_INDEX_KEY_CANDIDATES = ("idx", "index", "id", "tag_id", "token_id")

# メタ情報dict(スキーマF)の中からタグ名として使えそうなキーを探すための候補名。
_TAG_NAME_KEY_CANDIDATES = ("tag", "name", "tag_name", "label", "string", "text", "value")


def _log_vocab_shape(vocab: Any, tags_path: str) -> None:
    """
    load_tag_list() 冒頭で呼ぶデバッグログ。実データ構造の把握用であり、
    タグ名そのものは大量に出さず type/件数中心にする(指示書 2.1 の「追加検証」)。
    """
    top_type = type(vocab).__name__
    if isinstance(vocab, dict):
        key_count = len(vocab)
        sample_values = list(vocab.values())[:5]
        value_types = sorted({type(v).__name__ for v in sample_values})
        logger.info(
            "[TEW][TAGS] load_tag_list: path=%s top_level_type=dict key_count=%d "
            "sample_value_types=%s sample_keys=%s",
            tags_path, key_count, value_types, list(vocab.keys())[:5],
        )
        # 値がdictの場合、1件目のメタ情報のキー名一覧も出す(タグ値そのものは出さない)。
        # schema D/Fの判定ミスをログだけで診断できるようにするため(cl_v1で実際に必要になった)。
        #
        # 【実機で確認された不具合】schema A(cl_v2のmodel_vocabulary.json等)は
        # トップレベルが {"tag_to_idx": {全タグ→idx}, "idx_to_tag": {...}, ...} という構造で、
        # "tag_to_idx" の値そのものが「タグ名をキーとする巨大dict」(スキーマD/Fが想定する
        # 「2〜5件のメタ情報」ではなく、実質106536件のタグ名そのもの)である。
        # 修正前はここで sample_values[0].keys() を無条件にログへ出しており、schema Aの
        # 場合にvocab全体(=全タグ名)をログへ吐き出してしまっていた(cl_v2で実機確認)。
        # 常に先頭数件だけに切り詰め、かつ極端に多い場合は省略した旨だけ記録する。
        if sample_values and isinstance(sample_values[0], dict):
            meta_keys = list(sample_values[0].keys())
            _META_KEY_LOG_LIMIT = 20
            if len(meta_keys) > _META_KEY_LOG_LIMIT:
                logger.info(
                    "[TEW][TAGS] load_tag_list: path=%s sample_meta_keys=(%d件のため省略、先頭%d件のみ)%s ...",
                    tags_path, len(meta_keys), _META_KEY_LOG_LIMIT, meta_keys[:_META_KEY_LOG_LIMIT],
                )
            else:
                logger.info(
                    "[TEW][TAGS] load_tag_list: path=%s sample_meta_keys=%s",
                    tags_path, meta_keys,
                )
    elif isinstance(vocab, list):
        sample_types = sorted({type(v).__name__ for v in vocab[:5]})
        logger.info(
            "[TEW][TAGS] load_tag_list: path=%s top_level_type=list length=%d sample_types=%s",
            tags_path, len(vocab), sample_types,
        )
    else:
        logger.info("[TEW][TAGS] load_tag_list: path=%s top_level_type=%s", tags_path, top_type)


def _find_index_in_meta(meta: dict) -> Optional[int]:
    """スキーマD(tag -> メタ情報dict)の1エントリから、順序として使える整数値を探す。"""
    lower_map = {str(k).lower(): v for k, v in meta.items()}
    for key in _INDEX_KEY_CANDIDATES:
        if key in lower_map:
            try:
                return int(lower_map[key])
            except (TypeError, ValueError):
                continue
    return None


def _find_name_key_in_meta(meta: dict) -> Optional[str]:
    """スキーマF(idx -> メタ情報dict)の1エントリから、タグ名として使える実際のキー名を探す。"""
    lower_to_actual = {str(k).lower(): k for k in meta.keys()}
    for key in _TAG_NAME_KEY_CANDIDATES:
        if key in lower_to_actual:
            return lower_to_actual[key]
    return None


def _normalize_vocab_to_tag_list(vocab: Any, tags_path: str) -> List[str]:
    """
    JSONから読み込んだ生データ(vocab)の実際の構造を検出し、モデル出力の並び順に
    対応するタグ名リストへ正規化する。単純な `dict.items()` を無条件にsortしない
    (指示書 2.1 の必須修正/特に禁止)。

    対応スキーマ:
      A) {"idx_to_tag": {"0": "tag0", "1": "tag1", ...}, ...}
         ラッパーキー付きの idx(文字列)->tag。cl_tagger_v2 の model_vocabulary.json 形式。
      B) {"tag0": 0, "tag1": 1, ...}
         tag -> 整数インデックスの単純な辞書。DanbooruTagQuery の tag_to_id.json 形式。
      C) {"0": "tag0", "1": "tag1", ...}
         ラッパーキー無しで、トップレベルが直接 idx(文字列)->tag になっている形式。
      D) {"tag0": {"idx": 0, ...}, "tag1": {"idx": 1, ...}, ...}
         tag -> メタ情報dict。値がdictのため単純な数値ソートができない。
         メタ情報dict内から idx/index/id 等のキーを探して並べ替える。
         インデックスキーが1件でも見つからない場合は、安全側に倒してJSON記載順
         (Python 3.7+ の dict は挿入順を保持する)をそのまま採用し、警告ログを出す。
      E) ["tag0", "tag1", ...]
         トップレベルが文字列のリスト。記載順をそのまま採用する。
      F) {"0": {"name": "tag0", ...}, "1": {"name": "tag1", ...}, ...}
         idx(文字列キー) -> メタ情報dict。cl_v1 の tag_mapping.json の実データがこの形式
         (キーがインデックス、タグ名はdict内のフィールドとして格納されている)。
         メタ情報dict内から tag/name/label 等のキーを探してタグ名として採用し、
         idx(キー)の数値順に並べる。
    """
    _log_vocab_shape(vocab, tags_path)

    if isinstance(vocab, list):
        if vocab and not all(isinstance(v, str) for v in vocab):
            raise ValueError(
                f"タグリストJSONの形式を認識できません(list内に非文字列要素があります): {tags_path}"
            )
        tags = [str(v) for v in vocab]
        logger.info("[TEW][TAGS] schema=E(list) tag_count=%d path=%s", len(tags), tags_path)
        return tags

    if not isinstance(vocab, dict):
        raise ValueError(f"タグリストJSONのトップレベルがdict/listのどちらでもありません: {tags_path}")

    if not vocab:
        raise ValueError(f"タグリストJSONが空です: {tags_path}")

    # スキーマA: ラッパーキー付き idx_to_tag
    if "idx_to_tag" in vocab and isinstance(vocab["idx_to_tag"], dict):
        items = sorted(vocab["idx_to_tag"].items(), key=lambda kv: int(kv[0]))
        tags = [str(tag) for _idx, tag in items]
        logger.info("[TEW][TAGS] schema=A(idx_to_tag) tag_count=%d path=%s", len(tags), tags_path)
        return tags

    sample_values = list(vocab.values())
    value_types = {type(v) for v in sample_values}
    keys_are_numeric = all(str(k).lstrip("-").isdigit() for k in vocab.keys())

    # スキーマC: トップレベルが直接 idx(文字列) -> tag。キーが全て数字でvalueが全て文字列。
    if keys_are_numeric and value_types == {str}:
        items = sorted(vocab.items(), key=lambda kv: int(kv[0]))
        tags = [str(tag) for _idx, tag in items]
        logger.info("[TEW][TAGS] schema=C(idx->tag, no wrapper) tag_count=%d path=%s", len(tags), tags_path)
        return tags

    # スキーマF: idx(文字列キー) -> メタ情報dict。キーが数値でvalueがdictの場合はこちらを
    # 先に判定する(cl_v1のtag_mapping.jsonで実際に観測された形式)。
    # 「キー=タグ名、値=メタ情報」のスキーマDとは主従が逆(キー=インデックス、値の中にタグ名)。
    if keys_are_numeric and value_types <= {dict}:
        sample_meta = sample_values[0] if sample_values else {}
        name_key = _find_name_key_in_meta(sample_meta) if isinstance(sample_meta, dict) else None

        if name_key is None:
            raise ValueError(
                f"タグリストJSON({tags_path})はidx->メタ情報dict形式(schema F)ですが、"
                f"タグ名として使えそうなフィールド({_TAG_NAME_KEY_CANDIDATES})が"
                f"メタ情報内に見つかりません。実際のメタ情報のキー一覧: "
                f"{list(sample_meta.keys()) if isinstance(sample_meta, dict) else sample_meta!r}。"
                f"backends/preprocess.py の _TAG_NAME_KEY_CANDIDATES にこのモデル固有の"
                f"フィールド名を追加してください。"
            )

        items = sorted(vocab.items(), key=lambda kv: int(kv[0]))
        tags = []
        missing_name_count = 0
        for _idx, meta in items:
            if isinstance(meta, dict) and name_key in meta and meta[name_key] is not None:
                tags.append(str(meta[name_key]))
            else:
                missing_name_count += 1
                tags.append(f"__unknown_tag_idx{_idx}__")

        if missing_name_count:
            logger.warning(
                "[TEW][TAGS] schema=F: %d/%d件のエントリで name_key='%s' が欠落していたため、"
                "プレースホルダタグ名を割り当てました(該当タグは実質使用不能です): %s",
                missing_name_count, len(items), name_key, tags_path,
            )
        logger.info(
            "[TEW][TAGS] schema=F(idx->meta dict, name_key='%s') tag_count=%d path=%s",
            name_key, len(tags), tags_path,
        )
        return tags

    # スキーマB: tag -> 整数インデックス
    if value_types <= {int, float}:
        items = sorted(vocab.items(), key=lambda kv: kv[1])
        tags = [str(tag) for tag, _idx in items]
        logger.info("[TEW][TAGS] schema=B(tag->int idx) tag_count=%d path=%s", len(tags), tags_path)
        return tags

    # スキーマD: tag -> メタ情報dict(キーが非数値=タグ名そのもの)。単純な kv[1] ソートはできない。
    if value_types <= {dict}:
        indices: Dict[str, int] = {}
        missing = []
        for tag, meta in vocab.items():
            idx = _find_index_in_meta(meta)
            if idx is None:
                missing.append(tag)
            else:
                indices[tag] = idx

        if not missing:
            ordered = sorted(vocab.keys(), key=lambda t: indices[t])
            tags = [str(t) for t in ordered]
            logger.info(
                "[TEW][TAGS] schema=D(tag->meta dict, index key found) tag_count=%d path=%s",
                len(tags), tags_path,
            )
            return tags

        # インデックスキーが見つからない(=モデル固有の未知フォーマット)場合は、
        # 数値強制キャストでごまかさず、JSON記載順を安全側のフォールバックとして採用する。
        tags = [str(t) for t in vocab.keys()]
        logger.warning(
            "[TEW][TAGS] schema=D だがメタ情報dict内にインデックスキー(%s)が見つかりません "
            "(該当%d/%d件)。JSON記載順をそのままタグ順として採用しますが、モデル出力次元との "
            "対応がずれている可能性があります。手動で %s の実データ形式を確認してください。",
            _INDEX_KEY_CANDIDATES, len(missing), len(vocab), tags_path,
        )
        return tags

    # どのスキーマにも一致しない未知の混在型
    raise ValueError(
        f"タグリストJSONの値の型が認識できる形式(int/dict/文字列idx)のいずれでもありません: "
        f"path={tags_path} value_types={sorted(t.__name__ for t in value_types)}。"
        f"backends/preprocess.py の _normalize_vocab_to_tag_list() にこのモデルのスキーマ対応を追加してください。"
    )


def load_tag_list(tags_path: str, category_path: Optional[str] = None) -> List[str]:
    """
    タグリストを読み込む。以下形式に対応する:
      - .txt: 1行1タグ
      - .csv: ヘッダから name 列を検出して読み込む(WD/idolsankaku/OppaiOracle系の
              "tag_id,name,category,count" 形式含む)
      - .json: 実データ構造を検出して正規化する(_normalize_vocab_to_tag_list() 参照)。
          既知の代表例:
          (a) cl_tagger_v2 の model_vocabulary.json 形式:
              {"idx_to_tag": {"0": "tag0", ...}, "tag_to_category": {...}}
          (b) DanbooruTagQuery の tag_to_id.json 形式:
              {"tag0": 0, "tag1": 1, ...} (タグ名->インデックスの単純な辞書)
          (d) tag -> メタ情報dict 形式。
          (f) idx -> メタ情報dict 形式(cl_v1 の tag_mapping.json の実データがこの形式)。
          category_path を指定すると、(b)/(d)と対になる category ファイル
          (DanbooruTagQueryの tag_category.json 等、{"tag0": category_id, ...}) も
          load_tag_categories() 側で読み込める。

    onnx_backend.py / torch_backend.py の双方から共通で利用する。
    """
    import os

    if not os.path.exists(tags_path):
        raise FileNotFoundError(f"タグリストファイルが見つかりません: {tags_path}")

    if tags_path.lower().endswith(".csv"):
        data_rows, name_idx, _category_idx = _read_tag_csv(tags_path)
        tags = [row[name_idx].strip() for row in data_rows if row and len(row) > name_idx and row[name_idx].strip()]
        logger.info("[TEW][TAGS] schema=CSV tag_count=%d path=%s", len(tags), tags_path)
        return tags

    if tags_path.lower().endswith(".json"):
        vocab = _load_json_vocab(tags_path)
        tags = _normalize_vocab_to_tag_list(vocab, tags_path)
        if not tags:
            raise ValueError(f"タグリストの正規化結果が空になりました: {tags_path}")
        return tags

    with open(tags_path, "r", encoding="utf-8") as f:
        tags = [line.strip() for line in f if line.strip()]
        logger.info("[TEW][TAGS] schema=TXT tag_count=%d path=%s", len(tags), tags_path)
        return tags


# タグカテゴリの文字列ラベル -> 標準カテゴリ番号のマッピング。
# tagcomplete/Danbooru/animetimm互換の番号体系(0=general,1=artist,3=copyright,4=character,5=meta)
# だが、JSON形式のタグリストの中には数値ではなく文字列ラベルでカテゴリを記録しているものがある
# (cl_v1のtag_mapping.jsonで実機確認)。
_CATEGORY_LABEL_TO_ID = {
    "general": 0, "0": 0,
    "artist": 1, "1": 1,
    "copyright": 3, "series": 3, "3": 3,
    "character": 4, "characters": 4, "char": 4, "4": 4,
    "meta": 5, "metadata": 5, "5": 5,
    "rating": 9, "9": 9,
}


def _parse_category_value(raw: Any) -> Optional[int]:
    """
    カテゴリ値を標準カテゴリ番号(int)へ変換する。数値そのもの・数値文字列("3")・
    既知の文字列ラベル("character"等)のいずれにも対応する。認識できない場合はNoneを返す
    (int()に単純に投げて例外で弾くと、文字列ラベル形式を「見つからなかった」扱いにしてしまい、
    cl_v1で全タグがgeneral扱いになる不具合の原因になっていた)。
    """
    if isinstance(raw, bool):  # bool は int のサブクラスなので先に弾く
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if not s:
            return None
        if s in _CATEGORY_LABEL_TO_ID:
            return _CATEGORY_LABEL_TO_ID[s]
    return None


def load_tag_categories(
    tags_path: str,
    category_path: Optional[str] = None,
    expected_tag_count: Optional[int] = None,
) -> dict:
    """
    タグ名 -> カテゴリ番号(int) の辞書を返す。
    category列/情報を持たない形式の場合は空辞書を返す
    (呼び出し側は「未分類 = general扱い」としてフォールバックすること)。

    カテゴリ値の意味(tagcomplete/Danbooru/animetimm互換):
      0=general, 1=artist, 3=copyright(series扱い), 4=character, 5=meta

    category_path が指定されている場合、DanbooruTagQueryの tag_category.json のような
    「tags_path(タグ名一覧)とは別ファイルのカテゴリ情報」を読み込む
    (両ファイルとも {tag: category_id} 形式の単純な辞書を想定)。

    指示書14: categoryファイルが「存在することを期待されているモデル」
    (=category_pathが明示的に指定されている)場合、読み込み失敗を空辞書へ
    サイレントにフォールバックせず、原因(missing/invalid JSON/invalid schema/
    tag count mismatch)を明示した警告ログを必ず出す。
    expected_tag_count を渡すと、カテゴリ件数とタグ総数の乖離も検出できる。
    """
    import os

    if category_path:
        if not os.path.exists(category_path):
            logger.warning(
                "[TEW][TAGS] category_path=missing: 指定されたカテゴリファイルが見つかりません: %s。"
                "カテゴリ情報無し(=全タグgeneral扱い)として続行します。",
                category_path,
            )
            return {}

        try:
            with open(category_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "[TEW][TAGS] category_path=invalid JSON: カテゴリファイルの読み込みに失敗しました: "
                "%s (%s)。カテゴリ情報無しとして続行します。",
                category_path, exc,
            )
            return {}

        if not isinstance(raw, dict):
            logger.warning(
                "[TEW][TAGS] category_path=invalid schema: JSONトップレベルがdictではありません"
                "(型=%s): %s。カテゴリ情報無しとして続行します。",
                type(raw).__name__, category_path,
            )
            return {}

        categories: Dict[str, int] = {}
        invalid_values = 0
        for tag, cat in raw.items():
            try:
                categories[tag] = int(cat)
            except (TypeError, ValueError):
                invalid_values += 1

        if invalid_values:
            logger.warning(
                "[TEW][TAGS] category_path=invalid schema: 整数へ変換できないカテゴリ値が%d件あり、"
                "該当タグをスキップしました: %s",
                invalid_values, category_path,
            )

        if expected_tag_count is not None and len(categories) != expected_tag_count:
            logger.warning(
                "[TEW][TAGS] category_path=tag count mismatch: カテゴリ件数(%d)がタグ総数(%d)と"
                "一致しません: %s。一部タグはカテゴリ未設定(general扱い)になります。",
                len(categories), expected_tag_count, category_path,
            )

        logger.info(
            "[TEW][TAGS] category schema=external_file category_count=%d path=%s",
            len(categories), category_path,
        )
        return categories

    if tags_path.lower().endswith(".json"):
        vocab = _load_json_vocab(tags_path)
        if isinstance(vocab, dict):
            tag_to_category = vocab.get("tag_to_category")
            if isinstance(tag_to_category, dict):
                # cl_v1のschema F修正時と同じ問題がこの直接指定パスにも存在していた:
                # 値が数値ではなく文字列ラベル("general"/"character"等)の場合、単純な
                # int()変換のみだと全件失敗し、辞書全体を空にして握りつぶしていた
                # (cl_v2の実機ログで確認: 106536件全タグがgeneral扱いになる不具合)。
                # _parse_category_value()で1件ずつ解釈し、失敗したエントリだけをスキップする
                # (1件の失敗で全件を捨てない)。
                categories: Dict[str, int] = {}
                unparsed = 0
                unparsed_samples: List[Any] = []
                for tag, cat in tag_to_category.items():
                    parsed = _parse_category_value(cat)
                    if parsed is not None:
                        categories[tag] = parsed
                    else:
                        unparsed += 1
                        if len(unparsed_samples) < 5:
                            unparsed_samples.append(cat)

                if unparsed:
                    logger.warning(
                        "[TEW][TAGS] tag_to_category: %d/%d件の値を解釈できませんでした: %s。"
                        "解釈できなかった生の値の例(先頭%d件): %s。"
                        "_CATEGORY_LABEL_TO_ID にこのモデル固有のラベル表記を追加してください。",
                        unparsed, len(tag_to_category), tags_path, len(unparsed_samples), unparsed_samples,
                    )
                from collections import Counter
                logger.info(
                    "[TEW][TAGS] category schema=tag_to_category category_count=%d value_distribution=%s path=%s",
                    len(categories), dict(Counter(categories.values())), tags_path,
                )
                return categories

            sample_values = list(vocab.values())[:1]
            if sample_values and isinstance(sample_values[0], dict):
                keys_are_numeric = all(str(k).lstrip("-").isdigit() for k in vocab.keys())
                cat_key_candidates = ("category", "category_id", "cat")

                if keys_are_numeric:
                    # スキーマF(idx -> メタ情報dict、cl_v1のtag_mapping.json等)。
                    # 辞書のキーはインデックスでありタグ名ではないため、categories辞書は
                    # メタ情報内のタグ名フィールド(_find_name_key_in_metaで検出)をキーにして
                    # 構築する必要がある。キー=インデックスのままcategories[idx]=cat として
                    # 返すと、実際のタグ名では絶対にヒットせず全タグがgeneral扱いに
                    # なってしまう不具合があった(cl_v1で実機確認)。
                    name_key = _find_name_key_in_meta(sample_values[0])
                    if name_key is None:
                        logger.warning(
                            "[TEW][TAGS] category schema=F だがタグ名フィールドが見つからず、"
                            "カテゴリをタグ名に対応付けできません: %s。カテゴリ情報無し"
                            "(=全タグgeneral扱い)として続行します。",
                            tags_path,
                        )
                        return {}

                    categories: Dict[str, int] = {}
                    missing_cat = 0
                    unparsed_raw_samples: List[Any] = []
                    for _idx, meta in vocab.items():
                        if not isinstance(meta, dict) or name_key not in meta:
                            continue
                        tag_name = str(meta[name_key])
                        lower_map = {str(k).lower(): v for k, v in meta.items()}
                        found = False
                        for key in cat_key_candidates:
                            if key in lower_map:
                                parsed = _parse_category_value(lower_map[key])
                                if parsed is not None:
                                    categories[tag_name] = parsed
                                    found = True
                                elif len(unparsed_raw_samples) < 5:
                                    unparsed_raw_samples.append(lower_map[key])
                                break
                        if not found:
                            missing_cat += 1

                    if missing_cat:
                        logger.warning(
                            "[TEW][TAGS] category schema=F: %d/%d件のタグでcategory値を解釈できませんでした: "
                            "%s。値の型/内容の例(先頭%d件、認識できなかった生の値): %s。"
                            "_CATEGORY_LABEL_TO_ID にこのモデル固有のラベル表記を追加してください。",
                            missing_cat, len(vocab), tags_path, len(unparsed_raw_samples), unparsed_raw_samples,
                        )
                    from collections import Counter
                    logger.info(
                        "[TEW][TAGS] category schema=F(name_key='%s') category_count=%d "
                        "value_distribution=%s path=%s",
                        name_key, len(categories), dict(Counter(categories.values())), tags_path,
                    )
                    return categories

                # スキーマD(tag -> メタ情報dict、キー自体がタグ名)。
                categories = {}
                for tag, meta in vocab.items():
                    if not isinstance(meta, dict):
                        continue
                    lower_map = {str(k).lower(): v for k, v in meta.items()}
                    for key in cat_key_candidates:
                        if key in lower_map:
                            parsed = _parse_category_value(lower_map[key])
                            if parsed is not None:
                                categories[tag] = parsed
                            break
                return categories
        return {}

    if not tags_path.lower().endswith(".csv"):
        logger.info(
            "[TEW][TAGS] category schema=none(no category_path, tags_path is not .csv/.json): "
            "全タグgeneral扱いになります path=%s",
            tags_path,
        )
        return {}

    data_rows, name_idx, category_idx = _read_tag_csv(tags_path)
    if category_idx is None:
        logger.warning(
            "[TEW][TAGS] category schema=csv だが 'category' 列がヘッダに見つかりませんでした: %s。"
            "全タグがgeneral扱いになり、threshold_character/threshold_copyrightが実質無視されます。"
            "CSVの実際のヘッダ行を確認してください(WD/idolsankaku系は "
            "'tag_id,name,category,count' 形式を想定)。",
            tags_path,
        )
        return {}

    categories = {}
    invalid_rows = 0
    for row in data_rows:
        if not row or len(row) <= max(name_idx, category_idx):
            invalid_rows += 1
            continue
        tag = row[name_idx].strip()
        raw_cat = row[category_idx].strip()
        parsed = _parse_category_value(raw_cat) if tag else None
        if parsed is None:
            invalid_rows += 1
            continue
        categories[tag] = parsed

    if invalid_rows:
        logger.warning(
            "[TEW][TAGS] category schema=csv: %d行がcategory列の値を解釈できずスキップされました: %s",
            invalid_rows, tags_path,
        )

    if expected_tag_count is not None and len(categories) != expected_tag_count:
        logger.warning(
            "[TEW][TAGS] category schema=csv: tag count mismatch: カテゴリ件数(%d)がタグ総数(%d)と"
            "一致しません: %s",
            len(categories), expected_tag_count, tags_path,
        )

    # カテゴリ値の分布(0=general等)もログに出す。診断用であり、タグ名そのものは出さない。
    from collections import Counter
    distribution = dict(Counter(categories.values()))
    logger.info(
        "[TEW][TAGS] category schema=csv category_count=%d value_distribution=%s path=%s",
        len(categories), distribution, tags_path,
    )
    return categories
