"""
backends/onnx_backend.py

ONNX Runtime (CUDAExecutionProvider優先、CPUへ自動フォールバック) を用いた
Heavyタガーバックエンド。

実装指示書 リスク評価 #3 (onnxruntime CUDA EPのVRAMリーク) 対策として、
InferenceSession生成時に arena_extend_strategy を明示指定する
(gpu_mem_limitは実機検証で cl_v2 のセッション生成を破壊することが判明したため撤去済み。
モデル単位のVRAM予算は tew_utils.vram_manager.ensure_capacity() が別レイヤーで管理する)。
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from tew_backends.base import ModelBase
from tew_backends.preprocess import MODEL_REGISTRY, load_tag_list, log_preprocess_spec, preprocess

logger = logging.getLogger("ComfyUI_Tagger_Ensemble_Worker")

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore


class OnnxInputInfo:
    """1つのONNX入力について、name/shape/type等をまとめて保持する(指示書 3.1)。"""

    __slots__ = ("name", "shape", "elem_type", "is_pixel_values")

    def __init__(self, name: str, shape: list, elem_type: str):
        self.name = name
        self.shape = shape  # 例: [1, 3, 448, 448] や ['batch', 3, 'height', 'width']
        self.elem_type = elem_type  # 例: "tensor(float)"
        self.is_pixel_values = False

    def __repr__(self) -> str:
        return f"OnnxInputInfo(name={self.name!r}, shape={self.shape!r}, elem_type={self.elem_type!r})"


# --- モデル別の補助input(pixel_values以外)ビルダー登録用レジストリ(指示書 3.4) ---------
# key: model_id, value: (image_shape: tuple[int,int], preprocessed: np.ndarray, input_info: OnnxInputInfo) -> np.ndarray
# 特定モデルの補助入力の意味(padding_mask等)がREADME/モデルカードから確認でき次第、
# ここに専用ビルダーを登録することで _default_auxiliary_input_builder() の汎用推定を上書きできる。
AUXILIARY_INPUT_BUILDERS: Dict[str, Dict[str, Callable]] = {}


def register_auxiliary_input_builder(model_id: str, input_name: str, builder: Callable) -> None:
    AUXILIARY_INPUT_BUILDERS.setdefault(model_id, {})[input_name] = builder


_ORT_DTYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
}


def _resolve_dim(dim, fallback: int) -> int:
    """ONNXのshape要素は int か シンボリック文字列(バッチ次元名等)の場合がある。"""
    if isinstance(dim, int) and dim > 0:
        return dim
    return fallback


def _default_auxiliary_input_builder(
    model_id: str,
    input_info: "OnnxInputInfo",
    pixel_values: np.ndarray,
) -> np.ndarray:
    """
    pixel_values 以外の必須ONNX入力(padding_mask等)に対する汎用フォールバック生成。

    【重要な前提と限界】(指示書 3.3 に対する対応方針)
    このプロジェクトの preprocess() は常に画像をモデルの input_size ぴったりへ
    リサイズ/パディング済みにしてから渡している(可変長入力・動的パディングは行わない)。
    そのため「有効領域=画像全体、パディング領域=無し」という前提のもとでは、
    padding_mask 系の入力は「全要素が"パディングなし"を表す値」で埋めるのが妥当な近似となる。
    ただし maskの値の向き(0=有効か、1=有効か)はモデルによって逆の可能性があるため、
    ONNX入力のdtypeがbool/int系の場合は「全てFalse/0」を採用し(=パディング無し、が
    典型的な慣例)、float系の場合は「全て1.0」(=フルアテンション)を採用する。
    この関数は np.zeros((1, 1)) のような固定形状のごまかしではなく、
    実際にONNXグラフが要求する shape/dtype をそのまま使って生成する。

    この推定が実際のモデル仕様と異なる場合は、モデルカード確認後に
    register_auxiliary_input_builder() で model_id 専用のビルダーを登録して上書きすること。
    """
    shape = input_info.shape
    batch = pixel_values.shape[0]
    img_h = pixel_values.shape[-2] if pixel_values.ndim >= 2 else 1
    img_w = pixel_values.shape[-1] if pixel_values.ndim >= 2 else 1

    resolved_shape = []
    for i, dim in enumerate(shape):
        if i == 0:
            resolved_shape.append(batch)
        elif isinstance(dim, int) and dim > 0:
            resolved_shape.append(dim)
        else:
            # シンボリック次元。pixel_valuesのH/Wと一致しそうな位置ならそれを流用し、
            # それ以外(パッチ数等、次元数から特定できないもの)は1にフォールバックする。
            resolved_shape.append(img_h if i == len(shape) - 2 else (img_w if i == len(shape) - 1 else 1))

    np_dtype = _ORT_DTYPE_TO_NUMPY.get(input_info.elem_type, np.float32)
    if np_dtype in (np.bool_, np.int64, np.int32, np.uint8, np.int8):
        fill_value = False if np_dtype is np.bool_ else 0
    else:
        fill_value = 1.0

    logger.warning(
        "[TEW][ONNX] model_id=%s: 補助入力 '%s' (shape=%s, dtype=%s) をモデル固有仕様不明のため "
        "汎用フォールバック(shape=%s, fill=%r)で生成します。精度に影響する可能性があるため、"
        "配布元のモデルカードで意味を確認し、必要なら onnx_backend.register_auxiliary_input_builder() "
        "でこのmodel_id専用の生成関数を登録してください。",
        model_id, input_info.name, shape, input_info.elem_type, resolved_shape, fill_value,
    )
    return np.full(resolved_shape, fill_value, dtype=np_dtype)


class OnnxBackend(ModelBase):
    """
    ONNX形式のタガーモデル用バックエンド。

    Args:
        model_id: MODEL_REGISTRY / VRAM_TABLE のキーと一致させること。
        model_path: .onnx ファイルへのパス。
        tags_path: タグリスト(.txt または .csv)へのパス。
        apply_sigmoid: モデル出力がロジット(sigmoid未適用)の場合True。
                       モデルグラフに既にSigmoidが含まれる場合はFalseにする。
        device: 実行デバイスの希望("auto"|"cuda"|"directml"|"cpu"、大文字小文字は問わない)。
                "auto"(既定)は従来通りCUDA優先→CPUフォールバック。
                "cuda"/"directml"は該当EPが利用できない場合CPUへフォールバックし警告ログを出す
                (onnxruntimeは1venvにつき1EPビルドしか持てないため、要求EPが無いのは異常では
                なく設定ミスの可能性が高いという前提で、明確に警告する)。
        device_id: CUDAExecutionProvider使用時のCUDAデバイスID。
        dml_device_id: DmlExecutionProvider使用時のデバイスID(Windowsのタスクマネージャ
                       「パフォーマンス」タブのGPU番号やDXGIアダプタ列挙順に対応。環境依存のため
                       正しい値は実機のactual_providers/選択結果ログで確認すること)。
    """

    # onnxruntimeのCUDA/DirectML EPはtorchのキャッシュアロケータを経由しないため、
    # torch.cuda.memory_allocated()の差分では実際のVRAM使用量を反映できない(指示書5)。
    vram_measurement_kind = "device_free_estimate"

    _VALID_DEVICE_PREFS = ("auto", "gpu", "cpu")
    # CUDAとDirectMLは同一onnxruntimeインストールに同時共存できない(ビルドが別物)ため、
    # ユーザーに「CUDAかDirectMLか」を選ばせる意味が無い。どちらが入っていても
    # "gpu"/"auto"はこの優先順位で自動的に見つかった方を使う。
    _GPU_EP_PRIORITY = ("CUDAExecutionProvider", "DmlExecutionProvider")

    def __init__(
        self,
        model_id: str,
        model_path: str,
        tags_path: str,
        apply_sigmoid: bool = True,
        device: str = "auto",
        device_id: int = 0,
        dml_device_id: int = 0,
    ):
        super().__init__(model_id)
        if model_id not in MODEL_REGISTRY:
            raise KeyError(f"backends/preprocess.py の MODEL_REGISTRY に '{model_id}' の前処理仕様がありません")
        self.model_path = model_path
        self.tags_path = tags_path
        self.apply_sigmoid = apply_sigmoid
        self.device_id = device_id
        self.dml_device_id = dml_device_id

        device_pref = (device or "auto").strip().lower()
        if device_pref not in self._VALID_DEVICE_PREFS:
            raise ValueError(
                f"model_id={model_id}: 未知のdevice指定です: {device!r} "
                f"(許容値: {self._VALID_DEVICE_PREFS})"
            )
        self.device_pref = device_pref

        self.session: Optional["ort.InferenceSession"] = None
        self.tags: List[str] = []
        self._input_name: Optional[str] = None
        self._input_is_nchw: bool = False
        self._inputs: List[OnnxInputInfo] = []
        self._pixel_input: Optional[OnnxInputInfo] = None
        self.actual_providers: List[str] = []
        self._last_raw_output_dim: Optional[int] = None

    # -- ModelBase実装 -------------------------------------------------

    def _do_load(self) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime がインストールされていません")

        providers: List[str] = []
        provider_options: List[dict] = []

        available = ort.get_available_providers()

        def _add_cuda() -> None:
            providers.append("CUDAExecutionProvider")
            provider_options.append(
                {
                    "device_id": self.device_id,
                    # 実機でcl_v2(語彙数106536、cl_v1の約2倍)のみ再現する
                    # "Available memory of 4166656 is smaller than requested bytes of 19832832"
                    # のような失敗は、実際の空きVRAM(GB単位)からするとあり得ないほど小さい値であり、
                    # 真のVRAM枯渇ではなくBFCArenaの断片化が原因と判断した。
                    # kSameAsRequested(リクエストされた分だけ確保する控えめな戦略)は、
                    # cl_v2のような巨大な分類ヘッドを持つモデルで多様なサイズの確保要求が
                    # 連続すると、使用済みブロックの隙間に次のリクエストが収まる連続空き領域が
                    # 無くなる断片化を起こしやすい。onnxruntime本来のデフォルトである
                    # kNextPowerOfTwo(次の2の冪乗サイズで多めに確保し断片化を避ける)に変更する。
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    # 【実機で撤去】gpu_mem_limit(MAX_VRAM_GBの50%、例: 2GB)を明示指定していたが、
                    # 実機検証で「gpu_mem_limitを外した素のCUDAExecutionProviderならcl_v2の
                    # セッション生成が成功する」ことが確認された。kNextPowerOfTwoは次の2の冪乗まで
                    # 多めに確保しようとするため、cl_v2のような巨大な分類ヘッドを持つモデルでは
                    # 実際に必要な量がこの上限を超えてセッション生成自体が失敗していたと判断した。
                    # モデル単位のVRAM予算は既にtew_utils.vram_manager.ensure_capacity()
                    # (LRUエビクション)が別レイヤーで管理しているため、CUDA EPのアリーナ自体に
                    # 追加でハード上限をかけるのは二重制御であり、cl_v2のように必要量が読みにくい
                    # モデルではエビクション予算より先にアリーナ確保自体が壊れるリスクの方が大きい。
                    # PyTorch(ComfyUI本体)はcudaMallocAsyncで独自にVRAMプールを管理しており、
                    # onnxruntimeのCUDA EP(別のBFCArena)と同じGPUコンテキストを共有する際、
                    # cuDNNの畳み込みアルゴリズム探索が「使える最大ワークスペース」を要求すると、
                    # その瞬間の断片化した空き領域と噛み合わずBFCArenaの確保に失敗することがある。
                    # cudnn_conv_use_max_workspaceを無効化し、保守的なワークスペースサイズで
                    # アルゴリズムを選ばせることでこれも合わせて回避する。
                    "cudnn_conv_use_max_workspace": "0",
                    "cudnn_conv_algo_search": "HEURISTIC",
                }
            )

        def _add_directml() -> None:
            providers.append("DmlExecutionProvider")
            provider_options.append({"device_id": self.dml_device_id})

        _GPU_EP_ADDER = {"CUDAExecutionProvider": _add_cuda, "DmlExecutionProvider": _add_directml}

        def _try_add_first_available_gpu_ep() -> Optional[str]:
            """_GPU_EP_PRIORITY の順に available をチェックし、最初に見つかったGPU EPを追加する。
            見つかった場合はそのEP名を、無ければNoneを返す。"""
            for ep_name in self._GPU_EP_PRIORITY:
                if ep_name in available:
                    _GPU_EP_ADDER[ep_name]()
                    return ep_name
            return None

        selected_gpu_ep: Optional[str] = None
        if self.device_pref == "cpu":
            pass  # 下で無条件にCPUExecutionProviderが追加される(GPU EPの初期化自体を試みない)
        elif self.device_pref == "gpu":
            selected_gpu_ep = _try_add_first_available_gpu_ep()
            if selected_gpu_ep is None:
                logger.warning(
                    "model_id=%s: device='GPU' が指定されましたが、CUDA/DirectMLのいずれのEPも"
                    "利用できません(onnxruntime-gpu/onnxruntime-directml未導入の可能性)。"
                    "CPUへフォールバックします。available_providers=%s",
                    self.model_id, available,
                )
        else:  # auto: GPU EPが使えれば使う、無ければ静かにCPU(後方互換の緩やかな挙動)
            selected_gpu_ep = _try_add_first_available_gpu_ep()

        providers.append("CPUExecutionProvider")
        provider_options.append({})

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        gpu_ep_requested = any(p != "CPUExecutionProvider" for p in providers)
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers,
                provider_options=provider_options,
            )
        except Exception as exc:
            if not gpu_ep_requested:
                raise  # CPUのみで既に失敗している場合はフォールバック先が無いのでそのまま送出
            # GPU EP(CUDA/DirectML)でのセッション生成自体が失敗するケースは、致命的なモデル不整合
            # だけでなく、他プロセス(ComfyUI本体のPyTorch)とのVRAM/アロケータ競合による一時的な
            # 失敗であることも多い(実機でcl_v2のBFCArena確保失敗として観測)。ノード全体を
            # クラッシュさせるより、CPUへ自動フォールバックして処理を継続する方が実用的なため、
            # ここで一度だけCPU-onlyで再試行する。フォールバックした事実はログに明示し、
            # 隠さない(禁止事項: CUDA fallbackを隠す、に抵触しないため)。
            logger.warning(
                "model_id=%s: GPU EP(%s)でのONNXセッション生成に失敗しました。CPUへ自動フォールバック"
                "して再試行します。原因: %s",
                self.model_id, [p for p in providers if p != "CPUExecutionProvider"], exc,
            )
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
                provider_options=[{}],
            )
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        # --- 指示書 4.1: providerを正しく記録・診断する ---------------------------------
        self.actual_providers = list(self.session.get_providers())
        top_provider = self.actual_providers[0] if self.actual_providers else "unknown"

        if self.device_pref == "cpu":
            diagnosis = "CPU明示指定"
        elif selected_gpu_ep is not None and selected_gpu_ep in self.actual_providers:
            diagnosis = f"{selected_gpu_ep}_READY"
        elif selected_gpu_ep is not None:
            diagnosis = f"{selected_gpu_ep} present but session selected {top_provider}"
        else:
            diagnosis = (
                "CUDA/DirectMLのいずれのEPも利用できません "
                "(onnxruntime-gpu/onnxruntime-directml未導入 or Runtime不整合の可能性)"
            )

        try:
            provider_options_actual = self.session.get_provider_options()
        except Exception:
            provider_options_actual = {}

        logger.info(
            "[TEW][ONNX] model_id=%s backend=onnxruntime\n"
            "  requested_device=%s\n"
            "  available_providers=%s\n"
            "  selected_providers=%s\n"
            "  provider_options=%s\n"
            "  diagnosis=%s",
            self.model_id, self.device_pref, available, self.actual_providers, provider_options_actual, diagnosis,
        )

        if self.device_pref != "cpu" and top_provider == "CPUExecutionProvider":
            # 「ロード完了=希望通りのデバイスで動作」の誤表示を避けるため、この時点で明示的に警告する。
            # ユーザー向けの実際の表示は node_heavy.py 側の status 出力でも provider を明示する。
            logger.warning(
                "model_id=%s: 要求device=%s でしたが実際は CPUExecutionProvider にフォールバックしました "
                "(diagnosis=%s)。",
                self.model_id, self.device_pref, diagnosis,
            )

        # --- 指示書 3.1: 全ONNX入力を取得し、pixel_values相当の入力を特定する -----------------
        raw_inputs = self.session.get_inputs()
        self._inputs = [
            OnnxInputInfo(name=meta.name, shape=list(meta.shape), elem_type=meta.type)
            for meta in raw_inputs
        ]

        logger.info(
            "[TEW][ONNX] model_id=%s ONNX inputs:\n%s",
            self.model_id,
            "\n".join(f"  {info.name}: shape={info.shape} dtype={info.elem_type}" for info in self._inputs),
        )

        pixel_input = self._resolve_pixel_input(self._inputs)
        pixel_input.is_pixel_values = True
        self._pixel_input = pixel_input
        self._input_name = pixel_input.name
        # shape例: [1, 3, H, W] (NCHW) か [1, H, W, 3] (NHWC) かを軸から推定する。
        shape = pixel_input.shape
        self._input_is_nchw = len(shape) == 4 and (shape[1] == 3 or shape[1] == "3")

        outputs_meta = self.session.get_outputs()
        logger.info(
            "[TEW][ONNX] model_id=%s ONNX outputs:\n%s",
            self.model_id,
            "\n".join(f"  {o.name}: shape={list(o.shape)} dtype={o.type}" for o in outputs_meta),
        )

        self.tags = load_tag_list(self.tags_path)
        log_preprocess_spec(self.model_id)

    @staticmethod
    def _resolve_pixel_input(inputs: List["OnnxInputInfo"]) -> "OnnxInputInfo":
        """
        複数入力の中から画像本体(pixel_values相当)の入力を特定する。
        指示書3.1の方針どおり「入力は1つだけ」という前提を捨て、名前とshapeの両方から判定する。
        """
        if not inputs:
            raise RuntimeError("ONNXモデルに入力が1つもありません")

        # 1) 名前に "pixel_values" / "input" / "image" を含むものを優先
        for candidate_name in ("pixel_values", "input", "images", "image"):
            for info in inputs:
                if info.name.lower() == candidate_name:
                    return info

        # 2) shapeが4次元で、チャンネル軸(3)を持つものを画像入力とみなす
        for info in inputs:
            if len(info.shape) == 4 and (3 in info.shape or "3" in [str(d) for d in info.shape]):
                return info

        # 3) それでも判別できない場合は先頭を採用し、警告する
        logger.warning(
            "ONNX入力からpixel_values相当の入力を自動判別できませんでした。先頭の入力(%s)を採用します: %s",
            inputs[0].name, [i.name for i in inputs],
        )
        return inputs[0]

    def _get_provider_status(self) -> Optional[str]:
        """指示書15: 実際に選択されたExecution Providerで判定する。"""
        if not self.actual_providers:
            return "UNKNOWN"
        top = self.actual_providers[0]
        if top == "CUDAExecutionProvider":
            return "CUDA_READY"
        if top == "DmlExecutionProvider":
            return "DIRECTML_READY"
        return "CPU_FALLBACK"

    def _do_unload(self) -> None:
        self.session = None  # onnxruntimeはPythonオブジェクト破棄でネイティブリソースを解放する

    def _build_input_feed(self, batched_pixel_values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        指示書 3.2: ONNX input定義を取得済みの self._inputs を基に、pixel_values以外の
        必須入力(padding_mask等)も含めた完全なinput feedを動的に構築する。
        単一入力({self._input_name: batched})決め打ちにしない。
        """
        pixel_dtype = _ORT_DTYPE_TO_NUMPY.get(self._pixel_input.elem_type, np.float32)
        feed: Dict[str, np.ndarray] = {self._input_name: batched_pixel_values.astype(pixel_dtype)}

        custom_builders = AUXILIARY_INPUT_BUILDERS.get(self.model_id, {})
        for input_info in self._inputs:
            if input_info.name == self._input_name:
                continue
            builder = custom_builders.get(input_info.name, _default_auxiliary_input_builder)
            feed[input_info.name] = builder(self.model_id, input_info, batched_pixel_values)

        return feed

    def infer(self, image: Image.Image) -> Dict[str, float]:
        if self.session is None:
            raise RuntimeError(f"モデルが未ロードです: model_id={self.model_id} (先にload()を呼んでください)")

        self._touch()
        _t0 = time.perf_counter()

        arr = preprocess(image, self.model_id)  # (H, W, 3) float32, NHWC
        if self._input_is_nchw:
            arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)
        batched = np.expand_dims(arr, axis=0).astype(np.float32)  # バッチ次元を追加

        feed = self._build_input_feed(batched)
        outputs = self.session.run(None, feed)
        logits = outputs[0][0]  # (num_tags,)

        if self.apply_sigmoid:
            probs = 1.0 / (1.0 + np.exp(-logits))
        else:
            probs = logits

        self._last_raw_output_dim = len(probs)
        if len(probs) != len(self.tags):
            logger.warning(
                "model_id=%s: 出力次元(%d)とタグ数(%d)が一致しません。先頭の一致する範囲のみ使用します",
                self.model_id, len(probs), len(self.tags),
            )

        n = min(len(probs), len(self.tags))
        result = {self.tags[i]: float(probs[i]) for i in range(n)}

        elapsed_ms = (time.perf_counter() - _t0) * 1000.0
        logger.info(
            "[TEW][INFER] model_id=%s input_shapes=%s output_shapes=%s elapsed_ms=%.1f "
            "raw_score_range=[min=%.4f, max=%.4f, mean=%.4f]",
            self.model_id, {k: list(v.shape) for k, v in feed.items()}, [list(o.shape) for o in outputs], elapsed_ms,
            float(np.min(probs)), float(np.max(probs)), float(np.mean(probs)),
        )
        return result
