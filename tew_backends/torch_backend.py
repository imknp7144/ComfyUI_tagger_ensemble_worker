"""
backends/torch_backend.py

timm(DINOv3/EVA02/ConvNeXt等)ベースのHeavyタガーバックエンド。
ONNX配布のないモデル(dtq_l16/dtq_b16等)向け。

torch.compile はモデル単位でON/OFFを切り替えられるようにする(実装指示書 Phase 2):
  - dtq_b16 のような小型モデルはcompileで高速化が見込める
  - at_convnext_huge はcompileでVRAMが+1GB程度増えるため、デフォルトOFFを推奨
    (実際のデフォルト値はnode_heavy.py側のUIチェックボックスで制御する。
     このバックエンドクラス自体はuse_compileフラグをそのまま受け取るだけ)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from tew_backends.base import ModelBase
from tew_backends.preprocess import MODEL_REGISTRY, load_tag_list, log_preprocess_spec, preprocess

logger = logging.getLogger("ComfyUI_Tagger_Ensemble_Worker")

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    import timm
except ImportError:
    timm = None  # type: ignore


class TorchBackend(ModelBase):
    """
    timm経由でロードするタガーモデル用バックエンド。

    Args:
        model_id: MODEL_REGISTRY / VRAM_TABLE のキーと一致させること。
        timm_model_name: timm.create_model() に渡すアーキテクチャ名
                          (例: "eva02_large_patch14_448.dbv4_full")。
        weights_path: .safetensors または .pth の重みファイルパス。
        tags_path: タグリスト(.txt または .csv)へのパス。
        use_compile: Trueの場合 torch.compile() を適用する。
        device_pref: 実行デバイスの希望("auto"|"gpu"|"cpu"、大文字小文字は問わない)。
                     "auto"(既定)は従来通りCUDA優先→CPUフォールバック。"gpu"は明示要求
                     (CUDA利用不可ならCPUへフォールバックし警告)。旧値"cuda"も"gpu"の
                     同義語として引き続き受け付ける(後方互換)。
                     PyTorchのDirectMLサポート(torch-directml)は別パッケージが必要で
                     このプロジェクトでは未導入のため、"directml"は現状onnx_backend.py
                     (OnnxBackend)専用。TorchBackendで"directml"を指定した場合は
                     警告ログを出しautoにフォールバックする。
    """

    _VALID_DEVICE_PREFS = ("auto", "gpu", "cpu")
    _LEGACY_ALIASES = {"cuda": "gpu"}  # 旧device値からの後方互換読み替え

    def __init__(
        self,
        model_id: str,
        timm_model_name: str,
        weights_path: str,
        tags_path: str,
        use_compile: bool = False,
        device_pref: str = "auto",
    ):
        super().__init__(model_id)
        if model_id not in MODEL_REGISTRY:
            raise KeyError(f"backends/preprocess.py の MODEL_REGISTRY に '{model_id}' の前処理仕様がありません")
        self.timm_model_name = timm_model_name
        self.weights_path = weights_path
        self.tags_path = tags_path
        self.use_compile = use_compile

        normalized = (device_pref or "auto").strip().lower()
        normalized = self._LEGACY_ALIASES.get(normalized, normalized)
        if normalized == "directml":
            logger.warning(
                "model_id=%s: device='DirectML'が指定されましたが、TorchBackendはtorch-directml"
                "未導入のため未対応です。autoにフォールバックします"
                "(DirectMLはONNXバックエンドのモデルでのみ現在サポートされています)。",
                model_id,
            )
            normalized = "auto"
        if normalized not in self._VALID_DEVICE_PREFS:
            raise ValueError(
                f"model_id={model_id}: 未知のdevice指定です: {device_pref!r} "
                f"(許容値: {self._VALID_DEVICE_PREFS}, 'directml'はautoへ読み替え)"
            )
        self.device_pref = normalized

        self.model = None
        self.tags: List[str] = []
        self.device: Optional["torch.device"] = None
        self._compiled = False
        self._last_raw_output_dim: Optional[int] = None
        self._load_missing_keys: List[str] = []
        self._load_unexpected_keys: List[str] = []

    # -- ModelBase実装 -------------------------------------------------

    def _do_load(self) -> None:
        if torch is None:
            raise RuntimeError("torch がインストールされていません")
        if timm is None:
            raise RuntimeError("timm がインストールされていません")

        self.tags = load_tag_list(self.tags_path)

        model = timm.create_model(self.timm_model_name, pretrained=False, num_classes=len(self.tags))
        state_dict = self._load_state_dict(self.weights_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        self._load_missing_keys = list(missing)
        self._load_unexpected_keys = list(unexpected)

        # 指示書11: strict=Falseで進めた場合でも、重大なmissing/unexpected keyがあれば
        # 単なるwarningで推論続行させず、ロード失敗として扱う。
        # shape不一致自体はPyTorchがload_state_dict内で例外を送出するため既にここで捕捉される
        # (strict=Falseはキーの有無のみを緩和し、shape不一致までは許容しない)が、
        # キー数が多い場合はアーキテクチャ名と重みファイルの対応自体が誤っている可能性が高いため
        # 明示的な閾値でロード失敗扱いにする。
        _MAX_TOLERABLE_MISMATCHED_KEYS = 5
        if len(missing) > _MAX_TOLERABLE_MISMATCHED_KEYS or len(unexpected) > _MAX_TOLERABLE_MISMATCHED_KEYS:
            raise RuntimeError(
                f"model_id={self.model_id}: state_dictの不一致が閾値を超えています "
                f"(missing={len(missing)}, unexpected={len(unexpected)})。"
                f"timm_model_name='{self.timm_model_name}' と weights_path='{self.weights_path}' の"
                f"対応関係が誤っている可能性が高いため、ロードを中止します。"
                f"missing例={missing[:5]} unexpected例={unexpected[:5]}"
            )
        if missing or unexpected:
            logger.warning(
                "model_id=%s: state_dictの一部が一致しませんでした (missing=%d, unexpected=%d)。"
                "アーキテクチャ名や重みファイルの対応関係を確認してください。missing=%s unexpected=%s",
                self.model_id, len(missing), len(unexpected), missing, unexpected,
            )
        model.eval()

        cuda_available = torch.cuda.is_available()
        if self.device_pref == "cpu":
            self.device = torch.device("cpu")
        elif self.device_pref == "gpu":
            if cuda_available:
                self.device = torch.device("cuda")
            else:
                logger.warning(
                    "model_id=%s: device='GPU'が指定されましたがCUDAが利用できません"
                    "(torch.cuda.is_available()=False)。CPUへフォールバックします。",
                    self.model_id,
                )
                self.device = torch.device("cpu")
        else:  # auto: 既存の優先順位(CUDA優先)を維持(後方互換)
            self.device = torch.device("cuda" if cuda_available else "cpu")

        model.to(self.device)

        logger.info(
            "[TEW][TORCH] model_id=%s requested_device=%s actual_device=%s cuda_available=%s",
            self.model_id, self.device_pref, self.device, cuda_available,
        )

        if self.use_compile:
            try:
                model = torch.compile(model)
                self._compiled = True
            except Exception:
                logger.warning("model_id=%s: torch.compile()に失敗したため通常モードで続行します", self.model_id, exc_info=True)
                self._compiled = False

        self.model = model
        log_preprocess_spec(self.model_id)

    def _get_provider_status(self) -> Optional[str]:
        """指示書15: 実際にモデルがCUDAデバイスへ配置されたかどうかで判定する。"""
        if self.device is None:
            return "UNKNOWN"
        return "CUDA_READY" if self.device.type == "cuda" else "CPU_FALLBACK"

    def _do_unload(self) -> None:
        if self.model is not None and torch is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
        self.model = None
        self._compiled = False

    def infer(self, image: Image.Image) -> Dict[str, float]:
        if self.model is None or torch is None:
            raise RuntimeError(f"モデルが未ロードです: model_id={self.model_id} (先にload()を呼んでください)")

        self._touch()
        _t0 = time.perf_counter()

        arr = preprocess(image, self.model_id)  # (H, W, 3) float32, NHWC
        arr = np.transpose(arr, (2, 0, 1))  # (3, H, W) NCHW
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        self._last_raw_output_dim = len(probs)
        n = min(len(probs), len(self.tags))
        if len(probs) != len(self.tags):
            logger.warning(
                "model_id=%s: 出力次元(%d)とタグ数(%d)が一致しません。先頭の一致する範囲のみ使用します",
                self.model_id, len(probs), len(self.tags),
            )
        result = {self.tags[i]: float(probs[i]) for i in range(n)}

        elapsed_ms = (time.perf_counter() - _t0) * 1000.0
        logger.info(
            "[TEW][INFER] model_id=%s input_shapes=%s output_shapes=%s elapsed_ms=%.1f "
            "raw_score_range=[min=%.4f, max=%.4f, mean=%.4f]",
            self.model_id, list(tensor.shape), list(logits.shape), elapsed_ms,
            float(np.min(probs)), float(np.max(probs)), float(np.mean(probs)),
        )
        return result

    # -- 内部ユーティリティ ---------------------------------------------

    @staticmethod
    def _load_state_dict(weights_path: str):
        if weights_path.lower().endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(weights_path)
        return torch.load(weights_path, map_location="cpu")
