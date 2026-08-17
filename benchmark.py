"""
benchmark.py

複数のHeavyタガーモデルを同一画像セットに対して実行し、
推論時間・VRAM使用量・OOM発生率・出力タグを比較するための単体スクリプト。
ComfyUIノードではなく、コマンドラインから直接実行する(セットアップ後の動作確認用)。

使い方:
    python benchmark.py --images_dir path/to/test_images [--model_ids cl_v2,wd_eva02_l] [--repeat 3]

models.json に登録済みの(= TaggerWorkerSetupノードで一度セットアップ済みの)モデルが対象。
--model_ids を省略すると登録済みの全モデルを対象にする。
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tew_backends.base import VRAM_TABLE, ModelBase
from tew_backends.onnx_backend import OnnxBackend
from tew_backends.torch_backend import TorchBackend
from tew_utils import model_registry, vram_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

try:
    import torch
except ImportError:
    torch = None  # type: ignore


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass
class ModelBenchmarkResult:
    model_id: str
    load_time_sec: Optional[float] = None
    load_error: Optional[str] = None
    vram_gb: Optional[float] = None
    infer_times_sec: List[float] = field(default_factory=list)
    oom_count: int = 0
    error_count: int = 0
    total_images: int = 0
    sample_tags: dict = field(default_factory=dict)  # {image_filename: [(tag, prob), ...]} 上位5件のみ保持

    @property
    def success_count(self) -> int:
        return len(self.infer_times_sec)

    def summary_row(self) -> dict:
        times = self.infer_times_sec
        return {
            "model_id": self.model_id,
            "vram_gb": round(self.vram_gb, 3) if self.vram_gb is not None else None,
            "load_time_sec": round(self.load_time_sec, 3) if self.load_time_sec is not None else None,
            "infer_avg_sec": round(statistics.mean(times), 4) if times else None,
            "infer_min_sec": round(min(times), 4) if times else None,
            "infer_max_sec": round(max(times), 4) if times else None,
            "success": self.success_count,
            "total": self.total_images,
            "oom_count": self.oom_count,
            "error_count": self.error_count,
            "load_error": self.load_error,
        }


def _load_images(images_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in _IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


def _build_backend(model_id: str) -> ModelBase:
    config = model_registry.get_model_config(model_id)
    backend_type = config["backend"]
    if backend_type == "onnx":
        return OnnxBackend(
            model_id=model_id,
            model_path=config["model_path"],
            tags_path=config["tags_path"],
            apply_sigmoid=config.get("apply_sigmoid", True),
        )
    if backend_type == "torch":
        return TorchBackend(
            model_id=model_id,
            timm_model_name=config["timm_name"],
            weights_path=config["model_path"],
            tags_path=config["tags_path"],
        )
    raise ValueError(f"未知のbackend種別です: {backend_type}")


def _is_oom_error(exc: Exception) -> bool:
    if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def run_benchmark(model_ids: List[str], image_paths: List[str], repeat: int) -> List[ModelBenchmarkResult]:
    results: List[ModelBenchmarkResult] = []

    for model_id in model_ids:
        result = ModelBenchmarkResult(model_id=model_id, total_images=len(image_paths) * repeat)
        logger.info("=== ベンチマーク開始: %s ===", model_id)

        try:
            backend = _build_backend(model_id)
        except Exception as exc:  # noqa: BLE001
            result.load_error = f"バックエンド構築に失敗: {exc}"
            logger.error(result.load_error)
            results.append(result)
            continue

        try:
            vram_manager.ensure_capacity(VRAM_TABLE.get(model_id, 1.0))
            t0 = time.perf_counter()
            backend.load()
            result.load_time_sec = time.perf_counter() - t0
            result.vram_gb = backend.vram_weight
        except Exception as exc:  # noqa: BLE001
            result.load_error = f"モデルロードに失敗: {exc}"
            logger.error(result.load_error)
            results.append(result)
            continue

        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                logger.warning("画像を開けませんでした: %s (%s)", image_path, exc)
                continue

            for _ in range(repeat):
                try:
                    t0 = time.perf_counter()
                    probs = backend.infer(image)
                    elapsed = time.perf_counter() - t0
                    result.infer_times_sec.append(elapsed)

                    if os.path.basename(image_path) not in result.sample_tags:
                        top5 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
                        result.sample_tags[os.path.basename(image_path)] = top5
                except Exception as exc:  # noqa: BLE001
                    if _is_oom_error(exc):
                        result.oom_count += 1
                        logger.warning("OOM発生: model_id=%s image=%s", model_id, image_path)
                    else:
                        result.error_count += 1
                        logger.warning("推論エラー: model_id=%s image=%s error=%s", model_id, image_path, exc)

        backend.unload()
        results.append(result)
        logger.info("=== ベンチマーク終了: %s (成功%d/%d) ===", model_id, result.success_count, result.total_images)

    return results


def _print_summary_table(results: List[ModelBenchmarkResult]) -> None:
    rows = [r.summary_row() for r in results]
    headers = ["model_id", "vram_gb", "load_time_sec", "infer_avg_sec", "infer_min_sec", "infer_max_sec",
               "success", "total", "oom_count", "error_count"]
    widths = {h: max(len(h), max((len(str(row.get(h, ""))) for row in rows), default=0)) for h in headers}

    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))
        if row.get("load_error"):
            print(f"  -> load_error: {row['load_error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Heavyタガーのベンチマーク(推論時間・VRAM・OOM率)")
    parser.add_argument("--images_dir", required=True, help="テスト画像が入ったディレクトリ")
    parser.add_argument("--model_ids", default=None, help="カンマ区切りのmodel_idリスト(省略時は登録済み全モデル)")
    parser.add_argument("--repeat", type=int, default=3, help="各画像あたりの推論回数(既定3回)")
    parser.add_argument("--output_json", default=None, help="結果をJSONで保存するパス(任意)")
    args = parser.parse_args()

    if args.model_ids:
        model_ids = [m.strip() for m in args.model_ids.split(",") if m.strip()]
    else:
        model_ids = model_registry.available_model_ids()

    if not model_ids:
        logger.error("対象モデルがありません。TaggerWorkerSetupノードで先にモデルをセットアップしてください。")
        return

    image_paths = _load_images(args.images_dir)
    if not image_paths:
        logger.error("images_dir に画像が見つかりませんでした: %s", args.images_dir)
        return

    logger.info("対象モデル: %s", model_ids)
    logger.info("テスト画像数: %d, repeat=%d", len(image_paths), args.repeat)

    results = run_benchmark(model_ids, image_paths, args.repeat)

    print()
    _print_summary_table(results)

    if args.output_json:
        payload = {
            "results": [r.summary_row() for r in results],
            "sample_tags": {r.model_id: r.sample_tags for r in results},
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("結果をJSONに保存しました: %s", args.output_json)


if __name__ == "__main__":
    main()
