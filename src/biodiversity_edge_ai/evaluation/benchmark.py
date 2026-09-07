"""Benchmark the integrated TFLite pipeline on the target device."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
from PIL import Image

from biodiversity_edge_ai.pipeline import BiodiversityPipeline


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _load_metadata(path: str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["filename"]: row for row in rows}


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads < 1 or args.repetitions < 1 or args.warmup < 0:
        raise ValueError("threads/repetitions must be positive and warmup nonnegative")
    net_power_w = getattr(args, "net_power_w", None)
    if net_power_w is not None and net_power_w <= 0:
        raise ValueError("net power must be positive")
    images_root = Path(args.images)
    images = sorted(
        path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"no images found in {args.images}")
    metadata = _load_metadata(args.metadata_csv)
    rss_before = _rss_mb()
    load_started = time.perf_counter()
    pipeline = BiodiversityPipeline.from_files(
        vision_model=args.vision_model,
        vision_manifest=args.vision_manifest,
        class_map=args.class_map,
        geo_model=args.geo_model,
        geo_manifest=args.geo_manifest,
        num_threads=args.threads,
        fusion_mode=args.fusion_mode,
        alpha=args.alpha,
    )
    rss_after = _rss_mb()
    load_ms = (time.perf_counter() - load_started) * 1000.0

    warmup_path = images[0]
    warmup_row = metadata.get(warmup_path.relative_to(images_root).as_posix(),
                              metadata.get(warmup_path.name, {}))
    warmup_latitude = float(warmup_row["latitude"]) if warmup_row.get("latitude") else None
    warmup_longitude = float(warmup_row["longitude"]) if warmup_row.get("longitude") else None
    warmup_date = warmup_row.get("date") or None
    with Image.open(warmup_path) as image:
        for _ in range(args.warmup):
            pipeline.predict(
                image,
                latitude=warmup_latitude,
                longitude=warmup_longitude,
                observation_date=warmup_date,
                k=1,
            )

    end_to_end: list[float] = []
    invoke: list[float] = []
    correct = 0
    correct_top5 = 0
    labelled = 0
    for _ in range(args.repetitions):
        for path in images:
            relative_name = path.relative_to(images_root).as_posix()
            row = metadata.get(relative_name, metadata.get(path.name, {}))
            latitude = float(row["latitude"]) if row.get("latitude") else None
            longitude = float(row["longitude"]) if row.get("longitude") else None
            date = row.get("date") or None
            location_used = (
                pipeline.geo_runner is not None
                and latitude is not None
                and longitude is not None
                and date is not None
            )
            with Image.open(path) as image:
                started = time.perf_counter()
                predictions = pipeline.predict(
                    image,
                    latitude=latitude,
                    longitude=longitude,
                    observation_date=date,
                    k=5,
                )
                end_to_end.append((time.perf_counter() - started) * 1000.0)
            model_ms = float(getattr(pipeline.vision_runner, "last_inference_ms", 0.0))
            if location_used:
                model_ms += float(getattr(pipeline.geo_runner, "last_inference_ms", 0.0))
            invoke.append(model_ms)
            if row.get("label_id") not in (None, ""):
                labelled += 1
                correct += int(predictions[0].class_id == int(row["label_id"]))
                correct_top5 += int(any(p.class_id == int(row["label_id"]) for p in predictions))

    result: dict[str, Any] = {
        "vision_model": str(args.vision_model),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "load_ms": load_ms,
        "vision_model_id": pipeline.vision_manifest.model_id,
        "vision_optimization": pipeline.vision_manifest.optimization,
        "geo_model": str(args.geo_model) if args.geo_model else None,
        "geo_model_id": pipeline.geo_manifest.model_id if pipeline.geo_manifest else None,
        "geo_optimization": (
            pipeline.geo_manifest.optimization if pipeline.geo_manifest else None
        ),
        "fusion_mode": args.fusion_mode if args.geo_model else None,
        "alpha": args.alpha if args.geo_model else None,
        "threads": args.threads,
        "images": len(images),
        "repetitions": args.repetitions,
        "samples": len(end_to_end),
        "vision_model_bytes": Path(args.vision_model).stat().st_size,
        "geo_model_bytes": Path(args.geo_model).stat().st_size if args.geo_model else 0,
        "rss_delta_mb": (rss_after - rss_before) if None not in (rss_before, rss_after) else None,
        "invoke_median_ms": statistics.median(invoke),
        "invoke_mean_ms": statistics.mean(invoke),
        "invoke_p90_ms": _percentile(invoke, 90),
        "invoke_p95_ms": _percentile(invoke, 95),
        "end_to_end_median_ms": statistics.median(end_to_end),
        "end_to_end_p95_ms": _percentile(end_to_end, 95),
        "end_to_end_images_per_second": 1000.0 / statistics.median(end_to_end),
        "end_to_end_mean_ms": statistics.mean(end_to_end),
        "mean_images_per_second": 1000.0 / statistics.mean(end_to_end),
        "top1_accuracy": correct / labelled if labelled else None,
        "top5_accuracy": correct_top5 / labelled if labelled else None,
        "labelled_samples": labelled,
    }
    if net_power_w is not None:
        result.update({"net_power_w": net_power_w,
                       "energy_per_inference_mj": net_power_w * statistics.mean(invoke),
                       "invoke_fps_per_w": 1000.0 / statistics.mean(invoke) / net_power_w})
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--metadata-csv")
    parser.add_argument("--vision-model", required=True)
    parser.add_argument("--vision-manifest", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--geo-model")
    parser.add_argument("--geo-manifest")
    parser.add_argument("--fusion-mode", choices=("bayes", "log_linear"), default="log_linear")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--net-power-w", type=float,
                        help="measured active-minus-idle power for derived invocation energy")
    return parser


def main() -> None:
    print(json.dumps(benchmark(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
