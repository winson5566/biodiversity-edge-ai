"""Run one vision-only or geo-fused prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from biodiversity_edge_ai.pipeline import BiodiversityPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--vision-model", required=True)
    parser.add_argument("--vision-manifest", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--geo-model")
    parser.add_argument("--geo-manifest")
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument("--date")
    parser.add_argument("--fusion-mode", choices=("bayes", "log_linear"), default="log_linear")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.geo_model) != bool(args.geo_manifest):
        raise SystemExit("--geo-model and --geo-manifest must be supplied together")
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
    with Image.open(Path(args.image)) as image:
        predictions = pipeline.predict(
            image,
            latitude=args.latitude,
            longitude=args.longitude,
            observation_date=args.date,
            k=args.top_k,
        )
    print(json.dumps([prediction.__dict__ for prediction in predictions], indent=2))


if __name__ == "__main__":
    main()
