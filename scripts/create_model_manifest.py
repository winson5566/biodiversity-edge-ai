"""Inspect a TFLite artifact and create its compatibility manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from biodiversity_edge_ai.inference import TFLiteRunner
from biodiversity_edge_ai.manifest import ModelManifest, class_map_sha256
from biodiversity_edge_ai.pipeline import load_class_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--role", choices=("vision", "geo_prior"), required=True)
    parser.add_argument("--input-scale", required=True)
    parser.add_argument("--optimization", required=True)
    parser.add_argument("--source", default="external TFLite artifact")
    args = parser.parse_args()
    runner = TFLiteRunner(args.model)
    shape = [int(value) for value in runner.input_details["shape"][1:]]
    classes = load_class_names(args.class_map)
    manifest = ModelManifest(
        model_id=args.model_id,
        role=args.role,
        format="tflite",
        num_classes=len(classes),
        class_map_sha256=class_map_sha256(classes),
        input_shape=shape,
        input_dtype=np.dtype(runner.input_details["dtype"]).name,
        input_scale=args.input_scale,
        optimization=args.optimization,
        source=args.source,
    )
    manifest.save(args.output)
    print(Path(args.output))


if __name__ == "__main__":
    main()
