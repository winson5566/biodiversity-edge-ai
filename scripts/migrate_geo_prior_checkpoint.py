"""Convert the legacy FCNet checkpoint to a clean inference-only Keras model."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from biodiversity_edge_ai.pipeline import load_class_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo-repo", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-dim", type=int, default=256)
    args = parser.parse_args()
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for checkpoint migration") from exc

    module_path = Path(args.geo_repo) / "geo_prior" / "models.py"
    spec = importlib.util.spec_from_file_location("legacy_geo_models", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import legacy model module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class DummyRandomGenerator:
        def get_rand_samples(self, batch_size):
            return tf.zeros((batch_size, 6), dtype=tf.float32)

    classes = load_class_names(args.class_map)
    legacy = module.FCNet(
        num_inputs=6,
        embed_dim=args.embedding_dim,
        num_classes=len(classes),
        rand_sample_generator=DummyRandomGenerator(),
        num_users=0,
        use_bn=False,
    )
    legacy(tf.zeros((1, 6), dtype=tf.float32))
    legacy.load_weights(args.checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    legacy.model.save(output)
    print(output)


if __name__ == "__main__":
    main()
