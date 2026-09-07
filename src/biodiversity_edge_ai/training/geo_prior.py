"""Train the six-feature presence-only geo-prior used by the original project."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from biodiversity_edge_ai.metadata import encode_geo_features
from biodiversity_edge_ai.models.geo_prior import build_geo_prior_model


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for geo-prior training") from exc
    return tf


def load_observations(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load columns `latitude`, `longitude`, `date`, and `label_id` from CSV."""
    features: list[np.ndarray] = []
    labels: list[int] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            valid = row.get("valid", "true").strip().lower() not in {"0", "false", "no"}
            if not valid:
                continue
            features.append(
                encode_geo_features(row["latitude"], row["longitude"], row["date"])
            )
            labels.append(int(row["label_id"]))
    if not features:
        raise ValueError(f"no valid observations found in {path}")
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def _random_features(tf: Any, batch_size: Any) -> Any:
    random_values = tf.random.uniform((batch_size, 3), dtype=tf.float32)
    longitude = random_values[:, 0] * 2.0 - 1.0
    theta = tf.acos(random_values[:, 1] * 2.0 - 1.0)
    latitude = 1.0 - 2.0 * theta / np.pi
    date = random_values[:, 2] * 2.0 - 1.0
    return tf.stack(
        [
            tf.sin(np.pi * longitude),
            tf.cos(np.pi * longitude),
            tf.sin(np.pi * latitude),
            tf.cos(np.pi * latitude),
            tf.sin(np.pi * date),
            tf.cos(np.pi * date),
        ],
        axis=1,
    )


def train(args: argparse.Namespace) -> Path:
    tf = _tensorflow()
    features, labels = load_observations(args.observations)
    if labels.min() < 0 or labels.max() >= args.num_classes:
        raise ValueError("label_id lies outside configured class range")
    dataset = (
        tf.data.Dataset.from_tensor_slices((features, labels))
        .shuffle(len(labels), seed=args.seed, reshuffle_each_iteration=True)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    model = build_geo_prior_model(
        num_classes=args.num_classes,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    epsilon = tf.constant(1e-5, dtype=tf.float32)

    for epoch in range(args.epochs):
        losses: list[float] = []
        for batch_features, batch_labels in dataset:
            batch_size = tf.shape(batch_features)[0]
            random_features = _random_features(tf, batch_size)
            targets = tf.one_hot(batch_labels, args.num_classes)
            with tf.GradientTape() as tape:
                combined = tf.concat([batch_features, random_features], axis=0)
                predictions = model(combined, training=True)
                positive = predictions[:batch_size]
                negative = predictions[batch_size:]
                positive_loss = -(
                    args.num_classes * targets * tf.math.log(positive + epsilon)
                    + (1.0 - targets) * tf.math.log(1.0 - positive + epsilon)
                )
                negative_loss = -tf.math.log(1.0 - negative + epsilon)
                loss = tf.reduce_mean(positive_loss) + tf.reduce_mean(negative_loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            losses.append(float(loss))
        print(f"epoch={epoch + 1} loss={np.mean(losses):.6f}")

    if args.validation_observations:
        val_features, val_labels = load_observations(args.validation_observations)
        if val_labels.min() < 0 or val_labels.max() >= args.num_classes:
            raise ValueError("validation label_id lies outside configured class range")
        val_predictions = model.predict(
            val_features, batch_size=args.batch_size, verbose=0
        )
        val_accuracy = float(
            np.mean(np.argmax(val_predictions, axis=1) == val_labels)
        )
        print(f"validation_top1_accuracy={val_accuracy:.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--validation-observations")
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    print(train(build_parser().parse_args()))


if __name__ == "__main__":
    main()
