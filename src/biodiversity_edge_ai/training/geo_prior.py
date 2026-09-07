"""Train the six-feature presence-only Geo Prior."""

from __future__ import annotations

import argparse
import csv
import json
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


def balanced_epoch_indices(labels: np.ndarray, cap: int, rng: Any) -> np.ndarray:
    """Sample classes with weight min(class_count, cap), then cycle each class.

    Equivalent class weights to the capped repeated-category input streams;
    drawing order is generated with NumPy, not the old TFRecord/JSON loader.
    """
    if cap < 1:
        return rng.permutation(len(labels))
    classes, counts = np.unique(labels, return_counts=True)
    weights = np.minimum(counts, cap)
    chosen = rng.choice(len(classes), size=int(weights.sum()), p=weights / weights.sum())
    indices = np.empty(len(chosen), dtype=np.int64)
    label_order = np.argsort(labels, kind="stable")
    class_ends = np.cumsum(counts)
    class_starts = class_ends - counts
    draw_order = np.argsort(chosen, kind="stable")
    draw_counts = np.bincount(chosen, minlength=len(classes))
    draw_ends = np.cumsum(draw_counts)
    draw_starts = draw_ends - draw_counts
    for class_index, label in enumerate(classes):
        positions = draw_order[draw_starts[class_index]:draw_ends[class_index]]
        candidates = label_order[class_starts[class_index]:class_ends[class_index]]
        samples = []
        while len(samples) < len(positions):
            samples.extend(rng.permutation(candidates).tolist())
        indices[positions] = samples[:len(positions)]
    return indices


def train(args: argparse.Namespace) -> Path:
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    features, labels = load_observations(args.observations)
    if labels.min() < 0 or labels.max() >= args.num_classes:
        raise ValueError("label_id lies outside configured class range")
    model = build_geo_prior_model(
        num_classes=args.num_classes,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    epsilon = tf.constant(1e-5, dtype=tf.float32)
    history = []
    learning_rates = []
    rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        rate = args.learning_rate * args.lr_decay ** epoch
        optimizer.learning_rate.assign(rate)
        learning_rates.append(rate)
        indices = balanced_epoch_indices(labels, args.max_per_class, rng)
        if args.max_per_class > 0 and len(indices) < args.batch_size:
            raise ValueError("balanced geo epoch is smaller than batch_size; reduce geo_batch_size")
        dataset = tf.data.Dataset.from_tensor_slices((features[indices], labels[indices])).batch(
            args.batch_size, drop_remainder=args.max_per_class > 0).prefetch(tf.data.AUTOTUNE)
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
        history.append(float(np.mean(losses)))

    val_accuracy = None
    if args.validation_observations:
        val_features, val_labels = load_observations(args.validation_observations)
        if val_labels.min() < 0 or val_labels.max() >= args.num_classes:
            raise ValueError("validation label_id lies outside configured class range")
        correct = 0
        for start in range(0, len(val_labels), args.batch_size):
            stop = start + args.batch_size
            predictions = model(val_features[start:stop], training=False).numpy()
            correct += int(np.sum(np.argmax(predictions, axis=1) == val_labels[start:stop]))
        val_accuracy = correct / len(val_labels)
        print(f"validation_top1_accuracy={val_accuracy:.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    Path(f"{output}.history.json").write_text(json.dumps({
        "configuration": vars(args), "loss": history, "learning_rates": learning_rates,
        "validation_top1": val_accuracy,
    }, indent=2, default=str) + "\n", encoding="utf-8")
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
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--max-per-class", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    print(train(build_parser().parse_args()))


if __name__ == "__main__":
    main()
