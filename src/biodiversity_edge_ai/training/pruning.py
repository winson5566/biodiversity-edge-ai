"""Magnitude-pruning utilities for vision model optimization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for pruning") from exc
    return tf


def prunable_weights(model: Any) -> list[Any]:
    return [
        weight
        for weight in model.trainable_weights
        if len(weight.shape) >= 2 and "kernel" in weight.name.lower()
    ]


def build_global_masks(model: Any, sparsity: float) -> list[tuple[Any, Any]]:
    """Build fixed global magnitude masks for kernel tensors."""
    tf = _tensorflow()
    if not 0.0 < sparsity < 1.0:
        raise ValueError("sparsity must be between 0 and 1")
    weights = prunable_weights(model)
    if not weights:
        raise ValueError("model contains no prunable kernel weights")
    magnitudes = tf.concat([tf.reshape(tf.abs(weight), [-1]) for weight in weights], 0)
    count = int(magnitudes.shape[0])
    prune_count = max(1, min(count - 1, int(round(count * sparsity))))
    threshold = tf.sort(magnitudes)[prune_count - 1]
    return [(weight, tf.cast(tf.abs(weight) > threshold, weight.dtype)) for weight in weights]


def apply_masks(masks: list[tuple[Any, Any]]) -> float:
    tf = _tensorflow()
    for weight, mask in masks:
        weight.assign(weight * mask)
    weights = [weight for weight, _ in masks]
    zeros = sum(int(tf.math.count_nonzero(weight == 0)) for weight in weights)
    total = sum(int(tf.size(weight)) for weight in weights)
    return zeros / total


def apply_global_magnitude_pruning(model: Any, sparsity: float) -> float:
    """Set the globally smallest kernel magnitudes to zero in-place."""
    return apply_masks(build_global_masks(model, sparsity))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sparsity", required=True, type=float)
    parser.add_argument("--data-dir")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--input-scale", default="minus1_1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    tf = _tensorflow()
    model = tf.keras.models.load_model(args.keras_model, compile=False)
    masks = build_global_masks(model, args.sparsity)
    achieved = apply_masks(masks)
    if args.finetune_epochs:
        if not args.data_dir:
            raise SystemExit("--data-dir is required when --finetune-epochs is non-zero")
        from biodiversity_edge_ai.training.vision import _scale_images, load_prepared_classes

        prepared_directories, _ = load_prepared_classes(args.data_dir)
        images_dir = Path(args.data_dir) / "images" if prepared_directories else Path(args.data_dir)

        train_ds = tf.keras.utils.image_dataset_from_directory(
            images_dir / "train",
            image_size=(args.input_size, args.input_size),
            batch_size=args.batch_size,
            label_mode="int",
            seed=args.seed,
            class_names=prepared_directories,
            crop_to_aspect_ratio=True,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            images_dir / "val",
            image_size=(args.input_size, args.input_size),
            batch_size=args.batch_size,
            label_mode="int",
            shuffle=False,
            class_names=prepared_directories,
            crop_to_aspect_ratio=True,
        )
        scale = _scale_images(tf, args.input_scale)
        train_ds = train_ds.map(scale, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.map(scale, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        class EnforceMasks(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
                apply_masks(masks)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.finetune_epochs,
            callbacks=[EnforceMasks()],
        )
        achieved = apply_masks(masks)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(f"saved: {output}")
    print(f"kernel sparsity: {achieved:.4f}")
    print("Note: zero weights do not guarantee lower device latency; benchmark the TFLite artifact.")


if __name__ == "__main__":
    main()
