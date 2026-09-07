"""Teaching-scale transfer learning for a directory-based image dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from biodiversity_edge_ai.models.vision import build_vision_model


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for vision training") from exc
    return tf


def _scale_images(tf: Any, mode: str):
    def apply(images: Any, labels: Any) -> tuple[Any, Any]:
        images = tf.cast(images, tf.float32)
        if mode == "0_1":
            images = images / 255.0
        elif mode == "minus1_1":
            images = images / 127.5 - 1.0
        elif mode == "imagenet":
            images = images / 255.0
            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            images = (images - mean) / std
        return images, labels
    return apply


def load_prepared_classes(data_dir: str | Path) -> tuple[list[str] | None, list[str] | None]:
    """Return directory keys and display names from a prepared dataset manifest."""
    manifest_path = Path(data_dir) / "dataset_manifest.json"
    if not manifest_path.is_file():
        return None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = sorted(manifest["classes"], key=lambda item: int(item["label_id"]))
    label_ids = [int(item["label_id"]) for item in classes]
    if label_ids != list(range(len(classes))):
        raise ValueError("prepared dataset label IDs must be contiguous and zero-based")
    return (
        [str(item["directory"]) for item in classes],
        [str(item["display_name"]) for item in classes],
    )


def train(args: argparse.Namespace) -> Path:
    tf = _tensorflow()
    prepared_directories, prepared_names = load_prepared_classes(args.data_dir)
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
    directory_names = list(train_ds.class_names)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        images_dir / "val",
        image_size=(args.input_size, args.input_size),
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False,
        class_names=prepared_directories,
        crop_to_aspect_ratio=True,
    )
    if list(val_ds.class_names) != directory_names:
        raise ValueError("train and validation class order differs")
    class_names = prepared_names or directory_names
    autotune = tf.data.AUTOTUNE
    scale = _scale_images(tf, args.input_scale)
    train_ds = train_ds.map(scale, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(scale, num_parallel_calls=autotune).prefetch(autotune)

    model = build_vision_model(
        backbone=args.backbone,
        num_classes=len(directory_names),
        input_size=args.input_size,
        alpha=args.width_multiplier,
        weights=None if args.weights == "none" else args.weights,
        trainable=False,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.head_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs)

    base = model.layers[1]
    base.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.finetune_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    class_map = Path(args.class_map)
    class_map.parent.mkdir(parents=True, exist_ok=True)
    class_map.write_text(json.dumps(class_names, ensure_ascii=False, indent=2) + "\n")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--backbone", default="mobilenet-v2")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--input-scale", default="minus1_1")
    parser.add_argument("--width-multiplier", type=float, default=0.5)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    print(train(build_parser().parse_args()))


if __name__ == "__main__":
    main()
