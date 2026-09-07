"""Transfer learning for a directory-based image dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from biodiversity_edge_ai.inference import preprocess_image
from biodiversity_edge_ai.manifest import ModelManifest, class_map_sha256
from biodiversity_edge_ai.models.catalog import BACKBONES, input_scale_for
from biodiversity_edge_ai.models.vision import build_vision_model


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for vision training") from exc
    return tf


def image_dataset(tf: Any, directory: Path, classes: list[str], manifest: ModelManifest,
                  batch_size: int, seed: int, shuffle: bool) -> Any:
    """Use the device's PIL crop, resize and scaling during training as well."""
    paths, labels = [], []
    for label, name in enumerate(classes):
        images = sorted(p for p in (directory / name).glob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
        if not images:
            raise ValueError(f"no images for class {name} in {directory}")
        paths.extend(str(p) for p in images)
        labels.extend([label] * len(images))

    def read(path: bytes) -> np.ndarray:
        with Image.open(path.decode("utf-8")) as image:
            return preprocess_image(image, manifest)

    def decode(path: Any, label: Any) -> tuple[Any, Any]:
        values = tf.numpy_function(read, [path], tf.float32)
        values.set_shape(manifest.input_shape)
        return values, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=seed)
    return dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size).prefetch(tf.data.AUTOTUNE)


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
    tf.keras.utils.set_random_seed(args.seed)
    expected_scale = input_scale_for(args.backbone)
    if args.input_scale not in ("auto", expected_scale):
        raise ValueError(f"{args.backbone} requires --input-scale {expected_scale}")
    args.input_scale = expected_scale
    prepared_directories, prepared_names = load_prepared_classes(args.data_dir)
    images_dir = Path(args.data_dir) / "images" if prepared_directories else Path(args.data_dir)
    directory_names = prepared_directories or sorted(
        p.name for p in (images_dir / "train").iterdir() if p.is_dir())
    class_names = prepared_names or directory_names
    class_map = Path(args.class_map)
    if class_map.exists() and json.loads(class_map.read_text()) != class_names:
        raise ValueError("existing class map differs from prepared dataset")
    manifest = ModelManifest(
        model_id=args.backbone, role="vision", format="keras",
        num_classes=len(class_names), class_map_sha256=class_map_sha256(class_names),
        input_shape=[args.input_size, args.input_size, 3], input_dtype="float32",
        input_scale=args.input_scale,
    )
    train_ds = image_dataset(tf, images_dir / "train", directory_names, manifest,
                             args.batch_size, args.seed, True)
    val_ds = image_dataset(tf, images_dir / "val", directory_names, manifest,
                           args.batch_size, args.seed, False)

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
    head = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs)

    base = model.layers[1]
    base.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.finetune_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    fine = model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    manifest.save(f"{output}.manifest.json")
    Path(f"{output}.history.json").write_text(json.dumps({
        "configuration": vars(args), "head": head.history, "finetune": fine.history,
    }, indent=2, default=str) + "\n", encoding="utf-8")
    class_map.parent.mkdir(parents=True, exist_ok=True)
    class_map.write_text(json.dumps(class_names, ensure_ascii=False, indent=2) + "\n")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--backbone", choices=tuple(BACKBONES), default="mobilenet-v2")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--input-scale", default="auto")
    parser.add_argument("--width-multiplier", type=float, default=1.0)
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
