"""Transfer learning for a directory-based image dataset."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from biodiversity_edge_ai.inference import preprocess_image
from biodiversity_edge_ai.manifest import ModelManifest, class_map_sha256
from biodiversity_edge_ai.models.catalog import BACKBONES, input_scale_for
from biodiversity_edge_ai.models.vision import build_vision_model
from biodiversity_edge_ai.training.recipe import (
    configure_trainable_layers, learning_rate_at_step, stages,
)


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for vision training") from exc
    return tf


def image_dataset(tf: Any, directory: Path, classes: list[str], manifest: ModelManifest,
                  batch_size: int, seed: int, shuffle: bool, *, randaug_layers: int = 0,
                  randaug_magnitude: int = 0, categorical: bool = False,
                  drop_remainder: bool = False) -> Any:
    """Use augmentation for early stages; evaluation shares device preprocessing."""
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
        if randaug_layers:
            from biodiversity_edge_ai.training.randaugment import distort_image_with_randaugment
            values = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
            values.set_shape([None, None, 3])
            begin, size, _ = tf.image.sample_distorted_bounding_box(
                tf.shape(values), bounding_boxes=[[[0., 0., 1., 1.]]],
                min_object_covered=0.5, aspect_ratio_range=(0.75, 1.33),
                area_range=(0.08, 1.0), max_attempts=100, use_image_if_no_bounding_boxes=True,
                seed=seed)
            values = tf.slice(values, begin, size)
            values = tf.image.resize(tf.cast(values, tf.float32), manifest.input_shape[:2])
            values = tf.image.random_flip_left_right(values, seed=seed)
            values = distort_image_with_randaugment(
                tf.cast(tf.clip_by_value(values, 0, 255), tf.uint8),
                randaug_layers, randaug_magnitude)
            values = tf.cast(values, tf.float32)
            if manifest.input_scale == "minus1_1":
                values = values / 127.5 - 1.0
            elif manifest.input_scale == "caffe":
                values = tf.reverse(values, [-1]) - [103.939, 116.779, 123.68]
            elif manifest.input_scale != "0_255":
                raise ValueError(f"unsupported augmentation scale: {manifest.input_scale}")
        else:
            values = tf.numpy_function(read, [path], tf.float32)
        values.set_shape(manifest.input_shape)
        if categorical:
            label = tf.one_hot(label, len(classes))
        return values, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=seed)
    return dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)


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
    if args.initial_weights:
        if not args.initial_class_map:
            raise ValueError("initial weights require --initial-class-map")
        initial_names = json.loads(Path(args.initial_class_map).read_text())
        if initial_names != class_names:
            raise ValueError("initial class map differs from prepared dataset label order")
    plan = stages(args)
    if not plan:
        raise ValueError("at least one training stage is required")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_weights = None
    histories, stage_records = {}, []
    model = None
    for stage in plan:
        size = stage["input_size"]
        manifest = replace(manifest, input_shape=[size, size, 3])
        if model is None or int(model.input_shape[1]) != size:
            if model is not None:
                previous_weights = model.get_weights()
            model = build_vision_model(
                backbone=args.backbone, num_classes=len(directory_names), input_size=size,
                alpha=args.width_multiplier,
                weights=(None if previous_weights is not None or args.initial_weights
                         or args.weights == "none" else args.weights),
                trainable=False,
            )
            if previous_weights is not None:
                model.set_weights(previous_weights)
            elif args.initial_weights:
                model.load_weights(args.initial_weights)
        configure_trainable_layers(model, stage["unfreeze"])
        train_ds = image_dataset(
            tf, images_dir / "train", directory_names, manifest, args.batch_size, args.seed, True,
            randaug_layers=args.randaug_layers if stage["augment"] else 0,
            randaug_magnitude=args.randaug_magnitude, categorical=True,
            drop_remainder=args.optimizer == "sgd",
        )
        val_ds = image_dataset(tf, images_dir / "val", directory_names, manifest,
                               args.batch_size, args.seed, False, categorical=True)
        batches = int(tf.data.experimental.cardinality(train_ds))
        if batches < 1:
            raise ValueError("training split is smaller than batch_size; reduce batch_size")
        rate = stage["learning_rate"] * (args.batch_size / 256 if args.scale_learning_rate else 1)
        optimizer = (tf.keras.optimizers.SGD(rate, momentum=args.momentum)
                     if args.optimizer == "sgd" else tf.keras.optimizers.Adam(rate))
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
                      metrics=["accuracy"])
        rates = []

        class BatchSchedule(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.step = 0

            def on_train_batch_begin(self, batch, logs=None):
                self.step += 1
                value = learning_rate_at_step(
                    rate, self.step, stage["epochs"] * batches,
                    int(args.lr_warmup_epochs * batches), args.cosine_decay)
                self.model.optimizer.learning_rate.assign(value)
                rates.append(value)

        callbacks = [BatchSchedule()]
        checkpoint = output.parent / f"{output.stem}_{stage['name']}.weights.h5"
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            checkpoint, save_weights_only=True, save_freq="epoch"))
        print(f"stage={stage['name']} epochs={stage['epochs']} size={size} "
              f"optimizer={args.optimizer} scaled_learning_rate={rate}", flush=True)
        history = model.fit(train_ds, validation_data=val_ds, epochs=stage["epochs"],
                            callbacks=callbacks)
        histories[stage["name"]] = history.history
        stage_records.append({
            **stage, "effective_initial_learning_rate": rate, "steps_per_epoch": batches,
            "trainable_variables": len(model.trainable_variables),
            "first_learning_rate": rates[0], "last_learning_rate": rates[-1],
        })

    model.save(output)
    manifest.save(f"{output}.manifest.json")
    Path(f"{output}.history.json").write_text(json.dumps({
        "configuration": vars(args), **histories, "stages": stage_records,
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
    parser.add_argument("--resolution-epochs", type=int, default=0)
    parser.add_argument("--resolution-learning-rate", type=float, default=0.008)
    parser.add_argument("--resolution-input-size", type=int, default=300)
    parser.add_argument("--unfreeze-layers", type=int, default=18)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--scale-learning-rate", action="store_true")
    parser.add_argument("--cosine-decay", action="store_true")
    parser.add_argument("--lr-warmup-epochs", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--randaug-layers", type=int, default=0)
    parser.add_argument("--randaug-magnitude", type=int, default=0)
    parser.add_argument("--initial-weights")
    parser.add_argument("--initial-class-map")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    print(train(build_parser().parse_args()))


if __name__ == "__main__":
    main()
