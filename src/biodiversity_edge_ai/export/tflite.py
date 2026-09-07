"""Export a Keras vision or geo-prior model to TFLite with a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
from PIL import Image

from biodiversity_edge_ai.inference import preprocess_image
from biodiversity_edge_ai.manifest import ModelManifest, class_map_sha256
from biodiversity_edge_ai.pipeline import load_class_names


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for model export") from exc
    return tf


def _image_samples(
    directory: Path,
    draft_manifest: ModelManifest,
    limit: int,
) -> Iterable[list[np.ndarray]]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    discovered = sorted(path for pattern in patterns for path in directory.rglob(pattern))
    by_parent: dict[Path, list[Path]] = {}
    for path in discovered:
        by_parent.setdefault(path.parent, []).append(path)
    paths: list[Path] = []
    depth = 0
    parents = sorted(by_parent)
    while len(paths) < limit:
        added = False
        for parent in parents:
            candidates = by_parent[parent]
            if depth < len(candidates):
                paths.append(candidates[depth])
                added = True
                if len(paths) == limit:
                    break
        if not added:
            break
        depth += 1
    if not paths:
        raise ValueError(f"no representative images found in {directory}")
    for path in paths:
        with Image.open(path) as image:
            yield [preprocess_image(image, draft_manifest)[None, ...].astype(np.float32)]


def _array_samples(path: Path, limit: int) -> Iterable[list[np.ndarray]]:
    values = np.load(path)
    if values.ndim < 2:
        raise ValueError("representative NPY must contain a batch dimension")
    for row in values[:limit]:
        yield [np.asarray(row, dtype=np.float32)[None, ...]]


def export_model(args: argparse.Namespace) -> tuple[Path, Path]:
    tf = _tensorflow()
    class_names = load_class_names(args.class_map)
    training_manifest = Path(f"{args.keras_model}.manifest.json")
    if args.role == "vision" and training_manifest.is_file():
        contract = ModelManifest.load(training_manifest)
        if contract.class_map_sha256 != class_map_sha256(class_names):
            raise ValueError("export class map differs from training class map")
        if args.input_scale not in ("auto", contract.input_scale):
            raise ValueError("export input scaling differs from training")
        args.input_scale = contract.input_scale
    elif args.input_scale == "auto":
        raise ValueError("model has no training manifest; supply --input-scale explicitly")
    model = tf.keras.models.load_model(args.keras_model, compile=False)
    if int(model.output_shape[-1]) != len(class_names):
        raise ValueError("model output size does not match the class map")
    if args.role == "geo_prior" and args.input_scale != "encoded_geo":
        raise ValueError("geo-prior exports must use --input-scale encoded_geo")
    if args.role == "vision" and args.input_scale == "encoded_geo":
        raise ValueError("vision exports cannot use encoded_geo input scaling")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keras 3 models loaded by TensorFlow 2.16 do not expose the private
    # `_get_save_spec` used by from_keras_model. A SavedModel boundary is
    # compatible with both Keras 2 and 3 and avoids relying on that API.
    with tempfile.TemporaryDirectory(prefix="biodiversity_saved_model_") as directory:
        if hasattr(model, "export"):
            model.export(directory)
        else:
            tf.saved_model.save(model, directory)
        converter = tf.lite.TFLiteConverter.from_saved_model(directory)

        if args.optimization == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif args.optimization == "drq":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif args.optimization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            draft = ModelManifest(
                model_id=args.model_id,
                role=args.role,
                format="tflite",
                num_classes=len(class_names),
                class_map_sha256=class_map_sha256(class_names),
                input_shape=list(model.input_shape[1:]),
                input_dtype="float32",
                input_scale=args.input_scale,
                optimization=args.optimization,
                source=args.source,
            )
            if args.representative_images:
                converter.representative_dataset = lambda: _image_samples(
                    Path(args.representative_images), draft, args.representative_limit
                )
            elif args.representative_npy:
                converter.representative_dataset = lambda: _array_samples(
                    Path(args.representative_npy), args.representative_limit
                )
            else:
                raise ValueError("INT8 export requires representative images or NPY data")
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif args.optimization != "fp32":
            raise ValueError(f"unsupported optimization: {args.optimization}")

        output.write_bytes(converter.convert())

    interpreter = tf.lite.Interpreter(model_path=str(output))
    interpreter.allocate_tensors()
    details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    if int(output_details["shape"][-1]) != len(class_names):
        raise ValueError("TFLite output size does not match the class map")
    input_shape = [int(value) for value in details["shape"][1:]]
    manifest = ModelManifest(
        model_id=args.model_id,
        role=args.role,
        format="tflite",
        num_classes=len(class_names),
        class_map_sha256=class_map_sha256(class_names),
        input_shape=input_shape,
        input_dtype=np.dtype(details["dtype"]).name,
        input_scale=args.input_scale,
        optimization=args.optimization,
        source=args.source,
        notes=args.notes,
    )
    manifest_path = Path(args.manifest or f"{output}.json")
    manifest.save(manifest_path)
    return output, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--role", required=True, choices=("vision", "geo_prior"))
    parser.add_argument(
        "--optimization", default="fp32", choices=("fp32", "fp16", "drq", "int8")
    )
    parser.add_argument(
        "--input-scale",
        default="auto",
        choices=("auto", "0_255", "0_1", "minus1_1", "imagenet", "caffe", "encoded_geo"),
    )
    parser.add_argument("--representative-images")
    parser.add_argument("--representative-npy")
    parser.add_argument("--representative-limit", type=int, default=200)
    parser.add_argument("--source", default="new")
    parser.add_argument("--notes", default="")
    return parser


def main() -> None:
    output, manifest = export_model(build_parser().parse_args())
    print(f"model: {output}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
