"""Portable TFLite inference and image preprocessing."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

from .manifest import ModelManifest


def _default_interpreter_factory(**kwargs: Any) -> Any:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "TFLite runtime is unavailable. Install tflite-runtime on Raspberry Pi "
                "or the training extra on a workstation."
            ) from exc
    return Interpreter(**kwargs)


def preprocess_image(image: Image.Image, manifest: ModelManifest) -> np.ndarray:
    """Apply the preprocessing declared by a model manifest."""
    if len(manifest.input_shape) != 3 or manifest.input_shape[-1] != 3:
        raise ValueError(f"vision input_shape must be [height, width, 3]: {manifest.input_shape}")
    height, width, _ = manifest.input_shape
    image = image.convert("RGB")
    source_width, source_height = image.size
    side = min(source_width, source_height)
    left = (source_width - side) // 2
    top = (source_height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)

    scale = manifest.input_scale
    if scale == "0_255":
        return array
    if scale == "0_1":
        return array / 255.0
    if scale == "minus1_1":
        return array / 127.5 - 1.0
    if scale == "imagenet":
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        return (array / 255.0 - mean) / std
    raise ValueError(f"unsupported input_scale: {scale}")


def _quantize_input(array: np.ndarray, details: dict[str, Any]) -> np.ndarray:
    dtype = details["dtype"]
    if np.issubdtype(dtype, np.floating):
        return array.astype(dtype, copy=False)
    scale, zero_point = details.get("quantization", (0.0, 0))
    if not scale:
        raise ValueError("integer input tensor has no quantization scale")
    quantized = np.rint(array / float(scale) + float(zero_point))
    limits = np.iinfo(dtype)
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def _dequantize_output(array: np.ndarray, details: dict[str, Any]) -> np.ndarray:
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32, copy=False)
    scale, zero_point = details.get("quantization", (0.0, 0))
    if not scale:
        return array.astype(np.float32)
    return (array.astype(np.float32) - float(zero_point)) * float(scale)


class TFLiteRunner:
    """One-input, one-output TFLite runner used for vision and geo-prior models."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        num_threads: int = 1,
        interpreter_factory: Callable[..., Any] | None = None,
    ) -> None:
        factory = interpreter_factory or _default_interpreter_factory
        self.model_path = Path(model_path)
        self.interpreter = factory(
            model_path=str(self.model_path), num_threads=int(num_threads)
        )
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("integrated runtime requires exactly one input and one output")
        self.input_details = inputs[0]
        self.output_details = outputs[0]
        self.last_inference_ms = 0.0

    def predict(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        expected_rank = len(self.input_details["shape"])
        if array.ndim == expected_rank - 1:
            array = array[None, ...]
        if array.ndim != expected_rank:
            raise ValueError(
                f"input rank {array.ndim} does not match model rank {expected_rank}"
            )
        prepared = _quantize_input(array, self.input_details)
        self.interpreter.set_tensor(self.input_details["index"], prepared)
        started = time.perf_counter()
        self.interpreter.invoke()
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        output = self.interpreter.get_tensor(self.output_details["index"])
        output = _dequantize_output(output, self.output_details)
        return output[0] if output.shape[0] == 1 else output
