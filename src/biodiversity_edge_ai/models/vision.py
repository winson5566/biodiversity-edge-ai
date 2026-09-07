"""Single canonical vision model builder for training and export."""

from __future__ import annotations

from typing import Any


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for vision model construction") from exc
    return tf


def build_vision_model(
    *,
    backbone: str,
    num_classes: int,
    input_size: int = 224,
    alpha: float = 1.0,
    weights: str | None = "imagenet",
    trainable: bool = True,
) -> Any:
    """Build the architecture used by both training and TFLite export.

    Inputs are float tensors. Input scaling remains external and is recorded in
    the model manifest so that training and inference remain consistent.
    """
    tf = _tensorflow()
    key = backbone.lower().replace("_", "-")
    shape = (input_size, input_size, 3)
    if key == "efficientnet-b0":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=shape, include_top=False, weights=weights
        )
    elif key == "mobilenet-v2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=shape, alpha=alpha, include_top=False, weights=weights
        )
    elif key == "mobilenet-v3-large":
        base = tf.keras.applications.MobileNetV3Large(
            input_shape=shape, alpha=alpha, include_top=False, weights=weights
        )
    elif key == "resnet-50":
        base = tf.keras.applications.ResNet50(
            input_shape=shape, include_top=False, weights=weights
        )
    elif key == "resnet-101":
        base = tf.keras.applications.ResNet101(
            input_shape=shape, include_top=False, weights=weights
        )
    elif key == "convnext-tiny":
        base = tf.keras.applications.ConvNeXtTiny(
            input_shape=shape, include_top=False, weights=weights
        )
    elif key == "convnext-small":
        base = tf.keras.applications.ConvNeXtSmall(
            input_shape=shape, include_top=False, weights=weights
        )
    else:
        raise ValueError(f"unsupported backbone: {backbone}")
    base.trainable = trainable
    inputs = tf.keras.Input(shape=shape, dtype=tf.float32, name="image")
    features = base(inputs)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(features)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="species_probabilities"
    )(features)
    return tf.keras.Model(inputs, outputs, name=key.replace("-", "_"))
