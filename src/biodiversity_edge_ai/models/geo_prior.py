"""Canonical inference model for the legacy six-feature geo prior."""

from __future__ import annotations

from typing import Any


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for geo-prior construction") from exc
    return tf


def build_geo_prior_model(
    *,
    num_classes: int,
    input_features: int = 6,
    embedding_dim: int = 256,
    residual_blocks: int = 4,
    dropout: float = 0.5,
) -> Any:
    """Build the FCNet described in the original project report."""
    tf = _tensorflow()
    inputs = tf.keras.Input(shape=(input_features,), dtype=tf.float32, name="geo_features")
    x = tf.keras.layers.Dense(embedding_dim, name="geo_projection")(inputs)
    x = tf.keras.layers.Activation("relu", name="geo_projection_relu")(x)
    for block in range(residual_blocks):
        residual = x
        x = tf.keras.layers.Dense(embedding_dim, name=f"geo_block_{block}_dense_1")(x)
        x = tf.keras.layers.Activation("relu", name=f"geo_block_{block}_relu_1")(x)
        x = tf.keras.layers.Dropout(dropout, name=f"geo_block_{block}_dropout")(x)
        x = tf.keras.layers.Dense(embedding_dim, name=f"geo_block_{block}_dense_2")(x)
        x = tf.keras.layers.Activation("relu", name=f"geo_block_{block}_relu_2")(x)
        x = tf.keras.layers.Add(name=f"geo_block_{block}_add")([residual, x])
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="sigmoid",
        use_bias=False,
        name="species_geo_scores",
    )(x)
    return tf.keras.Model(inputs, outputs, name="geo_prior_fcnet")
