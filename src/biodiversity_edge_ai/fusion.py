"""Vision and geo-prior fusion with explicit normalization and fallback."""

from __future__ import annotations

import numpy as np


def _as_batch(probabilities: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(probabilities, dtype=np.float32)
    squeezed = array.ndim == 1
    if squeezed:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape [classes] or [batch, classes]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(array < 0):
        raise ValueError(f"{name} contains negative values")
    return array, squeezed


def normalize_scores(scores: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    denominator = scores.sum(axis=-1, keepdims=True)
    if np.any(denominator <= epsilon):
        raise ValueError("scores must contain positive mass for every sample")
    return scores / denominator


def fuse_probabilities(
    vision_probabilities: np.ndarray,
    geo_probabilities: np.ndarray,
    *,
    alpha: float = 0.3,
    mode: str = "log_linear",
    location_valid: bool | np.ndarray = True,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Fuse class scores and return a normalized probability distribution.

    `alpha` is the vision weight for log-linear fusion. Invalid location rows
    return the normalized vision-only distribution.
    """
    vision, squeezed = _as_batch(vision_probabilities, "vision_probabilities")
    geo, geo_squeezed = _as_batch(geo_probabilities, "geo_probabilities")
    if vision.shape != geo.shape:
        raise ValueError(f"vision and geo shapes differ: {vision.shape} != {geo.shape}")
    if squeezed != geo_squeezed:
        raise ValueError("vision and geo inputs must use the same rank")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    vision = normalize_scores(vision, epsilon)
    if mode == "bayes":
        combined = vision * geo
    elif mode == "log_linear":
        combined = np.exp(
            alpha * np.log(np.clip(vision, epsilon, None))
            + (1.0 - alpha) * np.log(np.clip(geo, epsilon, None))
        )
    else:
        raise ValueError(f"unsupported fusion mode: {mode}")

    combined = normalize_scores(combined, epsilon)
    valid = np.asarray(location_valid, dtype=bool)
    if valid.ndim == 0:
        valid = np.full((vision.shape[0], 1), bool(valid))
    elif valid.ndim == 1 and valid.shape[0] == vision.shape[0]:
        valid = valid[:, None]
    else:
        raise ValueError("location_valid must be a scalar or one value per batch row")
    result = np.where(valid, combined, vision)
    return result[0] if squeezed else result


def top_k(probabilities: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    probabilities, squeezed = _as_batch(probabilities, "probabilities")
    if not 1 <= k <= probabilities.shape[1]:
        raise ValueError(f"k must be between 1 and {probabilities.shape[1]}")
    indices = np.argsort(-probabilities, axis=1)[:, :k]
    scores = np.take_along_axis(probabilities, indices, axis=1)
    if squeezed:
        return indices[0], scores[0]
    return indices, scores
