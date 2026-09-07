"""End-to-end prediction pipeline shared by CLI and Raspberry Pi UI."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .fusion import fuse_probabilities, normalize_scores, top_k
from .inference import TFLiteRunner, preprocess_image
from .manifest import ModelManifest, class_map_sha256, validate_fusion_compatibility
from .metadata import encode_geo_features


class Predictor(Protocol):
    def predict(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Prediction:
    class_id: int
    name: str
    score: float


def load_class_names(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    categories = raw.get("categories", raw) if isinstance(raw, dict) else raw
    if not isinstance(categories, list):
        raise ValueError("class map must be a list or an object containing 'categories'")
    if categories and isinstance(categories[0], dict):
        ordered = sorted(categories, key=lambda item: int(item.get("id", 0)))
        ids = [int(item.get("id", index)) for index, item in enumerate(ordered)]
        if ids != list(range(len(ids))):
            raise ValueError("class IDs must be contiguous and zero-based")
        return [
            str(item.get("common_name") or item.get("name") or f"class_{ids[index]}")
            for index, item in enumerate(ordered)
        ]
    return [str(value) for value in categories]


class BiodiversityPipeline:
    def __init__(
        self,
        *,
        vision_runner: Predictor,
        vision_manifest: ModelManifest,
        class_names: list[str],
        geo_runner: Predictor | None = None,
        geo_manifest: ModelManifest | None = None,
        fusion_mode: str = "log_linear",
        alpha: float = 0.3,
    ) -> None:
        if vision_manifest.role != "vision":
            raise ValueError("vision_manifest role must be 'vision'")
        if len(class_names) != vision_manifest.num_classes:
            raise ValueError("class map length does not match vision model")
        if class_map_sha256(class_names) != vision_manifest.class_map_sha256:
            raise ValueError("class map does not match vision manifest hash")
        if (geo_runner is None) != (geo_manifest is None):
            raise ValueError("geo_runner and geo_manifest must be provided together")
        if geo_manifest is not None:
            validate_fusion_compatibility(vision_manifest, geo_manifest)
        self.vision_runner = vision_runner
        self.vision_manifest = vision_manifest
        self.class_names = class_names
        self.geo_runner = geo_runner
        self.geo_manifest = geo_manifest
        self.fusion_mode = fusion_mode
        self.alpha = alpha

    @classmethod
    def from_files(
        cls,
        *,
        vision_model: str | Path,
        vision_manifest: str | Path,
        class_map: str | Path,
        geo_model: str | Path | None = None,
        geo_manifest: str | Path | None = None,
        num_threads: int = 1,
        fusion_mode: str = "log_linear",
        alpha: float = 0.3,
    ) -> "BiodiversityPipeline":
        vision_info = ModelManifest.load(vision_manifest)
        geo_info = ModelManifest.load(geo_manifest) if geo_manifest else None
        return cls(
            vision_runner=TFLiteRunner(vision_model, num_threads=num_threads),
            vision_manifest=vision_info,
            class_names=load_class_names(class_map),
            geo_runner=(
                TFLiteRunner(geo_model, num_threads=num_threads) if geo_model else None
            ),
            geo_manifest=geo_info,
            fusion_mode=fusion_mode,
            alpha=alpha,
        )

    def predict(
        self,
        image: Image.Image,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        observation_date: str | dt.date | dt.datetime | None = None,
        k: int = 5,
    ) -> list[Prediction]:
        image_values = preprocess_image(image, self.vision_manifest)
        vision = normalize_scores(self.vision_runner.predict(image_values))
        location_valid = (
            self.geo_runner is not None
            and latitude is not None
            and longitude is not None
            and observation_date is not None
        )
        if location_valid:
            features = encode_geo_features(latitude, longitude, observation_date)
            geo = self.geo_runner.predict(features)
            probabilities = fuse_probabilities(
                vision,
                geo,
                alpha=self.alpha,
                mode=self.fusion_mode,
                location_valid=True,
            )
        else:
            probabilities = vision
        indices, scores = top_k(probabilities, k=k)
        return [
            Prediction(int(index), self.class_names[int(index)], float(score))
            for index, score in zip(indices, scores)
        ]
