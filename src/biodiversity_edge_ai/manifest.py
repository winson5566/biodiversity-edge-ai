"""Model metadata and compatibility checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def class_map_sha256(classes: Iterable[str]) -> str:
    canonical = json.dumps(list(classes), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    role: str
    format: str
    num_classes: int
    class_map_sha256: str
    input_shape: list[int]
    input_dtype: str
    input_scale: str
    optimization: str = "none"
    source: str = "new"
    notes: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelManifest":
        return cls(**value)

    @classmethod
    def load(cls, path: str | Path) -> "ModelManifest":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(asdict(self), stream, ensure_ascii=False, indent=2)
            stream.write("\n")


def validate_fusion_compatibility(
    vision: ModelManifest,
    geo: ModelManifest,
) -> None:
    errors: list[str] = []
    if vision.role != "vision":
        errors.append(f"vision manifest role is {vision.role!r}")
    if geo.role != "geo_prior":
        errors.append(f"geo manifest role is {geo.role!r}")
    if vision.num_classes != geo.num_classes:
        errors.append(
            f"class counts differ: {vision.num_classes} != {geo.num_classes}"
        )
    if vision.class_map_sha256 != geo.class_map_sha256:
        errors.append("class-map hashes differ")
    if errors:
        raise ValueError("incompatible model manifests: " + "; ".join(errors))
