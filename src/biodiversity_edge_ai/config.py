"""Validated experiment settings; paths are relative to the working directory."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

from .models.catalog import BACKBONES


@dataclass(frozen=True)
class Experiment:
    name: str = "mini"
    annotations: str = "raw/inat2021/train_mini.json"
    images_root: str = "raw/inat2021"
    synthetic: bool = False
    num_classes: int = 10000
    min_per_class: int = 20
    max_per_class: int | None = None
    seed: int = 42
    backbone: str = "mobilenet-v2"
    input_size: int = 224
    width_multiplier: float = 1.0
    weights: str = "imagenet"
    batch_size: int = 32
    head_epochs: int = 3
    finetune_epochs: int = 5
    geo_epochs: int = 30
    geo_embedding_dim: int = 256
    formats: tuple[str, ...] = ("fp32", "drq")
    representative_limit: int = 200
    threads: int = 1
    warmup: int = 10
    repetitions: int = 1
    alpha: float = 0.3

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "Experiment":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("configuration must be a JSON object")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        value.update({k: v for k, v in overrides.items() if v is not None})
        if "formats" in value:
            value["formats"] = tuple(value["formats"])
        experiment = cls(**value)
        experiment.validate()
        return experiment

    def validate(self) -> None:
        if self.backbone not in BACKBONES:
            raise ValueError(f"unsupported backbone: {self.backbone}")
        if self.weights not in ("imagenet", "none"):
            raise ValueError("weights must be imagenet or none")
        for key in ("num_classes", "min_per_class", "input_size", "batch_size",
                    "geo_epochs", "geo_embedding_dim", "representative_limit",
                    "threads", "repetitions"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.input_size < 32 or self.num_classes < 2 or self.min_per_class < 3:
            raise ValueError("require input_size >=32, num_classes >=2, min_per_class >=3")
        if self.max_per_class is not None and (
            type(self.max_per_class) is not int or self.max_per_class < 3
        ):
            raise ValueError("max_per_class must be null or an integer >=3")
        for key in ("head_epochs", "finetune_epochs", "warmup", "seed"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.head_epochs + self.finetune_epochs < 1:
            raise ValueError("at least one vision training epoch is required")
        if not 0 <= self.alpha <= 1 or self.width_multiplier <= 0:
            raise ValueError("alpha must be in [0,1] and width_multiplier positive")
        if not self.formats or len(set(self.formats)) != len(self.formats) or (
            set(self.formats) - {"fp32", "drq", "int8"}
        ):
            raise ValueError("formats must be a nonempty unique list of fp32, drq, int8")
        if self.synthetic and (self.num_classes != 2 or self.max_per_class != 12):
            raise ValueError("synthetic fixture uses exactly 2 classes and 12 images/class")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))
