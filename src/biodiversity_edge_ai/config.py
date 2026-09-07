"""Validated experiment settings; paths are relative to the working directory."""

from dataclasses import asdict, dataclass, field, fields
import json
import math
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
    head_learning_rate: float = 1e-3
    finetune_learning_rate: float = 1e-4
    resolution_epochs: int = 0
    resolution_learning_rate: float = 0.008
    resolution_input_size: int = 300
    unfreeze_layers: int = 18
    optimizer: str = "adam"
    momentum: float = 0.0
    scale_learning_rate: bool = False
    cosine_decay: bool = False
    lr_warmup_epochs: float = 0.0
    label_smoothing: float = 0.0
    randaug_layers: int = 0
    randaug_magnitude: int = 0
    initial_weights: str | None = None
    initial_class_map: str | None = None
    requires_initial_weights: bool = False
    recipe_status: str = "demo"
    parameter_source: dict = field(default_factory=dict)
    geo_epochs: int = 30
    geo_batch_size: int = 32
    geo_learning_rate: float = 5e-4
    geo_lr_decay: float = 1.0
    geo_max_per_class: int | None = None
    geo_embedding_dim: int = 256
    formats: tuple[str, ...] = ("fp32", "drq")
    representative_limit: int = 200
    threads: int = 1
    warmup: int = 10
    repetitions: int = 1
    alpha: float = 0.3

    @classmethod
    def load(cls, path: str | Path, *, data_source: str | None = None,
             **overrides) -> "Experiment":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("configuration must be a JSON object")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        if data_source is not None:
            if data_source not in ("mini", "full"):
                raise ValueError("data_source must be mini or full")
            if value.get("synthetic", False):
                raise ValueError("data_source cannot override a synthetic experiment")
            filename = "train_mini.json" if data_source == "mini" else "train.json"
            image_root = overrides.get("images_root") or value.get("images_root", cls.images_root)
            value["annotations"] = str(Path(image_root) / filename)
            value["name"] = data_source
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
                    "geo_epochs", "geo_batch_size", "geo_embedding_dim", "representative_limit",
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
        for key in ("head_epochs", "finetune_epochs", "resolution_epochs", "unfreeze_layers",
                    "randaug_layers", "randaug_magnitude", "warmup", "seed"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.head_epochs + self.finetune_epochs + self.resolution_epochs < 1:
            raise ValueError("at least one vision training epoch is required")
        for key in ("head_learning_rate", "finetune_learning_rate", "resolution_learning_rate",
                    "geo_learning_rate", "geo_lr_decay"):
            value = getattr(self, key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be a finite positive number")
        if self.optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be adam or sgd")
        if type(self.resolution_input_size) is not int or self.resolution_input_size < 32:
            raise ValueError("resolution_input_size must be an integer >=32")
        if not 0 <= self.momentum < 1 or not 0 <= self.label_smoothing < 1:
            raise ValueError("momentum and label_smoothing must be in [0,1)")
        if not math.isfinite(self.lr_warmup_epochs) or self.lr_warmup_epochs < 0:
            raise ValueError("lr_warmup_epochs must be finite and nonnegative")
        if self.randaug_magnitude > 10:
            raise ValueError("randaug_magnitude must be in [0,10]")
        if self.geo_lr_decay > 1:
            raise ValueError("geo_lr_decay must be <=1")
        if self.geo_max_per_class is not None and (
            type(self.geo_max_per_class) is not int or self.geo_max_per_class < 1
        ):
            raise ValueError("geo_max_per_class must be null or a positive integer")
        if self.recipe_status not in ("recovered", "missing", "demo", "custom"):
            raise ValueError("recipe_status must be recovered, missing, demo or custom")
        for key in ("scale_learning_rate", "cosine_decay", "requires_initial_weights"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
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
