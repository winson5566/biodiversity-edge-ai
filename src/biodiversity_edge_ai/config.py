"""Small JSON configuration loader with predictable path handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be an object: {source}")
    return value


def resolve_from_config(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent / path).resolve()
