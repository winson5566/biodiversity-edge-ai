"""Prepare a deterministic teaching dataset from iNaturalist-style JSON metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import random
import re
import shutil
from typing import Any, Iterable

from PIL import Image

from biodiversity_edge_ai.metadata import encode_geo_features


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CSV_FIELDS = [
    "filename",
    "image_id",
    "source_category_id",
    "label_id",
    "class_name",
    "latitude",
    "longitude",
    "date",
    "valid",
    "source_file",
]


@dataclass(frozen=True)
class SourceRecord:
    image_id: int
    category_id: int
    source: Path
    latitude: float | None
    longitude: float | None
    date: str | None
    geo_valid: bool


def _safe_text(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._") or "class"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    required = {"images", "annotations", "categories"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"annotation JSON is missing keys: {sorted(missing)}")
    return value


def _annotation_index(annotations: Iterable[dict[str, Any]]) -> dict[int, int]:
    result: dict[int, int] = {}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        previous = result.get(image_id)
        if previous is not None and previous != category_id:
            raise ValueError(f"image {image_id} has multiple category labels")
        result[image_id] = category_id
    return result


def _valid_geo(image: dict[str, Any]) -> tuple[float | None, float | None, str | None, bool]:
    latitude = image.get("latitude")
    longitude = image.get("longitude")
    date = image.get("date")
    if latitude in (None, "") or longitude in (None, "") or not date:
        return None, None, None, False
    try:
        latitude_value = float(latitude)
        longitude_value = float(longitude)
        encode_geo_features(latitude_value, longitude_value, str(date))
    except (TypeError, ValueError):
        return None, None, None, False
    return latitude_value, longitude_value, str(date), True


def load_source_records(
    annotations_path: str | Path,
    images_root: str | Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[SourceRecord]], dict[str, int]]:
    """Read and validate the classification join from a source annotation file."""
    annotations_path = Path(annotations_path)
    images_root = Path(images_root)
    data = _load_json(annotations_path)
    categories = {int(item["id"]): item for item in data["categories"]}
    labels = _annotation_index(data["annotations"])
    records: dict[int, list[SourceRecord]] = {category_id: [] for category_id in categories}
    skipped = {"unlabelled": 0, "unknown_category": 0, "missing_file": 0, "unsupported_file": 0}
    seen_ids: set[int] = set()
    seen_sources: dict[Path, int] = {}

    for image in data["images"]:
        image_id = int(image["id"])
        if image_id in seen_ids:
            raise ValueError(f"duplicate image id: {image_id}")
        seen_ids.add(image_id)
        category_id = labels.get(image_id)
        if category_id is None:
            skipped["unlabelled"] += 1
            continue
        if category_id not in categories:
            skipped["unknown_category"] += 1
            continue
        relative = Path(str(image["file_name"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe image path in annotation: {relative}")
        source = (images_root / relative).resolve()
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            skipped["unsupported_file"] += 1
            continue
        if not source.is_file():
            skipped["missing_file"] += 1
            continue
        previous_category = seen_sources.get(source)
        if previous_category is not None:
            if previous_category != category_id:
                raise ValueError(f"the same source file has conflicting labels: {source}")
            continue
        seen_sources[source] = category_id
        latitude, longitude, date, geo_valid = _valid_geo(image)
        records[category_id].append(
            SourceRecord(
                image_id=image_id,
                category_id=category_id,
                source=source,
                latitude=latitude,
                longitude=longitude,
                date=date,
                geo_valid=geo_valid,
            )
        )
    return categories, records, skipped


def select_categories(
    records: dict[int, list[SourceRecord]],
    *,
    category_ids: list[int] | None,
    num_classes: int | None,
    min_per_class: int,
) -> list[int]:
    eligible = [category_id for category_id, rows in records.items() if len(rows) >= min_per_class]
    if category_ids:
        requested = sorted(set(category_ids))
        missing = [value for value in requested if value not in records]
        too_small = [value for value in requested if value in records and len(records[value]) < min_per_class]
        if missing:
            raise ValueError(f"unknown category IDs: {missing}")
        if too_small:
            raise ValueError(f"categories below --min-per-class: {too_small}")
        return requested
    ranked = sorted(eligible, key=lambda value: (-len(records[value]), value))
    if num_classes is not None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if len(ranked) < num_classes:
            raise ValueError(
                f"only {len(ranked)} categories have at least {min_per_class} usable images"
            )
        ranked = ranked[:num_classes]
    if not ranked:
        raise ValueError("no categories satisfy the selection rules")
    return sorted(ranked)


def _split_counts(count: int, val_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("validation and test fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must sum to less than 1")
    val_count = int(round(count * val_fraction))
    test_count = int(round(count * test_fraction))
    if val_fraction > 0 and val_count == 0 and count >= 3:
        val_count = 1
    if test_fraction > 0 and test_count == 0 and count >= 3:
        test_count = 1
    overflow = val_count + test_count - (count - 1)
    while overflow > 0 and (val_count > 0 or test_count > 0):
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        overflow -= 1
    return count - val_count - test_count, val_count, test_count


def split_records(
    rows: list[SourceRecord],
    *,
    seed: int,
    category_id: int,
    val_fraction: float,
    test_fraction: float,
    max_per_class: int | None,
) -> dict[str, list[SourceRecord]]:
    shuffled = list(rows)
    random.Random(f"{seed}:{category_id}").shuffle(shuffled)
    if max_per_class is not None:
        if max_per_class <= 0:
            raise ValueError("max_per_class must be positive")
        shuffled = shuffled[:max_per_class]
    train_count, val_count, _ = _split_counts(len(shuffled), val_fraction, test_fraction)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def _transfer(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"unsupported transfer mode: {mode}")


def _verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output}; choose a new directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    categories, records, skipped = load_source_records(args.annotations, args.images_root)
    selected = select_categories(
        records,
        category_ids=args.category_ids,
        num_classes=args.num_classes,
        min_per_class=args.min_per_class,
    )

    class_entries: list[dict[str, Any]] = []
    class_names: list[str] = []
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    geo_valid_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    seen_output_names: set[str] = set()

    for label_id, category_id in enumerate(selected):
        category = categories[category_id]
        class_name = str(category.get("common_name") or category.get("name") or category_id)
        directory = f"{label_id:05d}"
        class_names.append(class_name)
        class_entries.append(
            {
                "label_id": label_id,
                "source_category_id": category_id,
                "name": str(category.get("name") or class_name),
                "display_name": class_name,
                "directory": directory,
            }
        )
        splits = split_records(
            records[category_id],
            seed=args.seed,
            category_id=category_id,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            max_per_class=args.max_per_class,
        )
        for split, source_rows in splits.items():
            for record in source_rows:
                if args.verify_images:
                    _verify_image(record.source)
                filename = f"{record.image_id}_{_safe_text(record.source.name)}"
                relative = Path(directory) / filename
                output_key = f"{split}/{relative.as_posix()}"
                if output_key in seen_output_names:
                    raise ValueError(f"duplicate output filename: {output_key}")
                seen_output_names.add(output_key)
                _transfer(record.source, output / "images" / split / relative, args.transfer_mode)
                split_rows[split].append(
                    {
                        "filename": relative.as_posix(),
                        "image_id": record.image_id,
                        "source_category_id": category_id,
                        "label_id": label_id,
                        "class_name": class_name,
                        "latitude": record.latitude if record.geo_valid else "",
                        "longitude": record.longitude if record.geo_valid else "",
                        "date": record.date if record.geo_valid else "",
                        "valid": str(record.geo_valid).lower(),
                        "source_file": str(record.source),
                    }
                )
                split_counts[split] += 1
                geo_valid_counts[split] += int(record.geo_valid)

    for split, rows in split_rows.items():
        rows.sort(key=lambda row: (int(row["label_id"]), str(row["filename"])))
        _write_csv(output / "metadata" / f"{split}.csv", rows)

    class_map = output / "class_map.json"
    class_map.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "source": {
            "annotations": str(Path(args.annotations).resolve()),
            "images_root": str(Path(args.images_root).resolve()),
            "annotation_sha256": hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),
        },
        "selection": {
            "seed": args.seed,
            "min_per_class": args.min_per_class,
            "max_per_class": args.max_per_class,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "transfer_mode": args.transfer_mode,
        },
        "classes": class_entries,
        "counts": split_counts,
        "geo_valid_counts": geo_valid_counts,
        "skipped_source_records": skipped,
        "paths": {
            "class_map": "class_map.json",
            "images": "images/{train,val,test}/{class_directory}/",
            "metadata": "metadata/{train,val,test}.csv",
        },
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="iNaturalist-style JSON")
    parser.add_argument("--images-root", required=True, help="root joined with image file_name")
    parser.add_argument("--output", required=True, help="must be new or empty")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--category-ids", nargs="+", type=int)
    selection.add_argument("--num-classes", type=int)
    parser.add_argument("--min-per-class", type=int, default=20)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--transfer-mode", choices=("symlink", "hardlink", "copy"), default="symlink"
    )
    parser.add_argument(
        "--verify-images", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    manifest = prepare_dataset(build_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
