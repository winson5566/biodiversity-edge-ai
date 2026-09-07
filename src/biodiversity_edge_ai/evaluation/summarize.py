"""Combine benchmark JSON files into a directly comparable trade-off table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "vision_model_id",
    "vision_optimization",
    "top1_accuracy",
    "vision_model_bytes",
    "geo_model_bytes",
    "invoke_median_ms",
    "invoke_p95_ms",
    "end_to_end_median_ms",
    "end_to_end_p95_ms",
    "end_to_end_images_per_second",
    "rss_delta_mb",
    "threads",
    "samples",
]


def load_results(paths: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        with path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        missing = [field for field in FIELDS if field not in result]
        if missing:
            raise ValueError(f"{path} is missing benchmark fields: {missing}")
        results.append(result)
    return sorted(
        results,
        key=lambda item: (str(item["vision_model_id"]), str(item["vision_optimization"])),
    )


def write_csv(path: str | Path, results: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _format(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: str | Path, results: list[dict[str, Any]]) -> None:
    columns = [
        ("Model", "vision_model_id"),
        ("Optimization", "vision_optimization"),
        ("Top-1", "top1_accuracy"),
        ("Vision bytes", "vision_model_bytes"),
        ("Invoke median ms", "invoke_median_ms"),
        ("End-to-end P95 ms", "end_to_end_p95_ms"),
        ("Images/s", "end_to_end_images_per_second"),
        ("RSS delta MB", "rss_delta_mb"),
    ]
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for result in results:
        lines.append(
            "| " + " | ".join(_format(result[field]) for _, field in columns) + " |"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--markdown", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = load_results(args.inputs)
    write_csv(args.csv, results)
    write_markdown(args.markdown, results)
    print(args.csv)
    print(args.markdown)


if __name__ == "__main__":
    main()
