"""Create a tiny deterministic iNaturalist-style source dataset for smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--images-per-class", type=int, default=12)
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.classes < 2 or args.images_per_class < 3 or args.size < 32:
        raise SystemExit("smoke data requires >=2 classes, >=3 images/class, and size >=32")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    images: list[dict[str, object]] = []
    annotations: list[dict[str, int]] = []
    categories: list[dict[str, object]] = []
    image_id = 0
    for category_id in range(args.classes):
        categories.append(
            {
                "id": category_id,
                "name": f"Synthetic species {category_id}",
                "common_name": f"Smoke class {category_id}",
            }
        )
        for index in range(args.images_per_class):
            relative = Path("images") / f"class_{category_id}" / f"sample_{index:03d}.png"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            base = np.zeros((args.size, args.size, 3), dtype=np.uint8)
            channel = category_id % 3
            base[..., channel] = 190
            base += rng.integers(0, 35, size=base.shape, dtype=np.uint8)
            image = Image.fromarray(base, mode="RGB")
            draw = ImageDraw.Draw(image)
            if category_id % 2:
                draw.ellipse((8, 8, args.size - 8, args.size - 8), fill=(240, 240, 40))
            else:
                draw.rectangle((8, 8, args.size - 8, args.size - 8), fill=(40, 240, 240))
            image.save(destination)
            images.append(
                {
                    "id": image_id,
                    "file_name": relative.as_posix(),
                    "width": args.size,
                    "height": args.size,
                    "latitude": -43.5 + category_id * 20 + index * 0.01,
                    "longitude": 172.6 - category_id * 40 + index * 0.01,
                    "date": f"2025-{category_id + 1:02d}-{index % 9 + 1:02d}",
                }
            )
            annotations.append(
                {"id": image_id, "image_id": image_id, "category_id": category_id}
            )
            image_id += 1
    source = {
        "info": {"description": "synthetic smoke-test data"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    annotation_path = output / "annotations.json"
    annotation_path.write_text(
        json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(annotation_path)


if __name__ == "__main__":
    main()
