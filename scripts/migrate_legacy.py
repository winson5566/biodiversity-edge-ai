"""Collect legacy model artifacts without modifying the three source projects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from biodiversity_edge_ai.pipeline import load_class_names
from biodiversity_edge_ai.manifest import ModelManifest, class_map_sha256


def transfer(source: Path, destination: Path, copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    if copy:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-repo", required=True)
    parser.add_argument("--geo-repo", required=True)
    parser.add_argument("--device-repo", required=True)
    parser.add_argument("--destination", default="artifacts/legacy")
    parser.add_argument("--copy", action="store_true", help="copy files instead of linking")
    parser.add_argument("--legacy-input-size", type=int, default=224)
    parser.add_argument("--legacy-input-scale", default="0_255")
    args = parser.parse_args()

    vision_repo = Path(args.vision_repo)
    geo_repo = Path(args.geo_repo)
    device_repo = Path(args.device_repo)
    destination = Path(args.destination)
    sources = {
        "vision_repo": str(vision_repo.resolve()),
        "geo_repo": str(geo_repo.resolve()),
        "device_repo": str(device_repo.resolve()),
    }
    for source in (vision_repo, geo_repo, device_repo):
        if not source.is_dir():
            raise SystemExit(f"legacy repository not found: {source}")

    category_source = device_repo / "inat2021" / "categories.json"
    class_names = load_class_names(category_source)
    class_map = destination / "class_maps" / "inat2021_10000.json"
    class_map.parent.mkdir(parents=True, exist_ok=True)
    class_map.write_text(json.dumps(class_names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    transferred: list[dict[str, object]] = []
    for source in sorted((device_repo / "model").glob("*.tflite")):
        target = destination / "vision_models" / source.name
        transfer(source, target, args.copy)
        optimization = "drq" if "_drq" in source.stem else "fp32"
        manifest_path = target.with_suffix(".manifest.json")
        ModelManifest(
            model_id=f"{source.stem}_legacy",
            role="vision",
            format="tflite",
            num_classes=len(class_names),
            class_map_sha256=class_map_sha256(class_names),
            input_shape=[args.legacy_input_size, args.legacy_input_size, 3],
            input_dtype="float32",
            input_scale=args.legacy_input_scale,
            optimization=optimization,
            source="original project artifact",
            notes="Shape and input scale were recovered from legacy export configs; verify on target runtime.",
        ).save(manifest_path)
        transferred.append(
            {
                "kind": "vision_tflite",
                "path": str(target),
                "manifest": str(manifest_path),
                "bytes": source.stat().st_size,
            }
        )

    geo_checkpoint = geo_repo / "model" / "geo_prior_ckp"
    if geo_checkpoint.exists():
        target = destination / "geo_prior_checkpoint"
        transfer(geo_checkpoint, target, args.copy)
        transferred.append({"kind": "geo_checkpoint", "path": str(target)})

    inventory = {
        "sources": sources,
        "mode": "copy" if args.copy else "symlink",
        "class_count": len(class_names),
        "class_map": str(class_map),
        "artifacts": transferred,
        "warning": "Legacy manifests use the original 224x224/0-255 export configuration and should be verified on the target runtime.",
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
