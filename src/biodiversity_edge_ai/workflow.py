"""Run a configured experiment with isolated artifacts and resumable stages."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from .config import Experiment

STAGES = ("prepare", "train", "export", "benchmark")


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Workflow:
    def __init__(self, config: Experiment, run: str | None = None, dry_run: bool = False):
        self.config = config
        self.name = run or f"{config.name}-{config.backbone}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.name):
            raise ValueError("run name must contain only letters, digits, hyphens or underscores")
        self.root = Path("artifacts/runs") / self.name
        self.data = self.root / "dataset"
        self.models = self.root / "models"
        self.results = self.root / "results"
        self.class_map = self.data / "class_map.json"
        self.dry_run = dry_run
        self.state = {}

    def initialize(self) -> None:
        if self.dry_run:
            return
        lock = self.root / "configuration.json"
        effective = self.config.to_dict()
        effective["working_directory"] = str(Path.cwd().resolve())
        package = Path(__file__).parent
        effective["source_files"] = {str(p.relative_to(package)): digest(p)
                                     for p in sorted(package.rglob("*.py"))}
        effective["python_version"] = sys.version
        effective["packages"] = {}
        for distribution in ("numpy", "Pillow", "tensorflow", "keras", "tflite-runtime"):
            try:
                effective["packages"][distribution] = version(distribution)
            except PackageNotFoundError:
                pass
        if lock.exists() and json.loads(lock.read_text()) != effective:
            raise ValueError(f"configuration, source or environment changed for {self.name}; "
                             "choose a new --run name")
        if not lock.exists() and self.root.exists() and any(self.root.iterdir()):
            raise ValueError(f"unmanaged output directory: {self.root}; choose a new --run name")
        for directory in (self.root, self.models, self.results, self.root / "logs"):
            directory.mkdir(parents=True, exist_ok=True)
        write_json(lock, effective)
        state_path = self.root / "state.json"
        if state_path.exists():
            self.state = json.loads(state_path.read_text())

    def step(self, name: str, module: str, options: list, outputs: list[Path],
             inputs: list[Path] | None = None) -> None:
        command = [sys.executable, "-m", f"biodiversity_edge_ai.{module}", *map(str, options)]
        if self.dry_run:
            print(shlex.join(command))
            return
        signature = {"command": command,
                     "inputs": {str(p): digest(p) for p in inputs or []}}
        previous = self.state.get(name)
        if previous:
            actual = {str(p): digest(p) for p in outputs if p.is_file()}
            if previous["signature"] != signature or previous["outputs"] != actual:
                raise ValueError(f"{name}: saved inputs or outputs changed; choose a new --run name")
            print(f"[reuse] {name}", flush=True)
            return
        log = self.root / "logs" / f"{name}.log"
        print(f"[run] {name} -> {log}", flush=True)
        started = time.monotonic()
        with log.open("w") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            print(log.read_text(errors="replace")[-6000:], file=sys.stderr)
            raise RuntimeError(f"{name} failed (exit {completed.returncode}); see {log}")
        self.state[name] = {"signature": signature,
                            "outputs": {str(p): digest(p) for p in outputs},
                            "seconds": time.monotonic() - started}
        write_json(self.root / "state.json", self.state)
        print(f"[done] {name}", flush=True)

    def execute(self, stage: str = "all") -> None:
        self.initialize()
        c = self.config
        annotations, images_root = Path(c.annotations), Path(c.images_root)
        if c.synthetic:
            images_root = self.root / "source"
            annotations = images_root / "annotations.json"
            self.step("source", "data.synthetic", ["--output", images_root, "--seed", c.seed],
                      [annotations])
        manifest = self.data / "dataset_manifest.json"
        metadata = [self.data / "metadata" / f"{s}.csv" for s in ("train", "val", "test")]
        options = ["--annotations", annotations, "--images-root", images_root,
                   "--output", self.data, "--num-classes", c.num_classes,
                   "--min-per-class", c.min_per_class, "--seed", c.seed,
                   "--val-fraction", .15, "--test-fraction", .15]
        if c.max_per_class is not None:
            options += ["--max-per-class", c.max_per_class]
        self.step("prepare", "data.prepare", options, [manifest, self.class_map, *metadata],
                  [annotations])
        if stage == "prepare":
            return
        if not self.dry_run:
            prepared = json.loads(manifest.read_text())
            if any(prepared["geo_valid_counts"][split] == 0 for split in ("train", "val")):
                raise ValueError("Geo Prior needs valid latitude, longitude and date in train and val; "
                                 "inspect dataset metadata before training")
        vision, geo = self.models / "vision.keras", self.models / "geo_prior.keras"
        self.step("train-vision", "training.vision", [
            "--data-dir", self.data, "--output", vision, "--class-map", self.class_map,
            "--backbone", c.backbone, "--input-size", c.input_size,
            "--width-multiplier", c.width_multiplier, "--weights", c.weights,
            "--batch-size", c.batch_size, "--head-epochs", c.head_epochs,
            "--finetune-epochs", c.finetune_epochs, "--seed", c.seed,
            "--head-learning-rate", c.head_learning_rate,
            "--finetune-learning-rate", c.finetune_learning_rate,
        ], [vision, Path(f"{vision}.manifest.json"), Path(f"{vision}.history.json")],
            [manifest, self.class_map, *metadata[:2]])
        self.step("train-geo", "training.geo_prior", [
            "--observations", metadata[0], "--validation-observations", metadata[1],
            "--output", geo, "--num-classes", c.num_classes, "--epochs", c.geo_epochs,
            "--embedding-dim", c.geo_embedding_dim, "--batch-size", c.geo_batch_size,
            "--learning-rate", c.geo_learning_rate,
            "--seed", c.seed,
        ], [geo, Path(f"{geo}.history.json")], [manifest, *metadata[:2]])
        if stage == "train":
            return
        for fmt in c.formats:
            output = self.models / f"vision_{fmt}.tflite"
            options = ["--keras-model", vision, "--output", output,
                       "--manifest", f"{output}.manifest.json", "--class-map", self.class_map,
                       "--model-id", f"{c.backbone}_{fmt}", "--role", "vision",
                       "--optimization", fmt]
            if fmt == "int8":
                options += ["--representative-images", self.data / "images/train",
                            "--representative-limit", c.representative_limit]
            self.step(f"export-{fmt}", "export.tflite", options,
                      [output, Path(f"{output}.manifest.json")],
                      [vision, Path(f"{vision}.manifest.json"), self.class_map])
        geo_tflite = self.models / "geo_prior_fp32.tflite"
        self.step("export-geo", "export.tflite", [
            "--keras-model", geo, "--output", geo_tflite,
            "--manifest", f"{geo_tflite}.manifest.json", "--class-map", self.class_map,
            "--model-id", "geo_prior_fp32", "--role", "geo_prior",
            "--optimization", "fp32", "--input-scale", "encoded_geo",
        ], [geo_tflite, Path(f"{geo_tflite}.manifest.json")], [geo, self.class_map])
        if not self.dry_run:
            shutil.copy2(self.class_map, self.models / "class_map.json")
        if stage == "export":
            return
        results = []
        for fmt in c.formats:
            model = self.models / f"vision_{fmt}.tflite"
            for mode in ("vision", "fused"):
                output = self.results / f"{mode}_{fmt}.json"
                options = ["--images", self.data / "images/test", "--metadata-csv", metadata[2],
                           "--vision-model", model, "--vision-manifest", f"{model}.manifest.json",
                           "--class-map", self.class_map, "--threads", c.threads,
                           "--warmup", c.warmup, "--repetitions", c.repetitions, "--output", output]
                dependencies = [model, Path(f"{model}.manifest.json"), self.class_map, metadata[2]]
                if mode == "fused":
                    options += ["--geo-model", geo_tflite, "--geo-manifest",
                                f"{geo_tflite}.manifest.json", "--alpha", c.alpha]
                    dependencies += [geo_tflite, Path(f"{geo_tflite}.manifest.json")]
                self.step(f"benchmark-{mode}-{fmt}", "evaluation.benchmark", options,
                          [output], dependencies)
                results.append(output)
        self.step("summarize", "evaluation.summarize", [
            *results, "--csv", self.results / "tradeoffs.csv",
            "--markdown", self.results / "tradeoffs.md",
        ], [self.results / "tradeoffs.csv", self.results / "tradeoffs.md"], results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models/mobilenet-v2.json")
    parser.add_argument("--run", help="unique experiment name; changed settings require a new name")
    parser.add_argument("--data-source", choices=("mini", "full"),
                        help="override the dataset source, preserving model training settings")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--dry-run", action="store_true", help="print commands without writing files")
    parser.add_argument("--annotations")
    parser.add_argument("--images-root")
    args = parser.parse_args()
    try:
        config = Experiment.load(args.config, data_source=args.data_source,
                                 annotations=args.annotations, images_root=args.images_root)
        Workflow(config, args.run, args.dry_run).execute(args.stage)
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
