# Biodiversity Edge AI

This repository consolidates three legacy biodiversity projects into one reproducible embedded machine learning teaching example:

- vision training and TFLite export from `inat2021ufam-master`;
- the spatio-temporal FCNet from `geo_prior_tf-master`;
- Raspberry Pi camera inference and benchmarking from `AI_Camera-main`.

Use Python 3.10--3.12. Python 3.12 is recommended for the workstation training environment.

The original research contribution covered CNN training, dynamic-range quantization (DRQ), geo-prior fusion, and Raspberry Pi deployment. Pruning is a **new teaching extension**, not an original thesis result. See [docs/RESULTS_PROVENANCE.md](docs/RESULTS_PROVENANCE.md).

## System

```text
image -> vision.tflite -> P(species | image) ---------+
                                                      +-> log-linear fusion -> prediction
lat/lon/date -> geo_prior.tflite -> P(species | geo) -+
```

If location is unavailable, inference falls back to the vision prediction. Both models must use the same class-map hash; the application refuses to fuse incompatible artifacts.

## Quick start for the dependency-light core

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m biodiversity_edge_ai.device.predict --help
```

For model training and export:

```bash
python -m pip install -e '.[train,dev]'
```

## Prepare the data

The deterministic preparation command reads the source annotation JSON once and generates image splits, a contiguous class map, Geo Prior CSV files, device benchmark metadata, and a provenance manifest:

```bash
PYTHONPATH=src python scripts/prepare_data.py \
  --annotations raw/inat2021/train_mini.json \
  --images-root raw/inat2021 \
  --output data/prepared \
  --num-classes 10 --max-per-class 50 \
  --val-fraction 0.15 --test-fraction 0.15 --seed 42
```

Download, checksum, output layout, training, export, and evaluation commands are documented end to end in [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md).

After setting `RAW_JSON` and `IMAGES_ROOT`, the complete practical-scale workstation flow can also be run through the provided Makefile:

```bash
make setup
make workstation RAW_JSON=/path/to/train_mini.json IMAGES_ROOT=/path/to/extracted/data
```

Before downloading the real dataset, verify the entire software path with tiny generated data:

```bash
make smoke
```

The recorded stages and example outputs are in [docs/SMOKE_TEST.md](docs/SMOKE_TEST.md).

On Raspberry Pi, install the platform-provided `picamera2`, GPIO, and SPI packages, then install this project with the lightweight runtime. TensorFlow Lite Runtime may be installed separately when it is available for the Pi OS/Python combination.

## Reproducible workflow

1. Create a class map and fixed train/validation/test split.
2. Train a vision backbone with `scripts/train_vision.py`.
3. Train the geo-prior with `scripts/train_geo_prior.py`.
4. Export both models and manifests with `scripts/export_tflite.py`.
5. Tune the fusion weight on validation data only.
6. Evaluate the selected weight once on the held-out test set.
7. Run `scripts/benchmark_rpi.py` on the target Raspberry Pi.
8. Add the pruning extension only after the original baseline is reproduced.

The legacy repositories remain unchanged. Their roles and known incompatibilities are documented in [docs/INTEGRATION.md](docs/INTEGRATION.md).
The suggested 60--90 minute teaching sequence and practical deliverables are in [docs/LECTURE_DEMO.md](docs/LECTURE_DEMO.md).

## Link the existing artifacts

The migration command creates a normalized class map and links existing TFLite/checkpoint files into the ignored `artifacts/legacy` directory:

```bash
PYTHONPATH=src python scripts/migrate_legacy.py \
  --vision-repo /path/to/inat2021ufam-master \
  --geo-repo /path/to/geo_prior_tf-master \
  --device-repo /path/to/AI_Camera-main
```

Add `--copy` when preparing a self-contained transfer to Raspberry Pi. Linking is the default so the same 186 MB of legacy TFLite files is not duplicated during cleanup.

## Run an existing vision model

After migration, the existing CNN artifacts can be used in vision-only mode. This path does not require a Geo Prior model:

```bash
PYTHONPATH=src python scripts/predict.py \
  --image /path/to/photo.jpg \
  --vision-model artifacts/legacy/vision_models/model_efficientnet_b0_inat2021_drq.tflite \
  --vision-manifest artifacts/legacy/vision_models/model_efficientnet_b0_inat2021_drq.manifest.json \
  --class-map artifacts/legacy/class_maps/inat2021_10000.json
```

To enable fusion, first convert the legacy Geo Prior checkpoint as described in
[docs/INTEGRATION.md](docs/INTEGRATION.md), export it to TFLite, and add
`--geo-model`, `--geo-manifest`, `--latitude`, `--longitude`, and `--date`.

## Pruning extension experiment

This command creates a new magnitude-pruned Keras artifact. It is a classroom extension and must not be presented as work completed in the original report:

```bash
PYTHONPATH=src python scripts/prune_vision.py \
  --keras-model artifacts/models/vision_baseline.keras \
  --output artifacts/models/vision_pruned_50.keras \
  --sparsity 0.50 \
  --data-dir data/prepared \
  --finetune-epochs 3
```

Export the baseline and pruned models with the same TFLite mode, then compare accuracy, file size, and Raspberry Pi latency. Sparse weights alone do not guarantee a smaller or faster TFLite model.

## Raspberry Pi measurement

Use a self-contained artifact directory on the Pi, then measure the same image set for every model variant:

```bash
PYTHONPATH=src python scripts/benchmark_rpi.py \
  --images data/test \
  --vision-model artifacts/models/vision.tflite \
  --vision-manifest artifacts/models/vision.manifest.json \
  --class-map artifacts/class_map.json \
  --warmup 10 --repetitions 5 \
  --output results/pi_benchmark.json
```

For camera capture, run `scripts/rpi_camera.py` with the same model arguments. Fixed latitude and longitude are supported; live GPS ingestion is intentionally left as a separate hardware task.

## Current verification boundary

- Unit tests, Python byte-code compilation, command-line parsing, and wheel packaging pass in the current workspace.
- The original repositories were not modified; their existing artifacts are linked under ignored `artifacts/legacy` paths.
- TensorFlow/TFLite Runtime is not installed in this workspace, so model conversion and real inference must be run in the training environment or on the Pi.
- Raspberry Pi camera, display, latency, power, and live GPS behavior still require validation on the physical device.
