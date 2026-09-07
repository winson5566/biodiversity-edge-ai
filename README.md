# Biodiversity Edge AI

Biodiversity Edge AI is an end-to-end embedded machine learning system for offline species recognition. It includes image classification, a spatio-temporal prior, model optimization, multimodal fusion, and Raspberry Pi deployment in one Python package.

Use Python 3.10--3.12. Python 3.12 is recommended for workstation training.

## System

```text
image -> vision.tflite -> P(species | image) ---------+
                                                      +-> log-linear fusion -> prediction
lat/lon/date -> geo_prior.tflite -> P(species | geo) -+
```

If location is unavailable, inference falls back to the vision prediction. Both models must use the same class-map hash; the application refuses to fuse incompatible artifacts.

## Quick start

Install the training and development dependencies:

```bash
python -m pip install -e '.[train,dev]'
```

Run the unit tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Before downloading the full dataset, verify the complete pipeline with generated data:

```bash
make smoke
```

This creates a two-class dataset, trains the vision and Geo Prior models, exports FP32/DRQ/INT8 TFLite variants, prunes the vision model, benchmarks every variant, and builds a trade-off table. The recorded reference run is in [docs/SMOKE_TEST.md](docs/SMOKE_TEST.md).

## Prepare a dataset

The preparation command reads iNaturalist-format annotations and produces fixed image splits, a contiguous class map, Geo Prior CSV files, calibration inputs, benchmark metadata, and a dataset manifest:

```bash
PYTHONPATH=src python scripts/prepare_data.py \
  --annotations raw/inat2021/train_mini.json \
  --images-root raw/inat2021 \
  --output data/prepared \
  --num-classes 10 --max-per-class 50 \
  --val-fraction 0.15 --test-fraction 0.15 --seed 42
```

Dataset download, checksum verification, required fields, output layout, and validation commands are documented in [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md).

## Run the end-to-end workflow

After setting the annotation and image paths, run:

```bash
make workstation \
  RAW_JSON=/path/to/train_mini.json \
  IMAGES_ROOT=/path/to/extracted/data
```

The workflow:

1. creates a deterministic class map and train/validation/test split;
2. trains the vision classifier and Geo Prior;
3. exports FP32, dynamic-range quantized, and full-INT8 models;
4. fine-tunes and exports a 50% magnitude-pruned vision model;
5. benchmarks the same held-out images for every variant;
6. generates CSV and Markdown performance trade-off tables.

Individual stages are available as `make prepare`, `make train`, `make export`, `make optimize`, and `make benchmark`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the package structure and model-compatibility rules.

## Run inference

Vision-only prediction:

```bash
PYTHONPATH=src python scripts/predict.py \
  --image /path/to/photo.jpg \
  --vision-model artifacts/models/vision_drq.tflite \
  --vision-manifest artifacts/models/vision_drq.tflite.manifest.json \
  --class-map data/prepared/class_map.json
```

Add `--geo-model`, `--geo-manifest`, `--latitude`, `--longitude`, and `--date` to enable spatio-temporal fusion.

## Quantization, pruning, and trade-offs

The standard workflow produces four vision deployment variants:

- FP32 baseline;
- dynamic-range quantization (DRQ);
- full-integer INT8 quantization;
- 50% magnitude pruning followed by DRQ.

The pruning target can be changed when running its script directly:

```bash
PYTHONPATH=src python scripts/prune_vision.py \
  --keras-model artifacts/models/vision_baseline.keras \
  --output artifacts/models/vision_pruned_50.keras \
  --sparsity 0.50 \
  --data-dir data/prepared \
  --finetune-epochs 3
```

Compare accuracy, file size, invoke latency, end-to-end latency, memory, and energy on the target device. Sparse weights alone do not guarantee a smaller or faster TFLite model; the measured trade-off determines the deployment choice. The evaluation template is in [docs/PERFORMANCE_EVALUATION.md](docs/PERFORMANCE_EVALUATION.md).

## Raspberry Pi deployment

Install the platform-provided `picamera2`, GPIO, and SPI packages, then install this project with the lightweight runtime dependencies. Install TensorFlow Lite Runtime separately when it is available for the selected Pi OS and Python version.

Benchmark a model on the Pi:

```bash
PYTHONPATH=src python scripts/benchmark_rpi.py \
  --images data/prepared/images/test \
  --metadata-csv data/prepared/metadata/test.csv \
  --vision-model artifacts/models/vision_drq.tflite \
  --vision-manifest artifacts/models/vision_drq.tflite.manifest.json \
  --geo-model artifacts/models/geo_prior_fp32.tflite \
  --geo-manifest artifacts/models/geo_prior_fp32.tflite.manifest.json \
  --class-map data/prepared/class_map.json \
  --warmup 10 --repetitions 5 \
  --output artifacts/results/pi_benchmark.json
```

For live camera capture, use `scripts/rpi_camera.py` with the same model arguments. Fixed coordinates are supported; when coordinates are omitted, the application runs vision-only inference.

## Project guides

- [Data preparation and formats](docs/DATA_FORMATS.md)
- [Architecture and interfaces](docs/ARCHITECTURE.md)
- [Performance evaluation](docs/PERFORMANCE_EVALUATION.md)
- [Small-data full-pipeline test](docs/SMOKE_TEST.md)
- [Lecture and practical sequence](docs/LECTURE_DEMO.md)

Hardware latency, memory, power, camera, display, and GPS behavior must be measured on the target Raspberry Pi; workstation results are not a substitute for device measurements.
