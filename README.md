# Biodiversity Edge AI

**English** | [简体中文](README.zh-CN.md)

Offline species recognition with image classification, a spatio-temporal Geo Prior, quantization, pruning, and Raspberry Pi inference in one Python package.

[Quick start](#quick-start) · [Dataset](#dataset) · [Training](#training) · [Optimization](#optimization) · [Evaluation](#evaluation) · [Inference](#inference) · [Raspberry Pi](#raspberry-pi) · [Architecture](#architecture) · [Verification](#verification) · [Teaching](#teaching)

## Quick start

Use Python 3.10–3.12; Python 3.12 is recommended. On a macOS or Linux workstation, clone the repository, create an environment, and run the small-data pipeline:

```bash
git clone https://github.com/winson5566/biodiversity-edge-ai.git
cd biodiversity-edge-ai
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,dev]'
make smoke
```

Run subsequent workstation commands from the repository root with this environment active.

The smoke run generates **24 synthetic images in two classes**, prepares a 16/4/4 split, trains both models, exports FP32/DRQ/INT8, applies 50% pruning with fine-tuning, and benchmarks fused predictions. It does not need an iNaturalist download or ImageNet weights.

Outputs: `artifacts/smoke/models/` and `artifacts/smoke/results/tradeoffs.md`. Synthetic accuracy and workstation timings verify the software path and do not establish real species accuracy or Pi performance. The exact verified stages and results are listed in [Verification](#verification).

## Dataset

The project uses **iNaturalist 2021**, distributed through the AWS Open Data Program. Training defaults to **Train Mini: 500,000 images, 10,000 classes**, using every usable image without a per-class cap. Full Train is available through `DATA_SOURCE=full`. See the [official dataset page and terms](https://github.com/visipedia/inat_comp/tree/master/2021).

### Download catalog

| File | Size | MD5 |
|---|---:|---|
| [Train Mini images (default)](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz) | 42 GB | `db6ed8330e634445efc8fec83ae81442` |
| [Train Mini annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz) | 45 MB | `395a35be3651d86dc3b0d365b8ea5f92` |
| [Train images](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz) | 224 GB | `e0526d53c7f7b2e3167b2b43bb2690ed` |
| [Train annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz) | 221 MB | `38a7bb733f7a09214d44293460ec0021` |
| [Validation images](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz) | 8.4 GB | `f6f6e0e242e3d4c9569ba56400938afc` |
| [Validation annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz) | 9.4 MB | `4d761e0f6a86cc63e8f7afc91f6a8f0b` |
| [Test images](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.tar.gz) | 43 GB | `7124b949fe79bfa7f7019a15ef3dbd06` |
| [Test info](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.json.tar.gz) | 21 MB | `7a9413db55c6fa452824469cc7dd9d3d` |

All image archives contain JPEG images with a maximum dimension of 500 pixels. Extraction can take time; allow storage for both archives and extracted files.

<details>
<summary>S3 locations for all dataset files</summary>

```text
s3://ml-inat-competition-datasets/2021/train_mini.tar.gz
s3://ml-inat-competition-datasets/2021/train_mini.json.tar.gz
s3://ml-inat-competition-datasets/2021/train.tar.gz
s3://ml-inat-competition-datasets/2021/train.json.tar.gz
s3://ml-inat-competition-datasets/2021/val.tar.gz
s3://ml-inat-competition-datasets/2021/val.json.tar.gz
s3://ml-inat-competition-datasets/2021/public_test.tar.gz
s3://ml-inat-competition-datasets/2021/public_test.json.tar.gz
```

</details>

### Download and prepare Mini

Download the default pair:

```bash
mkdir -p raw/inat2021
cd raw/inat2021
curl -C - -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz
curl -C - -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz
```

Before extraction, compare both hashes with the table. On Linux:

```bash
md5sum train_mini.tar.gz train_mini.json.tar.gz
```

On macOS:

```bash
md5 train_mini.tar.gz train_mini.json.tar.gz
```

Extract only after the hashes match:

```bash
tar -xzf train_mini.tar.gz
tar -xzf train_mini.json.tar.gz
cd ../..
```

For full Train, use the `train.tar.gz` and `train.json.tar.gz` links above and extract both into `raw/inat2021`. Validation and Test are optional downloads, extracted into the same directory.

The layout below shows all available sources; only the selected training pair is required:

```text
raw/inat2021/
  train_mini.json
  train_mini/category/image.jpg
  train.json
  train/category/image.jpg
  val.json
  val/category/image.jpg
  public_test.json
  public_test/image.jpg
```

Use `raw/inat2021` as `IMAGES_ROOT`: the annotation paths already contain `train_mini/`, `train/`, or `val/`.

Prepare the default Mini source:

```bash
make prepare
```

This writes `data/prepared/class_map.json`, `dataset_manifest.json`, `images/{train,val,test}/`, and `metadata/{train,val,test}.csv`. The manifest records the input hash, class mapping, selection rules, counts, and skipped records.

### Which splits does this project use?

The default pipeline creates deterministic local train/validation/test splits from the selected **training source**, approximately 70%/15%/15%, rounded per class. The official Validation and public Test archives in the catalog are not automatically used by `make workstation`.

Official public Test info has no ground-truth annotations, so it cannot be passed to the supervised preparation command or used to calculate local Top-1 accuracy. Official Validation uses a separate labeled source; evaluation must align its categories with the trained class map. See the [dataset annotation notes](https://github.com/visipedia/inat_comp/tree/master/2021#annotation-format-notes).

The preparation command accepts COCO-style JSON with `images`, `annotations`, and `categories`. It uses `image.file_name`, `image.latitude`, `image.longitude`, `image.date`, and `annotation.category_id`. Invalid or missing location/date data remains available to vision training but is excluded from Geo Prior training.

## Training

### Default: Mini

After `make prepare`, train the image classifier and six-feature Geo Prior:

```bash
make train
```

The default vision model is MobileNetV2 with ImageNet initialization, 128×128 input, and width multiplier 0.5. The Geo Prior is a residual FCNet using longitude, latitude, and date encodings. Both models share the same class map.

To run preparation, training, export, pruning, and evaluation together:

```bash
make workstation
```

### Full Train

After downloading and extracting the full pair:

```bash
make workstation DATA_SOURCE=full
```

| Mode | Prepared data | Models | Results |
|---|---|---|---|
| Mini | `data/prepared` | `artifacts/models` | `artifacts/results` |
| Full | `data/prepared_full` | `artifacts/models_full` | `artifacts/results_full` |

Pass `DATA_SOURCE=full` to individual stages too, for example `make train DATA_SOURCE=full`. The remaining examples use Mini paths; substitute the corresponding full or custom paths for other runs.

### Smaller runs and custom paths

Select 10 classes with at most 50 images per class:

```bash
make workstation NUM_CLASSES=10 MAX_PER_CLASS=50 \
  DATASET=data/prepared_demo \
  MODEL_DIR=artifacts/models_demo RESULT_DIR=artifacts/results_demo
```

Point to a dataset stored elsewhere:

```bash
make workstation RAW_JSON=/path/to/train_mini.json IMAGES_ROOT=/path/to/raw
```

Use fresh data, model, and result directories when changing source paths, class selection, or split settings. Make reuses existing artifacts and does not detect changes to command-line variables.

`HEAD_EPOCHS`, `FINETUNE_EPOCHS`, `GEO_EPOCHS`, and `BATCH_SIZE` control training. `MAX_PER_CLASS` is empty by default (uncapped); `MIN_PER_CLASS=20` controls class eligibility. Full-source preparation builds indexes in memory, so memory and storage needs grow with the input size. Full-dataset training has not been run as part of the smoke test.

## Optimization

Export the trained baseline and Geo Prior, then create the pruned variant:

```bash
make export
make optimize
```

| Vision artifact | Optimization |
|---|---|
| `vision_fp32.tflite` | FP32 baseline |
| `vision_drq.tflite` | Dynamic-range weight quantization |
| `vision_int8.tflite` | Full INT8, calibrated using training images |
| `vision_pruned_50_drq.tflite` | 50% magnitude pruning, fine-tuning, then DRQ |

Files are written to `artifacts/models/`; each TFLite file has a `.tflite.manifest.json` companion. The Geo Prior export is `geo_prior_fp32.tflite`.

For another sparsity target:

```bash
PYTHONPATH=src python scripts/prune_vision.py \
  --keras-model artifacts/models/vision_baseline.keras \
  --output artifacts/models/vision_pruned_30.keras \
  --sparsity 0.30 --data-dir data/prepared \
  --input-size 128 --input-scale minus1_1 --finetune-epochs 3
```

Export that Keras model with `scripts/export_tflite.py` using a distinct artifact name and model ID.

## Evaluation

Run the four standard vision variants with the same Geo Prior and test images:

```bash
make benchmark
```

Outputs include raw JSON files plus `artifacts/results/tradeoffs.csv` and `tradeoffs.md`. Compare Top-1 accuracy, model bytes, invoke latency, end-to-end latency, throughput, and RSS memory change. Energy measurements require an external measurement method.

The Makefile uses a fixed fusion weight of `alpha=0.3`. It does not tune that value automatically. Select the weight on validation data before a final test comparison, and keep the split, preprocessing, thread count, warm-up, and repetitions fixed across variants.

For a controlled device run:

```bash
PYTHONPATH=src python scripts/benchmark_rpi.py \
  --images data/prepared/images/test \
  --metadata-csv data/prepared/metadata/test.csv \
  --vision-model artifacts/models/vision_drq.tflite \
  --vision-manifest artifacts/models/vision_drq.tflite.manifest.json \
  --geo-model artifacts/models/geo_prior_fp32.tflite \
  --geo-manifest artifacts/models/geo_prior_fp32.tflite.manifest.json \
  --class-map data/prepared/class_map.json \
  --alpha 0.3 --threads 1 --warmup 10 --repetitions 5 \
  --output artifacts/results/pi_benchmark.json
```

Sparse weights alone do not guarantee smaller files or lower latency. Base the deployment choice on target-device measurements. The comparison protocol is in [Evaluation](#evaluation).

## Inference

Predict from one image:

```bash
PYTHONPATH=src python scripts/predict.py \
  --image /path/to/photo.jpg \
  --vision-model artifacts/models/vision_drq.tflite \
  --vision-manifest artifacts/models/vision_drq.tflite.manifest.json \
  --class-map data/prepared/class_map.json
```

For geo-fused inference, append these arguments to the same command, using the observation's actual coordinates and date:

```bash
--geo-model artifacts/models/geo_prior_fp32.tflite \
--geo-manifest artifacts/models/geo_prior_fp32.tflite.manifest.json \
--latitude -43.5321 --longitude 172.6362 --date 2026-09-08
```

Missing location or date triggers vision-only inference. Models with incompatible class maps are rejected.

## Raspberry Pi

Copy the project, selected TFLite files, their manifests, and `class_map.json` to the Pi, preserving the paths used in your commands. For benchmarking, also copy the prepared test images and metadata; materialize symlinked images when transferring them to another machine.

Install the OS-provided Picamera2, GPIO, and SPI dependencies, then install the lightweight project dependencies:

```bash
python -m pip install -e '.[rpi]'
```

Install a compatible TensorFlow Lite Runtime for your Pi OS and Python version. If using a virtual environment, ensure it can access the OS-provided camera libraries.

Capture one frame and predict:

```bash
PYTHONPATH=src python scripts/rpi_camera.py \
  --vision-model artifacts/models/vision_drq.tflite \
  --vision-manifest artifacts/models/vision_drq.tflite.manifest.json \
  --class-map data/prepared/class_map.json
```

Add `--display` for the ST7789 display. Geo fusion accepts `--geo-model`, `--geo-manifest`, `--latitude`, and `--longitude`; camera inference uses today's date. Live GPS ingestion is not implemented. Camera/display operation, latency, and power need validation on physical hardware.

## Architecture

```text
Image ─────────────> Vision TFLite ────┐
                                      ├──> Fusion ──> Species prediction
Latitude/longitude/date ──> Geo Prior ─┘
```

The workstation trains and exports models; the Pi loads TFLite artifacts. Shared manifests declare input shape, scaling, dtype, optimization, and the class-map hash so that the two prediction vectors remain aligned.

### Components and compatibility

| Capability | Implementation |
|---|---|
| Dataset preparation and class map | `scripts/prepare_data.py`, `pipeline.py` |
| Vision model and training | `models/vision.py`, `training/vision.py` |
| Geo Prior and metadata encoding | `metadata.py`, `models/geo_prior.py`, `training/geo_prior.py` |
| TFLite export and manifests | `export/tflite.py`, `manifest.py` |
| Quantization and pruning | `export/tflite.py`, `training/pruning.py` |
| Fusion | `fusion.py`, `inference.py` |
| Benchmarking | `evaluation/benchmark.py`, `scripts/summarize_benchmarks.py` |
| Pi camera and display | `device/rpi_camera.py`, `device/display.py` |

Before fusion, the application checks that both output dimensions match the class-map length and that both manifests have the same class-map SHA-256 value. It also records input shape, dtype, scaling, role, and optimization mode. Geo features have a fixed order: sine/cosine longitude, sine/cosine latitude, and sine/cosine date. An incompatible artifact fails explicitly.

## Verification

The following small-data run was completed on 8 September 2026 using Python 3.12.8, NumPy 1.26.4, TensorFlow 2.16.2, Keras 3.8.0, and an Apple M4 workstation:

1. Created 24 synthetic source images and a COCO-style annotation file.
2. Prepared deterministic 16/4/4 train/validation/test splits with a shared two-class map.
3. Trained MobileNetV2 and the six-feature Geo Prior.
4. Exported vision FP32, DRQ, and full-INT8 TFLite models plus the Geo Prior TFLite model.
5. Applied 50% magnitude pruning, fine-tuned with fixed masks, and exported the pruned DRQ model.
6. Loaded every artifact through the integrated runtime, ran fused predictions, and generated JSON, CSV, and Markdown trade-off results.

| Model | Optimization | Input | Size | Test Top-1 | End-to-end median |
|---|---|---|---:|---:|---:|
| `vision_fp32` | FP32 | float32 | 2,761,512 bytes | 50% | 0.263 ms |
| `vision_drq` | DRQ | float32 | 870,752 bytes | 50% | 0.237 ms |
| `vision_int8` | full INT8 | int8 | 973,752 bytes | 50% | 0.192 ms |
| `vision_pruned_50_drq` | pruning + DRQ | float32 | 862,096 bytes | 50% | 0.230 ms |

The common Geo Prior model was 15,392 bytes. The run used four held-out synthetic images, one warm-up, and one repetition. These figures only verify the runtime path; they are not real-data accuracy results or Raspberry Pi measurements.

### Measurement protocol

For a deployment comparison, keep the test split, class map, image preprocessing, input resolution, Geo Prior artifact, fusion weight, Pi model, OS image, cooling, power supply, CPU governor, TFLite thread count, warm-up count, and measured repetitions fixed. Record the model manifests, raw JSON outputs, ambient conditions, and power-meter method.

| Variant | Top-1 | Model bytes | Invoke median | End-to-end P95 | RSS delta | Energy/image |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | TBD | TBD | TBD | TBD | TBD | TBD |
| DRQ | TBD | TBD | TBD | TBD | TBD | TBD |
| Full INT8 | TBD | TBD | TBD | TBD | TBD | TBD |
| Pruned 50% + DRQ | TBD | TBD | TBD | TBD | TBD | TBD |

Choose the smallest model that stays within the agreed accuracy loss and meets the P95 latency target. Optional experiments can add 30% or 70% pruning and pruning plus full INT8.

## Teaching

This project supports a 60–90 minute session plus a practical lab:

1. Explain offline species recognition and Pi constraints.
2. Trace image preprocessing, vision inference, softmax, and Top-K output.
3. Encode latitude, longitude, and day of year, then compare vision-only and log-linear fusion.
4. Export FP32, DRQ, full INT8, and 30%/50%/70% pruned variants.
5. Benchmark each artifact on the same Pi workload and justify a model choice from accuracy, size, latency, memory, and energy.

For a live session, use 5–10 classes, 128×128 MobileNetV2, and a pre-exported large-class model for scale. Each group should submit the fixed class map and split description, model manifests, raw Pi benchmark JSON, a completed trade-off table, and a recommendation backed by measured results.

Run core tests with:

```bash
make test
```
