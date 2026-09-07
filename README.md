# Biodiversity Edge AI

**English** · [简体中文](README.zh-CN.md)

An offline species-recognition system for Raspberry Pi. It trains an image classifier and a spatio-temporal Geo Prior, exports TFLite models, evaluates quantization, and runs camera inference on-device.

## Quick start

Use Python 3.10–3.12; Python 3.12 is recommended.

```bash
git clone https://github.com/winson5566/biodiversity-edge-ai.git
cd biodiversity-edge-ai
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,dev]'
make smoke
```

`make smoke` needs no download. It creates 24 synthetic images, trains both models, exports FP32/DRQ/full-INT8 variants, and writes a comparison table to `artifacts/smoke/results/tradeoffs.md`.

## Dataset

The default source is **iNaturalist 2021 Train Mini**: 500,000 images across 10,000 species. Full Train uses the same pipeline.

| Mode | Download | Run |
|---|---:|---|
| Mini (default) | 42 GB images + 45 MB annotations | `make workstation` |
| Full Train | 224 GB images + 221 MB annotations | `make workstation DATA_SOURCE=full` |

Download the Mini pair, verify it, and extract it into `raw/inat2021`:

```bash
mkdir -p raw/inat2021 && cd raw/inat2021
curl -C - -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz
curl -C - -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz
md5sum train_mini.tar.gz train_mini.json.tar.gz
tar -xzf train_mini.tar.gz
tar -xzf train_mini.json.tar.gz
cd ../..
```

Expected Mini MD5 values are `db6ed8330e634445efc8fec83ae81442` and `395a35be3651d86dc3b0d365b8ea5f92`. On macOS, use `md5` in place of `md5sum`.

<details>
<summary>All dataset downloads, hashes, and S3 locations</summary>

| File | Size | MD5 | S3 location |
|---|---:|---|---|
| [Train Mini images](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz) | 42 GB | `db6ed8330e634445efc8fec83ae81442` | `s3://ml-inat-competition-datasets/2021/train_mini.tar.gz` |
| [Train Mini annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz) | 45 MB | `395a35be3651d86dc3b0d365b8ea5f92` | `s3://ml-inat-competition-datasets/2021/train_mini.json.tar.gz` |
| [Train images](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz) | 224 GB | `e0526d53c7f7b2e3167b2b43bb2690ed` | `s3://ml-inat-competition-datasets/2021/train.tar.gz` |
| [Train annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz) | 221 MB | `38a7bb733f7a09214d44293460ec0021` | `s3://ml-inat-competition-datasets/2021/train.json.tar.gz` |
| [Validation images](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz) | 8.4 GB | `f6f6e0e242e3d4c9569ba56400938afc` | `s3://ml-inat-competition-datasets/2021/val.tar.gz` |
| [Validation annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz) | 9.4 MB | `4d761e0f6a86cc63e8f7afc91f6a8f0b` | `s3://ml-inat-competition-datasets/2021/val.json.tar.gz` |
| [Public Test images](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.tar.gz) | 43 GB | `7124b949fe79bfa7f7019a15ef3dbd06` | `s3://ml-inat-competition-datasets/2021/public_test.tar.gz` |
| [Public Test info](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.json.tar.gz) | 21 MB | `7a9413db55c6fa452824469cc7dd9d3d` | `s3://ml-inat-competition-datasets/2021/public_test.json.tar.gz` |

Images are JPEG files with a maximum dimension of 500 pixels. Extraction creates `train_mini/category/image.jpg`, `train/category/image.jpg`, or `val/category/image.jpg`. Read the [official dataset page and terms](https://github.com/visipedia/inat_comp/tree/master/2021) before downloading.

</details>

The standard pipeline makes a deterministic 70%/15%/15% split from the selected training source. It does not automatically use the official Validation or public Test downloads. Public Test has no labels, so it cannot calculate local accuracy.

## Train and export

Run the complete Mini workflow:

```bash
make workstation
```

This prepares data, trains the selected vision backbone (MobileNetV2 by default) and the six-feature Geo Prior, exports TFLite models, and creates benchmark tables.

MobileNetV2 is the default backbone. Select any evaluated architecture with `VISION_BACKBONE`: `efficientnet-b0`, `mobilenet-v3-large`, `mobilenet-v2`, `resnet-50`, `resnet-101`, `convnext-tiny`, or `convnext-small`.

```bash
make workstation VISION_BACKBONE=efficientnet-b0
```

Use a small reproducible subset:

```bash
make workstation NUM_CLASSES=10 MAX_PER_CLASS=50 \
  DATASET=data/prepared_demo \
  MODEL_DIR=artifacts/models_demo RESULT_DIR=artifacts/results_demo
```

| Stage | Command | Main output |
|---|---|---|
| Prepare | `make prepare` | fixed splits, metadata CSVs, `class_map.json` |
| Train | `make train` | `vision_baseline.keras`, `geo_prior.keras` |
| Export | `make export` | FP32, DRQ, INT8, and Geo Prior TFLite |
| Compare | `make benchmark` | JSON, CSV, and Markdown trade-off tables |

## Architecture

```mermaid
flowchart LR
  subgraph WS[Workstation]
    RAW[Images + labels + geo metadata] --> PREP[Prepare fixed splits]
    PREP --> VTRAIN[Train vision model]
    PREP --> GTRAIN[Train Geo Prior]
    VTRAIN --> VEXPORT[Export vision TFLite variants]
    GTRAIN --> GEXPORT[Export Geo Prior TFLite]
  end

  subgraph ART[Deployable artifacts]
    MAP[class_map.json]
    VM[vision.tflite + manifest]
    GM[geo_prior.tflite + manifest]
  end

  subgraph PI[Raspberry Pi]
    IMAGE[Camera frame or image] --> VINF[Vision inference]
    META[Latitude + longitude + date] --> GINF[Geo Prior inference]
    VINF --> FUSE[Validate artifacts and fuse]
    GINF --> FUSE
    FUSE --> OUT[Top-K species prediction]
  end

  VEXPORT --> VM
  GEXPORT --> GM
  MAP --> VINF
  MAP --> GINF
  VM --> VINF
  GM --> GINF
```

Fusion requires matching class-map hashes and output dimensions. Missing location or date uses vision-only inference.

## Evaluate and deploy

Benchmark the standard variants on the same held-out images:

```bash
make benchmark
```

Compare FP32, DRQ, and full INT8 by Top-1 accuracy, model size, invocation latency, end-to-end latency, memory, and energy. Keep the split, preprocessing, Geo Prior, fusion weight, Pi configuration, thread count, warm-up, and repetitions unchanged across runs.

Predict one image:

```bash
PYTHONPATH=src python scripts/predict.py \
  --image /path/to/photo.jpg \
  --vision-model artifacts/models/vision_drq.tflite \
  --vision-manifest artifacts/models/vision_drq.tflite.manifest.json \
  --class-map data/prepared/class_map.json
```

For a Raspberry Pi camera, install PiCamera2, GPIO/SPI support, this project's `.[rpi]` dependencies, and a compatible TensorFlow Lite Runtime. Then run:

```bash
PYTHONPATH=src python scripts/rpi_camera.py \
  --vision-model artifacts/models/vision_drq.tflite \
  --vision-manifest artifacts/models/vision_drq.tflite.manifest.json \
  --class-map data/prepared/class_map.json
```

Add `--display` for ST7789 output. Add `--geo-model`, `--geo-manifest`, `--latitude`, and `--longitude` for Geo Prior fusion.

## Reference hardware and reported results

The following values are reported reference measurements, not outputs reproduced by `make smoke`. The report compares seven vision architectures; the default reproducible workflow above uses MobileNetV2.

### Raspberry Pi configuration

| Component | Configuration |
|---|---|
| Compute | Raspberry Pi Zero 2 W — quad-core ARM Cortex-A53 at 1.0 GHz, 512 MB RAM; Raspberry Pi OS Lite 64-bit |
| Camera | Raspberry Pi CSI Sony IMX219, 8 MP |
| Display | 1.3-inch Waveshare IPS LCD (ST7789) |
| Location hardware | L76K GPS |
| Power | PiSugar 3 battery-management board with 1,200 mAh single-cell Li-ion battery |
| Storage | 32 GB microSD card |
| Bill of materials | NZ$158, excluding the custom enclosure and buttons |

The hardware configuration includes GPS. The current camera command accepts fixed `--latitude` and `--longitude`; live GPS acquisition is not implemented in this repository.

### iNat2021 validation accuracy

| Model | FP32 Top-1 | FP32 Top-5 | DRQ Top-1 | DRQ Top-5 |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 73.77% | 89.39% | 70.84% | 87.60% |
| MobileNetV3-Large | 71.63% | 87.70% | 69.64% | 86.75% |
| MobileNetV2 | 68.62% | 86.25% | 68.31% | 86.14% |
| ResNet-50 | 75.61% | 90.63% | 75.50% | 90.54% |
| ResNet-101 | 77.85% | 91.89% | 77.80% | 91.85% |
| ConvNeXt-Tiny | 81.89% | 94.13% | 81.74% | 94.06% |
| ConvNeXt-Small | 83.33% | 94.81% | 83.29% | 94.78% |

For EfficientNet-B0, Geo Prior fusion with log-linear α = 0.3 raises Top-1 from 73.77% to 83.26% (FP32), and from 70.84% to 81.33% (DRQ).

### Model size and complexity

| Model | FP32 size | DRQ size | Parameters | FLOPs |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 64.07 MB | 16.15 MB | 16.80 M | 0.81 G |
| MobileNetV3-Large | 47.95 MB | 12.07 MB | 12.57 M | 0.61 G |
| MobileNetV2 | 57.28 MB | 14.41 MB | 15.02 M | 0.63 G |
| ResNet-50 | 167.74 MB | 42.04 MB | 43.97 M | 8.20 G |
| ResNet-101 | 240.09 MB | 60.20 MB | 62.94 M | 15.60 G |
| ConvNeXt-Tiny | 135.44 MB | 34.05 MB | 35.50 M | 9.00 G |
| ConvNeXt-Small | 217.94 MB | 54.84 MB | 58.13 M | 17.40 G |

### Pi Zero 2 W latency and throughput

Each entry is mean latency in milliseconds / FPS. `—` means that format was not measured on the device.

| Model | FP32, 1 thread | FP32, 4 threads | DRQ, 1 thread | DRQ, 4 threads |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 903.95 / 1.11 | 753.08 / 1.33 | 626.26 / 1.60 | 461.50 / 2.17 |
| MobileNetV3-Large | 224.87 / 4.45 | 116.28 / 8.60 | 239.81 / 4.17 | 205.18 / 4.87 |
| MobileNetV2 | 221.25 / 4.52 | 114.18 / 8.76 | 311.95 / 3.21 | 182.61 / 5.48 |
| ResNet-50 | — | — | 1,731.15 / 0.58 | 653.47 / 1.53 |
| ResNet-101 | — | — | 3,213.36 / 0.31 | 1,192.17 / 0.84 |
| ConvNeXt-Tiny | — | — | 6,484.90 / 0.15 | 5,243.91 / 0.19 |
| ConvNeXt-Small | — | — | 11,302.33 / 0.09 | 10,884.12 / 0.09 |

### Energy and battery life

Energy measurements use four inference threads and a net device power of 1.5 W. Battery life uses a 30-second capture-infer-display cycle.

| Model | FP32: energy / FPS/W / life | DRQ: energy / FPS/W / life |
|---|---:|---:|
| EfficientNet-B0 | 1,127.8 mJ / 0.89 / 4.68 h | 691.2 mJ / 1.45 / 4.55 h |
| MobileNetV3-Large | 174.4 mJ / 5.73 / 4.66 h | 308.0 mJ / 3.25 / 4.62 h |
| MobileNetV2 | 171.2 mJ / 5.84 / 4.65 h | 273.7 mJ / 3.65 / 4.60 h |
| ResNet-50 | — | 980.4 mJ / 1.02 / 4.40 h |
| ResNet-101 | — | 1,785.7 mJ / 0.56 / 4.25 h |
| ConvNeXt-Tiny | — | 7,894.7 mJ / 0.13 / 3.93 h |
| ConvNeXt-Small | — | 16,666.7 mJ / 0.06 / 3.70 h |

## Verification

Run the full generated-data check with `make smoke`, and the unit tests with `make test`.
