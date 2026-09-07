# Data preparation: raw download to model-ready files

The project uses the iNaturalist 2021 classification data because one annotation file contains all information needed by both branches: image path, species category, latitude, longitude, and observation date. The preparation command does not download or redistribute images.

## 0. Create the training environment

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,dev]'
```

Run all following workstation commands from the repository root with this environment active.

After the raw data is available, the complete sequence below is also automated by:

```bash
make workstation
```

The default source is Train Mini, with 10,000 classes and no per-class image cap. Use `DATA_SOURCE=full` for full Train, or override `RAW_JSON` and `IMAGES_ROOT` for custom paths.

Use the individual `make prepare`, `make train`, `make export`, `make optimize`, and `make benchmark` targets when teaching or diagnosing one stage at a time. Run `make help` to see the available overrides.

## 1. Download the raw source

Read and accept the dataset terms on the [official iNaturalist 2021 page](https://github.com/visipedia/inat_comp/tree/master/2021) before downloading. The images are for non-commercial research and educational use and must not be redistributed.

Train Mini is the default source. It contains 500,000 images across 10,000 classes. Full Train is an optional larger source; both use the same annotation schema and preparation pipeline.

| Source | Archive | Download size | MD5 |
|---|---|---:|---|
| Mini (default) | [Images](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz) | 42 GB | `db6ed8330e634445efc8fec83ae81442` |
| Mini (default) | [Annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz) | 45 MB | `395a35be3651d86dc3b0d365b8ea5f92` |
| Full | [Images](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz) | 224 GB | `e0526d53c7f7b2e3167b2b43bb2690ed` |
| Full | [Annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz) | 221 MB | `38a7bb733f7a09214d44293460ec0021` |

Images are JPEG files with a maximum dimension of 500 pixels. Keep free space for both the compressed archives and extracted images. Selecting fewer classes reduces the prepared working set after extraction; it does not reduce the archive download size.

### Download Mini (default)

```bash
mkdir -p raw/inat2021
cd raw/inat2021

curl -C - -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz
curl -C - -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz

cd ../..
```

### Download full Train (optional)

```bash
mkdir -p raw/inat2021
cd raw/inat2021

curl -C - -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz
curl -C - -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz

cd ../..
```

### Verify and extract

Verify the downloaded pair against the MD5 table before extracting. For Mini, run:

```bash
# Linux
md5sum raw/inat2021/train_mini.tar.gz raw/inat2021/train_mini.json.tar.gz
# macOS
md5 raw/inat2021/train_mini.tar.gz raw/inat2021/train_mini.json.tar.gz
```

For full Train, use `train.tar.gz` and `train.json.tar.gz` in those commands. If a checksum differs, re-download that archive before continuing.

Extract the chosen pair:

```bash
# Mini
tar -xzf raw/inat2021/train_mini.tar.gz -C raw/inat2021
tar -xzf raw/inat2021/train_mini.json.tar.gz -C raw/inat2021

# Full Train (only if downloaded)
tar -xzf raw/inat2021/train.tar.gz -C raw/inat2021
tar -xzf raw/inat2021/train.json.tar.gz -C raw/inat2021
```

The resulting layout is:

```text
raw/inat2021/
  train_mini.json
  train_mini/category/image.jpg
  train.json                       # optional full source
  train/category/image.jpg         # optional full source
```

Use `raw/inat2021` as `--images-root` for either source. Annotation `file_name` values already include `train_mini/` or `train/`; pointing the root at either of those subdirectories would duplicate the prefix.

For a very small trial, the same preparation code also accepts any COCO-style JSON containing `images`, `annotations`, and `categories`, provided each `images[].file_name` points to a local image below `--images-root`.

## 2. Prepare the selected source

Default Mini preparation, retaining all usable images in all 10,000 classes:

```bash
PYTHONPATH=src python scripts/prepare_data.py \
  --annotations raw/inat2021/train_mini.json \
  --images-root raw/inat2021 \
  --output data/prepared \
  --num-classes 10000 \
  --val-fraction 0.15 \
  --test-fraction 0.15 \
  --seed 42 \
  --transfer-mode symlink
```

The Makefile equivalent is `make prepare`. Full-source preparation is `make prepare DATA_SOURCE=full`, which reads `raw/inat2021/train.json` and writes `data/prepared_full`.

Neither default applies `--max-per-class`. All usable images are assigned to the generated train, validation, or test split (approximately 70%/15%/15%, rounded per class). These are local splits of the selected training source, not the official validation/test splits. Check `dataset_manifest.json` for exact counts and skipped records.

For a smaller subset, set `NUM_CLASSES=10 MAX_PER_CLASS=50` and use separate dataset, model, and result directories:

```bash
make workstation NUM_CLASSES=10 MAX_PER_CLASS=50 \
  DATASET=data/prepared_demo \
  MODEL_DIR=artifacts/models_demo RESULT_DIR=artifacts/results_demo
```

`--num-classes 10` selects the ten eligible categories with the most usable images; ties are resolved by source category ID. To select specific species through `prepare_data.py`, replace it with `--category-ids 3 47 108 ...`. The two selection flags cannot be used together. `MIN_PER_CLASS` controls class eligibility separately from the optional `MAX_PER_CLASS` cap; its Makefile default is 20.

Use fresh output directories when changing the source, class selection, or split settings. Make reuses existing artifacts and does not detect a change to command-line variables.

The full-source loader reads the annotation JSON and builds record indexes in memory. Large runs require workstation memory and storage proportional to the source size. Full-dataset training has not been executed in the recorded smoke test.

The default transfer mode is `symlink`, which avoids duplicating the raw images. Use `--transfer-mode copy` for a self-contained archive, or `hardlink` when source and destination are on the same filesystem. The command refuses to write into a non-empty output directory, so an earlier split cannot be silently overwritten.

## 3. Generated files

```text
data/prepared/
  class_map.json
  dataset_manifest.json
  images/
    train/00000/*.jpg
    val/00000/*.jpg
    test/00000/*.jpg
    ...
  metadata/
    train.csv
    val.csv
    test.csv
```

`dataset_manifest.json` records the source paths, source annotation SHA-256, random seed, selection rules, original-to-contiguous label mapping, split counts, valid-location counts, and skipped source rows.

The numeric directory is the contiguous `label_id`. The human-readable output name is stored separately in `class_map.json`. This prevents alphabetical folder ordering from changing the model labels.

Every metadata CSV uses:

```csv
filename,image_id,source_category_id,label_id,class_name,latitude,longitude,date,valid,source_file
00003/123_photo.jpg,123,81,3,Example species,-43.5321,172.6362,2026-09-07,true,/raw/path/photo.jpg
```

- `filename` is relative to the corresponding `images/train`, `images/val`, or `images/test` directory.
- `source_category_id` preserves the source dataset ID.
- `label_id` is the new zero-based model output index.
- Invalid or missing location/date fields are blank and `valid=false`; those rows still train the vision model but are skipped by Geo Prior training.
- A source image ID and source file can occur in only one generated split.

## 4. Train both branches

The vision trainer detects `dataset_manifest.json`, reads images from the nested `images` directory, and preserves the display-name class map:

```bash
PYTHONPATH=src python scripts/train_vision.py \
  --data-dir data/prepared \
  --output artifacts/models/vision_baseline.keras \
  --class-map artifacts/models/class_map.json \
  --backbone mobilenet-v2 \
  --input-size 128 \
  --input-scale minus1_1
```

Train the Geo Prior on the exact same label IDs and report validation accuracy:

```bash
PYTHONPATH=src python scripts/train_geo_prior.py \
  --observations data/prepared/metadata/train.csv \
  --validation-observations data/prepared/metadata/val.csv \
  --num-classes 10000 \
  --output artifacts/models/geo_prior.keras
```

The value of `--num-classes` must equal the number of entries in `class_map.json`.

The individual commands below use the default Mini output paths. For full Train, the complete equivalent is `make workstation DATA_SOURCE=full`; use `data/prepared_full`, `artifacts/models_full`, and `artifacts/results_full` when running those stages individually.

Training, INT8 calibration, desktop inference, and Raspberry Pi inference all use the same centre-crop-then-resize policy and the same declared input scaling. These settings must not be changed for only one stage.

## 5. Export deployment models

FP32 vision model:

```bash
PYTHONPATH=src python scripts/export_tflite.py \
  --keras-model artifacts/models/vision_baseline.keras \
  --output artifacts/models/vision_fp32.tflite \
  --manifest artifacts/models/vision_fp32.manifest.json \
  --class-map artifacts/models/class_map.json \
  --model-id vision_fp32 \
  --role vision \
  --optimization fp32 \
  --input-scale minus1_1
```

Full-integer INT8 vision model. Representative images are selected round-robin across class directories rather than taking only the first class:

```bash
PYTHONPATH=src python scripts/export_tflite.py \
  --keras-model artifacts/models/vision_baseline.keras \
  --output artifacts/models/vision_int8.tflite \
  --manifest artifacts/models/vision_int8.manifest.json \
  --class-map artifacts/models/class_map.json \
  --model-id vision_int8 \
  --role vision \
  --optimization int8 \
  --input-scale minus1_1 \
  --representative-images data/prepared/images/train \
  --representative-limit 200
```

Geo Prior TFLite model:

```bash
PYTHONPATH=src python scripts/export_tflite.py \
  --keras-model artifacts/models/geo_prior.keras \
  --output artifacts/models/geo_prior_fp32.tflite \
  --manifest artifacts/models/geo_prior_fp32.manifest.json \
  --class-map artifacts/models/class_map.json \
  --model-id geo_prior_fp32 \
  --role geo_prior \
  --optimization fp32 \
  --input-scale encoded_geo
```

Both manifests contain the same class-map SHA-256. The inference application rejects fusion if their label spaces differ.

## 6. Create optimization variants

Dynamic-range quantization uses the same export command with `--optimization drq`. It quantizes eligible weights while retaining floating-point model input and output:

```bash
PYTHONPATH=src python scripts/export_tflite.py \
  --keras-model artifacts/models/vision_baseline.keras \
  --output artifacts/models/vision_drq.tflite \
  --manifest artifacts/models/vision_drq.manifest.json \
  --class-map artifacts/models/class_map.json \
  --model-id vision_drq \
  --role vision \
  --optimization drq \
  --input-scale minus1_1
```

Create a new 50% magnitude-pruned model and fine-tune while enforcing its mask:

```bash
PYTHONPATH=src python scripts/prune_vision.py \
  --keras-model artifacts/models/vision_baseline.keras \
  --output artifacts/models/vision_pruned_50.keras \
  --sparsity 0.50 \
  --data-dir data/prepared \
  --input-size 128 \
  --input-scale minus1_1 \
  --finetune-epochs 3
```

Export the pruned Keras model using the FP32, DRQ, or INT8 command above. Use a distinct model ID and output filename for every variant; never overwrite the baseline.

## 7. Evaluate the held-out test split

```bash
PYTHONPATH=src python scripts/benchmark_rpi.py \
  --images data/prepared/images/test \
  --metadata-csv data/prepared/metadata/test.csv \
  --vision-model artifacts/models/vision_fp32.tflite \
  --vision-manifest artifacts/models/vision_fp32.manifest.json \
  --geo-model artifacts/models/geo_prior_fp32.tflite \
  --geo-manifest artifacts/models/geo_prior_fp32.manifest.json \
  --class-map artifacts/models/class_map.json \
  --alpha 0.3 \
  --warmup 10 \
  --repetitions 5 \
  --output artifacts/results/fused_fp32.json
```

The benchmark searches class subdirectories recursively and joins metadata using the relative `filename`, avoiding collisions when two images share a basename.

Repeat the benchmark with each vision artifact while keeping the test split, Geo Prior, fusion alpha, thread count, warm-up, and repetitions fixed. This produces directly comparable accuracy, file-size, latency, and memory results.

`make benchmark` runs the FP32, DRQ, full INT8, and pruned-DRQ variants and produces both `artifacts/results/tradeoffs.csv` and `artifacts/results/tradeoffs.md`.

## 8. Reproducibility rules

1. Keep the raw archive and its checksum outside version control.
2. Commit `dataset_manifest.json`, `class_map.json`, and result JSON files, but do not redistribute dataset images.
3. Use the same seed and selected source category IDs for every model variant.
4. Choose fusion alpha and optimization settings on `val`; use `test` only for the final comparison.
5. Use only training images for INT8 representative calibration.
6. Never compare variants built from different class maps or splits.
