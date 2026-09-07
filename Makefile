PYTHON ?= python3
PYTHONPATH := src

DATA_SOURCE ?= mini
ifeq ($(DATA_SOURCE),mini)
RAW_JSON ?= raw/inat2021/train_mini.json
RUN_SUFFIX :=
else ifeq ($(DATA_SOURCE),full)
RAW_JSON ?= raw/inat2021/train.json
RUN_SUFFIX := _full
else
$(error DATA_SOURCE must be mini or full)
endif
IMAGES_ROOT ?= raw/inat2021
DATASET ?= data/prepared$(RUN_SUFFIX)
MODEL_DIR ?= artifacts/models$(RUN_SUFFIX)
RESULT_DIR ?= artifacts/results$(RUN_SUFFIX)

NUM_CLASSES ?= 10000
MIN_PER_CLASS ?= 20
# Empty means use every usable image in each selected class.
MAX_PER_CLASS ?=
INPUT_SIZE ?= 128
SEED ?= 42
VISION_WEIGHTS ?= imagenet
BATCH_SIZE ?= 32
HEAD_EPOCHS ?= 3
FINETUNE_EPOCHS ?= 5
GEO_EPOCHS ?= 30
GEO_EMBEDDING_DIM ?= 256
REPRESENTATIVE_LIMIT ?= 200
BENCHMARK_WARMUP ?= 10
BENCHMARK_REPETITIONS ?= 5

VISION_KERAS := $(MODEL_DIR)/vision_baseline.keras
GEO_KERAS := $(MODEL_DIR)/geo_prior.keras
VISION_FP32 := $(MODEL_DIR)/vision_fp32.tflite
VISION_DRQ := $(MODEL_DIR)/vision_drq.tflite
VISION_INT8 := $(MODEL_DIR)/vision_int8.tflite
GEO_FP32 := $(MODEL_DIR)/geo_prior_fp32.tflite

.PHONY: help setup prepare train export workstation benchmark test smoke

help:
	@echo "make setup       Install training and test dependencies"
	@echo "make prepare     Build deterministic image and metadata splits"
	@echo "make train       Train vision and Geo Prior Keras models"
	@echo "make export      Export FP32, DRQ, full INT8, and Geo TFLite models"
	@echo "make workstation Run prepare, train, export, and local benchmark"
	@echo "make benchmark   Benchmark and summarize all deployment variants"
	@echo "make smoke       Run the entire pipeline on tiny generated data"
	@echo "Default training: Mini source, 10000 classes, no per-class image cap"
	@echo "Full source: make workstation DATA_SOURCE=full"
	@echo "Small subset: make workstation NUM_CLASSES=10 MAX_PER_CLASS=50 DATASET=data/prepared_demo MODEL_DIR=artifacts/models_demo RESULT_DIR=artifacts/results_demo"
	@echo "Override paths, for example: make prepare RAW_JSON=/data/train_mini.json IMAGES_ROOT=/data"

setup:
	$(PYTHON) -m pip install -e '.[train,dev]'

prepare: $(DATASET)/dataset_manifest.json

$(DATASET)/dataset_manifest.json:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/prepare_data.py \
		--annotations $(RAW_JSON) \
		--images-root $(IMAGES_ROOT) \
		--output $(DATASET) \
		--num-classes $(NUM_CLASSES) \
		--min-per-class $(MIN_PER_CLASS) \
		$(if $(strip $(MAX_PER_CLASS)),--max-per-class $(MAX_PER_CLASS)) \
		--val-fraction 0.15 --test-fraction 0.15 \
		--seed $(SEED) --transfer-mode symlink

train: $(VISION_KERAS) $(GEO_KERAS)

$(VISION_KERAS): $(DATASET)/dataset_manifest.json
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_vision.py \
		--data-dir $(DATASET) --output $@ --class-map $(DATASET)/class_map.json \
		--backbone mobilenet-v2 --input-size $(INPUT_SIZE) --input-scale minus1_1 \
		--weights $(VISION_WEIGHTS) --batch-size $(BATCH_SIZE) \
		--head-epochs $(HEAD_EPOCHS) --finetune-epochs $(FINETUNE_EPOCHS)

$(GEO_KERAS): $(DATASET)/dataset_manifest.json
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_geo_prior.py \
		--observations $(DATASET)/metadata/train.csv \
		--validation-observations $(DATASET)/metadata/val.csv \
		--num-classes $(NUM_CLASSES) --output $@ \
		--embedding-dim $(GEO_EMBEDDING_DIM) --batch-size $(BATCH_SIZE) \
		--epochs $(GEO_EPOCHS)

export: $(VISION_FP32) $(VISION_DRQ) $(VISION_INT8) $(GEO_FP32)

$(VISION_FP32): $(VISION_KERAS)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_tflite.py \
		--keras-model $< --output $@ --manifest $@.manifest.json \
		--class-map $(DATASET)/class_map.json --model-id vision_fp32 \
		--role vision --optimization fp32 --input-scale minus1_1

$(VISION_DRQ): $(VISION_KERAS)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_tflite.py \
		--keras-model $< --output $@ --manifest $@.manifest.json \
		--class-map $(DATASET)/class_map.json --model-id vision_drq \
		--role vision --optimization drq --input-scale minus1_1

$(VISION_INT8): $(VISION_KERAS)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_tflite.py \
		--keras-model $< --output $@ --manifest $@.manifest.json \
		--class-map $(DATASET)/class_map.json --model-id vision_int8 \
		--role vision --optimization int8 --input-scale minus1_1 \
		--representative-images $(DATASET)/images/train \
		--representative-limit $(REPRESENTATIVE_LIMIT)

$(GEO_FP32): $(GEO_KERAS)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_tflite.py \
		--keras-model $< --output $@ --manifest $@.manifest.json \
		--class-map $(DATASET)/class_map.json --model-id geo_prior_fp32 \
		--role geo_prior --optimization fp32 --input-scale encoded_geo

BENCHMARK_RESULTS := \
	$(RESULT_DIR)/fused_fp32.json \
	$(RESULT_DIR)/fused_drq.json \
	$(RESULT_DIR)/fused_int8.json

benchmark: $(RESULT_DIR)/tradeoffs.md

$(RESULT_DIR)/fused_fp32.json: $(VISION_FP32) $(GEO_FP32)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/benchmark_rpi.py \
		--images $(DATASET)/images/test --metadata-csv $(DATASET)/metadata/test.csv \
		--vision-model $(VISION_FP32) --vision-manifest $(VISION_FP32).manifest.json \
		--geo-model $(GEO_FP32) --geo-manifest $(GEO_FP32).manifest.json \
		--class-map $(DATASET)/class_map.json --alpha 0.3 \
		--warmup $(BENCHMARK_WARMUP) --repetitions $(BENCHMARK_REPETITIONS) --output $@

$(RESULT_DIR)/fused_drq.json: $(VISION_DRQ) $(GEO_FP32)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/benchmark_rpi.py \
		--images $(DATASET)/images/test --metadata-csv $(DATASET)/metadata/test.csv \
		--vision-model $(VISION_DRQ) --vision-manifest $(VISION_DRQ).manifest.json \
		--geo-model $(GEO_FP32) --geo-manifest $(GEO_FP32).manifest.json \
		--class-map $(DATASET)/class_map.json --alpha 0.3 \
		--warmup $(BENCHMARK_WARMUP) --repetitions $(BENCHMARK_REPETITIONS) --output $@

$(RESULT_DIR)/fused_int8.json: $(VISION_INT8) $(GEO_FP32)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/benchmark_rpi.py \
		--images $(DATASET)/images/test --metadata-csv $(DATASET)/metadata/test.csv \
		--vision-model $(VISION_INT8) --vision-manifest $(VISION_INT8).manifest.json \
		--geo-model $(GEO_FP32) --geo-manifest $(GEO_FP32).manifest.json \
		--class-map $(DATASET)/class_map.json --alpha 0.3 \
		--warmup $(BENCHMARK_WARMUP) --repetitions $(BENCHMARK_REPETITIONS) --output $@

$(RESULT_DIR)/tradeoffs.md: $(BENCHMARK_RESULTS)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/summarize_benchmarks.py \
		$(BENCHMARK_RESULTS) \
		--csv $(RESULT_DIR)/tradeoffs.csv --markdown $@

workstation: prepare train export benchmark

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

data/smoke_source/annotations.json:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/create_smoke_source.py \
		--output data/smoke_source --classes 2 --images-per-class 12 --size 40 --seed 42

smoke: data/smoke_source/annotations.json
	$(MAKE) workstation \
		PYTHON=$(PYTHON) RAW_JSON=data/smoke_source/annotations.json \
		IMAGES_ROOT=data/smoke_source DATASET=data/smoke_prepared \
		MODEL_DIR=artifacts/smoke/models RESULT_DIR=artifacts/smoke/results \
		NUM_CLASSES=2 MIN_PER_CLASS=3 MAX_PER_CLASS=12 INPUT_SIZE=32 VISION_WEIGHTS=none \
		BATCH_SIZE=4 HEAD_EPOCHS=1 FINETUNE_EPOCHS=1 \
		GEO_EPOCHS=1 GEO_EMBEDDING_DIM=16 \
		REPRESENTATIVE_LIMIT=8 BENCHMARK_WARMUP=1 BENCHMARK_REPETITIONS=1
