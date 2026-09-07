PYTHON ?= python3
export PYTHONPATH := src
.NOTPARALLEL:
CONFIG ?= configs/models/efficientnet-b0.json
DATA_SOURCE ?=
ifneq ($(DATA_SOURCE),)
ifeq ($(filter $(DATA_SOURCE),mini full),)
$(error DATA_SOURCE must be mini or full)
endif
endif
ifneq ($(VISION_BACKBONE),)
$(error VISION_BACKBONE has been replaced; use CONFIG=configs/models/$(VISION_BACKBONE).json)
endif
RUN ?=
INITIAL_WEIGHTS ?=
INITIAL_CLASS_MAP ?=
RUN_ARGS = $(if $(RUN),--run "$(RUN)") $(if $(DATA_SOURCE),--data-source "$(DATA_SOURCE)") $(if $(INITIAL_WEIGHTS),--initial-weights "$(INITIAL_WEIGHTS)") $(if $(INITIAL_CLASS_MAP),--initial-class-map "$(INITIAL_CLASS_MAP)")
WORKFLOW = $(PYTHON) -m biodiversity_edge_ai.workflow

.PHONY: help setup prepare train export benchmark workstation smoke test
help:
	@echo "make setup                       Install workstation dependencies"
	@echo "make smoke                       Run a 24-image experiment, no downloads"
	@echo "make workstation                 Run Mini training, FP32/DRQ export and evaluation"
	@echo "make workstation DATA_SOURCE=full Use the full training source"
	@echo "make workstation CONFIG=configs/small_demo.json Run a 10-class subset"
	@echo "make prepare | train | export | benchmark Run through the selected stage"
	@echo "make workstation CONFIG=configs/models/efficientnet-b0.json Select a model profile"
	@echo "Options: CONFIG=path.json RUN=unique-name DATA_SOURCE=mini|full"

setup:
	$(PYTHON) -m pip install -c constraints-workstation.txt -e '.[train,dev]'

prepare train export benchmark:
	$(WORKFLOW) --config "$(CONFIG)" $(RUN_ARGS) --stage $@

workstation:
	$(WORKFLOW) --config "$(CONFIG)" $(RUN_ARGS)

smoke:
	$(WORKFLOW) --config configs/smoke.json $(RUN_ARGS)

test:
	$(PYTHON) -m unittest discover -s tests -v
