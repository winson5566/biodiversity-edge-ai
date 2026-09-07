PYTHON ?= python3
export PYTHONPATH := src
.NOTPARALLEL:
DATA_SOURCE ?= mini
ifeq ($(DATA_SOURCE),mini)
CONFIG ?= configs/mini.json
else ifeq ($(DATA_SOURCE),full)
CONFIG ?= configs/full_system.json
else
$(error DATA_SOURCE must be mini or full)
endif
RUN ?=
VISION_BACKBONE ?=
RUN_ARGS = $(if $(RUN),--run "$(RUN)") $(if $(VISION_BACKBONE),--backbone "$(VISION_BACKBONE)")
WORKFLOW = $(PYTHON) -m biodiversity_edge_ai.workflow

.PHONY: help setup prepare train export benchmark workstation smoke test
help:
	@echo "make setup                       Install workstation dependencies"
	@echo "make smoke                       Run a 24-image experiment, no downloads"
	@echo "make workstation                 Run Mini training, FP32/DRQ export and evaluation"
	@echo "make workstation DATA_SOURCE=full Use the full training source"
	@echo "make workstation CONFIG=configs/small_demo.json Run a 10-class subset"
	@echo "make prepare | train | export | benchmark Run through the selected stage"
	@echo "Options: CONFIG=path.json RUN=unique-name VISION_BACKBONE=efficientnet-b0"

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
