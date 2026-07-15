PAPER        := compose_dont_tune
PYTHON       ?= $(if $(wildcard ../.venv/bin/python),../.venv/bin/python,python3)
TECTONIC     ?= tectonic
BUILD_DIR    ?= build
COMPOSITION  ?= ../artifacts/paper_a_sft_v2/analysis/composition/composition.json
METADATA     ?= ../artifacts/paper_a_sft_v2/analysis/composition/composition_metadata.json
PROTOCOL     ?= config/prospective_protocol.template.json

.DEFAULT_GOAL := all
.PHONY: all evidence evidence-check protocol protocol-locked test pdf verify clean

all: verify pdf

evidence:  ## regenerate retrospective pilot TeX from the anchored clean-v2 result
	$(PYTHON) code/build_pilot_artifacts.py --composition $(COMPOSITION) --metadata $(METADATA) --out-dir generated

evidence-check:  ## fail if generated pilot inputs are absent or stale
	$(PYTHON) code/build_pilot_artifacts.py --composition $(COMPOSITION) --metadata $(METADATA) --out-dir generated --check

protocol:  ## validate the explicit draft protocol and print remaining warnings
	$(PYTHON) code/validate_protocol.py $(PROTOCOL)

protocol-locked:  ## require a complete claim-bearing protocol; expected to fail for the template
	$(PYTHON) code/validate_protocol.py $(PROTOCOL) --require-locked

test:  ## run Paper B workspace tests
	$(PYTHON) -m pytest -q tests

verify: evidence-check protocol test  ## verify anchored evidence, draft contract, and code

pdf: evidence  ## compile the visibly retrospective 11pt article draft
	mkdir -p $(BUILD_DIR)
	$(TECTONIC) --keep-logs --outdir $(BUILD_DIR) $(PAPER).tex
	cp $(BUILD_DIR)/$(PAPER).pdf $(PAPER).pdf

clean:  ## remove build-only files while preserving generated evidence inputs
	rm -rf $(BUILD_DIR)
