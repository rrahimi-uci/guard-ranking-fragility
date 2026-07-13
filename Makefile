# Paper A pipeline — "The Benchmark Chooses the Winner"
# Run every target from the repo root. Steps 1-3 and 6 are CPU-only; steps 4-5 need a GPU.
# The committed scores.parquet makes `make repro` reproduce all tables/figures without a GPU.

PY      ?= python
CONFIG   = configs/paper_a_sft.yaml
ROOT     = artifacts/paper_a_sft
MANIFESTS     = $(ROOT)/manifests
MANIFEST_JSON = $(MANIFESTS)/manifest.json
AUDIT    = $(ROOT)/audit
LOCK     = $(ROOT)/LOCK.json
SCORES   = $(ROOT)/scores/scores.parquet
ANALYSIS = $(ROOT)/analysis

.DEFAULT_GOAL := help
.PHONY: help install install-all manifests audit lock train eval analyze \
        repro selftest test paper paper-html explorer clean

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

install:      ## install the core library (CPU: analysis + tests)
	$(PY) -m pip install -e .
install-all:  ## install everything (training, figures, dev)
	$(PY) -m pip install -e ".[all]"

## --- pipeline (steps 1-6) ---------------------------------------------------
manifests:  ## 1. build the decontaminated data manifests (CPU, HF access)
	$(PY) experiments/prepare_paper_a_manifests.py --config $(CONFIG) --out $(MANIFESTS)
audit:      ## 2. recompute + hard-assert decontamination (CPU)
	$(PY) experiments/audit_paper_a_splits.py --config $(CONFIG) --manifest $(MANIFEST_JSON) --out $(AUDIT)
lock:       ## 3. freeze config + manifest hashes into LOCK.json (CPU)
	$(PY) experiments/lock_paper_a_sft.py --config $(CONFIG) --manifest $(MANIFEST_JSON) --audit $(AUDIT)/audit.json --out $(LOCK)
train:      ## 4. train the 4x5 LoRA-SFT panel (GPU)
	$(PY) experiments/run_paper_a_sft.py train --lock $(LOCK)
eval:       ## 5. score bases + adapters -> scores.parquet (GPU)
	$(PY) experiments/eval_paper_a_sft.py --lock $(LOCK)
analyze:    ## 6. macro-AP + bootstrap + claim gates -> tables/figures (CPU)
	$(PY) experiments/analyze_paper_a_sft.py --lock $(LOCK) --scores $(SCORES) --out $(ANALYSIS)

## --- reproduce / verify (no GPU) --------------------------------------------
repro: analyze  ## reproduce every table + figure from the committed scores.parquet
	@echo "reproduced tables/figures in $(ANALYSIS)"
selftest:   ## synthetic end-to-end check of the analysis (fast)
	$(PY) experiments/analyze_paper_a_sft.py --self-test
test:       ## run the unit tests
	$(PY) -m pytest

## --- documents -------------------------------------------------------------
paper:      ## build the paper PDF (needs tectonic)
	$(MAKE) -C paper-a
paper-html: ## build the HTML edition (needs pandoc, tectonic, pdftocairo)
	$(PY) paper-html/build.py
explorer:   ## regenerate the benchmark-explorer content samples (paper-html/explorer)
	$(PY) paper-html/explorer/build_content_samples.py

clean:      ## remove Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} + ; \
	find . -type f -name '*.pyc' -delete
