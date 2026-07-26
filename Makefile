# Paper A pipeline — "The Benchmark Chooses the Winner"
# Steps 1–3 and 6 are CPU-only; steps 4–5 need a GPU.
# The historical score bundle is reproduced only through the explicit legacy target.

PY              ?= python3
CONFIG           = configs/paper_a_sft.yaml
LEGACY_ROOT      = artifacts/paper_a_sft
V2_ROOT          ?= artifacts/paper_a_sft_v2
MANIFESTS        = $(V2_ROOT)/manifests
PUBLIC_MANIFESTS = $(V2_ROOT)/public_manifests
MANIFEST_JSON    = $(MANIFESTS)/manifest.json
AUDIT            = $(V2_ROOT)/audit
LOCK             ?= $(V2_ROOT)/LOCK.json
SCORES           = $(V2_ROOT)/scores/scores.parquet
ANALYSIS         = $(V2_ROOT)/analysis
LEGACY_LOCK      = $(LEGACY_ROOT)/LOCK.json
LEGACY_SCORES    = $(LEGACY_ROOT)/scores/scores.parquet
LEGACY_ANALYSIS  = $(LEGACY_ROOT)/analysis
COMPOSITION_ANALYSIS        ?= $(ANALYSIS)/composition
COMPOSITION_FULL_ANALYSIS   ?= $(ANALYSIS)/composition-full
LEGACY_COMPOSITION_ANALYSIS ?= $(LEGACY_ANALYSIS)/composition
RELEASE_DIR      ?= dist/paper-a-sft-v2-release
PAPER_ANALYSIS   ?= $(ANALYSIS)
PAPER_DIR        = papers/finetuning-specialization

.DEFAULT_GOAL := help
.PHONY: help install install-all manifests manifests-legacy audit lock relock \
        verify-lock verify-legacy-lock train validate-runs eval analyze analyze-release \
        analyze-legacy repro repro-release repro-legacy release-package paper-sync paper-verify \
        composition composition-full composition-legacy selftest test paper clean

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-17s\033[0m %s\n", $$1, $$2}'

install:      ## install CPU analysis plus tests
	$(PY) -m pip install -c requirements.txt -e ".[dev]"
install-all:  ## install training/scoring plus CPU analysis and tests
	$(PY) -m pip install -c requirements.txt -e ".[all]"

## --- clean v2 pipeline ------------------------------------------------------
manifests:  ## 1. pinned-HF, hash-ranked manifests + text-free public index
	$(PY) experiments/prepare_paper_a_manifests.py --config $(CONFIG) --out $(MANIFESTS) --public-out $(PUBLIC_MANIFESTS)
manifests-legacy:  ## reconstruct previously inspected seed-7 cohorts (explicit)
	$(PY) experiments/prepare_paper_a_manifests.py --config $(CONFIG) --out $(MANIFESTS) --public-out $(PUBLIC_MANIFESTS) --allow-legacy-frozen-cohorts
audit:      ## 2. independently recompute and fail-closed on split integrity
	$(PY) experiments/audit_paper_a_splits.py --config $(CONFIG) --manifest $(MANIFEST_JSON) --out $(AUDIT)
lock:       ## 3. write a clean v2 lock without overwriting the historical lock
	$(PY) experiments/lock_paper_a_sft.py --config $(CONFIG) --manifest $(MANIFEST_JSON) --manifests-dir $(MANIFESTS) --audit $(AUDIT)/audit.json --out $(LOCK) --require-tokenizer-probe
relock:     ## explicitly replace the clean v2-root LOCK.json (destructive)
	$(PY) experiments/lock_paper_a_sft.py --config $(CONFIG) --manifest $(MANIFEST_JSON) --manifests-dir $(MANIFESTS) --audit $(AUDIT)/audit.json --out $(LOCK) --require-tokenizer-probe --force
verify-lock:  ## verify lock self-hash and every bound file
	$(PY) -c "from experiments import paper_a_common as C; C.load_lock('$(LOCK)', verify_files=True); print('verified $(LOCK)')"
verify-legacy-lock:  ## verify historical self-hash without pretending it is v2
	$(PY) -c "from experiments import paper_a_common as C; C.load_lock('$(LEGACY_LOCK)', allow_legacy=True, verify_files=False); print('verified legacy $(LEGACY_LOCK)')"
train:      ## 4. train the 4x5 LoRA-SFT panel under the clean lock (GPU)
	$(PY) experiments/run_paper_a_sft.py train --lock $(LOCK)
validate-runs:  ## rehash and validate all 20 adapters against the clean lock
	$(PY) experiments/run_paper_a_sft.py validate-runs --lock $(LOCK) --strict
eval:       ## 5. score bases + validated adapters (GPU)
	$(PY) experiments/eval_paper_a_sft.py --lock $(LOCK)
analyze:    ## 6. strict complete-matrix analysis under the clean v2 contract
	$(PY) experiments/analyze_paper_a_sft.py --lock $(LOCK) --scores $(SCORES) --out $(ANALYSIS)
analyze-release:  ## verify and analyze a final v2 score-only release cache
	$(PY) experiments/analyze_paper_a_sft.py --release-cache --lock $(LOCK) --scores $(SCORES) --out $(ANALYSIS)

repro-release: analyze-release  ## regenerate v2 analysis and verify checked-in paper inputs
	$(MAKE) paper-verify PAPER_ANALYSIS=$(ANALYSIS)
	@echo "reproduced final v2 release-cache outputs in $(ANALYSIS); checked-in paper copies match"
repro: repro-release  ## primary no-GPU reproduction from a final v2 release cache

## --- historical score compatibility (no GPU) -------------------------------
analyze-legacy:  ## explicitly analyze the immutable v1 lock + committed scores
	$(PY) experiments/analyze_paper_a_sft.py --allow-legacy-lock --lock $(LEGACY_LOCK) --scores $(LEGACY_SCORES) --out $(LEGACY_ANALYSIS)
repro-legacy: analyze-legacy  ## regenerate archival v1 analysis without touching v2 paper files
	@echo "reproduced archival v1 outputs in $(LEGACY_ANALYSIS); publication paper files remain v2"

release-package:  ## stage the public v2 release from an explicit no-raw-prompt allowlist
	$(PY) experiments/package_paper_a_release.py --root $(V2_ROOT) --out $(RELEASE_DIR)

## --- generated paper inputs ------------------------------------------------
paper-sync:  ## explicit maintainer action: copy canonical outputs into paper-a
	cp $(PAPER_ANALYSIS)/tables/table3_primary.tex $(PAPER_DIR)/tab_primary_gen.tex
	cp $(PAPER_ANALYSIS)/tables/table4_per_benchmark.tex $(PAPER_DIR)/tab_sensitivity_gen.tex
	cp $(PAPER_ANALYSIS)/tables/table5_seed_values.tex $(PAPER_DIR)/tab_seed_values_gen.tex
	cp $(PAPER_ANALYSIS)/tables/results_macros.tex $(PAPER_DIR)/results_macros_gen.tex
	cp $(PAPER_ANALYSIS)/figures/specialization_plane.pdf $(PAPER_DIR)/figures/specialization_plane.pdf
paper-verify:  ## fail if a paper-consumed generated file is stale
	cmp $(PAPER_ANALYSIS)/tables/table3_primary.tex $(PAPER_DIR)/tab_primary_gen.tex
	cmp $(PAPER_ANALYSIS)/tables/table4_per_benchmark.tex $(PAPER_DIR)/tab_sensitivity_gen.tex
	cmp $(PAPER_ANALYSIS)/tables/table5_seed_values.tex $(PAPER_DIR)/tab_seed_values_gen.tex
	cmp $(PAPER_ANALYSIS)/tables/results_macros.tex $(PAPER_DIR)/results_macros_gen.tex
	cmp $(PAPER_ANALYSIS)/figures/specialization_plane.pdf $(PAPER_DIR)/figures/specialization_plane.pdf

## --- verification / documents ---------------------------------------------
composition:  ## primary composition analysis from the final v2 release cache
	$(PY) experiments/analyze_composition.py --release-cache --lock $(LOCK) --scores $(SCORES) --out $(COMPOSITION_ANALYSIS)
composition-full:  ## composition analysis with raw v2 manifests, run metadata, and adapters
	$(PY) experiments/analyze_composition.py --lock $(LOCK) --scores $(SCORES) --out $(COMPOSITION_FULL_ANALYSIS)
composition-legacy:  ## explicit archived-v1 composition compatibility analysis
	$(PY) experiments/analyze_composition.py --allow-legacy-lock --lock $(LEGACY_LOCK) --scores $(LEGACY_SCORES) --out $(LEGACY_COMPOSITION_ANALYSIS)
selftest:   ## synthetic end-to-end analysis check
	$(PY) experiments/analyze_paper_a_sft.py --self-test
test:       ## run unit and release-integrity tests
	$(PY) -m pytest
paper: paper-verify  ## verify generated inputs and build the PDF
	$(MAKE) -C $(PAPER_DIR)

clean:      ## remove Python and paper build caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	$(MAKE) -C $(PAPER_DIR) clean

# ─────────────────────────────────────────────────────────────── verification tiers
# Repository Layout v2, Phase 1. Every tier names one absolute interpreter, because
# nested Makefiles disagree on the variable name (PY vs PYTHON) and the local .venv is
# Python 3.14 while Paper A's documented release environment requires 3.12 -- local
# convenience success is not release verification.
PY_ABS ?= $(CURDIR)/.venv/bin/python

.PHONY: check-fast check-registry check-links check-locks check-papers check-data-local check-all

check-registry:  ## validate the registry + ledger and assert generated indexes are current
	$(PY_ABS) tools/validate_registries.py
	$(PY_ABS) tools/render_indexes.py --check

check-links:  ## verify relative Markdown links in indexes resolve
	$(PY_ABS) tools/check_markdown_links.py

check-fast: check-registry check-links  ## hermetic tests across every suite
	$(PY_ABS) -m pytest -q
	$(PY_ABS) -m pytest apps/benchmark-explorer/tests -q
	$(PY_ABS) apps/benchmark-explorer/src/build.py --target public --fixtures
	$(MAKE) -C mortgage-benchmark test PY=$(PY_ABS)
	$(MAKE) -C papers/base-adapter-composition test PYTHON=$(PY_ABS)
	@echo "--- Paper C suites: predecessor and successor ---"
	@$(MAKE) -C papers/paper_c test PYTHON=$(PY_ABS) \
	  || echo "NOTE paper_c predecessor: expected_fail, see studies/registry.yaml"
	@$(MAKE) -C papers/paper_c/specialize_then_align test PY=$(PY_ABS) \
	  || echo "NOTE specialize_then_align: expected_fail (candidate lock binds live source bytes)"
	@$(MAKE) -C studies/paper-c-specialize-align-mortgage-v1 test PY=$(PY_ABS) \
	  || echo "NOTE sta study package: expected_fail (same declared candidate-lock failure)"

check-locks:  ## contract-specific checks; expected_fail cases are declared in the registry
	$(PY_ABS) tools/validate_registries.py --run-verification

check-papers:  ## isolated manuscript builds plus link checks
	bash papers/paper_c/specialize_then_align/manuscript/build.sh
	$(PY_ABS) tools/check_markdown_links.py

check-data-local:  ## inventories over ignored local data; explicitly not hermetic
	@test -d data/benchmarks || { echo "data/benchmarks absent: local-only tier"; exit 1; }
	$(PY_ABS) -c "import pathlib,json; \
	  fs=sorted(pathlib.Path('data/benchmarks').rglob('*.jsonl')); \
	  print(f'{len(fs)} local corpora'); \
	  [print(f'  {f.name}: {sum(1 for _ in f.open())} rows') for f in fs]"

# Paper A's LOCK.json pins a runtime software fingerprint at Python 3.12. The default
# .venv is 3.14, so release reproduction needs its own interpreter and its own tier;
# a local pass under 3.14 would not be release verification.
PY_RELEASE ?= /opt/homebrew/bin/python3.12

check-release:  ## pinned-3.12 release reproduction; skips cleanly if that env lacks deps
	@# One shell: each recipe line gets its own, so a bare `exit 0` would not skip the rest.
	@set -e; \
	if [ ! -x "$(PY_RELEASE)" ]; then \
	  echo "SKIP check-release: no interpreter at $(PY_RELEASE)"; exit 0; fi; \
	"$(PY_RELEASE)" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)' \
	  || { echo "SKIP check-release: $(PY_RELEASE) is not Python 3.12"; exit 0; }; \
	"$(PY_RELEASE)" -c 'import numpy, pandas, sklearn' 2>/dev/null \
	  || { echo "SKIP check-release: the 3.12 env lacks scientific deps."; \
	       echo "  to enable: $(PY_RELEASE) -m pip install -r requirements.txt"; exit 0; }; \
	echo "3.12 env ready; running release reproduction"; \
	$(MAKE) repro-release PY="$(PY_RELEASE)"

check-all: check-fast check-locks check-papers  ## aggregate; check-data-local is local-only
	@echo "check-all complete"
