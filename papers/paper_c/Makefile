PYTHON ?= $(shell if [ -x ../../.venv/bin/python ]; then echo ../../.venv/bin/python; else echo python3; fi)
PROJECT_ROOT := $(CURDIR)
export PYTHONPATH := $(PROJECT_ROOT)/src
export PYTHONDONTWRITEBYTECODE := 1
export PYTHONPYCACHEPREFIX := $(PROJECT_ROOT)/build/pycache

.PHONY: help test validate doctor bootstrap protocol-lock power frontier paper clean

help:
	@echo "Paper C isolated targets"
	@echo "  make test       CPU-only contract tests"
	@echo "  make validate   validate candidate config and objective identities"
	@echo "  make doctor     inspect local GPU, HF token presence, and gcloud readiness"
	@echo "  make bootstrap  copy and hash parent inputs into ignored inputs/"
	@echo "  make power      report design MDE and FAIL if underpowered (run before Stage 1)"
	@echo "  make frontier   compute the primary frontier estimands from candidate scores"
	@echo "  make protocol-lock  issue immutable Stage-1 candidate protocol lock"
	@echo "  make paper      compile manuscript/main.tex into build/"
	@echo "  make clean      remove only generated build products in this folder"

test:
	$(PYTHON) -m pytest

validate:
	$(PYTHON) -m paper_c validate-config --config config/study.json
	$(PYTHON) -m paper_c loss-check --config config/study.json

doctor:
	$(PYTHON) -m paper_c doctor --config config/study.json

bootstrap:
	$(PYTHON) -m paper_c bootstrap-inputs --config config/study.json

# The design must state what it can detect before any GPU time is spent. This target fails
# closed, so an underpowered panel cannot be discovered after the compute is gone.
power:
	$(PYTHON) -m paper_c power --config config/study.json \
	  --seed-variance $(SEED_VARIANCE) --out artifacts/power/design_power.json --gate

frontier:
	$(PYTHON) -m paper_c frontier --config config/study.json \
	  --candidate-scores $(CANDIDATE_SCORES) --out artifacts/analysis/frontier.json

SEED_VARIANCE ?= artifacts/power/parent_seed_variance.jsonl
CANDIDATE_SCORES ?= artifacts/analysis/candidate_scores.jsonl

protocol-lock: bootstrap validate power
	$(PYTHON) -m paper_c create-protocol-lock --config config/study.json

paper:
	bash manuscript/build.sh

clean:
	$(PYTHON) -m paper_c clean-build --config config/study.json
