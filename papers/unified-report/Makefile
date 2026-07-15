PAPER      := unified_report
TECTONIC   ?= tectonic
BUILD_DIR  ?= build
PYTHON     ?= $(if $(wildcard ../../.venv/bin/python),../../.venv/bin/python,python3)

.DEFAULT_GOAL := pdf
.PHONY: pdf reproduce reproduce-check clean

reproduce:  ## regenerate every table/figure from committed per-row scores (no GPU/network)
	$(PYTHON) reproduce.py

reproduce-check:  ## fail if any regenerated table drifts from the committed generated/ copy
	$(PYTHON) reproduce.py --check

pdf: ## compile the report (tectonic); use `make reproduce` first to refresh generated tables
	mkdir -p $(BUILD_DIR)
	$(TECTONIC) --keep-logs --outdir $(BUILD_DIR) $(PAPER).tex
	cp $(BUILD_DIR)/$(PAPER).pdf $(PAPER).pdf

all: reproduce pdf  ## regenerate all results, then build the PDF

clean:
	rm -rf $(BUILD_DIR)
