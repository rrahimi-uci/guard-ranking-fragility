PAPER      := unified_report
TECTONIC   ?= tectonic
BUILD_DIR  ?= build
PYTHON     ?= $(if $(wildcard ../../.venv/bin/python),../../.venv/bin/python,python3)

.DEFAULT_GOAL := pdf
.PHONY: pdf regenerate verify verify-heavy reproduce reproduce-check clean

# Two clearly separated verbs. `regenerate` WRITES; `verify` compares against scratch copies and
# leaves the tracked tree byte-clean. The old single `reproduce` target did the first while the
# paper described it as if it did the second, so a reader who ran it overwrote the committed
# artifacts and got "regenerated" back instead of a comparison.
#
# One honest limit on `verify`, measured rather than assumed. Three upstream generators are not
# redirected to the scratch directory: build_pilot_artifacts.py, reeval_from_scores.py and
# emit_baseline_tex.py write their INTERMEDIATES in place (papers/base-adapter-composition/generated/,
# mortgage-benchmark/out_eval/, mortgage-benchmark/generated/). The tracked tree still ends
# byte-clean, because with no drift the rewrite is byte-identical -- but drift in one of those
# intermediates is silently REPAIRED rather than reported, so `verify` would say "0 failed".
# Everything under papers/unified-report/generated/ -- every artifact the report inputs -- is
# compared properly. Redirecting those three needs an --out-dir on reeval_from_scores.py.

regenerate:  ## rewrite every generated table/figure from committed per-row scores (no GPU/network)
	$(PYTHON) reproduce.py

verify:  ## byte-check every generated table/figure; leaves the tracked tree byte-clean; nonzero on drift
	$(PYTHON) reproduce.py --check

verify-heavy: verify  ## also re-derive h2h.json offline (2,000-replicate bootstrap; takes minutes)
	cd ../.. && $(PYTHON) experiments/eval_frontier_general_h2h.py --offline --out /tmp/h2h_verify
	$(PYTHON) tools/compare_h2h.py ../../artifacts/frontier_general_h2h/h2h.json /tmp/h2h_verify/h2h.json

# Back-compat aliases. `reproduce` was the write path, so it keeps that meaning.
reproduce: regenerate
reproduce-check: verify

pdf: ## compile the report (tectonic); use `make regenerate` first to refresh generated tables
	mkdir -p $(BUILD_DIR)
	$(TECTONIC) --keep-logs --outdir $(BUILD_DIR) $(PAPER).tex
	cp $(BUILD_DIR)/$(PAPER).pdf $(PAPER).pdf

all: regenerate pdf  ## regenerate all results, then build the PDF

clean:
	rm -rf $(BUILD_DIR)
