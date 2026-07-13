# The Benchmark Chooses the Winner — paper build
PAPER = benchmark_chooses_the_winner
.PHONY: all clean
all: $(PAPER).pdf
$(PAPER).pdf: $(PAPER).tex refs.bib tab_primary_gen.tex tab_sensitivity_gen.tex $(wildcard figures/*.pdf)
	tectonic $(PAPER).tex
# The figure + generated tables come from the analysis step, not from here:
#   make analyze   (repo root) -> artifacts/paper_a_sft/analysis/{tables,figures}
clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf
	rm -rf build
