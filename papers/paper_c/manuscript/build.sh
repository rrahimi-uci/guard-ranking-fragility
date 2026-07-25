#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p build/texmf-var build/texmf-config build/texmf-home

export TEXMFVAR="$ROOT/build/texmf-var"
export TEXMFCONFIG="$ROOT/build/texmf-config"
export TEXMFHOME="$ROOT/build/texmf-home"

PDFLATEX="${PDFLATEX:-/Library/TeX/texbin/pdflatex}"
BIBTEX="${BIBTEX:-/Library/TeX/texbin/bibtex}"

"$PDFLATEX" -interaction=nonstopmode -halt-on-error -output-directory=build manuscript/main.tex
"$BIBTEX" build/main
"$PDFLATEX" -interaction=nonstopmode -halt-on-error -output-directory=build manuscript/main.tex
"$PDFLATEX" -interaction=nonstopmode -halt-on-error -output-directory=build manuscript/main.tex

if grep -Eq 'undefined citations|There were undefined references' build/main.log; then
  echo "manuscript build has unresolved references" >&2
  exit 2
fi
cp build/main.pdf build/paper_c.pdf
echo "built $ROOT/build/paper_c.pdf"

