#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${STUDY_DIR}/build/manuscript"
TECTONIC_BUNDLED="/Users/rezarahimi/.codex/plugins/cache/openai-bundled/latex-tectonic/0.1.1/bin/tectonic"

mkdir -p "${OUTPUT_DIR}"

if [[ -x "${TECTONIC_BUNDLED}" ]]; then
  "${TECTONIC_BUNDLED}" -X compile "${SCRIPT_DIR}/main.tex" \
    --outdir "${OUTPUT_DIR}" --keep-intermediates
else
  PDFLATEX="/Library/TeX/texbin/pdflatex"
  BIBTEX="/Library/TeX/texbin/bibtex"
  "${PDFLATEX}" -interaction=nonstopmode -halt-on-error \
    -output-directory="${OUTPUT_DIR}" "${SCRIPT_DIR}/main.tex"
  (
    cd "${OUTPUT_DIR}"
    BIBINPUTS="${SCRIPT_DIR}:" "${BIBTEX}" main
  )
  "${PDFLATEX}" -interaction=nonstopmode -halt-on-error \
    -output-directory="${OUTPUT_DIR}" "${SCRIPT_DIR}/main.tex"
  "${PDFLATEX}" -interaction=nonstopmode -halt-on-error \
    -output-directory="${OUTPUT_DIR}" "${SCRIPT_DIR}/main.tex"
fi

test -s "${OUTPUT_DIR}/main.pdf"
echo "${OUTPUT_DIR}/main.pdf"
