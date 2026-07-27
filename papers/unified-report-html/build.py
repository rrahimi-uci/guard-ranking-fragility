#!/usr/bin/env python
"""Build the HTML edition of the unified report from the SAME LaTeX sources the PDF uses.

Design rule, inherited from the rest of this repository: no claim-bearing number is ever
retyped here. Every figure in the prose comes from `../unified-report/generated/*.tex`
(the macro files and table files that `unified_report.tex` itself `\\input`s), so the HTML
cannot disagree with the PDF. If an analysis is rerun and a table changes, this edition
changes with it on the next build.

Pipeline
  1. figures()      PDF figures -> SVG (pdftocairo), PNGs copied as-is
  2. flatten()      expand \\input, neutralize print-only LaTeX, mark the custom boxes
  3. pandoc         the flattened body -> an HTML fragment (math left for MathJax)
  4. postprocess()  number floats in document order, resolve \\Cref, render citations,
                    turn the marked boxes into styled callouts, build the TOC
  5. verify()       assert the float numbering matches the built PDF exactly

Usage:  python build.py [--check]
        --check : rebuild and fail if the emitted index.html differs from the committed one
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "unified-report"
GEN = SRC / "generated"
OUT = HERE / "index.html"
ASSETS = HERE / "assets"
FIG = ASSETS / "fig"

# One source of truth for the published URL: the canonical link, the Open Graph tags, the
# JSON-LD, robots.txt and sitemap.xml all interpolate this. A canonical that disagrees with
# where the page actually lives is a silent SEO failure, so a test asserts they match.
SITE_URL = "https://rrahimi-uci.github.io/safety-guard-dynamics/"

DESCRIPTION = (
    "What does fine-tuning actually buy a compact safety guard? A paired, same-checkpoint "
    "study on a fixed four-checkpoint panel: LoRA-SFT lifts trained-on ranking to a ceiling "
    "but not transfer, and at an equal false-alarm budget it catches less than half of what "
    "its own untuned base catches off-source. Plus a retraining-free composition repair, a "
    "preregistered ten-checkpoint replication, and a dual-labeled mortgage benchmark."
)
KEYWORDS = (
    "AI safety, safety guard, LLM guardrail, prompt safety classifier, small language model, "
    "LoRA fine-tuning, supervised fine-tuning, specialization, out-of-distribution transfer, "
    "average precision, calibration, false-alarm rate, model composition, model merging, "
    "jailbreak detection, prompt injection, HMDA, mortgage lending, fair lending, "
    "redlining proxy, regulated domains, benchmark design, reproducibility"
)


def seo_files() -> dict[str, str]:
    """robots.txt and sitemap.xml, which jekyll-sitemap used to generate on the old site.

    No <lastmod>: it would have to come from a clock or a file mtime, which makes the build
    nondeterministic and defeats `--check` -- for no crawler benefit on a one-page site.
    """
    return {
        "robots.txt": (
            "User-agent: *\n"
            "Allow: /\n"
            "\n"
            f"Sitemap: {SITE_URL}sitemap.xml\n"
        ),
        "sitemap.xml": (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            "  <url>\n"
            f"    <loc>{SITE_URL}</loc>\n"
            "    <changefreq>monthly</changefreq>\n"
            "    <priority>1.0</priority>\n"
            "  </url>\n"
            "</urlset>\n"
        ),
    }

# Sentinels chosen from a range LaTeX never emits and pandoc passes through untouched.
REF, CITE, BOX, ENDBOX, RAW = "\u27e6REF:", "\u27e6CITE:", "\u27e6BOX:", "\u27e6/BOX\u27e7", "\u27e6RAW:"
EQ = "\u27e6EQ:"
CLOSE = "\u27e7"


# --------------------------------------------------------------------------- 1. figures
def figures() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    n = 0
    for pdf in sorted((SRC / "figures").glob("*.pdf")):
        svg = FIG / f"{pdf.stem}.svg"
        if subprocess.run(["pdftocairo", "-svg", str(pdf), str(svg)]).returncode == 0:
            n += 1
    for png in sorted((SRC / "figures").glob("*.png")):
        shutil.copy2(png, FIG / png.name)
        n += 1
    return n


# ------------------------------------------------------------------------- 2. flatten
def _read(rel: str) -> str:
    for cand in (SRC / rel, SRC / f"{rel}.tex"):
        if cand.is_file():
            return cand.read_text()
    raise FileNotFoundError(rel)


def _expand_inputs(tex: str, depth: int = 0) -> str:
    """Inline \\input{...} recursively so one flat document reaches pandoc."""
    if depth > 6:
        return tex
    def sub(m):
        return _expand_inputs(_read(m.group(1).strip()), depth + 1)
    return re.sub(r"\\input\{([^}]+)\}", sub, tex)


def _balanced(tex: str, start: int) -> tuple[str, int]:
    """Return (contents of the {...} group beginning at `start`, index just past it)."""
    assert tex[start] == "{", tex[start:start + 20]
    depth, i = 0, start
    while i < len(tex):
        if tex[i] == "{" and (i == 0 or tex[i - 1] != "\\"):
            depth += 1
        elif tex[i] == "}" and tex[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return tex[start + 1:i], i + 1
        i += 1
    raise ValueError("unbalanced brace")


def _unwrap(tex: str, cmd: str, keep: int = 0) -> str:
    """Drop a command wrapper, keeping the `keep`-th brace group (or nothing)."""
    out, i = [], 0
    pat = re.compile(re.escape("\\" + cmd) + r"(?![A-Za-z])")
    while True:
        m = pat.search(tex, i)
        if not m:
            out.append(tex[i:])
            break
        out.append(tex[i:m.start()])
        j = m.end()
        groups = []
        while j < len(tex) and tex[j] in "{[":
            if tex[j] == "[":                                  # skip optional arg
                j = tex.index("]", j) + 1
                continue
            g, j = _balanced(tex, j)
            groups.append(g)
        if groups and keep < len(groups):
            out.append(groups[keep])
        i = j
    return "".join(out)


def _strip_colspec_padding(tex: str) -> str:
    r"""Remove @{...} inter-column padding wherever a column spec can appear.

    `\begin{tabular}{@{}l l cc cc@{}}` and `\multicolumn{5}{@{}l}{...}` are pure print
    typography, but pandoc's tabular reader gives up on either and emits a
    <div class="tabular"> of <br>-separated lines instead of a real table -- which silently
    dropped four tables (adaptation, datasets, and the two ensembling tables) from the first
    build. Every `@{...}` in this document is column padding (verified across all sources),
    so stripping them globally is safe and restores the tables.
    """
    return re.sub(r"@\{[^{}]*\}", "", tex)


class RedactionError(RuntimeError):
    """The redaction anchors moved, so we cannot prove the page is text-free."""


def redact_restricted_rows(tex: str) -> str:
    r"""Withhold the two verbatim benchmark rows the case study quotes.

    OPT-IN as of 2026-07-27, and off by default. v1_hmda2022 is now licensed CC BY 4.0 with
    `redistribution_decision: publish_text`, so the published page carries the full case study
    and this function is not applied. It is kept, and kept tested, because the situation it
    handles will recur: the next source whose licence is unresolved needs exactly this, and
    rebuilding it under time pressure is how prompt text ends up published by accident.

    When applied, it withholds the two verbatim benchmark rows and keeps everything else --
    the row id, the gold labels, the cited policy cards, the per-guard scores, and the ranks
    that carry the claim.

    Fails loudly rather than silently passing text through: if either anchor stops matching
    -- because the case study was regenerated or reworded -- the build stops, because a
    redaction that quietly does nothing is worse than no redaction.
    """
    notice = (
        r"\begin{quote}\small\emph{The prompt text of this row is withheld from the web "
        r"edition.} The row is \code{MGB-UD-00020} of the frozen \code{public\_test} split "
        r"(udaap / deceptive, difficulty \emph{hard}, gold action \textsc{block}). Its "
        r"labels, cited policy cards, per-guard scores and ranks are all reported below and "
        r"are original content; only the generated prompt is held back, because the "
        r"benchmark's redistribution decision is unresolved "
        r"(\code{benchmarks/registry/distribution.yaml}). The full text is in the PDF "
        r"edition and in a local build.\end{quote}"
    )

    # A. the block quotation: the paragraph ends "quoted verbatim:" and the quote follows.
    pat_block = re.compile(
        r"(quoted verbatim:\s*)\\begin\{quote\}.*?\\end\{quote\}", re.S)
    tex, n_block = pat_block.subn(lambda m: m.group(1) + notice, tex, count=1)
    if n_block != 1:
        raise RedactionError(
            "could not find the verbatim case-study quotation to withhold. The case study "
            "may have been regenerated; re-check what this page would publish before "
            "building, then update redact_restricted_rows().")

    # B. the inline quotation from the second row.
    pat_inline = re.compile(r"(MGB-FL-00028\}? asks whether )``.*?''", re.S)
    tex, n_inline = pat_inline.subn(
        lambda m: m.group(1) + "whether protected traits visible in a file may inform "
                              "pricing discretion (text withheld)", tex, count=1)
    if n_inline != 1:
        raise RedactionError(
            "could not find the second case-study quotation (MGB-FL-00028) to withhold.")

    # C. the same fragment recurs once in the Act III narrative. Two- and three-word
    #    characterizations of the row ("market fit", "resale stability") are our own analytic
    #    prose and stay; this one is a seven-word run of the generated prompt, so it goes.
    tex = tex.replace("(``surname, preferred language'')",
                      "(protected traits named outright)")
    if "surname, preferred language" in tex:
        raise RedactionError(
            "a quoted run of the MGB-FL-00028 prompt survives redaction; locate it before "
            "publishing. Longer verbatim runs must be withheld, not paraphrased in place.")
    return tex


def _boxes(tex: str) -> str:
    """\\begin{takeaway}{Title} ... \\end{takeaway}  ->  sentinel-delimited quote block."""
    for env, kind in (("takeaway", "takeaway"), ("background", "background"),
                      ("casebox", "case"), ("edbox", "editor")):
        pat = re.compile(r"\\begin\{" + env + r"\}\s*(?=\{)")
        while (m := pat.search(tex)):
            title, j = _balanced(tex, m.end())
            end = tex.index(f"\\end{{{env}}}", j)
            body = tex[j:end]
            tex = (tex[:m.start()]
                   + f"\n\n\\begin{{quote}}\n{BOX}{kind}:{title}{CLOSE}\n\n{body}\n\n{ENDBOX}\n\\end{{quote}}\n\n"
                   + tex[end + len(f"\\end{{{env}}}"):])
    return tex


PREAMBLE_SHIM = r"""\documentclass{article}
\usepackage{amsmath,amssymb,booktabs,graphicx,array,longtable}
% Cross-references and citations become sentinels the post-processor resolves against the
% float numbering it derives from document order -- so \Cref{tab:x} lands on the same
% number the PDF prints, rather than a hand-maintained guess.
\newcommand{\Cref}[1]{&REF;#1&CLOSE;}
\newcommand{\cref}[1]{&REF;#1&CLOSE;}
\newcommand{\citep}[1]{&CITE;#1&CLOSE;}
\newcommand{\citet}[1]{&CITE;#1&CLOSE;}
\newcommand{\citealp}[1]{&CITE;#1&CLOSE;}
\newcommand{\cite}[1]{&CITE;#1&CLOSE;}
\newcommand{\code}[1]{\texttt{#1}}
\newcommand{\edb}[1]{}
\newcommand{\draftwarning}{}
"""


_BARE_NUMBER = re.compile(r"^\$([^$\\]*)\$$")


def macro_shim() -> str:
    r"""Every \newcommand from generated/, with purely numeric bodies taken out of math mode.

    The generated macros wrap their values in math (`{$+0.129$}`) because the paper always
    uses them in text. But the prose also writes `$\AdaHGainLCB>0$`, which expands to a
    nested `$...$` -- LaTeX tolerates it, pandoc does not. Unwrapping the bodies that are
    just a signed number fixes both call sites and changes no value. Bodies containing real
    math (\KLTakeaway's KL term) are left exactly as written.
    """
    out = []
    for f in sorted(GEN.glob("*.tex")):
        for line in f.read_text().splitlines():
            if "newcommand" not in line:
                continue
            m = re.match(r"(\\newcommand\{\\\w+\})\{(.*)\}\s*$", line)
            if m and (inner := _BARE_NUMBER.match(m.group(2).strip())):
                line = f"{m.group(1)}{{{inner.group(1)}}}"
            out.append(line)
    return "\n".join(out)


def flatten(redact: bool = True) -> tuple[str, dict]:
    root = (SRC / "unified_report.tex").read_text()

    meta = {
        "title": " ".join(re.findall(r"\\title\{(.*?)\}\s*$", root, re.S | re.M)[:1]) or "",
        "author": (re.search(r"\\author\{(.*?)\n\}", root, re.S).group(1)
                   if re.search(r"\\author\{(.*?)\n\}", root, re.S) else ""),
    }
    body = root[root.index(r"\begin{document}") + len(r"\begin{document}"):root.rindex(r"\end{document}")]
    body = _expand_inputs(body)

    # --- title block: rendered by the template, not by pandoc
    body = re.sub(r"\\maketitle|\\thispagestyle\{[^}]*\}", "", body)
    abstract = ""
    if (m := re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)):
        abstract = m.group(1)
        body = body[:m.start()] + body[m.end():]

    # --- The tikz workflow flowchart is re-drawn as semantic HTML (accessible, selectable,
    #     and it reflows on a phone). Replace the WHOLE enclosing float: leaving the LaTeX
    #     figure wrapper in place would nest a <figure> in a <figure>, duplicating both the
    #     caption and the fig:gating id.
    body = re.sub(r"\\begin\{figure\}[^\n]*\n\s*\\begin\{tikzpicture\}.*?\\end\{figure\}",
                  f"\n\n{RAW}gating{CLOSE}\n\n", body, flags=re.S)

    # --- print-only commands with no HTML meaning
    for cmd in ("noindent", "smallskip", "medskip", "bigskip", "clearpage", "newpage",
                "FloatBarrier", "centering", "small", "footnotesize", "scriptsize",
                "bfseries", "itshape", "normalsize", "toprule", "midrule", "bottomrule",
                "hline", "arraybackslash", "raggedright", "sloppy"):
        body = re.sub(re.escape("\\" + cmd) + r"(?![A-Za-z])", " ", body)
    for cmd in ("rowcolor", "definecolor", "setlength", "renewcommand", "captionsetup",
                "addtocounter", "vspace", "hspace", "label@hack"):
        body = _unwrap(body, cmd, keep=99)                      # drop wrapper AND args
    body = re.sub(r"\\cmidrule(\([a-z]+\))?\{[\d-]+\}", " ", body)
    body = _strip_colspec_padding(body)
    body = re.sub(r"\\\\\s*\[[^\]]*\]", r"\\\\", body)          # \\[2pt] -> \\
    body = re.sub(r"\\begin\{sloppypar\}|\\end\{sloppypar\}", "", body)
    for cmd in ("resizebox", "textcolor", "colorbox"):
        body = _unwrap(body, cmd, keep=1 if cmd != "resizebox" else 2)
    body = body.replace(r"\ldots", "\u2026").replace(r"\textsc", r"\textbf")
    # `$1{,}200$` is a LaTeX idiom for a thin-space thousands separator, not real math.
    # Left alone it renders in MathJax's italic math font mid-sentence; as text it just
    # reads as 1,200 like the PDF does.
    body = re.sub(r"\$(\d[\d.]*(?:\{,\}\d{3})+)\$", lambda m: m.group(1).replace("{,}", ","), body)
    body = re.sub(r"(\d)\{,\}(\d{3})", r"\1,\2", body)
    # --- numbered equations: pandoc turns \begin{equation} into \[...\] and drops the
    #     \label, so \Cref{eq:score} had nowhere to land. Emit a sentinel just before each
    #     one and let the post-processor number and anchor it in document order.
    def _eq(m):
        inner = m.group(1)
        key = km.group(1) if (km := re.search(r"\\label\{(eq:[^}]+)\}", inner)) else ""
        inner = re.sub(r"\\label\{[^}]*\}", "", inner).strip()
        return f"\n\n{EQ}{key}{CLOSE}\n\n\\[{inner}\\]\n\n"
    body = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", _eq, body, flags=re.S)

    if redact:
        body = redact_restricted_rows(body)
    body = _boxes(body)

    tex = (PREAMBLE_SHIM + macro_shim()
           + "\n\\begin{document}\n" + body + "\n\\end{document}\n")
    # sentinel placeholders -> real sentinels (kept out of the shim so \n parsing is simple)
    tex = tex.replace("&REF;", REF).replace("&CITE;", CITE).replace("&CLOSE;", CLOSE)
    meta["abstract"] = abstract
    return tex, meta


def pandoc(tex: str) -> str:
    r = subprocess.run(["pandoc", "-f", "latex", "-t", "html5", "--mathjax", "--wrap=preserve"],
                       input=tex, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"pandoc failed:\n{r.stderr[-3000:]}")
    if r.stderr.strip():
        print("  pandoc notes:", r.stderr.strip().splitlines()[0][:120])
    return r.stdout


# --------------------------------------------------------------------- 3. bibliography
def bibliography() -> dict[str, dict]:
    """Minimal BibTeX read: enough for a numbered, alphabetized reference list."""
    raw = (SRC / "refs.bib").read_text()
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", raw):
        start = m.end()
        depth, i = 1, raw.index("{", m.start())
        i += 1
        while i < len(raw) and depth:
            depth += (raw[i] == "{") - (raw[i] == "}")
            i += 1
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"](.*?)[}\"],?\s*(?=\n\s*\w+\s*=|\n?\s*\}?\s*$)",
                              raw[start:i - 1], re.S):
            fields[fm.group(1).lower()] = " ".join(fm.group(2).split()).strip("{}")
        entries[m.group(2).strip()] = fields
    return entries


def _sortkey(f: dict) -> tuple:
    a = f.get("author", f.get("title", "zz"))
    return (a.split(" and ")[0].split(",")[0].split()[-1].lower(), f.get("year", ""))


def render_bib(entries: dict, cited: list[str]) -> tuple[str, dict[str, int]]:
    # Deduplicate in first-citation order, then sort by (surname, year, key). The cite key is
    # the final tiebreaker because two Wortsman 2022 entries tie on the first two, and sorting
    # a *set* on a partial key made the numbering flip between runs -- which broke --check.
    uniq = list(dict.fromkeys(k for k in cited if k in entries))
    keys = sorted(uniq, key=lambda k: (*_sortkey(entries[k]), k))
    num = {k: i + 1 for i, k in enumerate(keys)}
    rows = []
    for k in keys:
        f = entries[k]
        who = f.get("author", "").replace(" and ", ", ")
        venue = f.get("journal") or f.get("booktitle") or f.get("publisher") or ""
        bits = [b for b in (html.escape(who), f"<em>{html.escape(f.get('title',''))}</em>",
                            html.escape(venue), html.escape(f.get("year", ""))) if b.strip(" <em>/")]
        url = f.get("url") or (f"https://doi.org/{f['doi']}" if f.get("doi") else "")
        cite = ". ".join(bits)
        if url:
            cite += f'. <a href="{html.escape(url)}" rel="noopener">link</a>'
        rows.append(f'<li id="bib-{html.escape(k)}"><span class="bibnum">[{num[k]}]</span>'
                    f'<span class="bibtext">{cite}.</span></li>')
    return "<ol class=\"bib\">\n" + "\n".join(rows) + "\n</ol>", num


# ------------------------------------------------------------------ 4. post-processing
KINDS = {"tab": "Table", "fig": "Figure", "sec": "Section", "eq": "Equation",
         "app": "Appendix", "alg": "Algorithm"}


def postprocess(frag: str, meta: dict) -> str:
    from bs4 import BeautifulSoup, NavigableString
    soup = BeautifulSoup(frag, "html.parser")

    # ---- 4a. boxes: a quote whose first text is a BOX sentinel becomes a callout
    for q in soup.find_all("blockquote"):
        txt = q.get_text()
        if BOX not in txt:
            continue
        m = re.search(re.escape(BOX) + r"(\w+):(.*?)" + re.escape(CLOSE), txt, re.S)
        if not m:
            continue
        kind, title = m.group(1), " ".join(m.group(2).split())
        for p in q.find_all(["p", "div"]):
            if BOX in p.get_text() or ENDBOX in p.get_text():
                p.decompose()
        div = soup.new_tag("div", attrs={"class": f"callout callout-{kind}"})
        hd = soup.new_tag("div", attrs={"class": "callout-title"})
        hd.append(BeautifulSoup(title, "html.parser"))
        div.append(hd)
        body = soup.new_tag("div", attrs={"class": "callout-body"})
        for child in list(q.children):
            body.append(child.extract())
        div.append(body)
        q.replace_with(div)

    # ---- 4b. raw blocks (the workflow flowchart)
    for node in soup.find_all(string=re.compile(re.escape(RAW))):
        name = re.search(re.escape(RAW) + r"(\w+)" + re.escape(CLOSE), node)
        if name and name.group(1) == "gating":
            holder = node.find_parent(["p", "div", "figure"]) or node.parent
            holder.replace_with(BeautifulSoup(GATING_HTML, "html.parser"))

    # ---- 4c. section numbering.
    # pandoc maps \section->h1, \subsection->h2, \subsubsection->h3, \paragraph->h4.
    # The PDF numbers the first three and leaves \paragraph as an unnumbered run-in
    # heading, so h4 gets a class instead of a number.
    APPENDIX_STARTS_AT = "Related work"          # the first \section after \appendix
    top, sub, subsub, appx = 0, 0, 0, 0
    numbers: dict[str, str] = {}
    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        if h.name == "h4":
            h["class"] = h.get("class", []) + ["runin"]
            continue
        if h.name == "h1":
            if appx or h.get_text().strip().startswith(APPENDIX_STARTS_AT):
                appx += 1
                cur = chr(ord("A") + appx - 1)
            else:
                top += 1
                cur = str(top)
            sub = subsub = 0
        else:
            stem = chr(ord("A") + appx - 1) if appx else str(top)
            if h.name == "h2":
                sub += 1
                subsub = 0
                cur = f"{stem}.{sub}"
            else:
                subsub += 1
                cur = f"{stem}.{sub}.{subsub}"
        span = soup.new_tag("span", attrs={"class": "secnum"})
        span.string = cur
        h.insert(0, span)
        if h.get("id"):
            numbers[h["id"]] = cur
        h["data-num"] = cur

    # ---- 4d. float numbering in document order, cross-checked against the PDF
    counters = {"Table": 0, "Figure": 0}
    for el in soup.find_all(["table", "figure", "div"]):
        if el.name == "div" and not (el.get("id", "").startswith(("tab:", "fig:"))):
            continue
        if el.name == "table" and el.find_parent("div", id=re.compile("^tab:")):
            continue                                            # counted via its wrapper
        if el.name == "figure" and el.find_parent("figure"):
            continue
        # A bare <table> with no caption is a layout tabular inside another float -- the
        # case-study figure lays its four guard rows out that way. It is not Table N.
        if el.name == "table" and not el.find("caption"):
            el["class"] = el.get("class", []) + ["plaintable"]
            continue
        ident = el.get("id", "")
        inner = el.find(["table", "img"]) if el.name == "div" else el
        kind = ("Table" if (el.name == "table" or (inner and inner.name == "table"))
                else "Figure" if (el.name == "figure" or (inner and inner.name == "img"))
                else None)
        if kind is None:
            continue
        counters[kind] += 1
        n = counters[kind]
        if ident:
            numbers[ident] = str(n)
        cap = el.find(["caption", "figcaption"])
        if cap:
            lead = soup.new_tag("span", attrs={"class": "floatnum"})
            lead.string = f"{kind} {n}."
            cap.insert(0, lead)
            cap.insert(1, NavigableString(" "))
        if kind == "Table":
            # Restructure into <figure class=tablewrap><figcaption/><div class=tablescroll>.
            # A wide 8-column table has to be able to scroll sideways, but the caption must
            # NOT scroll with it -- so the overflow lives on an inner box, not the float.
            tbl = inner if el.name == "div" else el
            fig = soup.new_tag("figure", attrs={"class": "tablewrap"})
            if ident:
                fig["id"] = ident
            if cap:
                fc = soup.new_tag("figcaption")
                for child in list(cap.children):
                    fc.append(child.extract())
                fig.append(fc)
                cap.decompose()
            scroll = soup.new_tag("div", attrs={"class": "tablescroll"})
            scroll.append(tbl.extract())
            fig.append(scroll)
            fig["data-float"] = f"{kind} {n}"
            el.replace_with(fig)
        else:
            el["class"] = el.get("class", []) + ["figwrap"]
            el["data-float"] = f"{kind} {n}"

    # ---- 4d-bis. equations: number in document order, anchor, and tag the display math
    #      that follows so the number can sit in the right margin like the PDF's.
    eqn = 0
    for node in list(soup.find_all(string=re.compile(re.escape(EQ)))):
        m = re.search(re.escape(EQ) + r"(.*?)" + re.escape(CLOSE), str(node), re.S)
        if not m:
            continue
        eqn += 1
        key = m.group(1).strip()
        if key:
            numbers[key] = str(eqn)
        holder = node.find_parent("p") or node.parent
        math = holder.find_next(["p", "div"])
        wrap = soup.new_tag("div", attrs={"class": "eqwrap"})
        if key:
            wrap["id"] = key
        tag = soup.new_tag("span", attrs={"class": "eqnum"})
        tag.string = f"({eqn})"
        if math is not None:
            wrap.append(math.extract())
        wrap.append(tag)
        holder.replace_with(wrap)

    # ---- 4e. resolve \Cref / \cref sentinels
    for node in list(soup.find_all(string=re.compile(re.escape(REF)))):
        out, pos, txt = [], 0, str(node)
        for m in re.finditer(re.escape(REF) + r"(.*?)" + re.escape(CLOSE), txt, re.S):
            out.append(html.escape(txt[pos:m.start()]))
            parts = []
            for key in [k.strip() for k in m.group(1).split(",") if k.strip()]:
                pre = key.split(":")[0]
                kind = KINDS.get(pre, "Section")
                num = numbers.get(key)
                parts.append(f'<a class="xref" href="#{html.escape(key)}">{kind}&nbsp;{num}</a>'
                             if num else f'<span class="xref-missing">{kind}</span>')
            out.append(parts[0] if len(parts) == 1 else
                       ", ".join(parts[:-1]) + " and " + parts[-1])
            pos = m.end()
        out.append(html.escape(txt[pos:]))
        node.replace_with(BeautifulSoup("".join(out), "html.parser"))

    # ---- 4f. citations -> numbered superscript links
    cited: list[str] = []
    for node in soup.find_all(string=re.compile(re.escape(CITE))):
        for m in re.finditer(re.escape(CITE) + r"(.*?)" + re.escape(CLOSE), str(node), re.S):
            cited += [k.strip() for k in m.group(1).split(",") if k.strip()]
    bib_html, bibnum = render_bib(bibliography(), cited)
    for node in list(soup.find_all(string=re.compile(re.escape(CITE)))):
        out, pos, txt = [], 0, str(node)
        for m in re.finditer(re.escape(CITE) + r"(.*?)" + re.escape(CLOSE), txt, re.S):
            out.append(html.escape(txt[pos:m.start()]))
            keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
            links = [f'<a href="#bib-{html.escape(k)}">{bibnum[k]}</a>'
                     for k in keys if k in bibnum]
            out.append(f'<sup class="cite">[{", ".join(links)}]</sup>' if links else "")
            pos = m.end()
        out.append(html.escape(txt[pos:]))
        node.replace_with(BeautifulSoup("".join(out), "html.parser"))

    # ---- 4g. images: point at the converted assets.
    # pandoc emits <embed> rather than <img> for a .pdf graphic, because no browser renders
    # PDF in an <img>. Since we ship SVG conversions, rewrite those back to real <img>.
    for node in soup.find_all(["img", "embed"]):
        stem = Path(node.get("src", "")).stem
        target = next((c for c in (f"{stem}.svg", f"{stem}.png") if (FIG / c).exists()), None)
        if not target:
            continue
        img = soup.new_tag("img", src=f"assets/fig/{target}")
        img["loading"] = "lazy"
        cap = node.find_parent("figure")
        alt = cap.find("figcaption") if cap else None
        img["alt"] = (" ".join(alt.get_text().split())[:180] if alt
                      else f"Figure asset {stem}")
        node.replace_with(img)

    # ---- 4h. table of contents
    toc = ['<nav class="toc" aria-label="Contents"><div class="toc-head">Contents</div><ol>']
    for h in soup.find_all(["h1", "h2", "h3"]):
        if not h.get("id"):
            h["id"] = "s-" + re.sub(r"[^a-z0-9]+", "-", h.get_text().lower()).strip("-")[:48]
        depth = "sub" if h.name == "h3" else "top"
        label = h.get_text().replace(h.get("data-num", ""), "", 1).strip()
        toc.append(f'<li class="toc-{depth}"><a href="#{h["id"]}">'
                   f'<span class="toc-num">{h.get("data-num","")}</span>'
                   f'<span>{html.escape(label)}</span></a></li>')
    toc.append("</ol></nav>")

    return (soup.decode(), "\n".join(toc), bib_html,
            {"tables": counters["Table"], "figures": counters["Figure"]})


# ------------------------------------------------------------------------- 5. verify
def verify(counts: dict) -> list[str]:
    """The PDF is ground truth for float numbering; disagreement is a build defect."""
    pdf = SRC / "unified_report.pdf"
    problems = []
    if not shutil.which("pdftotext") or not pdf.exists():
        return ["skipped: pdftotext or unified_report.pdf unavailable"]
    txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
    for kind, key in (("Table", "tables"), ("Figure", "figures")):
        want = len(set(re.findall(rf"^{kind} (\d+):", txt, re.M)))
        if want != counts[key]:
            problems.append(f"{kind} count: HTML {counts[key]} vs PDF {want}")
    return problems


GATING_HTML = """
<figure id="fig:gating" class="figwrap flowchart">
<div class="flow">
  <div class="flow-step"><b>1. Freeze the candidate registry</b><span>base · SFT · KL-SFT ·
    composition — checkpoint, prompt, calibrator, threshold rule, owner</span></div>
  <div class="flow-arrow" aria-hidden="true"></div>
  <div class="flow-step"><b>2. Calibrate and choose a threshold</b><span>separately for
    <em>each</em> candidate, on target-regime calibration data</span></div>
  <div class="flow-arrow" aria-hidden="true"></div>
  <div class="flow-step"><b>3. Open the blind acceptance set once</b><span>paired rows, one shot</span></div>
  <div class="flow-arrow" aria-hidden="true"></div>
  <div class="flow-step flow-gate"><b>4. Required gates</b><span>absolute-AP floor · operating point
    (FPR / recall) · transfer retention vs. base · each domain separately · reliability ·
    service SLO · governance</span></div>
  <div class="flow-arrow" aria-hidden="true"></div>
  <div class="flow-dec">All required gates pass?</div>
  <div class="flow-split">
    <div class="flow-branch">
      <div class="flow-label flow-yes">yes</div>
      <div class="flow-step flow-ship"><b>Select</b><span>incumbent-first → shadow → canary →
        monitor → rollback-ready</span></div>
    </div>
    <div class="flow-branch">
      <div class="flow-label flow-no">no / missing evidence</div>
      <div class="flow-step flow-stop"><b>No ship</b><span><code>NO_FEASIBLE_THRESHOLD</code>:
        redesign, escalate, or change the requirement</span></div>
    </div>
  </div>
</div>
<figcaption><b>Gate candidates, not leaderboards</b> (recommended workflow; not validated end to end
by this study). Every candidate is calibrated and thresholded separately, evaluated once on a blind
acceptance set, and must clear <em>all</em> required gates; a missing required gate is a failure, and
an empty feasible set is a deliberate no-ship, not a relaxed cutoff.</figcaption>
</figure>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--redact-case-study", action="store_true",
                    help="withhold the two verbatim v1_hmda2022 rows. No longer needed: that "
                         "source is CC BY 4.0 as of 2026-07-27 and redistribution is approved. "
                         "Retained for any future source whose licence is unresolved.")
    ap.add_argument("--check", action="store_true",
                    help="fail if the rebuild differs from the committed index.html")
    args = ap.parse_args(argv)

    print("Building the HTML edition from ../unified-report ...")
    print(f"  figures converted/copied: {figures()}")
    tex, meta = flatten(redact=args.redact_case_study)
    frag = pandoc(tex)
    body, toc, bib, counts = postprocess(frag, meta)
    print(f"  floats numbered: {counts['tables']} tables, {counts['figures']} figures")

    problems = verify(counts)
    for p in problems:
        print(f"  VERIFY: {p}")

    abstract_tex = ((PREAMBLE_SHIM + macro_shim())
                    .replace("&REF;", REF).replace("&CITE;", CITE).replace("&CLOSE;", CLOSE)
                    + "\n\\begin{document}\n" + meta["abstract"] + "\n\\end{document}\n")
    abstract_html = pandoc(abstract_tex)
    from bs4 import BeautifulSoup
    asoup = BeautifulSoup(abstract_html, "html.parser")
    for node in list(asoup.find_all(string=re.compile(re.escape(REF) + "|" + re.escape(CITE)))):
        node.replace_with(re.sub(re.escape(REF) + r".*?" + re.escape(CLOSE), "",
                          re.sub(re.escape(CITE) + r".*?" + re.escape(CLOSE), "", str(node))))
    abstract_html = asoup.decode()

    page = (HERE / "template.html").read_text()
    for k, v in {"{{TITLE}}": "The Safety-Guard Benchmark Chooses the Winner",
                 "{{SUBTITLE}}": ("Measuring, Tuning, and Composing Small Safety Guards "
                                  "in High-Compliance Business Domains"),
                 "{{ABSTRACT}}": abstract_html, "{{TOC}}": toc,
                 "{{BODY}}": body, "{{BIB}}": bib,
                 "{{SITE_URL}}": SITE_URL, "{{DESCRIPTION}}": DESCRIPTION,
                 "{{KEYWORDS}}": KEYWORDS,
                 "{{NTAB}}": str(counts["tables"]), "{{NFIG}}": str(counts["figures"])}.items():
        page = page.replace(k, v)
    assert "{{" not in page, f"unsubstituted template placeholder: {page[page.index('{{'):][:40]}"

    emitted = {"index.html": page, **seo_files()}

    if args.check:
        stale = [n for n, want in emitted.items()
                 if not (HERE / n).exists() or (HERE / n).read_text() != want]
        if stale:
            print(f"CHECK FAILED: differs from a fresh build: {', '.join(sorted(stale))}")
            return 1
        print(f"  byte-identical to a fresh build: {', '.join(sorted(emitted))}")
        return 1 if problems and "skipped" not in problems[0] else 0

    for name, text in emitted.items():
        (HERE / name).write_text(text)
    print("  wrote " + ", ".join(f"{n} ({len(t)/1024:.0f} KB)" if n.endswith(".html")
                                 else n for n, t in emitted.items()))
    return 1 if problems and "skipped" not in problems[0] else 0


if __name__ == "__main__":
    raise SystemExit(main())
