"""The Pages deployment must stay gated on the distribution ledger.

A workflow that *can* publish is a capability, and this repository has already learned that a
publication capability guarded only by intent gets used: the withdrawn 55 MB explorer artifact
was authorized by a `.gitignore` comment. So the gate is checked here the same way the
generator is -- by what the workflow would actually do, not by whether someone wrote the word
"gate" in it.

Two of these tests assert the *current* policy state (refused). If a licensing decision is
made they will fail, which is the intended behaviour: the decision should be accompanied by a
deliberate update to these expectations, not silently absorbed.

One limit, stated plainly: this covers the workflow. It cannot see GitHub's repository
Settings, where "Deploy from a branch" would serve the tree with no gate at all. That path is
outside anything a test in the repository can reach.
"""

import json
import pathlib
import re
import subprocess
import sys

import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = _ROOT / ".github/workflows/pages.yml"
ARTIFACT = "papers/unified-report-html"
REQUIREMENTS = _ROOT / ARTIFACT / "PUBLICATION_REQUIREMENTS.json"
GATE = _ROOT / "tools/pages_authorized.py"


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert WORKFLOW.is_file(), f"{WORKFLOW.relative_to(_ROOT)} is missing"
    # PyYAML reads the `on:` key as the boolean True; that is fine, we address it as such.
    return yaml.safe_load(WORKFLOW.read_text())


FIXTURE_UNAPPROVED = "tests/fixtures/pages_artifact_unapproved"


def test_the_gate_authorizes_the_page_because_it_carries_no_restricted_text():
    """Authorized on an empty dependency set -- not because a source was approved.

    The page used to require `mortgage_benchmark_v1_hmda2022`, whose DATA_CARD still reads
    "LICENSE NOT YET SELECTED". That requirement was removed by withholding the quotation,
    which is why this passes while the ledger is untouched. If someone re-adds restricted
    text, PUBLICATION_REQUIREMENTS.json has to name the source again and this flips back to
    a refusal.
    """
    r = subprocess.run([sys.executable, str(GATE), "--artifact", ARTIFACT],
                       cwd=_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (
        f"the page is no longer authorized for publication.\n{r.stdout}\n{r.stderr}"
    )
    assert "AUTHORIZED" in r.stdout

    req = json.loads(REQUIREMENTS.read_text())
    assert req["requires_publication_approval_for"] == [], (
        "the artifact now requires a source approval, so publication must be re-reviewed"
    )
    assert len(req.get("no_approval_required_because", "").split()) >= 25, (
        "an empty requirement list must carry a substantive justification"
    )


def test_the_gate_still_refuses_an_artifact_that_needs_an_unapproved_source():
    """The refusal path must stay tested once the real artifact stops needing approval.

    Otherwise the only test of a refusal is the live policy state -- the one thing that
    changes -- and the gate could rot into always-yes without any test noticing.
    """
    r = subprocess.run([sys.executable, str(GATE), "--artifact", FIXTURE_UNAPPROVED],
                       cwd=_ROOT, capture_output=True, text=True)
    assert r.returncode == 1, (
        f"the gate authorized an artifact needing an unapproved source.\n{r.stdout}"
    )
    assert "REFUSED" in r.stdout
    assert "mortgage_guard_bench_2k_v0_1_0" in r.stdout


def test_the_ledger_hold_on_the_mortgage_benchmark_is_untouched():
    """Publishing the page must not have quietly relaxed the source's own decision."""
    ledger = yaml.safe_load((_ROOT / "benchmarks/registry/distribution.yaml").read_text())
    src = next(s for s in ledger["sources"]
               if s["source_id"] == "mortgage_benchmark_v1_hmda2022")
    assert src["redistribution_decision"] == "local_only", (
        "mortgage_benchmark_v1_hmda2022 is no longer local_only. If that is a real "
        "licensing decision, its DATA_CARD.md still says 'LICENSE NOT YET SELECTED' and "
        "names an FFIEC/CFPB terms-of-use precondition -- resolve both, and name a "
        "reviewer, in the same commit."
    )
    assert src["license"]["permits_redistribution"] is not True


def test_the_gate_refuses_rather_than_crashes_on_a_broken_declaration():
    """A gate that errors must not be read as permission."""
    r = subprocess.run([sys.executable, str(GATE), "--artifact", "papers/does-not-exist"],
                       cwd=_ROOT, capture_output=True, text=True)
    assert r.returncode != 0, "a missing requirements file was treated as authorization"


def test_every_deploying_job_depends_on_the_authorize_job(workflow):
    """Catch the capability, not the wording: trace `needs` back to the gate."""
    jobs = workflow["jobs"]
    assert "authorize" in jobs, "the pages workflow has no authorize job"

    def needs_of(name: str) -> list[str]:
        n = jobs[name].get("needs", [])
        return [n] if isinstance(n, str) else list(n)

    def reaches_authorize(name: str, seen: set[str]) -> bool:
        if name == "authorize":
            return True
        if name in seen:
            return False
        seen.add(name)
        return any(reaches_authorize(d, seen) for d in needs_of(name))

    publishing = [
        name for name, spec in jobs.items()
        if any("deploy-pages" in str(step.get("uses", ""))
               or "upload-pages-artifact" in str(step.get("uses", ""))
               for step in spec.get("steps", []))
    ]
    assert publishing, "no job uploads or deploys a Pages artifact; the workflow is inert"
    ungated = [n for n in publishing if not reaches_authorize(n, set())]
    assert not ungated, (
        f"these jobs can publish without the ledger gate: {ungated}. Every job that uploads "
        "or deploys a Pages artifact must depend, directly or transitively, on `authorize`."
    )


def test_the_authorize_job_actually_runs_the_gate(workflow):
    """`needs: authorize` is worthless if the authorize job does not check anything."""
    steps = workflow["jobs"]["authorize"].get("steps", [])
    runs = " ".join(str(s.get("run", "")) for s in steps)
    assert "tools/pages_authorized.py" in runs, (
        "the authorize job does not invoke tools/pages_authorized.py, so it gates nothing"
    )


def test_the_workflow_does_not_publish_on_push(workflow):
    """Publication must be a deliberate act, not a consequence of merging."""
    triggers = workflow[True] if True in workflow else workflow.get("on")
    assert triggers, "the pages workflow declares no triggers"
    keys = set(triggers) if isinstance(triggers, dict) else {triggers}
    assert "push" not in keys, (
        "the pages workflow triggers on push. While no source is approved for "
        "redistribution, publication must require an explicit workflow_dispatch."
    )


PAGE = _ROOT / "papers/unified-report-html/index.html"
WITHHELD_MARKER = "prompt text of this row is withheld"
# Verbatim runs of the two quoted benchmark rows. Two-word characterizations in our own
# analytic prose ("market fit", "resale stability") are not redistribution and stay.
VERBATIM_RUNS = (
    "Montana conventional purchase",
    "surname, preferred language",
    "align well with our portfolio",
    "help you frame the decision note",
)


def test_the_committed_page_is_the_redacted_edition():
    """Assert the artifact's property directly, not that a build was run correctly.

    The workflow publishes these committed bytes, so this is the check that stands between
    the repository and serving a row of an unlicensed benchmark. It deliberately does not
    depend on build.py having run, or on which pandoc produced the markup.
    """
    assert PAGE.is_file(), "the published artifact is missing"
    page = PAGE.read_text(encoding="utf-8", errors="replace")
    assert WITHHELD_MARKER in page, (
        "the committed page carries no withholding notice, so it was probably built with "
        "--with-restricted-text. That edition is for local reading and must not be committed "
        "or served while the ledger is unresolved."
    )
    leaked = [s for s in VERBATIM_RUNS if s in page]
    assert not leaked, (
        f"verbatim benchmark row text is present in the page that gets served: {leaked}. "
        "Rebuild without --with-restricted-text, and check redact_restricted_rows() still "
        "matches the case study."
    )


# href/src values, without needing an HTML parser: the root suite runs in a CI environment
# that installs no beautifulsoup4.
_LINK = re.compile(r"""\b(?:href|src)\s*=\s*["']([^"']+)["']""")
STAGED_PREFIX = "assets/"


def test_every_relative_link_in_the_page_resolves_inside_the_published_site():
    r"""A relative link that is right in the repo can still 404 on the site.

    Pages serves papers/unified-report-html/ AS the site root, so `../unified-report/…` --
    correct when browsing the repository -- resolved to
    https://rrahimi-uci.github.io/unified-report/… and 404'd. The published page's only
    relative links must therefore point at files that are actually staged, which is
    index.html plus assets/.
    """
    page = PAGE.read_text(encoding="utf-8", errors="replace")
    bad = []
    for raw in _LINK.findall(page):
        if raw.startswith(("#", "http://", "https://", "mailto:", "data:", "//")):
            continue
        if not raw.startswith(STAGED_PREFIX):
            bad.append(f"{raw} (not under {STAGED_PREFIX}, so it is not staged for the site)")
            continue
        if not (PAGE.parent / raw.split("#")[0].split("?")[0]).is_file():
            bad.append(f"{raw} (staged prefix but the file does not exist)")
    assert not bad, (
        "relative links in the published page do not resolve inside the site: "
        + "; ".join(bad)
        + ". Either stage the target under assets/, or make the link absolute -- a `../` "
        "link escapes the Pages site root even though it is correct in the repository."
    )


SEO_FILES = ("robots.txt", "sitemap.xml")


def test_the_site_url_agrees_everywhere_and_is_where_the_page_actually_lives():
    r"""A canonical URL that disagrees with reality fails silently -- crawlers just obey it.

    The predecessor site got these fields from jekyll-seo-tag against a `baseurl` in
    _config.yml. This site interpolates one SITE_URL constant instead, so the risk moves from
    "two configs disagree" to "the constant is stale". Pin it to the repository name, which is
    what GitHub Pages derives the path from.
    """
    page = PAGE.read_text(encoding="utf-8", errors="replace")
    expected = "https://rrahimi-uci.github.io/safety-guard-dynamics/"

    canonical = re.search(r'<link rel="canonical" href="([^"]+)"', page)
    assert canonical and canonical.group(1) == expected, (
        f"canonical is {canonical.group(1) if canonical else 'absent'}, expected {expected}"
    )
    og = re.search(r'property="og:url" content="([^"]+)"', page)
    assert og and og.group(1) == expected, "og:url disagrees with the canonical URL"

    for name in SEO_FILES:
        f = PAGE.parent / name
        assert f.is_file(), f"{name} is not generated; run build.py"
        assert expected in f.read_text(), f"{name} does not reference {expected}"

    # The JSON-LD must be valid, or search engines silently drop the structured data.
    ld = re.search(r'type="application/ld\+json">\s*(\{.*?\})\s*</script>', page, re.S)
    assert ld, "no JSON-LD block in the page"
    data = json.loads(ld.group(1))          # raises if malformed
    assert data["url"] == expected, "JSON-LD url disagrees with the canonical URL"
    assert data["@type"] == "ScholarlyArticle"
    assert data["image"].startswith(expected), "og/JSON-LD image is not on this site"
    assert (PAGE.parent / data["image"][len(expected):]).is_file(), (
        "the social preview image referenced by og:image / JSON-LD does not exist"
    )


def test_the_workflow_stages_the_seo_files(workflow):
    """robots.txt and sitemap.xml are useless if they are not actually served."""
    staging = " ".join(str(s.get("run", "")) for j in workflow["jobs"].values()
                       for s in j.get("steps", []))
    missing = [n for n in SEO_FILES if n not in staging]
    assert not missing, (
        f"the workflow does not stage {missing} into the published site, so crawlers will "
        "get a 404 for them"
    )


def test_the_workflow_stages_the_site_rather_than_the_source_directory(workflow):
    """Uploading the source directory also serves build.py, the README and the requirements."""
    upload = [s for j in workflow["jobs"].values() for s in j.get("steps", [])
              if "upload-pages-artifact" in str(s.get("uses", ""))]
    assert upload, "no step uploads a Pages artifact"
    paths = [str(s.get("with", {}).get("path", "")) for s in upload]
    assert all(p.strip("./") not in ("papers/unified-report-html",) for p in paths), (
        f"the Pages artifact is the source directory ({paths}); stage index.html and assets/ "
        "into a separate directory so the build tooling is not served as site content"
    )


def test_the_pdf_is_not_staged_into_the_site(workflow):
    """Linking to the repository copy is fine; serving it from the site is not.

    The PDF contains the case-study row this edition withholds, and the text probes cannot
    see inside it, so a staged PDF would be exposure the quotation budget could not catch.
    """
    staging = " ".join(str(s.get("run", "")) for j in workflow["jobs"].values()
                       for s in j.get("steps", []))
    assert "unified_report.pdf" not in staging, (
        "the workflow copies the PDF into the published site. It carries the withheld "
        "case-study row, and no text probe can see inside a PDF."
    )


def test_the_artifact_declares_what_it_needs_approved():
    """The per-source requirement is recorded next to the artifact, not inferred."""
    assert REQUIREMENTS.is_file(), f"{REQUIREMENTS.relative_to(_ROOT)} is missing"
    req = json.loads(REQUIREMENTS.read_text())
    assert req["artifact"].startswith(ARTIFACT)
    required = req["requires_publication_approval_for"]
    ledger = yaml.safe_load((_ROOT / "benchmarks/registry/distribution.yaml").read_text())
    known = {s["source_id"] for s in ledger["sources"]}
    unknown = [s for s in required if s not in known]
    assert not unknown, (
        f"the artifact requires approval for sources absent from the ledger: {unknown}"
    )
    # An empty list is the strongest claim the file can make, so it needs the most support:
    # a justification, and the record of what the requirement used to be and why it went.
    justification = req.get("no_approval_required_because", "") if not required else \
        req.get("rationale", "")
    assert len(justification.split()) >= 20, (
        "record why no source needs approval -- or why these and not others; a bare list rots"
    )
    if not required:
        history = req.get("history", [])
        assert history and all(len(h.get("why_it_no_longer_applies", "").split()) >= 15
                               for h in history), (
            "an empty requirement list must record what it replaced and why, so that "
            "'needs nothing approved' cannot be quietly asserted from the start"
        )
