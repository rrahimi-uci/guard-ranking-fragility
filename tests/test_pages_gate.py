"""The Pages deployment must stay gated on the distribution ledger.

A workflow that *can* publish is a capability, and this repository has already learned that a
publication capability guarded only by intent gets used: the withdrawn 55 MB explorer artifact
was authorized by a `.gitignore` comment. So the gate is checked here the same way the
generator is -- by what the workflow would actually do, not by whether someone wrote the word
"gate" in it.

Several of these tests assert the *current* policy state. That state has now changed twice --
v1_hmda2022 went from unresolved to withheld-by-redaction to licensed CC BY 4.0 -- and each
transition was caught by one of these tests failing, never by someone noticing. That is the
intended behaviour: a policy change should have to be accompanied by a deliberate update here,
not silently absorbed. Expect to edit this file when a licence changes.

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


def test_the_gate_authorizes_the_page_on_a_recorded_licence():
    """Authorized because the source it depends on is licensed -- the gate's real job.

    This has now asserted three different states, and each transition was forced by a test
    failing rather than noticed by luck: REFUSED while v1_hmda2022 was unresolved; AUTHORIZED
    on an empty dependency set once the quotation was withheld; and now AUTHORIZED on the
    licence itself, with the dependency declared again and satisfied on its merits. The
    invariant across all three is that the gate never authorizes silently -- the reason is
    always recorded and always checked.
    """
    r = subprocess.run([sys.executable, str(GATE), "--artifact", ARTIFACT],
                       cwd=_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (
        f"the page is no longer authorized for publication.\n{r.stdout}\n{r.stderr}"
    )
    assert "AUTHORIZED" in r.stdout
    assert "mortgage_benchmark_v1_hmda2022" in r.stdout, (
        "the gate authorized without naming the source it relied on; an unexplained yes is "
        "the failure mode this gate exists to prevent"
    )

    req = json.loads(REQUIREMENTS.read_text())
    assert req["requires_publication_approval_for"] == ["mortgage_benchmark_v1_hmda2022"], (
        "the declared dependency changed; publication must be re-reviewed"
    )
    assert req.get("history"), "the requirement's history must record what it replaced"


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


def test_the_mortgage_benchmark_licence_is_completely_recorded():
    """The licence decision must be complete, and the artifact must agree with the ledger.

    This replaces a test that asserted the source was still `local_only`. That assertion did
    its job: it failed the moment the decision changed, which forced the change to be made
    deliberately rather than absorbed. What matters now is the opposite risk -- a decision
    recorded in fragments, with a licence but no reviewer, or a ledger that says CC BY 4.0
    while the shipped data card still says no licence was selected. Both are worse than the
    original hold, because a redistributor reads the card.
    """
    ledger = yaml.safe_load((_ROOT / "benchmarks/registry/distribution.yaml").read_text())
    src = next(s for s in ledger["sources"]
               if s["source_id"] == "mortgage_benchmark_v1_hmda2022")

    assert src["redistribution_decision"] == "publish_text"
    assert src["license"]["permits_redistribution"] is True
    assert src["license"]["spdx_or_name"] == "CC-BY-4.0"
    assert src["reviewer"] != "unassigned", "an approved source needs a named reviewer"
    assert src["attribution_notice"], "CC BY 4.0 requires an attribution notice to pass on"

    # The data card ships with the data, so it is what a redistributor actually reads.
    card = (_ROOT / "mortgage-benchmark/benchmark/v1_hmda2022/DATA_CARD.md").read_text()
    assert "LICENSE NOT YET SELECTED" not in card, (
        "the ledger approves redistribution but the shipped DATA_CARD.md still says no "
        "licence was selected -- the artifact and the record must not disagree"
    )
    assert "CC BY 4.0" in card, "DATA_CARD.md does not state the licence it is released under"
    for caveat in ("not SME-adjudicated", "release bytes only"):
        assert caveat in card, (
            f"DATA_CARD.md dropped the {caveat!r} boundary; it survives the licence and has "
            "to travel with every redistribution"
        )


def test_the_release_checksums_still_cover_every_shipped_file():
    """Re-freezing the card's digest must not have disturbed the frozen data."""
    import hashlib
    d = _ROOT / "mortgage-benchmark/benchmark/v1_hmda2022"
    lines = [l.split(None, 1) for l in (d / "CHECKSUMS.txt").read_text().splitlines() if l.strip()]
    assert len(lines) >= 8, "CHECKSUMS.txt lost entries"
    bad = []
    for want, name in lines:
        f = d / name.strip().lstrip("./")
        if not f.is_file():
            bad.append(f"{name.strip()} missing")
            continue
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != want:
            bad.append(f"{name.strip()} {want[:12]} != {got[:12]}")
    assert not bad, "release checksums do not verify: " + "; ".join(bad)


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


def test_the_committed_page_carries_the_rows_and_the_licence_together():
    """The rows and their attribution are one unit. Either both are present, or neither.

    This replaces a test that required the withholding notice. That requirement was correct
    while v1_hmda2022 was unresolved and it failed the moment the redaction was lifted, which
    is what forced this rewrite instead of a silent divergence. The rule that replaces it is
    the one CC BY 4.0 actually imposes: a page may publish the rows, or withhold them, but it
    may not publish them stripped of the notice.

    Deliberately reads the committed artifact rather than trusting that build.py ran, and does
    not depend on which pandoc produced the markup.
    """
    assert PAGE.is_file(), "the published artifact is missing"
    page = PAGE.read_text(encoding="utf-8", errors="replace")

    quotes_rows = any(s in page for s in VERBATIM_RUNS)
    withheld = WITHHELD_MARKER in page
    assert quotes_rows != withheld, (
        "the page must either quote the case-study rows or carry the withholding notice, "
        f"not both and not neither (quotes_rows={quotes_rows}, withheld={withheld})"
    )

    if quotes_rows:
        for token in ("MortgageGuardBench", "CC BY 4.0"):
            assert token in page, (
                f"the page publishes v1_hmda2022 rows verbatim but is missing {token!r}. "
                "CC BY 4.0 permits redistribution on the condition of attribution; dropping "
                "the notice breaches the licence the ledger records."
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


def test_the_page_carries_the_attribution_its_licence_requires():
    """CC BY 4.0 is permissive but conditional; publishing rows without the notice breaches it.

    The notice is a build output rather than a manual step precisely so it cannot be forgotten
    on a rebuild, and this asserts that the build actually emitted it.
    """
    req = json.loads(REQUIREMENTS.read_text())
    attr = req.get("attribution_required")
    assert attr, "the artifact declares no attribution requirement"
    page = PAGE.read_text(encoding="utf-8", errors="replace")
    for token in ("MortgageGuardBench", "CC BY 4.0", "creativecommons.org/licenses/by/4.0"):
        assert token in page, (
            f"the published page is missing {token!r}. It redistributes v1_hmda2022 rows under "
            "CC BY 4.0, whose attribution condition is not optional."
        )
    # The caveats that survive the licence must travel with the rows.
    for caveat in ("not\n      SME-adjudicated", "SME-adjudicated"):
        if caveat in page:
            break
    else:
        raise AssertionError("the attribution dropped the not-SME-adjudicated boundary")


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
