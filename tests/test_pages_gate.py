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
