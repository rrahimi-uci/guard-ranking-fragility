"""Study navigation packages must stay subordinate to the registry.

`studies/<slug>/` gives each study one page answering what it asks, what it may claim,
where its code and evidence live, and how to verify it. The risk in adding such a page is
that it becomes a second place where study state is written down, and then disagrees with
`studies/registry.yaml`. These tests keep it subordinate: the content is generated, the
Makefile only delegates, and no package may exist that the registry does not know about.
"""

import pathlib
import subprocess
import sys

import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import render_indexes  # noqa: E402
import study_verify  # noqa: E402

# Owns its own Makefile and sources; it is a code package, not a generated one.
SELF_CONTAINED = {"paper-c-specialize-align-mortgage-v1"}


@pytest.fixture(scope="module")
def registry() -> dict:
    return yaml.safe_load((_ROOT / "studies/registry.yaml").read_text())


def packaged(registry) -> list[dict]:
    return [s for s in registry["studies"] if s.get("package_path")]


def test_every_declared_package_exists_and_is_generated(registry):
    assert packaged(registry), "expected at least one study navigation package"
    for study in packaged(registry):
        pkg = _ROOT / study["package_path"]
        assert pkg.is_dir(), f"{study['study_id']} declares a package that does not exist: {pkg}"
        readme = pkg / "README.md"
        assert readme.is_file(), f"{pkg} has no README.md"
        assert readme.read_text().startswith(render_indexes.BANNER), (
            f"{readme} is not marked generated; a hand-edited package page becomes a "
            "second source of study state."
        )
        assert (pkg / "Makefile").is_file(), f"{pkg} has no Makefile"


def test_generated_packages_are_current(registry):
    """Same freshness rule as the indexes, so a registry edit cannot leave pages stale."""
    for study in packaged(registry):
        pkg = _ROOT / study["package_path"]
        assert (pkg / "README.md").read_text() == render_indexes.package_readme(study, registry)
        assert (pkg / "Makefile").read_text() == render_indexes.package_makefile(study)


def test_package_makefiles_do_not_restate_the_verification_command(registry):
    """A package delegates; it never carries its own copy of the command."""
    for study in packaged(registry):
        body = (_ROOT / study["package_path"] / "Makefile").read_text()
        assert "study_verify.py" in body, f"{study['study_id']} does not delegate"
        # The verbatim command, not a token from it: `ls` as a token matches inside
        # "klsft", which is how the first version of this assertion failed on a
        # correctly-generated Makefile.
        assert study["verification_command"] not in body, (
            f"{study['study_id']}'s Makefile restates its verification command; it must "
            "look the command up from the registry instead."
        )


def test_no_unregistered_package_directory_exists(registry):
    """A directory under studies/ that the registry does not know about is orphaned state."""
    declared = {s["package_path"].rstrip("/").split("/")[-1] for s in packaged(registry)}
    on_disk = {p.name for p in (_ROOT / "studies").iterdir()
               if p.is_dir() and not p.name.startswith(".")}
    orphans = on_disk - declared - SELF_CONTAINED
    assert not orphans, (
        f"study directories absent from studies/registry.yaml: {sorted(orphans)}. "
        "Register them or remove them."
    )


def test_self_contained_package_is_not_declared_as_generated(registry):
    """Rendering over a code package would destroy its real build entry point.

    Learned the hard way: setting package_path on the Paper C study overwrote a Makefile
    carrying test/validate/grid/readiness/paper/lock targets with a two-line stub.
    """
    for study in packaged(registry):
        slug = study["package_path"].rstrip("/").split("/")[-1]
        assert slug not in SELF_CONTAINED, (
            f"{study['study_id']} is self-contained and must not declare package_path; "
            "rendering would overwrite its Makefile."
        )
    paper_c = _ROOT / "studies/paper-c-specialize-align-mortgage-v1/Makefile"
    if paper_c.is_file():
        body = paper_c.read_text()
        for target in ("test:", "validate:", "readiness:"):
            assert target in body, f"the Paper C package Makefile lost its {target} target"


def test_study_verify_resolves_every_registered_study(registry):
    """The delegator must know every study, or a package can point at nothing."""
    for study in registry["studies"]:
        assert study_verify.load_study(study["study_id"])["study_id"] == study["study_id"]
    with pytest.raises(SystemExit):
        study_verify.load_study("no_such_study")


def test_study_verify_inverts_the_expectation_for_declared_failures():
    """expected_fail must FAIL when its command passes -- a resolved blocker is a finding."""
    out = subprocess.run(
        [sys.executable, "tools/study_verify.py", "paper_c_specialize_align_mortgage_v1"],
        cwd=_ROOT, capture_output=True, text=True)
    assert out.returncode == 0, (
        "a study declared expected_fail should exit 0 while it still fails:\n" + out.stdout)
    assert "failed as declared" in out.stdout

    ok = subprocess.run(
        [sys.executable, "tools/study_verify.py", "expguard_external"],
        cwd=_ROOT, capture_output=True, text=True)
    assert ok.returncode == 0 and "ok" in ok.stdout
