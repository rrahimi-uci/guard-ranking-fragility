"""Fail-closed configuration, task, path, and identity contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import tempfile


ACTIONS = ("allow", "review", "intervene")
ALIGNMENT_ARMS = (
    "gold_sft",
    "soft_distill",
    "specialist_pairce",
    "generalist_cm_dpo",
    "specialist_cm_dpo",
)
PRIMARY_SEEDS = (42, 43, 44)
FAMILY_SPLITS = (
    "specialist_train",
    "alignment_pool",
    "calibration",
    "checkpoint_selection",
)
READINESS_GATES = (
    "mortgage_policy_sme_signed",
    "annotation_rubric_signed",
    "licence_ledger_complete",
    "power_pilot_complete",
    "sealed_cohorts_created",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """An input violates a preregistered scientific contract."""


def project_root() -> Path:
    """The study workspace root: the nearest ancestor holding this package's pyproject."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    return here.parents[2]  # last resort: the historical fixed depth


REPO_MARKERS = ("guard_research", "benchmarks/registry", ".git")


def repository_root(start: Path | None = None) -> Path:
    """Find the monorepo root by marker, not by counting parent directories.

    A fixed `parents[N]` walk is correct only at the depth it was written for. This
    workspace exists at two depths -- papers/paper_c/specialize_then_align (3) and
    studies/paper-c-specialize-align-mortgage-v1 (2) -- and at the shallower one the
    old parents[2] resolved to a directory *outside* the repository, which would have
    silently read corpora from the wrong place.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if any((candidate / marker).exists() for marker in REPO_MARKERS):
            return candidate
    raise ContractError(
        f"no repository marker {REPO_MARKERS} found above {here}; "
        "cannot locate the monorepo root"
    )


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ordered(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = canonical_json_bytes(value)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


STUDY_ID = "paper_c_specialize_align_mortgage_v1"
RUNS_SLUG = "paper-c-specialize-align-mortgage-v1"


def runs_root() -> Path:
    """The storage contract's transient-output location: ignored <repo>/runs/<slug>/.

    Named by study slug rather than by workspace directory name so both the
    predecessor tree and the migrated study package route to the same place -- the
    study is one study regardless of which path is executing it.
    """
    return repository_root() / "runs" / RUNS_SLUG


def output_path(value: str | Path, *, root: Path | None = None) -> Path:
    """Confine writes to this workspace or to the study's ignored runs/ directory.

    The workspace is the default so committed development output keeps resolving.
    <repo>/runs/<slug>/ is admitted as a second root because the storage contract
    puts transient execution there, and until this existed the contract was
    unreachable: an absolute repo-level runs/ path raised, and a relative "runs/x"
    silently landed *inside* the workspace instead. Documentation promising that
    routing was therefore describing behaviour the code rejected.

    Still fail-closed -- exactly two roots, and anything else raises. A caller
    passing `root=` narrows to that root alone, since an explicit root is a
    containment request and widening it would defeat the point.
    """
    raw = Path(value)
    if root is not None:
        base = Path(root).resolve()
        candidate = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
        if not _inside(base, candidate):
            raise ContractError(f"output escapes the requested root: {value}")
        return candidate

    base = project_root().resolve()
    candidate = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
    if _inside(base, candidate):
        return candidate
    try:
        runs = runs_root().resolve()
    except ContractError:
        runs = None
    if runs is not None and _inside(runs, candidate):
        return candidate
    allowed = f"{base}" + (f" or {runs}" if runs is not None else "")
    raise ContractError(
        f"output escapes Paper C v2 workspace: {value} (permitted roots: {allowed})"
    )


def read_json(path: str | Path, *, root: Path | None = None) -> dict:
    candidate = output_path(path, root=root)
    with candidate.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {candidate}")
    return value


def write_json(path: str | Path, value: object, *, root: Path | None = None) -> None:
    target = output_path(path, root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def _relative_workspace_path(value: object, field: str) -> str:
    text = str(value or "")
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{field} must be workspace-relative")
    output_path(path)
    return text


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer")
    return value


def _finite_positive(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise ContractError(f"{field} must be finite and positive")
    return number


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ContractError(f"{field} must be finite")
    return number


def _strict_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{field} must be boolean")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a nonempty string")
    return value


def _iso_date(value: object, field: str) -> date:
    text = _nonempty_string(value, field)
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ContractError(f"{field} must be an ISO date") from exc


def validate_taxonomy(taxonomy: Mapping) -> None:
    if taxonomy.get("schema_version") != 1:
        raise ContractError("taxonomy schema_version must be 1")
    if tuple(taxonomy.get("actions") or ()) != ACTIONS:
        raise ContractError(f"taxonomy actions must be {ACTIONS}")
    core = taxonomy.get("core_categories")
    heldout = taxonomy.get("heldout_transfer_categories")
    if not isinstance(core, Mapping) or not isinstance(heldout, Mapping):
        raise ContractError("taxonomy needs core and held-out category maps")
    if set(core) & set(heldout):
        raise ContractError("core and held-out categories overlap")
    for name, record in {**core, **heldout}.items():
        if not str(name).strip() or not isinstance(record, Mapping):
            raise ContractError("invalid taxonomy category")
        if record.get("domain") not in {"general_safety", "mortgage"}:
            raise ContractError(f"invalid domain for {name}")
    if sum(record.get("domain") == "mortgage" for record in core.values()) < 1:
        raise ContractError("primary taxonomy must include mortgage")
    if sum(record.get("domain") == "general_safety" for record in core.values()) < 1:
        raise ContractError("primary taxonomy must include general safety")


def validate_policy_snapshot(policy: Mapping, taxonomy: Mapping) -> None:
    if policy.get("schema_version") != 1:
        raise ContractError("mortgage policy schema_version must be 1")
    if policy.get("jurisdiction") != "US_federal":
        raise ContractError("mortgage policy must be scoped to US federal law")
    if policy.get("claim_boundary") != "risk_triage_not_legal_compliance_or_credit_decision":
        raise ContractError("mortgage claim boundary is missing")
    _nonempty_string(policy.get("snapshot_id"), "mortgage_policy.snapshot_id")
    _iso_date(policy.get("retrieved_on"), "mortgage_policy.retrieved_on")
    if policy.get("legal_review_status") not in {
        "candidate_not_sme_signed", "sme_signed"
    }:
        raise ContractError("unknown mortgage legal-review status")
    authorities = policy.get("authorities")
    if not isinstance(authorities, list) or not authorities:
        raise ContractError("policy snapshot has no authorities")
    valid_categories = set(taxonomy["core_categories"]) | set(
        taxonomy["heldout_transfer_categories"]
    )
    ids: set[str] = set()
    categories: set[str] = set()
    for record in authorities:
        if not isinstance(record, Mapping):
            raise ContractError("invalid policy authority")
        authority_id = str(record.get("id", ""))
        if not authority_id or authority_id in ids:
            raise ContractError("policy authority IDs must be unique and nonempty")
        ids.add(authority_id)
        category = str(record.get("category", ""))
        if category not in valid_categories:
            raise ContractError(f"policy authority has unknown category: {category}")
        categories.add(category)
        if not str(record.get("url", "")).startswith("https://"):
            raise ContractError(f"policy authority must use HTTPS: {authority_id}")
        effective_from = _iso_date(
            record.get("effective_as_of"),
            f"mortgage_policy.authorities.{authority_id}.effective_as_of",
        )
        if record.get("effective_through") is not None:
            effective_through = _iso_date(
                record.get("effective_through"),
                f"mortgage_policy.authorities.{authority_id}.effective_through",
            )
            if effective_through < effective_from:
                raise ContractError(f"policy authority has an inverted vintage: {authority_id}")
        _nonempty_string(record.get("title"), f"authority {authority_id} title")
        _nonempty_string(record.get("scope"), f"authority {authority_id} scope")
    mortgage_categories = {
        name for name, record in taxonomy["core_categories"].items()
        if record["domain"] == "mortgage"
    } | {
        name for name, record in taxonomy["heldout_transfer_categories"].items()
        if record["domain"] == "mortgage"
    }
    if not mortgage_categories <= categories:
        raise ContractError("each claimed mortgage category needs an authority")
    conflicts = policy.get("temporal_conflicts")
    if not isinstance(conflicts, list) or not conflicts:
        raise ContractError("policy snapshot must record known temporal conflicts")
    conflict_ids: set[str] = set()
    for conflict in conflicts:
        if not isinstance(conflict, Mapping):
            raise ContractError("invalid temporal conflict")
        conflict_id = _nonempty_string(conflict.get("id"), "temporal conflict ID")
        if conflict_id in conflict_ids:
            raise ContractError("temporal conflict IDs must be unique")
        conflict_ids.add(conflict_id)
        _nonempty_string(conflict.get("description"), f"temporal conflict {conflict_id}")
        if conflict.get("required_action") != "review":
            raise ContractError("policy conflict must route to semantic REVIEW")


def validate_general_safety_policy(policy: Mapping, taxonomy: Mapping) -> None:
    if policy.get("schema_version") != 1:
        raise ContractError("general-safety policy schema_version must be 1")
    if policy.get("review_status") not in {"candidate_not_human_signed", "human_signed"}:
        raise ContractError("invalid general-safety policy review status")
    _nonempty_string(policy.get("snapshot_id"), "general_safety_policy.snapshot_id")
    if set(policy.get("actions") or {}) != set(ACTIONS):
        raise ContractError("general-safety policy must define all three actions")
    if "exactly one" not in str(policy.get("focal_category_rule", "")).lower():
        raise ContractError("general-safety policy must freeze a focal-category rule")
    expected = {
        name for name, record in taxonomy["core_categories"].items()
        if record["domain"] == "general_safety"
    }
    records = policy.get("categories")
    if not isinstance(records, Mapping) or set(records) != expected:
        raise ContractError("general-safety policy categories differ from taxonomy")
    for category, record in records.items():
        if not isinstance(record, Mapping):
            raise ContractError(f"invalid general-safety rule: {category}")
        if not str(record.get("intervene_if", "")).strip():
            raise ContractError(f"missing intervention rule: {category}")
        if not record.get("allow_near_neighbors") or not str(record.get("review_if", "")).strip():
            raise ContractError(f"missing boundary rules: {category}")


def validate_authority_archive_manifest(
    archive: Mapping,
    policy_snapshot: Mapping,
) -> None:
    """Validate archived authority bytes and reviewer-authorized excerpt bindings."""
    required = {
        "schema_version",
        "archive_id",
        "snapshot_id",
        "snapshot_object_sha256",
        "authority_sources",
        "authorized_excerpts",
    }
    if set(archive) != required:
        raise ContractError("authority archive manifest fields are not exact")
    if archive.get("schema_version") != 1:
        raise ContractError("authority archive manifest schema_version must be 1")
    _nonempty_string(archive.get("archive_id"), "authority_archive.archive_id")
    if archive.get("snapshot_id") != policy_snapshot.get("snapshot_id"):
        raise ContractError("authority archive targets the wrong policy snapshot")
    if archive.get("snapshot_object_sha256") != canonical_sha256(policy_snapshot):
        raise ContractError("authority archive policy-snapshot hash drifted")

    policy_authority_ids = {
        record["id"] for record in policy_snapshot.get("authorities", [])
    }
    sources = archive.get("authority_sources")
    if not isinstance(sources, list) or not sources:
        raise ContractError("authority archive has no source-byte records")
    source_ids: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping) or set(source) != {
            "authority_id", "archived_path", "archived_sha256"
        }:
            raise ContractError("authority archive source fields are not exact")
        authority_id = _nonempty_string(
            source.get("authority_id"),
            f"authority_archive.authority_sources[{index}].authority_id",
        )
        if authority_id in source_ids:
            raise ContractError("authority archive source IDs must be unique")
        if authority_id not in policy_authority_ids:
            raise ContractError("authority archive contains an unknown authority ID")
        source_ids.add(authority_id)
        archived_path = _relative_workspace_path(
            source.get("archived_path"),
            f"authority_archive.authority_sources[{index}].archived_path",
        )
        expected_hash = source.get("archived_sha256")
        if not output_path(archived_path).is_file():
            raise ContractError("archived authority bytes are missing")
        if (
            not HEX64.fullmatch(str(expected_hash))
            or sha256_file(output_path(archived_path)) != expected_hash
        ):
            raise ContractError("archived authority byte hash drifted")

    excerpts = archive.get("authorized_excerpts")
    if not isinstance(excerpts, list) or not excerpts:
        raise ContractError("authority archive has no authorized excerpt records")
    excerpt_hashes: set[str] = set()
    for index, excerpt in enumerate(excerpts):
        if not isinstance(excerpt, Mapping) or set(excerpt) != {
            "content_sha256", "authority_ids"
        }:
            raise ContractError("authorized excerpt fields are not exact")
        content_hash = excerpt.get("content_sha256")
        if not HEX64.fullmatch(str(content_hash)):
            raise ContractError("authorized excerpt has an invalid content hash")
        if content_hash in excerpt_hashes:
            raise ContractError("authorized excerpt hashes must be unique")
        excerpt_hashes.add(str(content_hash))
        authority_ids = excerpt.get("authority_ids")
        if (
            not isinstance(authority_ids, list)
            or not authority_ids
            or any(not isinstance(item, str) or not item for item in authority_ids)
            or len(set(authority_ids)) != len(authority_ids)
            or not set(authority_ids) <= source_ids
        ):
            raise ContractError(
                f"authorized excerpt {index} is not bound to archived authority sources"
            )


def validate_authority_excerpt_binding(
    archive: Mapping,
    *,
    content_sha256: str,
    authority_ids: Iterable[str],
    gold_policy_ids: Iterable[str],
) -> None:
    """Require the exact excerpt record to cover every cited and gold authority."""
    matches = [
        record
        for record in archive.get("authorized_excerpts", [])
        if record.get("content_sha256") == content_sha256
    ]
    if len(matches) != 1:
        raise ContractError("policy excerpt hash is absent from the authority archive")
    covered_ids = set(matches[0].get("authority_ids", []))
    required_ids = set(authority_ids) | set(gold_policy_ids)
    if not required_ids <= covered_ids:
        raise ContractError(
            "exact policy excerpt record does not cover every cited authority"
        )


def load_taxonomy(path: str | Path = "config/taxonomy.json") -> dict:
    taxonomy = read_json(path)
    validate_taxonomy(taxonomy)
    return taxonomy


def load_policy_snapshot(
    path: str | Path = "config/mortgage_policy_snapshot.json",
    *, taxonomy: Mapping | None = None,
) -> dict:
    policy = read_json(path)
    validate_policy_snapshot(policy, taxonomy or load_taxonomy())
    return policy


def load_general_safety_policy(
    path: str | Path = "config/general_safety_policy.json",
    *,
    taxonomy: Mapping | None = None,
) -> dict:
    policy = read_json(path)
    validate_general_safety_policy(policy, taxonomy or load_taxonomy())
    return policy


def validate_policy_vintage_inventory(
    inventory: Mapping,
    taxonomy: Mapping,
) -> None:
    if inventory.get("schema_version") != 1:
        raise ContractError("policy-vintage inventory schema_version must be 1")
    _nonempty_string(inventory.get("inventory_id"), "policy_vintage.inventory_id")
    if inventory.get("jurisdiction") != "US_federal":
        raise ContractError("policy-vintage inventory must be US federal")
    cutoff = _iso_date(
        inventory.get("temporal_policy_cutoff"),
        "policy_vintage.temporal_policy_cutoff",
    )
    status = inventory.get("review_status")
    if status not in {"candidate_not_sme_signed", "sme_signed"}:
        raise ContractError("invalid policy-vintage inventory review status")
    complete = _strict_bool(inventory.get("complete"), "policy_vintage.complete")
    coverage = inventory.get("temporal_side_coverage")
    if not isinstance(coverage, Mapping) or set(coverage) != {
        "pre_cutoff", "post_cutoff"
    }:
        raise ContractError("policy-vintage inventory must declare both temporal sides")
    allowed_coverage = {"missing", "candidate_not_sme_signed", "sme_signed"}
    if any(value not in allowed_coverage for value in coverage.values()):
        raise ContractError("invalid policy-vintage side coverage")
    vintages = inventory.get("vintages")
    if not isinstance(vintages, list) or not vintages:
        raise ContractError("policy-vintage inventory has no vintages")
    snapshot_ids: set[str] = set()
    lock_ids: set[str] = set()
    represented_sides: set[str] = set()
    for index, record in enumerate(vintages):
        if not isinstance(record, Mapping):
            raise ContractError(f"invalid policy vintage: {index}")
        snapshot_id = _nonempty_string(
            record.get("snapshot_id"), f"policy_vintage.vintages[{index}].snapshot_id"
        )
        lock_id = _nonempty_string(
            record.get("policy_vintage_lock_id"),
            f"policy_vintage.vintages[{index}].policy_vintage_lock_id",
        )
        if snapshot_id in snapshot_ids or lock_id in lock_ids:
            raise ContractError("policy-vintage snapshot and lock IDs must be unique")
        snapshot_ids.add(snapshot_id)
        lock_ids.add(lock_id)
        snapshot_path = _relative_workspace_path(
            record.get("snapshot_path"),
            f"policy_vintage.vintages[{index}].snapshot_path",
        )
        snapshot = read_json(snapshot_path)
        validate_policy_snapshot(snapshot, taxonomy)
        if snapshot.get("snapshot_id") != snapshot_id:
            raise ContractError("policy-vintage entry and snapshot IDs disagree")
        if record.get("snapshot_object_sha256") != canonical_sha256(snapshot):
            raise ContractError("policy-vintage snapshot object hash drifted")
        valid_from = _iso_date(
            record.get("valid_from"), f"policy_vintage.vintages[{index}].valid_from"
        )
        valid_through_raw = record.get("valid_through")
        valid_through = (
            _iso_date(
                valid_through_raw,
                f"policy_vintage.vintages[{index}].valid_through",
            )
            if valid_through_raw is not None
            else None
        )
        if valid_through is not None and valid_through < valid_from:
            raise ContractError("policy-vintage validity interval is inverted")
        side = record.get("temporal_side")
        if side not in {"pre_cutoff", "post_cutoff"}:
            raise ContractError("invalid policy-vintage temporal side")
        if side == "pre_cutoff" and (valid_through is None or valid_through > cutoff):
            raise ContractError("pre-cutoff policy vintage crosses the cutoff")
        if side == "post_cutoff" and valid_from <= cutoff:
            raise ContractError("post-cutoff policy vintage begins before the cutoff")
        represented_sides.add(side)
        record_status = record.get("review_status")
        if record_status not in {"candidate_not_sme_signed", "sme_signed"}:
            raise ContractError("invalid policy-vintage record review status")
        if (
            record_status == "sme_signed"
            and snapshot.get("legal_review_status") != "sme_signed"
        ):
            raise ContractError(
                "SME-signed policy vintage requires an SME-signed snapshot"
            )
        archive_path = record.get("authority_archive_manifest_path")
        archive_hash = record.get("authority_archive_manifest_sha256")
        if (archive_path is None) != (archive_hash is None):
            raise ContractError("policy-vintage archive path and hash must appear together")
        if archive_path is not None:
            relative_archive = _relative_workspace_path(
                archive_path,
                f"policy_vintage.vintages[{index}].authority_archive_manifest_path",
            )
            archive_file = output_path(relative_archive)
            if not archive_file.is_file():
                raise ContractError("policy-vintage authority archive is missing")
            if (
                not HEX64.fullmatch(str(archive_hash))
                or sha256_file(archive_file) != archive_hash
            ):
                raise ContractError("policy-vintage authority archive hash drifted")
            validate_authority_archive_manifest(
                read_json(relative_archive),
                snapshot,
            )
        if record_status == "sme_signed" and archive_path is None:
            raise ContractError("SME-signed policy vintage requires an authority archive")
    for side in represented_sides:
        if coverage[side] == "missing":
            raise ContractError("policy-vintage coverage contradicts its entries")
    if complete:
        if status != "sme_signed" or set(coverage.values()) != {"sme_signed"}:
            raise ContractError("complete policy-vintage inventory must be SME signed")
        if represented_sides != {"pre_cutoff", "post_cutoff"}:
            raise ContractError("complete policy-vintage inventory needs both temporal sides")


def load_policy_vintage_inventory(
    path: str | Path = "config/mortgage_policy_vintage_inventory.json",
    *,
    taxonomy: Mapping | None = None,
) -> dict:
    inventory = read_json(path)
    validate_policy_vintage_inventory(inventory, taxonomy or load_taxonomy())
    return inventory


def require_frozen_allow_minimum(config: Mapping, *, stream: str) -> int:
    """Return the pilot-frozen ALLOW minimum, or refuse.

    The pre-pilot value is ``null``.  Any step that builds, validates, or scores a
    *primary* specificity cohort must call this first, so that an unfrozen cohort
    size can never be silently treated as "no minimum".
    """
    if stream not in {"calibration", "sealed"}:
        raise ContractError("allow-minimum stream must be calibration or sealed")
    validate_config(config)
    value = config["data"].get(f"{stream}_allow_min_per_core_category")
    if value is None:
        raise ContractError(
            f"{stream} ALLOW minimum is not frozen; the disjoint pilot must derive it "
            "from the FPR precision target before any primary cohort is built"
        )
    return _positive_int(value, f"data.{stream}_allow_min_per_core_category")


def validate_config(config: Mapping) -> None:
    if config.get("schema_version") != 2:
        raise ContractError("config schema_version must be 2")
    profile = config.get("profile")
    if profile not in {"primary", "smoke"}:
        raise ContractError("profile must be primary or smoke")
    _nonempty_string(config.get("study_id"), "study_id")
    marker = _relative_workspace_path(config.get("project_root_marker"), "project_root_marker")
    if not output_path(marker).is_file():
        raise ContractError("project root marker does not exist")
    for field in ("artifact_root", "input_root"):
        _relative_workspace_path(config.get(field), field)

    taxonomy_path = _relative_workspace_path(config.get("taxonomy_path"), "taxonomy_path")
    general_policy_path = _relative_workspace_path(
        config.get("general_safety_policy_path"), "general_safety_policy_path"
    )
    mortgage_policy_path = _relative_workspace_path(
        config.get("mortgage_policy_path"), "mortgage_policy_path"
    )
    vintage_inventory_path = _relative_workspace_path(
        config.get("mortgage_policy_vintage_inventory_path"),
        "mortgage_policy_vintage_inventory_path",
    )
    taxonomy = read_json(taxonomy_path)
    validate_taxonomy(taxonomy)
    validate_general_safety_policy(read_json(general_policy_path), taxonomy)
    validate_policy_snapshot(read_json(mortgage_policy_path), taxonomy)
    vintage_inventory = read_json(vintage_inventory_path)
    validate_policy_vintage_inventory(vintage_inventory, taxonomy)
    if tuple(config.get("actions") or ()) != ACTIONS:
        raise ContractError(f"config actions must be {ACTIONS}")

    task = config.get("task")
    if not isinstance(task, Mapping):
        raise ContractError("task contract is required")
    expected_unit = (
        "request", "proposed_response", "context", "jurisdiction",
        "policy_as_of", "policy_context",
    )
    expected_output = (
        "action", "category", "violation_tags", "policy_ids", "confidence",
    )
    if tuple(task.get("unit") or ()) != expected_unit:
        raise ContractError("task unit must include immutable policy context")
    if tuple(task.get("output") or ()) != expected_output:
        raise ContractError("task structured output contract drifted")
    if task.get("category_contract") != "one_adjudicated_focal_category_per_primary_event":
        raise ContractError("primary events require one adjudicated focal category")
    required_context = {
        "actor_role", "product", "transaction_stage", "applicable_regime",
        "coverage_facts",
    }
    if set(task.get("mortgage_required_context_fields") or ()) != required_context:
        raise ContractError("mortgage decisive-context fields drifted")
    if set(task.get("mortgage_exclusions") or ()) != {
        "credit_decision", "legal_opinion", "compliance_certification"
    }:
        raise ContractError("mortgage claim exclusions drifted")
    if profile == "primary" and task.get("mortgage_claim") != (
        "US_federal_mortgage_compliance_risk_triage"
    ):
        raise ContractError("primary mortgage claim boundary drifted")
    if profile == "smoke" and task.get("mortgage_claim") != "infrastructure_only":
        raise ContractError("smoke profile cannot support a mortgage claim")

    backbones = config.get("backbones")
    if not isinstance(backbones, Mapping) or len(backbones) < 2:
        raise ContractError("cross-model alignment requires at least two backbones")
    if profile == "primary" and len(backbones) != 2:
        raise ContractError("primary design freezes exactly two backbones")
    model_cells: set[tuple[str, str]] = set()
    model_ids: set[str] = set()
    for key, record in backbones.items():
        _nonempty_string(key, "backbone alias")
        if not isinstance(record, Mapping):
            raise ContractError(f"invalid backbone record: {key}")
        model_id = _nonempty_string(record.get("model_id"), f"backbones.{key}.model_id")
        revision = str(record.get("revision", ""))
        if not HEX40.fullmatch(revision):
            raise ContractError(f"backbone revision is not a pinned commit: {key}")
        if model_id in model_ids or (model_id, revision) in model_cells:
            raise ContractError("cross-model backbones must have distinct model identities")
        model_ids.add(model_id)
        model_cells.add((model_id, revision))

    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ContractError("seeds must be a nonempty unique list")
    for index, seed in enumerate(seeds):
        _positive_int(seed, f"seeds[{index}]")
    if profile == "primary" and tuple(seeds) != PRIMARY_SEEDS:
        raise ContractError(f"primary seeds must be {PRIMARY_SEEDS}")

    categories = config.get("core_categories")
    if (
        not isinstance(categories, list)
        or not categories
        or any(not isinstance(item, str) or not item for item in categories)
        or len(set(categories)) != len(categories)
    ):
        raise ContractError("core_categories must be a nonempty unique string list")
    taxonomy_core = set(taxonomy["core_categories"])
    if not set(categories) <= taxonomy_core:
        raise ContractError("config contains a category absent from taxonomy")
    if profile == "primary" and set(categories) != taxonomy_core:
        raise ContractError("primary config must cross every core taxonomy category")
    category_domains = {taxonomy["core_categories"][name]["domain"] for name in categories}
    if category_domains != {"general_safety", "mortgage"}:
        raise ContractError("every profile must exercise general-safety and mortgage categories")

    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ContractError("data configuration is required")
    _positive_int(data.get("family_split_seed"), "data.family_split_seed")
    split = data.get("family_split")
    if not isinstance(split, Mapping) or tuple(split) != FAMILY_SPLITS:
        raise ContractError("family split order and roles are frozen")
    split_values = [
        _finite_positive(value, f"data.family_split.{name}")
        for name, value in split.items()
    ]
    if not math.isclose(sum(split_values), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("family split fractions must sum to one")
    cutoff = _iso_date(data.get("temporal_policy_cutoff"), "data.temporal_policy_cutoff")
    if _iso_date(
        vintage_inventory.get("temporal_policy_cutoff"),
        "policy_vintage.temporal_policy_cutoff",
    ) != cutoff:
        raise ContractError("data and policy-vintage cutoffs disagree")
    if data.get("temporal_evaluation_eligibility_field") != (
        "temporal_evaluation_eligible"
    ):
        raise ContractError("temporal evaluation eligibility field drifted")
    if profile == "primary" and cutoff != date(2026, 7, 20):
        raise ContractError("primary temporal cutoff must precede the 2026-07-21 rule change")
    _positive_int(data.get("mortgage_target_families"), "data.mortgage_target_families")
    _positive_int(data.get("rows_per_mortgage_family"), "data.rows_per_mortgage_family")
    # The specificity-cohort size is a measured quantity, not an assertion: it is
    # frozen by the disjoint pilot from the FPR precision target, exactly as the
    # mortgage family count already is.  `null` is the honest pre-pilot state and
    # fails closed downstream -- `require_frozen_allow_minimum` refuses to build or
    # score a primary cohort until the pilot has supplied a number.
    for field in (
        "calibration_allow_min_per_core_category",
        "sealed_allow_min_per_core_category",
    ):
        value = data.get(field)
        if value is None:
            continue
        _positive_int(value, f"data.{field}")
    rule = data.get("allow_minimum_rule")
    if not isinstance(rule, str) or not rule.strip():
        raise ContractError("data.allow_minimum_rule must be a nonempty string")
    if profile == "primary" and rule != (
        "frozen_by_disjoint_pilot_from_fpr_precision_target_not_asserted"
    ):
        raise ContractError("primary allow-minimum rule must defer to the pilot")
    halfwidth = data.get("allow_minimum_target_fpr_halfwidth")
    if not isinstance(halfwidth, Real) or isinstance(halfwidth, bool):
        raise ContractError("data.allow_minimum_target_fpr_halfwidth must be numeric")
    if not 0 < float(halfwidth) < 1:
        raise ContractError("allow-minimum FPR half-width must lie in (0,1)")
    mixture = data.get("capacity_evaluation_mixture")
    if not isinstance(mixture, Mapping) or tuple(mixture) != ACTIONS:
        raise ContractError(
            "capacity-evaluation mixture must define the three actions in order"
        )
    mixture_values = [
        _finite_number(value, f"data.capacity_evaluation_mixture.{action}")
        for action, value in mixture.items()
    ]
    if any(value < 0 or value > 1 for value in mixture_values) or not math.isclose(
        sum(mixture_values), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ContractError(
            "capacity-evaluation mixture must be probabilities summing to one"
        )
    if _positive_int(data.get("mortgage_reviewers"), "data.mortgage_reviewers") < 2:
        raise ContractError("mortgage gold requires two reviewers")
    if _strict_bool(data.get("separate_adjudicator"), "data.separate_adjudicator") is not True:
        raise ContractError("mortgage disagreements require a separate adjudicator")
    sealed_required = _strict_bool(
        data.get("sealed_external_test_required"), "data.sealed_external_test_required"
    )
    if profile == "primary" and not sealed_required:
        raise ContractError("primary study requires an external sealed test")
    if data.get("legacy_mortgage_role") != "candidate_generation_only_not_gold":
        raise ContractError("legacy mortgage labels cannot be treated as gold")

    specialists = config.get("specialists")
    if not isinstance(specialists, Mapping):
        raise ContractError("specialist configuration is required")
    if specialists.get("grid") != "all_core_categories_x_all_backbones_x_all_seeds":
        raise ContractError("category/backbone crossover is mandatory")
    if specialists.get("out_of_expertise_behavior") != "abstain":
        raise ContractError("out-of-expertise behavior must be routing-layer abstention")
    lora = specialists.get("lora")
    if not isinstance(lora, Mapping):
        raise ContractError("specialist LoRA configuration is required")
    _positive_int(lora.get("r"), "specialists.lora.r")
    _positive_int(lora.get("alpha"), "specialists.lora.alpha")
    dropout = _finite_number(lora.get("dropout"), "specialists.lora.dropout")
    if not 0 <= dropout < 1:
        raise ContractError("specialists.lora.dropout must lie in [0,1)")
    _finite_positive(specialists.get("learning_rate"), "specialists.learning_rate")
    _positive_int(specialists.get("max_steps"), "specialists.max_steps")
    if specialists.get("calibration") != "temperature_scaling":
        raise ContractError("specialist confidence must be externally calibrated")

    preferences = config.get("preferences")
    if not isinstance(preferences, Mapping):
        raise ContractError("preference configuration is required")
    if preferences.get("teacher_mode") != "leave_one_backbone_out":
        raise ContractError("self-teaching is forbidden")
    minimum_seeds = _positive_int(
        preferences.get("minimum_distinct_teacher_seeds"),
        "preferences.minimum_distinct_teacher_seeds",
    )
    if minimum_seeds > len(seeds):
        raise ContractError("minimum teacher seeds exceeds available primary seeds")
    confidence = _finite_number(
        preferences.get("minimum_calibrated_confidence"),
        "preferences.minimum_calibrated_confidence",
    )
    if not 0 <= confidence <= 1:
        raise ContractError("teacher confidence threshold must lie in [0,1]")
    for field in (
        "require_adjudicated_gold_consistency",
        "require_complete_structured_candidates",
        "retain_agreement_and_disagreement_strata",
        "review_blinded_to_model_identity_and_order",
    ):
        if _strict_bool(preferences.get(field), f"preferences.{field}") is not True:
            raise ContractError(f"preferences.{field} must be true")
    if tuple(preferences.get("candidate_sources") or ()) != (
        "category_specialist", "joint_generalist"
    ):
        raise ContractError("matched specialist and generalist candidates are required")
    if preferences.get("candidate_calibration") != (
        "categorywise_temperature_scaling_for_both_source_types_on_calibration_split"
    ):
        raise ContractError("both candidate sources require locked category-wise calibration")
    _finite_positive(preferences.get("beta"), "preferences.beta")

    alignment = config.get("alignment")
    if not isinstance(alignment, Mapping):
        raise ContractError("alignment configuration is required")
    if tuple(alignment.get("arms") or ()) != ALIGNMENT_ARMS:
        raise ContractError(f"alignment arms must be {ALIGNMENT_ARMS}")
    if alignment.get("reference") != "joint_multitask_sft":
        raise ContractError("all alignment arms require the same joint SFT reference")
    _finite_positive(alignment.get("learning_rate"), "alignment.learning_rate")
    max_steps = _positive_int(alignment.get("max_steps"), "alignment.max_steps")
    checkpoints = alignment.get("checkpoint_steps")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ContractError("checkpoint ladder is required")
    for index, checkpoint in enumerate(checkpoints):
        _positive_int(checkpoint, f"alignment.checkpoint_steps[{index}]")
        if checkpoint > max_steps:
            raise ContractError("checkpoint step exceeds alignment.max_steps")
    if checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != max_steps:
        raise ContractError("checkpoint ladder must be unique, ordered, and end at max_steps")
    if alignment.get("soft_distillation_target") != (
        "three_calibrated_action_probabilities_only"
    ):
        raise ContractError("soft distillation target must be the three action probabilities")
    if alignment.get("pair_logprob_reduction") != (
        "sum_response_token_logprob_with_prompt_masked"
    ):
        raise ContractError("pair log-probability reduction drifted")
    if alignment.get("candidate_length_rule") != (
        "reject_pair_if_either_candidate_is_truncated"
    ):
        raise ContractError("candidate truncation rule drifted")
    if _strict_bool(
        alignment.get("same_examples_tokens_optimizer"),
        "alignment.same_examples_tokens_optimizer",
    ) is not True:
        raise ContractError("matched alignment inputs are mandatory")
    _finite_positive(
        alignment.get("category_dro_temperature"), "alignment.category_dro_temperature"
    )
    for field in ("lambda_gold", "lambda_retain"):
        value = _finite_number(alignment.get(field), f"alignment.{field}")
        if value < 0:
            raise ContractError(f"alignment.{field} must be nonnegative")

    if profile == "primary":
        pilot = config.get("pilot")
        if not isinstance(pilot, Mapping):
            raise ContractError("primary design requires a pilot contract")
        pilot_backbones = pilot.get("backbones")
        pilot_seeds = pilot.get("seeds")
        if not isinstance(pilot_backbones, list) or pilot_backbones != list(backbones):
            raise ContractError("pilot must exercise both pinned backbones in frozen order")
        if (
            not isinstance(pilot_seeds, list)
            or not pilot_seeds
            or len(set(pilot_seeds)) != len(pilot_seeds)
        ):
            raise ContractError("pilot seeds must be a nonempty unique list")
        for index, seed in enumerate(pilot_seeds):
            _positive_int(seed, f"pilot.seeds[{index}]")
        if set(pilot_seeds) & set(seeds):
            raise ContractError("pilot and primary seeds must be disjoint")
        if len(pilot_seeds) < minimum_seeds:
            raise ContractError("pilot needs enough seeds for teacher stability filtering")
        if pilot.get("categories") != "all_core_categories":
            raise ContractError("pilot must exercise all core categories")
        if _strict_bool(pilot.get("reuse_in_primary_panel"), "pilot.reuse_in_primary_panel"):
            raise ContractError("outcome-inspected pilot cells cannot enter confirmation")
        _nonempty_string(pilot.get("family_namespace"), "pilot.family_namespace")
        _nonempty_string(pilot.get("purpose"), "pilot.purpose")
        expected_pilot = (
            len(pilot_backbones) * len(pilot_seeds)
            + len(categories) * len(pilot_backbones) * len(pilot_seeds)
            + len(ALIGNMENT_ARMS) * len(pilot_backbones) * len(pilot_seeds)
        )
        if pilot.get("expected_training_cells") != expected_pilot:
            raise ContractError("pilot training-cell count is inconsistent")

    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ContractError("evaluation configuration is required")
    fpr_grid = evaluation.get("benign_fpr_grid")
    if not isinstance(fpr_grid, list) or not fpr_grid:
        raise ContractError("benign FPR grid is required")
    fprs = [
        _finite_number(value, f"evaluation.benign_fpr_grid[{index}]")
        for index, value in enumerate(fpr_grid)
    ]
    if any(not 0 < value < 1 for value in fprs) or fprs != sorted(set(fprs)):
        raise ContractError("benign FPR grid must be unique, increasing, and inside (0,1)")
    budget = evaluation.get("review_budget")
    if not isinstance(budget, Mapping):
        raise ContractError("review budget is required")
    maximum = _finite_number(budget.get("maximum"), "evaluation.review_budget.maximum")
    if not 0 <= maximum < 1:
        raise ContractError("review budget must lie in [0,1)")
    if budget.get("scope") != "capacity_evaluation_mixture":
        raise ContractError(
            "review budget applies only to the capacity-evaluation mixture"
        )
    if evaluation.get("threshold_fit_split") != "calibration":
        raise ContractError("thresholds must be fit on calibration families only")
    if evaluation.get("checkpoint_selection_split") != "checkpoint_selection":
        raise ContractError("checkpoint selection requires its own family split")
    if _strict_bool(
        evaluation.get("balanced_triplet_review_budget_applies"),
        "evaluation.balanced_triplet_review_budget_applies",
    ):
        raise ContractError(
            "capacity-mixture review budgets cannot be applied to balanced triplets"
        )
    _strict_bool(evaluation.get("mortgage_disaggregated"), "evaluation.mortgage_disaggregated")
    heldout = evaluation.get("heldout_transfer_categories")
    expected_heldout = {
        name for name, record in taxonomy["heldout_transfer_categories"].items()
        if record["domain"] == "mortgage"
    }
    if not isinstance(heldout, list) or len(set(heldout)) != len(heldout):
        raise ContractError("held-out categories must be a unique list")
    if profile == "primary":
        if set(heldout) != expected_heldout:
            raise ContractError("primary held-out mortgage taxonomy drifted")
        if evaluation.get("decision_rule") != (
            "intervene_if_p_intervene_ge_ti_else_review_if_one_minus_p_allow_ge_tr_else_allow"
        ):
            raise ContractError("primary evaluation requires the frozen two-threshold rule")
        if evaluation.get("checkpoint_rule") != (
            "maximize_development_worst_category_frontier_subject_to_capacity_mixture_fpr_and_"
            "review_constraints_ties_earliest"
        ):
            raise ContractError("checkpoint selection rule drifted")
        if evaluation.get("sealed_scoring") != "selected_checkpoint_once":
            raise ContractError("sealed data may score each selected checkpoint once")
        if tuple(evaluation.get("primary") or ()) != (
            "specialist_vs_generalist_cm_dpo_worst_category_frontier",
        ):
            raise ContractError("there must be exactly one frozen primary endpoint")
        if evaluation.get("primary_contrast") != (
            "specialist_cm_dpo_minus_generalist_cm_dpo"
        ):
            raise ContractError("primary contrast must isolate specialist candidate generation")
        if tuple(evaluation.get("secondary_contrasts") or ()) != (
            "specialist_cm_dpo_minus_specialist_pairce",
            "specialist_pairce_minus_soft_distill",
            "specialist_cm_dpo_minus_gold_sft",
        ):
            raise ContractError("secondary contrast decomposition drifted")
        if evaluation.get("cluster_unit") != (
            "family_with_fixed_backbones_and_fixed_named_seeds"
        ):
            raise ContractError("primary inference must resample families only")
        if evaluation.get("inferential_scope") != (
            "conditional_on_two_pinned_backbones_and_three_named_seeds"
        ):
            raise ContractError("primary scope must condition on fixed backbones and seeds")
        if evaluation.get("multiplicity") != (
            "simultaneous_one_sided_intervals_for_primary_and_holm_for_secondary"
        ):
            raise ContractError("multiplicity procedure drifted")
        power = config.get("power")
        if not isinstance(power, Mapping):
            raise ContractError("primary design requires a power contract")
        for field in ("target_primary_frontier_effect", "worst_category_noninferiority_margin"):
            value = _finite_positive(power.get(field), f"power.{field}")
            if value >= 1:
                raise ContractError(f"power.{field} must be smaller than one")
        alpha = _finite_number(power.get("one_sided_alpha"), "power.one_sided_alpha")
        target_power = _finite_number(power.get("target_power"), "power.target_power")
        if not 0 < alpha < 0.5 or not 0.5 < target_power < 1:
            raise ContractError("invalid alpha or target power")
        _nonempty_string(power.get("sample_size_rule"), "power.sample_size_rule")
    else:
        if heldout:
            raise ContractError("smoke profile cannot perform held-out transfer")
        if evaluation.get("sealed_scoring") != "disabled":
            raise ContractError("smoke profile cannot score sealed evidence")

    readiness = config.get("readiness")
    evidence = config.get("readiness_evidence")
    if not isinstance(readiness, Mapping) or set(readiness) != set(READINESS_GATES):
        raise ContractError("readiness must contain the exact frozen gates")
    if not isinstance(evidence, Mapping) or set(evidence) != set(READINESS_GATES):
        raise ContractError("readiness evidence must match the exact frozen gates")
    for gate in READINESS_GATES:
        ready = _strict_bool(readiness[gate], f"readiness.{gate}")
        record = evidence[gate]
        if not ready:
            if record is not None:
                raise ContractError(f"false readiness gate {gate} must have null evidence")
            continue
        if not isinstance(record, Mapping):
            raise ContractError(f"true readiness gate {gate} requires evidence")
        if record.get("gate") != gate:
            raise ContractError(f"readiness evidence names the wrong gate: {gate}")
        evidence_path = _relative_workspace_path(
            record.get("artifact_path"),
            f"readiness_evidence.{gate}.artifact_path",
        )
        if not output_path(evidence_path).is_file():
            raise ContractError(f"readiness evidence file is missing: {gate}")
        if not HEX64.fullmatch(str(record.get("artifact_sha256", ""))):
            raise ContractError(f"readiness evidence hash is invalid: {gate}")
        if sha256_file(evidence_path) != record["artifact_sha256"]:
            raise ContractError(f"readiness evidence hash drifted: {gate}")
        _nonempty_string(record.get("lock_id"), f"readiness_evidence.{gate}.lock_id")
        issued = _nonempty_string(
            record.get("issued_utc"), f"readiness_evidence.{gate}.issued_utc"
        )
        try:
            issued_at = datetime.fromisoformat(issued.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractError(f"readiness evidence has invalid issuance time: {gate}") from exc
        if issued_at.tzinfo is None:
            raise ContractError(f"readiness evidence issuance time lacks timezone: {gate}")
        approvers = record.get("approver_ids")
        minimum_approvers = 2 if gate == "mortgage_policy_sme_signed" else 1
        if (
            not isinstance(approvers, list)
            or len(approvers) < minimum_approvers
            or any(not isinstance(item, str) or not item.strip() for item in approvers)
            or len(set(approvers)) != len(approvers)
        ):
            raise ContractError(f"readiness evidence has invalid approvers: {gate}")


def load_config(path: str | Path = "config/study.json") -> dict:
    config = read_json(path)
    validate_config(config)
    return config


def expected_training_grid(config: Mapping) -> dict[str, list[dict]]:
    validate_config(config)
    backbones = list(config["backbones"])
    seeds = [int(seed) for seed in config["seeds"]]
    categories = list(config["core_categories"])
    arms = list(config["alignment"]["arms"])
    references = [
        {"backbone": backbone, "seed": seed}
        for backbone in backbones for seed in seeds
    ]
    specialists = [
        {"category": category, "backbone": backbone, "seed": seed}
        for category in categories for backbone in backbones for seed in seeds
    ]
    aligned = [
        {"arm": arm, "backbone": backbone, "seed": seed}
        for arm in arms for backbone in backbones for seed in seeds
    ]
    return {"references": references, "specialists": specialists, "aligned": aligned}


def readiness_blockers(config: Mapping) -> list[str]:
    validate_config(config)
    return sorted(key for key, value in config["readiness"].items() if not value)


def validate_task_event(
    event: Mapping,
    *,
    taxonomy: Mapping | None = None,
    policy_snapshot: Mapping | None = None,
    policy_vintage_inventory: Mapping | None = None,
    claim_bearing: bool = True,
) -> None:
    """Validate an event; mortgage validation is claim-bearing by default.

    ``claim_bearing=False`` is an explicit candidate-development escape hatch.
    Such rows cannot populate primary, calibration, checkpoint, or sealed evidence.
    """
    taxonomy = taxonomy or load_taxonomy()
    _strict_bool(claim_bearing, "claim_bearing")
    required = {
        "sample_id", "family_id", "content_family_id", "category", "domain", "request",
        "proposed_response", "context", "jurisdiction", "policy_as_of",
        "policy_context", "temporal_evaluation_eligible", "gold", "provenance",
    }
    missing = required - set(event)
    if missing:
        raise ContractError(f"task event missing fields: {sorted(missing)}")
    category = str(event["category"])
    records = {**taxonomy["core_categories"], **taxonomy["heldout_transfer_categories"]}
    if category not in records:
        raise ContractError(f"unknown category: {category}")
    domain = records[category]["domain"]
    if event["domain"] != domain:
        raise ContractError("event domain disagrees with taxonomy")
    _nonempty_string(event["sample_id"], "sample_id")
    _nonempty_string(event["family_id"], "family_id")
    _nonempty_string(event["content_family_id"], "content_family_id")
    _nonempty_string(event["request"], "request")
    temporal_eligible = _strict_bool(
        event["temporal_evaluation_eligible"], "temporal_evaluation_eligible"
    )
    gold = event["gold"]
    if not isinstance(gold, Mapping) or gold.get("action") not in ACTIONS:
        raise ContractError("invalid gold action")
    if "category" in gold and gold["category"] != category:
        raise ContractError("gold category disagrees with focal category")
    for field in ("violation_tags", "policy_ids"):
        values = gold.get(field)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)
        ):
            raise ContractError(f"gold {field} must be a unique string list")
    if not str(gold.get("rationale", "")).strip():
        raise ContractError("gold rationale is required")
    provenance = event["provenance"]
    if not isinstance(provenance, Mapping):
        raise ContractError("provenance is required")
    if provenance.get("contains_real_pii") is not False:
        raise ContractError("real PII is forbidden")
    _nonempty_string(provenance.get("source_id"), "provenance.source_id")
    _nonempty_string(provenance.get("licence_id"), "provenance.licence_id")
    _strict_bool(provenance.get("synthetic"), "provenance.synthetic")
    if not HEX64.fullmatch(str(provenance.get("content_sha256", ""))):
        raise ContractError("invalid content hash")
    if domain == "mortgage":
        if not isinstance(event.get("proposed_response"), str) or not event["proposed_response"].strip():
            raise ContractError("mortgage event requires a proposed response or action")
        context = event.get("context")
        if not isinstance(context, Mapping) or not context:
            raise ContractError("mortgage event requires decisive context")
        required_context = {
            "actor_role", "product", "transaction_stage", "applicable_regime",
            "coverage_facts",
        }
        missing_context = required_context - set(context)
        if missing_context:
            raise ContractError(
                f"mortgage event missing decisive context: {sorted(missing_context)}"
            )
        for field in required_context:
            value = context[field]
            if value is None or value == "" or value == [] or value == {}:
                raise ContractError(f"mortgage context is empty: {field}")
        if event.get("jurisdiction") != "US_federal":
            raise ContractError("mortgage event requires US-federal jurisdiction and policy date")
        policy_date = _iso_date(event.get("policy_as_of"), "policy_as_of")
        policy = policy_snapshot or load_policy_snapshot(taxonomy=taxonomy)
        validate_policy_snapshot(policy, taxonomy)
        vintage_inventory = policy_vintage_inventory or load_policy_vintage_inventory(
            taxonomy=taxonomy
        )
        validate_policy_vintage_inventory(vintage_inventory, taxonomy)
        policy_context = event.get("policy_context")
        if not isinstance(policy_context, Mapping):
            raise ContractError("mortgage event requires immutable policy context")
        policy_context_fields = {
            "snapshot_id", "snapshot_object_sha256", "policy_vintage_lock_id",
            "policy_as_of", "authority_ids", "policy_text", "content_sha256",
        }
        missing_policy_context = policy_context_fields - set(policy_context)
        if missing_policy_context:
            raise ContractError(
                f"policy context missing fields: {sorted(missing_policy_context)}"
            )
        extra_policy_context = set(policy_context) - policy_context_fields
        if extra_policy_context:
            raise ContractError(
                f"policy context has unknown fields: {sorted(extra_policy_context)}"
            )
        if policy_context.get("snapshot_id") != policy.get("snapshot_id"):
            raise ContractError("policy context snapshot ID is not the locked snapshot")
        if policy_context.get("snapshot_object_sha256") != canonical_sha256(policy):
            raise ContractError("policy context does not bind the snapshot object")
        vintage_lock_id = _nonempty_string(
            policy_context.get("policy_vintage_lock_id"),
            "policy_context.policy_vintage_lock_id",
        )
        if _iso_date(policy_context.get("policy_as_of"), "policy_context.policy_as_of") != policy_date:
            raise ContractError("event and policy-context dates disagree")
        vintage_records = [
            record for record in vintage_inventory["vintages"]
            if record["snapshot_id"] == policy["snapshot_id"]
            and record["snapshot_object_sha256"] == canonical_sha256(policy)
            and record["policy_vintage_lock_id"] == vintage_lock_id
        ]
        if len(vintage_records) != 1:
            raise ContractError("policy context is absent from the locked vintage inventory")
        vintage_record = vintage_records[0]
        valid_from = _iso_date(vintage_record["valid_from"], "policy vintage valid_from")
        valid_through = (
            _iso_date(vintage_record["valid_through"], "policy vintage valid_through")
            if vintage_record.get("valid_through") is not None
            else None
        )
        if policy_date < valid_from or (
            valid_through is not None and policy_date > valid_through
        ):
            raise ContractError("policy date lies outside the registered policy vintage")
        policy_text = _nonempty_string(
            policy_context.get("policy_text"), "policy_context.policy_text"
        )
        expected_text_hash = hashlib.sha256(policy_text.encode("utf-8")).hexdigest()
        if policy_context.get("content_sha256") != expected_text_hash:
            raise ContractError("policy-context text hash mismatch")
        authority_ids = policy_context.get("authority_ids")
        if (
            not isinstance(authority_ids, list)
            or not authority_ids
            or any(not isinstance(item, str) or not item for item in authority_ids)
            or len(set(authority_ids)) != len(authority_ids)
        ):
            raise ContractError("policy context requires unique authority IDs")
        authorities = {record["id"]: record for record in policy["authorities"]}
        if any(item not in authorities for item in authority_ids):
            raise ContractError("policy context contains an unknown authority")
        if any(authorities[item]["category"] != category for item in authority_ids):
            raise ContractError("policy-context authority disagrees with focal category")
        for authority_id in authority_ids:
            authority = authorities[authority_id]
            effective_from = _iso_date(
                authority.get("effective_as_of"),
                f"authority {authority_id} effective_as_of",
            )
            if effective_from > policy_date:
                raise ContractError("policy context cites an authority not yet effective")
            effective_through = authority.get("effective_through")
            if effective_through is not None and _iso_date(
                effective_through, f"authority {authority_id} effective_through"
            ) < policy_date:
                raise ContractError("policy context cites an expired authority vintage")
        if not set(gold["policy_ids"]) <= set(authority_ids):
            raise ContractError("gold policy IDs are absent from immutable policy context")
        if claim_bearing:
            if (
                vintage_inventory.get("complete") is not True
                or vintage_inventory.get("review_status") != "sme_signed"
                or vintage_record.get("review_status") != "sme_signed"
            ):
                raise ContractError("claim-bearing mortgage rows require a complete SME-signed vintage inventory")
            archive_path = vintage_record.get("authority_archive_manifest_path")
            archive_hash = vintage_record.get("authority_archive_manifest_sha256")
            if archive_path is None or archive_hash is None:
                raise ContractError("claim-bearing mortgage rows require archived authority bytes")
            archive = read_json(archive_path)
            if sha256_file(output_path(archive_path)) != archive_hash:
                raise ContractError("authority archive manifest hash drifted")
            validate_authority_archive_manifest(archive, policy)
            validate_authority_excerpt_binding(
                archive,
                content_sha256=policy_context["content_sha256"],
                authority_ids=authority_ids,
                gold_policy_ids=gold["policy_ids"],
            )
        reviewers = gold.get("reviewer_ids")
        if (
            not isinstance(reviewers, list)
            or any(not isinstance(item, str) or not item.strip() for item in reviewers)
            or len(set(reviewers)) < 2
        ):
            raise ContractError("mortgage gold requires two distinct reviewers")
        adjudicator = gold.get("adjudicator_id")
        if not isinstance(adjudicator, str) or not adjudicator.strip() or adjudicator in set(reviewers):
            raise ContractError("mortgage gold requires a separate adjudicator")
        if not gold.get("policy_ids"):
            raise ContractError("mortgage gold requires policy IDs")
    else:
        if temporal_eligible:
            raise ContractError("request-screening rows cannot enter mortgage policy-time tests")
        if event.get("proposed_response") is not None or event.get("context") is not None:
            raise ContractError("request-screening events require null response and context")
        if event.get("jurisdiction") is not None or event.get("policy_as_of") is not None:
            raise ContractError("general-safety events require null legal scope")
        if event.get("policy_context") is not None or gold["policy_ids"]:
            raise ContractError("general-safety events cannot cite mortgage policy context")
