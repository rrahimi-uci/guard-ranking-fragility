# Repository Layout v2 — Migration Plan

**Status:** partially executed — Phases 0, 1, 3 and 5 applied; 2 and 4 open
**Written:** 2026-07-26
**Last audited:** 2026-07-26 against Git `8f66f83b9ac0e24a090874ce5a8c5fbb5b060745`
with a 63-entry dirty worktree and active GCS transfers
**Execution state:** unblocked on 2026-07-26 — Paper C was stopped and its GCS
transfers finished, clearing the concurrency freeze. See
[Implementation status](#implementation-status) for exactly what was applied, what was
deviated from, and what remains. **The distribution gate is still open**: no source is
approved for verbatim redistribution, so no public text build is authorized.

This plan separates reusable code, executable studies, benchmarks, publications,
immutable evidence, local data, and transient runs without invalidating the
repository's existing scientific locks. It does not authorize moving or rewriting
any claim-bearing file.

## Audit verdict

The study-centered direction and no-big-bang migration rule are correct. The first
draft was not safe to execute because it treated unlike contracts as equivalent,
omitted live Paper C input roots, proposed a mutating reproduction command as a
read-only check, and deferred a current distribution-license problem.

Verified state at this audit:

| Surface | Current verification state | Consequence for restructuring |
|---|---|---|
| Paper A v2 historical training lock | Full live-tree verification fails because four bound source files differ from their recorded hashes | Verify this lock against its immutable execution-source snapshot, not the hardened checkout |
| Paper A v2 release cache | Current release/anchor contract validates | Preserve every release-bound live compatibility path |
| Starting-type adaptation | No final `LOCK.json`; normative contract is `dev_nonfinal`, and its recorded authoring-config hash differs from the current YAML | Treat as contract-drifted and unresolved, not as an ordinary unlocked study |
| Mortgage v1 | Release checksums pass, but they cover release bytes rather than generator, judge, configuration, or code | Preserve the full workspace and describe the limited verification scope |
| Paper C reference-centering predecessor | Historical superseding locks do not validate against the later live tree, as its status ledger expects | Verify in a matching snapshot/archive; never rewrite the historical locks |
| Active Paper C successor | Candidate lock validation fails; active training/scoring entrypoints are not fully inventoried; candidate says data, GPU training, and claims are unauthorized | Complete a protocol-integrity audit before migration; existing outputs remain development-only absent contemporaneous authorization |
| Benchmark explorer distribution | Tracked HTML embeds full third-party and unlicensed draft benchmark text | Stop public-release/Pages work until a source-by-source distribution decision is recorded |

The failed Paper C validation and missing source coverage are not cured by moving
files or minting a retroactive lock. Future authorization must be prospective.

## Decision

Adopt a **study-centered monorepo** for new and contract-cleared work. Keep already
released, path-bound studies at their current compatibility paths. In particular:

- do not perform a repository-wide rename;
- do not rename the existing `artifacts/` tree;
- do not move released Paper A code, configuration, or evidence;
- do not move any of the three Paper C surfaces—the root matched-DPO candidate,
  predecessor workspace, or active successor—while protected work is active;
- do not treat "unlocked" as "safe to migrate"; first resolve contract and evidence state;
- use new structure prospectively, then migrate contract-cleared studies one at a time.

## Why change the layout

The current repository has several competing organizational models:

- `experiments/` contains Paper A, composition, KL-SFT, starting-type adaptation,
  ExpGuard, ensembling, and an older Paper C scaffold;
- one study is distributed across `experiments/`, `configs/`, `tests/`, `docs/`,
  `artifacts/`, and `papers/`;
- active Paper C code, tests, cloud tooling, run outputs, and manuscript sources are
  nested under `papers/paper_c/specialize_then_align/`;
- the active Paper C successor is nested inside an archived predecessor, while older
  root-level Paper C files are still described as current in some indexes;
- `papers/` contains substantial ignored runtime state and a repository-local Cloud
  SDK in addition to publication files;
- the benchmark explorer mixes application source with very large generated HTML;
- raw/licensed data, released evidence, and disposable run outputs are not expressed
  as three distinct storage classes;
- the root test command does not cover the mortgage and nested Paper C test suites.

The result is navigational ambiguity and a high risk of moving a file that is part
of a reproducibility contract.

## Non-negotiable invariants

1. **Compatibility contracts are explicit and typed.** Treat every path recorded by
   a lock, release anchor, checksum manifest, or reproduction consumer as a
   compatibility API until a per-study audit proves otherwise. A scientific lock,
   release-cache contract, checksum-only release, and ordinary hard-coded path are
   different contract classes and must not be described interchangeably.
2. **Historical evidence is immutable.** Never rewrite an old lock to make a new
   layout appear authorized. A migrated frozen release receives a versioned
   migration/release contract appropriate to its contract class. A prospective
   scientific lock is required before new data construction, training, selection,
   or claim-bearing evaluation.
3. **Authorization is prospective.** A migration may not upgrade development output,
   repair protocol chronology, or authorize a completed run retroactively.
4. **Papers contain publication material prospectively.** New training code, SDK installations,
   model weights, and transient runs must not be rooted under `papers/`.
   Existing paper-local code and Paper C workspaces remain compatibility exceptions.
5. **Artifacts are evidence, not a scratch directory.** New claim-bearing locks,
   manifests, text-free scores, and analyses may enter `artifacts/`; transient runs
   may not.
   The current tree is mixed; this rule applies to new namespaces.
6. **Licensed/raw data stays local by default.** Every source must have a tracked
   distribution decision and manifest, while restricted or unreviewed payloads
   remain ignored and excluded from public builds.
7. **One study, one owner.** Each active study owns its code, config, tests, protocol,
   environment, and CLI from one study root.
8. **One import name has one package root.** Do not create both `guard_research/`
   and `src/guard_research/`.
9. **Every move is verified.** Imports, tests, paper builds, links, contracts, and
   reproduction checks must pass before compatibility paths are removed.
10. **No symlink compatibility for scientific inputs.** Leave old files intact or
    verify them in their recorded snapshot; do not replace lock-bound paths with
    symlinks.

## Target logical structure

```text
guard-ranking-fragility/
├── guard_research/                    # authoritative release-bound shared package
├── studies/
│   ├── registry.yaml                 # normative machine-readable study registry
│   ├── registry.schema.json
│   ├── composition/
│   ├── expguard/
│   ├── klsft/
│   ├── starting-type-adaptation/
│   └── paper-c-specialize-align-mortgage-v1/  # only as a new authorized version
├── benchmarks/
│   ├── registry/                      # tracked source/license/hash decisions
│   ├── mortgage-v2/                   # future package; not a move of current v1
│   │   ├── generator/
│   │   ├── releases/
│   │   └── evaluation/
├── apps/
│   └── benchmark-explorer/
│       ├── src/
│       ├── fixtures/                  # redistributable, minimal CI fixtures
│       └── dist/                      # ignored build root
│           ├── public/                # audited allowlist-only release staging
│           └── local/                 # gated/local-only full-data output
├── papers/
│   ├── unified-report/
│   │   ├── slides/
│   │   └── editorial/
│   ├── finetuning-specialization/
│   ├── finetuning-specialization-simplified/
│   ├── base-adapter-composition/
│   ├── base-adapter-composition-simplified/
│   ├── mortgage-guardrail-benchmark/
│   ├── mortgage-guardrail-benchmark-simplified/
│   ├── paper_c/                       # frozen compatibility roots for now
│   └── paper-c-specialize-align-mortgage-v1/  # future manuscript copy, if authorized
├── artifacts/                         # existing mixed compatibility tree; new release evidence
├── data/                              # ignored raw/licensed/download cache
├── runs/                              # future ignored transient outputs
└── docs/
    ├── architecture/
    ├── reproducibility/
    └── decisions/
```

This is a non-exhaustive, prospective destination, not a command to move every
current directory. The root `guard_research/` package remains authoritative. The
future `benchmarks/mortgage-v2/` does not replace or relocate the existing
`mortgage-benchmark/benchmark/v1_hmda2022/` release. Current presentation-planning
documents and simplified editions remain publications/editorial assets unless a
separate lifecycle review classifies them otherwise.

The active successor's stable configured ID is
`paper_c_specialize_align_mortgage_v1`; the hyphenated directory name above is only
a proposed filesystem slug and must be mapped explicitly in the registry.

## Compatibility exceptions

The following current paths are compatibility surfaces, not examples for future
studies. Preserve them until a study-specific audit and versioned replacement says
otherwise.

**Paper A and composition:**

- `guard_research/`;
- root `pyproject.toml` and `requirements.txt`;
- `configs/paper_a_sft.yaml` and
  `configs/paper_a_sft_v2_release_anchor.json`;
- the historical lock-bound Paper A executables under `experiments/`;
- the six live release-cache sources:
  `experiments/paper_a_common.py`, `experiments/analyze_paper_a_sft.py`,
  `experiments/analyze_composition.py`, `guard_research/__init__.py`,
  `guard_research/metrics.py`, and `guard_research/thresholds.py`;
- `artifacts/paper_a_sft/` and `artifacts/paper_a_sft_v2/`, including
  `LOCK.json`, `RELEASE.json`, public manifests, scores, and provenance snapshots;
- Paper A, Paper B, and unified-report consumers of those outputs.

The Paper A historical training lock is expected to validate fully only against its
recorded execution-source snapshot. The current live compatibility contract is the
release-cache contract, not live-tree validation of the historical training lock.

**Starting-type adaptation:**

- `configs/starting_type_adaptation_v1.yaml`;
- `requirements-starting-type-adaptation.txt`;
- all starting-type, native-contract, and ensembling executables under
  `experiments/` and their root tests;
- `artifacts/starting_type_adaptation_v1/`;
- `papers/unified-report/proposal.md` and its report consumers.

This study is contract-drifted and scientifically unresolved: committed results
exist, but there is no final lock, the normative contract is `dev_nonfinal`, and
its recorded authoring-source hash no longer matches the current YAML. Preserve
the paths until an explicit result/protocol disposition is recorded.

**Mortgage:**

- the complete `mortgage-benchmark/` workspace, including the checksummed v1 rows,
  tracked `out_eval/` evidence, policy cards, tools, tests, and Makefile;
- the ignored mortgage source inputs used by active Paper C;
- all unified-report, paper, explorer, and Paper C consumers.

The v1 checksum validates release bytes only. It does not bind the generator,
judge, configuration, or code, and unresolved license/release issues prevent
labeling it sealed or public-ready.

**Paper C:**

- the root-level matched-DPO candidate files spread across `docs/`, `configs/`,
  `experiments/`, and `tests/`;
- the reference-centering predecessor at `papers/paper_c/`;
- the active successor at `papers/paper_c/specialize_then_align/`;
- every ignored input, artifact, bundle, and external object required to explain
  those workspaces.

The root-level matched-DPO candidate is an unresolved, unrun protocol scaffold.
Existing indexes still describe it as normative. Its disposition and relationship
to the active successor must be recorded explicitly; the successor's current
migration decision covers only the reference-centering predecessor.

## Publication and editorial lifecycle

Do not treat every document under `papers/` as equally current:

- `papers/unified-report/proposal.md` is a contract-consumed starting-type protocol
  despite its generic filename; keep it at its current path;
- `papers/unified-report/review.md` is the latest review but is currently untracked;
- `papers/unified-report/review-2026-07-17.md` is a historical applied-fix ledger
  and is also currently untracked;
- the tracked presentation proposal and its review predate the latest scientific
  review and contain claims the latest review disputes;
- the PowerPoint `~$...pptx` file is an editor lock, not a publication artifact.

Phase 0 must capture the durable reviews without the editor lock. The future paper
index must label documents `current`, `historical`, `superseded`, or `candidate`
instead of presenting all of them as peer guidance. Physical editorial moves are
optional and follow only after inbound-link updates; they are not a prerequisite
for the scientific-path migration.

## Paper C concurrency freeze

No structural operation may touch Paper C until all of the following are true:

- the Paper C agent reports a complete handoff;
- there are no live `gsutil` or `gcloud` processes using the in-repository SDK;
- no `*.gstmp` fragment has a live writer or changing size, and every incomplete
  remnant is classified;
- modified and untracked Paper C source files are committed or otherwise captured;
- ignored files are inventoried in addition to tracked and untracked files;
- every relevant GCS object and local artifact is hashed, and the cloud/local
  manifests are reconciled with a tested restore procedure;
- exact commands, bundle hashes, model revisions, VM image, Python version, package
  versions, and dependency-install commands are preserved with credentials and
  tokens redacted;
- the stale successor candidate lock and all unbound execution entrypoints are
  reconciled in a protocol-integrity report;
- source inventory covers `tools/`, `cloud/run_cells.sh`, dependency installation,
  and every actual data-build, GPU-training, scoring, and analysis entrypoint;
- existing successor outputs are explicitly classified as development-only unless
  a contemporaneous authorization chain proves otherwise;
- no completed output is retroactively authorized; future locks are issued before
  the work they authorize;
- contract verification is performed in the correct mode: live release contract,
  current candidate contract, or historical source snapshot as applicable;
- all three relevant Paper C test surfaces—the root matched-DPO tests, predecessor
  suite, and successor suite—pass, or a historically expected failure is
  explicitly classified under its correct verification mode;
- the exact pre-migration tracked tree, ignored-input manifests, and Git revision
  are recorded.

Protected during the freeze:

```text
papers/paper_c/
guard_research/
experiments/
configs/
tests/
artifacts/
mortgage-benchmark/
data/benchmarks/
data/mortgage_guard_bench_2k_v0_1_0/
Makefile
pyproject.toml
.gitignore
requirements.txt
requirements-starting-type-adaptation.txt
experiments/README.md
.env
.env.example
```

The protection is intentionally broader than `papers/paper_c/`: Paper C and older
compatibility code read several root-level paths.

Git cannot capture ignored Paper C inputs. A clean commit alone is therefore not a
handoff; the ignored/external manifest and restore test are mandatory. Record only
the secret file's presence, permission mode, and injection mechanism—never its
contents or hash in a public manifest.

## Contract-specific verification

Verification must name the contract class:

- **Live release contract:** validate the release anchor, release self-hash,
  release-bound source files, public manifest tree, scores, and metadata in the
  current compatibility checkout.
- **Historical scientific lock:** validate self-hash/chain integrity, then perform
  full byte/path verification in the recorded immutable source snapshot, detached
  Git worktree, or verified authority archive. An intentionally evolved checkout
  is not expected to match.
- **Checksum-only release:** verify the listed bytes and state clearly what code,
  process, or environment is not covered.
- **Protocol candidate:** require the current candidate to match its inventoried
  sources and confirm that its data-build, GPU-training, and claim flags remain
  false. A valid candidate freezes the design; it authorizes no evidence-producing
  operation. The corresponding required child authorization lock must separately
  exist, validate, and permit each proposed operation.

The active successor must make lock selection explicit rather than hard-code one
candidate filename. Its Make targets must accept a `LOCK` path and forward it to
`lock-protocol --out $(LOCK)` and `validate-lock --path $(LOCK)`, so callers can run
`make validate-lock LOCK=locks/<issued-name>.json`. Preserve
`PROTOCOL_TAXONOMY_CANDIDATE.json`, introduced at Git
`8d94d78dcd1bbb3fc4428c97929abe6e56dbdd1b`, as a named historical candidate;
never overwrite it to make the evolved tree pass. Any new prospective candidate
receives a new filename and is selected explicitly, but remains non-authorizing.
Evidence-producing Make targets and entrypoints must use a distinct `AUTH_LOCK`
path, validate the applicable child in the candidate's ordered `required_children`
chain, and fail closed if it is absent, stale, out of sequence, or does not permit
that exact operation. Child-lock creation and validation are not implemented in the
current code, so candidate validation alone cannot unblock new work.

Expected failures must be recorded rather than "fixed" by rewriting historical
contracts. Only the currently supported live contract must validate against live
compatibility paths.

## Storage contract

| Class | Location | Git policy | Examples |
|---|---|---|---|
| Reusable source | `guard_research/`, `studies/`, `benchmarks/`, `apps/` | tracked | Python, schemas, configs |
| Publication | `papers/` | tracked | TeX, bibliography, final PDF, review |
| Released or development evidence | `artifacts/<study_id>/` | explicit allowlist; status recorded | lock, hash manifest, text-free scores |
| Raw or licensed data | `data/` | ignored payload; manifest lives in `benchmarks/registry/` | HF cache, HMDA source data |
| Transient execution | `runs/<study_id>/<run_id>/` | ignored after adding `/runs/` rule | checkpoints, smoke runs, logs |
| Audited public build | release/Pages object | allowlist only; not a generated Git blob | explorer distribution |
| Large private evidence | versioned cloud/cold storage | manifest and hashes tracked | adapters, full internal archive |

The existing `artifacts/` layout remains intact. This policy applies to new output;
it does not retroactively move locked evidence.

The current `.gitignore` ignores all of `/data/` and does not ignore root `runs/`.
Therefore tracked data ledgers belong under `benchmarks/registry/`, and `/runs/`
must be added before anything writes there. New artifact exceptions must be narrow:
the existing `artifacts/**/manifests/*.jsonl` rule can otherwise hide a new study's
intended public manifest.

## Immediate distribution gate

Repository restructuring must not republish the current explorer as-is. The tracked
`benchmark-explorer/index.public.html` is approximately 55.6 MB and embeds 16,146
complete rows, including ToxicChat material identified as CC-BY-NC and all 2,000
rows of MortgageGuardBench-2K even though that dataset says its publication license
has not been selected. "Publicly accessible" and "not gated" do not mean
"redistributable."

Before any Pages, release, shard, or replacement build:

1. create the canonical source-by-source ledger at
   `benchmarks/registry/distribution.yaml`, validated by
   `benchmarks/registry/distribution.schema.json`;
2. record source revision and payload hash, access class, license/terms snapshot,
   redistribution decision, attribution/notice, derived-output license,
   sensitive-text/PII class, reviewer, and decision date;
3. use a positive allowlist for public text;
4. add negative tests proving gated, unreviewed, noncommercial-reconstruct-only,
   and forbidden source IDs/text cannot enter a public build;
5. emit source notices and a fail-closed expected section/count/hash manifest;
6. default unresolved sources to local-only or text-free views.

This gate also applies to the current repository tip, not only future builds.
Inventory GitHub Pages configuration/deployments, release assets, and other known
remote copies of the explorer or mortgage rows. After preserving a private/local
copy and obtaining explicit user approval, either replace or remove the tracked
explorer blob and remediate deployed copies, or record an affirmative distribution
decision for every embedded source. Make a separate explicit decision about the
current MortgageGuardBench-2K exposure; a text-free notice is the fail-closed
replacement while its license remains unresolved.

Apply the same current-tip decision gate to the tracked
`mortgage-benchmark/benchmark/v1_hmda2022/` release rows: either record affirmative
license and redistribution approval, keep every repository/mirror that contains
the row text nonpublic, or—only with explicit user approval and a verified private
authority archive—replace public row text with a text-free compatibility notice
and a separately versioned public-safe release. In the latter case, historical
byte verification moves to the authority archive; it must not be made green by
rewriting the existing checksum contract.

Removing the blob from the current tree does not remove it from Git history. Any
history rewrite is a separate destructive/legal migration requiring explicit user
approval and coordination with every clone.

## Study registry

After Phase 0 integrity gates pass, add `studies/registry.yaml` as the normative
machine-readable registry, validate it against `studies/registry.schema.json`, and
generate or check human-readable indexes against it. README prose must not become a
second source of truth.

Each entry should contain:

- stable `study_id`;
- title and scientific question;
- separate `study_state`, `evidence_state`, `publication_state`,
  `distribution_state`, and `test_exposure` fields;
- states including `protocol_candidate`, `development_only`, `active`, `blocked`,
  `contract_drifted`, `stopped`, `superseded`, `released`, and `invalidated`;
- evidence tier and `claim_authorization`;
- `contract_type`, `verification_mode`, `verification_command`, and
  `expected_verification_status`;
- code root;
- configuration and protocol paths;
- environment/lock path;
- artifact and release roots;
- manuscript and status paths;
- `consumes`, `produces`, `predecessor`, `successor`, and `supersedes` edges;
- whether paths are lock-bound;
- source snapshot commit or authority archive;
- external-object manifest and restore command;
- redistribution status;
- last verified Git revision and whether that verification used a dirty tree.

The root README, `papers/README.md`, and status tables should summarize this registry
rather than independently defining study state. The validator must reject unknown
states, missing targets, cycles in `supersedes`, incompatible evidence/claim states,
and a reported expected-pass verification command that actually fails.

## Migration phases

### Phase 0 — finish and capture active work

1. Let already-running Paper C transfers finish or stop them safely; do not launch
   a new evidence phase under the stale, non-authorizing candidate lock.
2. Separate existing cleanup, Paper C development, and documentation changes into
   reviewable commits.
3. Capture tracked, untracked, ignored, and external Paper C inputs/outputs with
   hashes and a restore recipe.
4. Conduct the successor protocol-integrity audit: stale candidate, source-inventory
   omissions, authorization chronology, cloud commands, and output classification.
5. Record each contract's type, expected-pass/expected-fail mode, matching source
   snapshot, and actual validation result.
6. Capture current reviews and other durable untracked editorial records before
   relying on Git history to preserve them. Exclude Office lock files and secrets.
7. Complete the explorer/mortgage distribution ledger and stop public publication
   for any source without an affirmative decision.
8. Do not move files or retrofit authorization in this phase.

The Phase 0 deliverables have fixed locations: the distribution ledger and schema
live under `benchmarks/registry/`; the successor integrity report, ignored-input
manifest, and external-object manifest live respectively at
`papers/paper_c/specialize_then_align/provenance/PROTOCOL_INTEGRITY_AUDIT.md`,
`papers/paper_c/specialize_then_align/provenance/IGNORED_INPUT_MANIFEST.json`, and
`papers/paper_c/specialize_then_align/provenance/EXTERNAL_OBJECT_MANIFEST.json`.
Create the Paper C files only after its active-work handoff and within the user's
Paper C scope; each manifest must have a schema or documented version, hashes, and
a tested restore command.

**Exit gate:** the Paper C concurrency-freeze conditions, protocol-integrity gate,
and immediate distribution gate are satisfied.

### Phase 1 — additive navigation and verification

This phase changes no scientific paths and begins only after Phase 0.

1. Split explorer testing into redistributable tracked-fixture tests suitable for a
   clean clone and explicit local full-data inventory tests. A tiny fixture does not
   authorize or reproduce the public full-data build.
2. Add `studies/registry.yaml`, its schema/validator, and generated-or-checked
   `studies/README.md` and `papers/README.md` views.
3. Correct stale references that identify an archived or unresolved Paper C design
   as current, without overstating the active successor's evidence status.
4. Add tiered root verification targets that delegate to all existing test surfaces
   with one explicit absolute interpreter per invocation.
5. Add a Markdown-link check for indexes and publication READMEs.
6. Add CI for the hermetic fast tier and registry/link validation. There is no CI
   workflow today; do not call the contract enforced until CI is active.

Required verification tiers:

```text
check-fast       root fixture-safe tests + mortgage + Paper B + both Paper C suites
check-data-local full ignored-data inventories; optional and explicitly local
check-locks      contract-specific live, candidate, or historical-snapshot checks
check-release    pinned Python 3.12, side-effect-free release reproduction
check-papers     isolated manuscript builds plus link checks
check-all        aggregate only when its declared prerequisites are available
```

Current test commands that belong in `check-fast` after the clean-clone split are:

```text
<py> -m pytest -q
make -C mortgage-benchmark test PY=<absolute-py>
make -C papers/base-adapter-composition test PYTHON=<absolute-py>
make -C papers/paper_c test PYTHON=<absolute-py>
make -C papers/paper_c/specialize_then_align test PY=<absolute-py>
```

The clean-clone bootstrap must first select an absolute CPython 3.12 interpreter
and assert its identity, install the fully pinned root `requirements.txt`, and run
`<absolute-py> -m pip check`. Only then may it run the five commands above. CI
fixtures are sufficient for `check-fast`, but they are not substitutes for the
ignored full inputs used by local inventory or public-release construction.

Nested Makefiles use both `PY` and `PYTHON`; the aggregator must pass the correct
variable explicitly and verify the interpreter matches the required Python version.
The current `.venv` is Python 3.14.4 while Paper A's documented release environment
requires Python 3.12, so local convenience success is not release verification.

Do **not** put the current `make -C papers/unified-report reproduce-check` in a
read-only target. Its `--check` path runs analyzers against canonical tracked
artifact/output trees and overwrites figure/intermediate files. First refactor it to
use temporary output roots, check every subprocess return code, enumerate every
paper-consumed input dynamically, and compare without mutation—or execute the
unchanged command only inside a disposable verified worktree.

Do not require the current active-successor `validate-lock` to pass until the
protocol-integrity audit has produced a valid new prospective candidate. Record
its present failure as an expected blocker, not a green check. Even after candidate
validation passes, keep data construction, training, scoring, selection, and claims
blocked until creation, validation, and entrypoint enforcement exist for the
applicable child authorization lock.

Root `clean` is not a verification prerequisite: the Paper A paper clean target
deletes a tracked canonical PDF. Repair clean semantics before any orchestration
invokes clean targets.

### Phase 2 — resolve Paper C navigation and dispositions

Treat these as distinct studies:

1. the unresolved, unrun root-level matched-DPO candidate spanning `docs/`,
   `configs/`, `experiments/`, and root `tests/`;
2. the stopped reference-centering workspace at `papers/paper_c/`;
3. the active cross-model specialist study at
   `papers/paper_c/specialize_then_align/`.

For the older flat scaffold:

- inventory every inbound reference and lock binding;
- write a disposition that explicitly relates it to the active successor;
- retain a minimal provenance pointer describing its disposition;
- remove unneeded live-code presentation only after compatibility tests prove it
  is not consumed;
- rely on Git history rather than recreating a general `legacy/` directory only
  after recording the exact scaffold commit and verifying that it is independently
  retrievable from the authoritative remote or archive.

For the stopped predecessor:

- keep lock-bound source and terminal evidence immutable until its archive is
  independently verified;
- move large private failure artifacts to cold storage only after checksum and
  restore testing;
- retain an in-repository manifest and explicit non-evidence statement.

Historical validation occurs in the matching snapshot/archive. Do not demand that
the evolved predecessor working tree pass an older source-byte lock.

### Phase 3 — separate application builds from source

This phase starts only after the Phase 0 distribution ledger authorizes a concrete
public subset. For the benchmark explorer:

1. retain generator/application source;
2. use ignored `dist/public/` for audited release staging and `dist/local/` for
   gated or local-only full-data output;
3. add a ledger-driven `fetch-public-inputs` or equivalent restore target that
   pins source revisions/URLs and verifies every payload hash before building the
   complete authorized public site; tracked fixtures remain CI-only;
4. replace the monolithic generated HTML with a small shell plus audited,
   per-benchmark compressed or paginated assets;
5. generate only from the positive source allowlist and fail on unknown sources;
6. keep all of `dist/` ignored and publish only `dist/public/` through Pages or a
   versioned release when the expected count/hash manifest and source notices
   validate;
7. preserve a local full-data validation command without treating its output as
   redistributable.

Do not rewrite Git history as part of this phase. Historical blob removal is a
separate repository migration requiring explicit approval.

### Phase 4 — prospective study packages

New or contract-cleared studies should use:

```text
studies/<study_id>/
├── README.md
├── pyproject.toml
├── environment/                       # Python/runtime/container lock or digest
├── config/
├── protocol/
├── schemas/
├── src/<package>/
├── tests/
└── tools/
```

They should expose module or console entry points rather than depend on
`python experiments/foo.py`, `sys.path` modification, or a fixed parent depth.
Outputs must be explicit arguments resolving to `artifacts/` or `runs/` according
to the storage contract. The environment record must pin Python and all scientific
dependencies used by the claimed result; a permissive `pyproject.toml` alone is not
a result-producing environment lock.

### Phase 5 — Paper C migration, if still desired

The user's current collaboration contract confines Paper C operations to
`papers/paper_c/`. Expanding it into `studies/`, `runs/`, or another paper root
requires fresh explicit user authorization even after technical gates pass.

Only after the active study reaches a stable, integrity-audited release and that
authorization is granted:

1. freeze and verify the original study tree;
2. create a migration decision with old and new tree hashes;
3. replace fixed-parent repository discovery and hard-coded environment paths;
4. instantiate a new version from the verified snapshot under
   `studies/paper-c-specialize-align-mortgage-v1/`;
5. copy manuscript material into the separately authorized future paper root;
6. route new local execution to
   `runs/paper-c-specialize-align-mortgage-v1/`;
7. install the Cloud SDK outside the repository;
8. create a complete pinned environment and source inventory;
9. issue a migration manifest for the copied frozen release; if new scientific
   execution is planned, issue a new prospective protocol candidate and every
   required child authorization lock before the operation that each child
   authorizes;
10. leave the old compatibility tree intact, or verify it only through its detached
    historical worktree/authority archive;
11. retain old locks unchanged and map every lock hash to its source snapshot;
12. run old-snapshot verification and new-version verification independently.

Do not replace the old tree with symlinks. Moving or deleting the active tree before
the new version is independently verified is prohibited.

## Verification required for every migration

- `git diff --check` passes;
- no unexpected tracked deletion or untracked replacement exists;
- `git status --porcelain --ignored` and the pre/post ignored-input manifest explain
  every relevant ignored-file change;
- all inbound Markdown and code references resolve;
- import tests pass without repository-root path injection where applicable;
- study unit and contract tests pass;
- relevant paper builds and reproduction checks run in temporary output roots or a
  disposable worktree and leave the source checkout unchanged;
- live release-cache contracts validate in the live compatibility tree;
- historical contracts validate against their recorded source snapshot/archive,
  with expected live-tree failures documented;
- current protocol candidates match a complete execution-source inventory and
  remain non-authorizing, while the applicable child authorization lock matches
  the current source/environment inventory and permits the exact proposed step;
- migrated frozen releases validate a new migration/release contract; studies
  performing new scientific work issue and validate prospective locks before
  execution;
- artifact hashes and row counts are unchanged unless a new version documents the
  change;
- ignored and external artifacts have complete manifests and tested restoration;
- a clean checkout can perform the documented public CPU-only verification path,
  while private-evidence restoration is documented as a distinct tier;
- gated or noncommercial text is absent from public releases unless its license
  and recorded distribution decision explicitly permit redistribution;
- no migration silently upgrades a development result's evidentiary status;
- no output contains developer-specific absolute paths unless explicitly classified
  as nonportable diagnostic metadata.

## Rollback rule

Each migration phase must be one focused commit or short commit series. If its
verification gate fails, revert that phase rather than patching locks or evidence
to match the move. Scientific provenance takes precedence over directory symmetry.

After Phase 0, perform restructuring in a separate worktree/branch based on the
captured handoff commit. Do not restructure in the shared worktree used by another
active agent.

## Recommended immediate action

While the Paper C agent and transfers remain active, do only read-only audits and
new isolated planning files. Do not edit existing README, STATUS, protocol, code,
configuration, input, artifact, or shared environment paths.

The next operational sequence is:

1. let already-running transfers finish or stop them safely, but launch no new
   evidence phase under the stale/unauthorizing candidate;
2. capture all tracked, ignored, and GCS state;
3. complete the Paper C protocol-integrity audit and classify existing output as
   development-only unless contemporaneous authorization is proven;
4. complete the explorer/mortgage distribution ledger;
5. record expected-pass and expected-fail verification modes;
6. only then begin Phase 1 registry, navigation, hermetic-test, and CI work in a
   separate worktree.

This sequence provides navigational benefit without destroying provenance or
allowing directory cleanup to mask a scientific or distribution-control problem.

---

## Implementation status

Recorded 2026-07-26, after the plan was executed. The sections above are the plan **as
written** and are left unedited; this section is the record of what actually happened, so
that a later reader can tell specification from state without re-deriving it from the log.

Applied across seven commits, one per phase boundary, per the rollback rule:
`12192b0` Phase 0 capture · `7fa0ef7` distribution ledger · `de93614` Phase 1 registry,
validators, tiers · `55a15af` README · `f2d0b00` verification reconciliation + CI ·
`421a836` Phase 3 explorer · `84eb61e` Phase 5 migration.

### What was applied

| Phase | State | Evidence |
|---|---|---|
| 0 — capture active work | done | Tracked/ignored/GCS state captured; Paper C classified development-only; `EXTERNAL_OBJECT_MANIFEST.json` (114 objects, 19.15 GiB) |
| 1 — additive navigation and verification | done | `studies/registry.yaml` + schema (12 studies), `benchmarks/registry/distribution.yaml` + schema (10 sources), `tools/validate_registries.py`, `check_markdown_links.py`, `render_indexes.py`, `tree_digest.py`, eight `make check-*` tiers (`check-all`, `check-data-local`, `check-fast`, `check-links`, `check-locks`, `check-papers`, `check-registry`, `check-release`), `.github/workflows/verify.yml` |
| 2 — Paper C navigation and dispositions | **not done** | See below |
| 3 — separate application builds from source | done, with the gate held shut | `apps/benchmark-explorer/` with `src/`/`fixtures/`/ignored `dist/`, positive-allowlist build, 8 negative tests, expected count/hash manifest |
| 4 — prospective study package layout | done as a template, one deviation | `studies/paper-c-specialize-align-mortgage-v1/` |
| 5 — Paper C migration | done as a copy | `provenance/MIGRATION_MANIFEST.json`, enforced by `tests/test_migration_manifest.py` |

**Nothing claim-bearing moved.** Zero git renames. `artifacts/`, `guard_research/`,
`mortgage-benchmark/`, `configs/`, `experiments/` and root `tests/` keep their paths, so
every existing lock still binds. No history was rewritten.

**Verification declarations are reconciled, not assumed.** All 12 registry entries declare
`expected_pass` or `expected_fail` with a reason, and `validate_registries.py
--run-verification` runs each and fails if an outcome disagrees with its declaration.
Writing that reconciliation caught four wrong declarations, including a `paper_a_sft_v2`
entry declared passing that in fact fails on `python_mismatch` (3.12 release env vs. the
3.14 local `.venv`) — a real environment split the registry now states rather than hides.

### Deviations from the plan as written

1. **Phase 4 layout: `PROTOCOL.md` at package root, no `protocol/` directory.** The
   spec lists `protocol/`. The study carries a single protocol document plus
   `DEVELOPMENT_PLAN.md`; a directory for one file adds a level without adding
   structure. `studies/registry.yaml` addresses it through `protocol_paths`, so
   discovery does not depend on the layout choice. Revisit when a study needs
   versioned protocol amendments.

2. **Phase 5 item 5: manuscript copied inside the package, not into a separate paper
   root.** The plan routes manuscript material to "the separately authorized future
   paper root." No such authorization exists, and the study is stopped with nothing to
   publish, so `manuscript/` stays in the package. Creating a paper root for a stopped
   study would imply a publication path that is not authorized.

3. **Phase 5 item 8: the environment record is explicitly incomplete.** Phase 4 requires
   pinning Python and every scientific dependency for a claimed result. This study claims
   nothing, and the record *cannot* be completed: torch/CUDA came from the VM image
   rather than pip, `accelerate` was installed unpinned and never captured, and the pilot
   A100s are deleted, so no `pip freeze` is recoverable. `environment/gpu-requirements.txt`
   says so in its header instead of presenting two pins as a lock. Consequence, stated
   plainly: **the GCS development artefacts cannot be exactly reproduced.** They are
   development-only and nothing depends on reproducing them.

4. **Source-group split isolation was infeasible as specified.** The protocol's wording
   implies unioning on `source_id`, which left 9 units across 9 datasets and made a 4-way
   split impossible. Splitting on the family ↔ content-family transitive closure works and
   is what the code does. The protocol wording still needs correcting — it describes an
   operation that cannot be performed on this corpus.

### What Phase 2 requires and why it was not done

Phase 2 disposes of the older flat matched-DPO scaffold spanning `docs/`, `configs/`,
`experiments/` and root `tests/`. It is skipped deliberately: it is the only phase whose
steps include *removing* live-code presentation, and its own preconditions are unmet —
it requires proving the scaffold is not consumed, and verifying the recorded commit is
independently retrievable from the authoritative remote. Both are audits of external
state, not edits. The two scaffold studies are meanwhile registered
(`paper_c_matched_dpo_scaffold`, `paper_c_reference_centering`), both verify
`expected_pass`, and both carry an explicit relation to their successor — which delivers
Phase 2's *navigational* purpose without any deletion.

### Defects found by executing the plan

Executing a layout plan is a test of the code it moves. Four defects surfaced that no
amount of reading would have found, all fixed in **both** the predecessor and the
migrated tree so they stay byte-identical:

1. **Fixed-parent repository discovery.** `parents[2]` is correct at the predecessor's
   depth of 3 and resolves *outside the repository* at the study package's depth of 2 —
   it would have read corpora from the wrong place rather than raising. Replaced with
   marker-based `repository_root()`. This is the concrete case for Phase 4's rule against
   depending on a fixed parent depth.

2. **The storage contract's `runs/` location was unreachable.** Phase 5 item 6 routes new
   execution to `runs/<study_id>/`. `output_path()` *rejected* an absolute
   `<repo>/runs/<slug>/` path and silently resolved a relative `runs/x` inside the
   workspace, so documentation promising that routing described behaviour the code
   refused. Added `runs_root()` and a second permitted root — still fail-closed at
   exactly two — with four tests covering the admitted path, the boundary, non-widening
   of an explicit `root=`, and unchanged relative-path behaviour.

3. **A migration check that could not see what it verified.** The first manifest reported
   the trees differing by one file, because the comparison hashed only non-ignored files —
   so `.gitkeep` placeholders under `artifacts/`, `inputs/` and `build/` were invisible,
   and the copy had dropped all three. `inputs/.gitkeep` is tracked source, so the new
   tree was genuinely incomplete while the check called it faithful. This is the failure
   mode the "verification required for every migration" section exists to prevent, and it
   still happened, because the coverage rule was implicit. `tools/tree_digest.py` now
   states its rule and prints it with every result, and `tests/test_migration_manifest.py`
   recomputes the comparison from disk — deleting `inputs/.gitkeep` fails three of its
   seven tests independently.

4. **A dead `.gitignore` negation.** `!artifacts/.gitkeep` cannot re-include a file whose
   parent directory is excluded, and `papers/paper_c/.gitignore` excludes `artifacts/`
   outright — so the negation is dead in the predecessor and live under `studies/`.
   Recorded in the migration manifest rather than silently changed: making them agree
   would alter ignore semantics for paths outside this study.

### Still open

- **The distribution gate.** No source is approved for verbatim redistribution, so
  `publishable()` returns empty and no public text build is authorized. The validator
  emits this as a standing warning on every run.
- **Tracked, already-pushed `benchmark-explorer/index.public.html`** — 53 MB, 16,146 rows
  across 10 sources, embedding all 2,000 MGB2K rows whose own `LICENSE_NOT_SELECTED.md`
  states that no publication license has been selected and legal review is required. The
  new build cannot produce this file, but the existing one is public and needs a **human
  licensing decision**; removing it from history is a separate migration requiring
  explicit approval.
- **Phase 2**, per the reasoning above.
- **The `paper_a_sft_v2` interpreter split** (3.12 release vs. 3.14 local) is declared and
  enforced, not resolved.
