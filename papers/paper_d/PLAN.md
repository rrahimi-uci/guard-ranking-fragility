# From Policy Documents to Guardrail Decisions

## An End-to-End Experimental Workflow for Regulated-Domain Safety Guards

**Paper D research and execution plan**

**Status:** protocol proposal; no Paper D experiment has run and no result is claimed

**Primary domain:** U.S. mortgage policy, with Fannie Mae seller-policy and federal consumer-finance regulation kept as separate authority classes

**Primary online system:** compact, policy-conditioned safety guards that emit `ALLOW`, `REVIEW`, or `INTERVENE`

**Date:** 2026-07-27

**Novelty review:** refreshed against adjacent work available through 2026-07-27

**Open-data audit:** refreshed against downloadable benchmark cards and the repository distribution ledger through 2026-07-27

**Upstream implementation audit:** neighboring `policy-to-knowledge` repository at commit `484afd4ff4994811bcbe11568a39f9498f12b2b9` and its current Fannie Mae stage artifacts inspected 2026-07-27

**Plan review 2026-07-27:** every quantitative claim in Section 7.1 was re-verified directly against
the sibling checkout and reproduced exactly — commit `484afd4` present and clean; A1 506 chunks, 51
merges, median 658 / max 2,845 words; A3 290 entity-nested + 104 relationship-nested = 394; A4 394;
A5 384 after 10 removals across 9 duplicate groups; 339 edges as 87 explicit / 105 implicit / 27
inferred / 120 workflow; 394 `text_match_score` objects with 64 below 0.5; 27 `reference_verified:
false`; 360 placeholder jurisdictions (309 `EXAMPLE_POLICIES`, 50 `SAMPLE_GUIDELINES`, 1
`EXAMPLE_POLICICIES`); graph SHA-256 `1497ce97…6be0d4`; source `fannie_mae.pdf` SHA-256
`7e399d96…578cc`, 1,191 pages, titled "Fannie Mae Selling Guide March 4, 2026"; seed rule
`BR_MORTGAGE_FILE_CONSTRAINT_008_004` present with `text_match_score` 0.988. Four defects were
fixed: the policy-context arms were renamed `P0`–`P6` → `R0`–`R6` to end a namespace collision with
the `P0`–`P9` lock-chain phases; `D_visible` gained the population definition it lacked (Section
14.2); the repository layout gained `locks/`, `provenance/`, `environment/` and a per-study
`.gitignore` (Section 19); and registration in `studies/registry.yaml` — absent entirely, and
load-bearing in this repository — became a P0 deliverable (Section 19.1).

**Citation audit 2026-07-27:** all 24 external citations in Section 2 were fetched and checked
against the papers themselves, several via locally extracted PDF text rather than fetch summaries.
**No citation is fabricated** — zero not-found, and no identifier resolving to a different paper.
Four were misdescribed and are corrected in place: ComplianceNLP (its Appendix N already runs a
paired policy-perturbation experiment, so the plan claimed ground that is occupied — the most
damaging finding); TEMPO (an RL training method with no benchmark, where “policy” means policy
*optimization*); PolicyShiftGuard (an **image** benchmark about held-out policy *definitions*, not
temporal vintages); and ProvenanceGuard (whose paper explicitly disclaims the adversarial and
policy-compliance territory the plan conceded to it). The former three-work temporal row was split,
because two of its three members did not belong in it. Six name-versus-title mismatches and four
bare-name collision families — “PolicyGuard” alone names three distinct systems — are now cited by
full title plus identifier. **The kill rule NARROWS rather than fires:** no neighbour scores more
than two of the construct's four load-bearing elements, and observability-as-a-manipulated-variable
and asymmetric scoring are unoccupied across the entire set, but *pairing* and *policy perturbation*
are now partly claimed, so Section 2.2 was restated as a four-part conjunction. Still open: the
construction half of the kill rule (P5 has not run) and five adjacent works surfaced but not read,
listed in Section 2.3.

**Recommended submission-facing title:**

**From Policy Documents to Guardrail Decisions: Counterfactual Integrity Tests for Compact Mortgage-Policy Guards**

---

## 1. Executive decision

Paper D should test an end-to-end scientific question, not merely describe a document-processing pipeline:

> When a compact guard must triage a proposed action under a changing regulated policy, where does reliability come from—and where does it fail—between the source document, extracted rule representation, retrieval layer, and final guard decision?

The paper should contain two linked experiments rather than treating one realistic pipeline comparison as causal evidence.

**Experiment A — identifiable policy-context integrity interventions.** Starting from an independently verified canonical policy packet, hold the case, facts, guard, prompt, retrieved rule nodes, and token budget fixed while changing exactly one of:

1. representation: semantically matched flat rules versus the same nodes plus typed graph edges;
2. policy integrity: canonical packet versus one prespecified silent semantic defect; or
3. evidence visibility: the same defect with versus without source/provenance evidence that makes the integrity failure observable.

**Experiment B — deployment-oriented end-to-end comparison.** Hold the guard, cases, prompt, retrieval budget, and scoring protocol fixed while changing the policy material supplied to the guard:

1. no policy context;
2. token-matched raw-document retrieval;
3. automatically extracted flat rules;
4. automatically extracted rule graph;
5. source-verified flat rules;
6. source-verified rule graph; and
7. an oracle policy packet containing exactly the independently adjudicated governing rules.

```mermaid
flowchart TB
    FIXED_A["Fixed case, facts, guard, prompt,<br/>retrieved nodes, and token budget"]
    FIXED_A --> REP["Representation intervention"]
    FIXED_A --> DEFECT["Integrity and visibility intervention"]
    REP --> A_REP["A0 — canonical flat<br/>A1 — canonical graph"]
    DEFECT --> A_DEFECT["A2 — silent defect<br/>A3 — same defect + visible evidence"]
    A_REP --> CAUSAL["Eligible for paired,<br/>causal contrasts"]
    A_DEFECT --> CAUSAL

    FIXED_B["Fixed guard, cases, prompt,<br/>retrieval budget, and scoring"]
    FIXED_B --> AUTO["No or automatically compiled context"]
    FIXED_B --> TRUSTED["Verified or oracle context"]
    AUTO --> B_AUTO["B0 — no context<br/>B1 — raw RAG<br/>B2 — automatic flat<br/>B3 — automatic graph"]
    TRUSTED --> B_TRUSTED["B4 — verified flat<br/>B5 — verified graph<br/>B6 — oracle packet"]
    B_AUTO --> BUNDLED["System comparisons;<br/>not isolated causal effects"]
    B_TRUSTED --> BUNDLED
```

The automatic rule and graph arms will use the real neighboring **Policy to Knowledge (P2K)** implementation, not a new Paper D reimplementation. Paper D imports its stage outputs read-only, pins the upstream Git commit and runtime configuration, and evaluates the outputs at Agent 3 (candidate rules), Agent 4 (merged graph), and Agent 5 (deduplicated rules plus inferred dependencies). The existing historical Fannie output remains a frozen descriptive baseline; a controlled rerun on Paper D's locked source bundle is required for primary comparisons.

Policy to Knowledge is an experimental system under test, not policy authority, gold annotation, or the claimed paper contribution. Paper D owns the source archive, independent adjudication, fail-closed verification, benchmark construction, retrieval controls, guard scoring, and statistical analysis.

Together these create six quantities, but only the controlled quantities from Experiment A receive a causal interpretation:

- the value of having policy context at all;
- the value of graph structure beyond semantically identical flat rules;
- the downstream amplification of a silent upstream defect;
- the ability to route a provenance-visible integrity defect to `REVIEW`;
- the practical effect of moving from raw retrieval to a verified representation; and
- the remaining guard-reasoning gap after policy retrieval is made nearly oracle.

The raw-RAG versus verified-graph comparison is a useful **system contrast**, but it is not the primary causal contrast because it jointly changes extraction fidelity, selection, serialization, and graph structure.

The study is **mortgage-first and depth-first**. The agreed title remains a working title, but a scope gate applies:

- if only mortgage is independently validated, retain “regulated-domain” as an adjectival description only if the abstract says “a mortgage case study”; otherwise use the narrower subtitle **“A Mortgage-Policy Case Study”**;
- claim cross-domain generality only after a second domain has its own authoritative source archive, schema mapping, qualified reviewers, and sealed evaluation families;
- the existing finance/health/law ExpGuard rows may be used as an external behavior check, but they do not establish document-to-decision validity because they are not currently bound to Paper D source snapshots and extracted rules.

Open datasets have sharply different roles in this design:

- HMDA and CFPB complaint data may improve scenario realism, but they provide no policy-compliance gold;
- SafePyramid, PolyGuard, DynaBench, and PolicyGuardBench may test external policy-conditioned guard behavior, but they cannot validate mortgage extraction or authority fidelity;
- FinSafeGuard is a noncommercial synthetic BFSI resource and may be used only under its license and never as authoritative mortgage gold;
- Zillow's full Fair Housing Guardrail dataset is request-gated rather than openly downloadable;
- the repository's two mortgage benchmarks remain local-only until licensing is affirmatively resolved, and their current labels are not substitutes for qualified Paper D adjudication.

No external dataset is pooled with the primary mortgage families. Each retains its own construct, license, label provenance, output mapping, and uncertainty statement.

Paper D does not propose an automated compliance officer. The guard evaluates a proposed assistant response or workflow action and decides whether it may proceed, needs qualified review, or should be constrained. It never approves a mortgage, denies credit, gives a legal determination, or certifies institutional compliance.

---

## 2. Updated related-work and novelty audit

The broad story “policy documents become a grounded compliance or guard system” is already crowded. The closest work now occupies almost every individual pipeline stage. Paper D must not use “first end-to-end,” “first document-grounded guard,” “first regulatory knowledge graph,” “first compact policy guard,” or “first temporal policy benchmark” language.

### 2.1 Closest-work matrix

| Work | What it already establishes | Collision with the original plan | Boundary Paper D can still test |
| --- | --- | --- | --- |
| [Business as Rulesual: A Benchmark and Framework for Business Rule Flow Modeling with LLMs](https://aclanthology.org/2026.acl-long.1625/) (BREX) | 409 real documents, 2,855 expert rules, structured dependencies and executable grounding | Rule extraction and structured rule-flow evaluation are not novel by themselves | Critical-field fidelity tied to downstream guard behavior under locked source versions |
| [AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models](https://aclanthology.org/2026.findings-acl.727/) | Ingests unstructured policy documents to generate and evolve safety evaluations | Document-to-benchmark generation is occupied | Independently adjudicated, source-version-bound guard decisions rather than autonomous benchmark generation |
| [GuardSet-X, formerly PolyGuard](https://papers.neurips.cc/paper_files/paper/2025/hash/11ed9cdc955e23684a1beae9cb8da059-Abstract-Datasets_and_Benchmarks_Track.html) (NeurIPS 2025 D&B; the camera-ready is retitled, and is distinct from the multilingual [arXiv:2504.04377](https://arxiv.org/abs/2504.04377) PolyGuard) | Authentic policy-grounded guard data across eight domains and 19 guard models; some policies name **real regulators**, so the “fictional policy” caveat applies to SafePyramid only | Multi-domain policy-grounded guard benchmarking is occupied | One authority-sensitive mortgage case study with explicit source lineage and upstream-error interventions |
| [SafePyramid: A Hierarchical Benchmark for In-context Policy Guardrailing](https://arxiv.org/abs/2606.29887) | 1,000 conversations, 3,000 policies, 61,699 rules (the HF card says 61,639), dependency levels L0/L1/L2, five policy-configurable guards. **Structurally a paired design**: the same conversation is reused across L0/L1/L2, and L1 rules must express an L0-not-violated → L1-violated conflict | In-context policy reasoning, dependency evaluation, **and paired policy-context variation over a fixed case** are occupied | The **integrity and authenticity** of the packet. Decisive separator: SafePyramid's edited policy is *true* and its ground truth **moves with the edit** (measuring policy-execution competence); Paper D's edited policy is *false* relative to a verified source and the reference decision stays **pinned to the canonical answer** (measuring corruption susceptibility). Opposite assumptions about who is trustworthy. It also has no abstain channel — refusals are discarded from its metrics |
| [DynaGuard: A Dynamic Guardian Model With User-Defined Policies](https://arxiv.org/abs/2509.02563) (1.7B / 4B / **8B**) and [Learning Efficient Guardrails for Compliance](https://arxiv.org/abs/2510.03485) (which contains the *PolicyGuardBench* benchmark and exactly **one** model, PolicyGuard-4B) | Compact models conditioned on user-defined policies or agent-policy trajectories | “Small policy-conditioned guard” is occupied | Test these as baselines under verified, corrupted, and evidence-visible regulatory packets |
| [FinGuard: Detecting Financial Regulatory Non-Compliance in LLM Interactions](https://arxiv.org/abs/2605.29427) | Regulation-driven benchmark generation and an 8B guard adapting to unseen institutional policy documents. Its policy object is **already in the inference input and trusted**; all eight adversarial dimensions rewrite the **user query**, never the policy. Already uses the word “paired” for violative/compliant instances | Financial regulation-to-guard is directly occupied — do not soften this | Paper D's attack surface exists structurally in FinGuard's design and is simply never attacked. Decisive separator: FinGuard changes the policy by changing the **weights** (self-play RL on new documents); Paper D changes it by changing the **context**. Cite the authors' own concession that it is not stress-tested against jailbreak attacks |
| [PolicyGuard: From Organizational Policies to Neuro-Symbolic Compliance Review Engines](https://arxiv.org/abs/2606.32004) — a **concurrent unrefereed preprint** (v1 30 Jun 2026) on a tiny closed corpus (5 NDAs, 95 guidelines, 475 decisions, unreleasable policy text), not settled prior art | Converts organizational policies into typed logic and uses local evidence questions plus symbolic evaluation for NDA review | Explicit policy compilation and stage separation are occupied | Learned online guard behavior when the compiled policy substrate is imperfect, stale, or provenance-inconsistent |
| [PolicyGuard: A Dialogue-Grounded Sub-Agent Verifier for Policy Adherence in LLM Agents](https://arxiv.org/abs/2606.29225) | A context-sharing verifier for multi-turn agent policy adherence and remediation; results are single-domain (τ²-bench airline only, 4 trials) | External policy-verifier framing is occupied | Source-document fidelity and policy-packet trust boundaries rather than dialogue-only policy adherence |
| [Safeguarding LLM Agents from Misalignment through Provenance Analysis](https://arxiv.org/abs/2607.01236) (system name: ProvenanceGuard; **not** the differently-authored [arXiv:2606.18037](https://arxiv.org/abs/2606.18037) of the same name) | Formalizes misalignment detection as whether a proposed tool call is supported by traceable evidence in the agent's context. Its guard takes exactly three inputs — query, tool docs, call history — **all trusted; there is no policy object in the formalism** | **Benign, non-adversarial intent-grounding** is occupied. The paper explicitly disclaims "harmful, adversarial, or policy-noncompliant behavior" and walls itself off from GuardAgent/ShieldAgent and AgentDojo/CaMeL, so it must **not** be cited as evidence that security-oriented provenance guarding is taken | Treat the compiled **policy context itself** as the corrupted object; bind it to authoritative source bytes and version lineage. **Differentiate explicitly from its §8 limitation** — "if that plan is itself already misaligned … a locally justified action may still be globally misaligned" — which is the nearest published statement of this intuition, but is a limitation paragraph with no experiment, no metric, and the agent's own reconstructed plan rather than an authoritative policy packet as its object |
| [Reason Less, Verify More: Deterministic Gates Recover a Silent Policy-Violation Failure Mode](https://arxiv.org/abs/2607.07405) | Deterministic pre-execution gates for silent policy-violating tool actions; coins “the silent policy-violation class.” Its canonical deceptive task #48 has a **user assert a false timestamp**, so the model is misled about state rather than ignorant of the rule | Deterministic gating, the “silent policy violation” framing, **and misleading context in agent policy enforcement** are occupied | Its loud-versus-silent distinction is **descriptive across domains, not a manipulated variable** — Paper D manipulates observability as an arm. Note the mechanism runs *opposite*: its gate wins by ignoring asserted context and consulting database state and policy, and that trusted fallback substrate is exactly what Paper D corrupts |
| [ComplianceNLP: Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection](https://arxiv.org/abs/2604.23585) | Regulatory change monitoring, obligation extraction, KG-augmented RAG, gap detection, 70B-to-8B distillation, and error propagation **attributed by error type** (Table 12 breaks the 87.7→83.4 drop into NER boundary −2.9, cross-reference −1.0, deontic −0.4). **Appendix N already runs a controlled policy-perturbation experiment**: replacing institutional policies with synthetically perturbed variants on the same 150 GapBench pairs, extraction unaffected, gap-detection F1 −3.2 moderate / −6.8 aggressive | Generic KG-versus-RAG, regulatory monitoring, compact deployment, **per-error-type propagation attribution, and paired policy-perturbation robustness probing** are all occupied | Narrower than previously stated, and defensible on three specific distinctions: **typed** defect classes rather than graded perturbation intensity; **injected** defects with per-defect amplification rather than decomposition of the system's own naturally occurring errors; and an **observable-versus-silent axis, which is absent from this work entirely** |
| [Knowledge Graph Representations for LLM-Based Policy Compliance Reasoning](https://arxiv.org/abs/2604.27713) | Two KG schemas over three policies and five-model policy QA | “KG helps policy reasoning” is occupied | Token- and node-matched structure effect on action triage, separated from extraction and retrieval |
| [LogiSafetyGen](https://arxiv.org/abs/2601.08196) (framework; the benchmark it produces is *LogiSafetyBench*) | Converts regulations into temporal-logic oracles for tool-use evaluation | Executable regulatory oracles are occupied | Use an executable shadow only as a control; study imperfect natural-language policy compilation |
| [STALE: Can LLM Agents Know When Their Memories Are No Longer Valid?](https://arxiv.org/abs/2605.06527) | Agent-memory temporal-validity benchmark; the one genuinely temporal citation of the three formerly grouped here. One probing dimension is named "Implicit Policy Adaptation," where *policy* means the agent's operating procedure | Temporal-validity probing of stale context is occupied | Verified rule-version lineage bound to authoritative source bytes, effective dates, and supersession, rather than memory staleness |
| [TEMPO: Temporal Enforcement via Mode-Separated Policy Optimization for Trustworthy LLM Backtesting](https://arxiv.org/abs/2605.18843) | A **GRPO training method** plus convergence theory, with a two-mode reward that drives post-cutoff claims to zero. Releases **no benchmark and no rows**; "Policy" is *policy optimization* in the reinforcement-learning sense, **not** a safety or content policy. Subject is knowledge-cutoff contamination as a threat to backtest validity | Temporal-leakage control **via training** is occupied | Not a benchmark and not about policy shift; cite as prior work on temporal-leakage control, never as a temporal evaluation resource — there is nothing here to evaluate on |
| [PolicyShiftGuard: Benchmarking and Improving Policy-Adaptive Image Guardrails](https://arxiv.org/abs/2607.05910) | An **image** guardrail benchmark (cs.CV) plus a trained 7B guard and a two-stage RP-SFT / BP-Adapt recipe. "Policy shift" means generalizing to **held-out policy definitions at a single point in time** — the Shift Split withholds policies, not vintages. No cutoff date, no before/after split | Policy-**definition** generalization is occupied, in the **vision** modality | Cannot support any claim about text guards or temporal vintages. PolicyShiftBench is licensed **non-commercial research only**, which given this repository's purge history is a hard constraint, not a formality |
| [Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty](https://aclanthology.org/2026.eacl-long.291/) (artifact: MedAbstain) | Explicit abstention methods in high-stakes medical QA | Abstention itself is occupied | `REVIEW` as an adjudicated workflow state with reason codes and a capacity constraint, not confidence rejection |

This matrix is a protocol-screening aid, not a claim of exhaustive systematic review. P0 still requires database queries, backward/forward citation chasing, deduplication, inclusion criteria, and a dated search ledger.

### 2.2 Novelty verdict

**Weak or already occupied ideas:** document ingestion, rule extraction, policy KG construction, generic KG-versus-RAG comparison, synthetic policy-grounded benchmark generation, compact policy-conditioned guards, provenance-based action guarding, deterministic action gates, abstention, temporal evaluation, error propagation **including per-error-type source attribution**, **paired policy-context variation over a fixed case**, and **policy-perturbation robustness probing**.

The last three were added after the 2026-07-27 citation audit. Each was previously claimed as
remaining ground and is not: ComplianceNLP attributes propagation by error type (Table 12) and runs
a paired policy-perturbation experiment (Appendix N); SafePyramid reuses one conversation across
L0/L1/L2 with conflict-expressing rules. **The words “paired,” “policy-context manipulation,” and
“counterfactual audit of policy-context integrity” can no longer carry the novelty on their own.**

**Defensible candidate if executed rigorously:** not pairing, and not policy perturbation, but the
**conjunction** of four properties — the first two now contested, the last two verified unoccupied
across every audited neighbour:

1. **Typed defect classes**, not graded perturbation intensity. ComplianceNLP grades its
   perturbation moderate/aggressive; Paper D contrasts named, prespecified defect operators, and
   injects them rather than decomposing errors the system happened to make.
2. **The reference decision pinned to the canonical answer.** This is the sharpest separator and
   was absent from earlier drafts. SafePyramid's edited policy is *true* and its ground truth
   **moves with the edit**, so it measures policy-execution competence. Paper D's edited policy is
   *false* relative to a verified source and the reference decision stays **pinned to canonical**,
   so it measures corruption susceptibility. The two designs make opposite assumptions about who is
   trustworthy.
3. **Observability as an independently manipulated variable** (`C2` withheld versus `C3` present,
   and the V0–V3 evidence ladder). No audited work manipulates it; the nearest, *Reason Less,
   Verify More*, uses loud-versus-silent descriptively across domains rather than as an arm.
4. **Asymmetric scoring** that does not penalize a guard for corruption it cannot observe, with
   adjudicated `REVIEW` reason codes. Also unoccupied: none of the six collision-critical works has
   any abstain or review channel at all, and SafePyramid explicitly discards refusals from its
   metrics.

The novel object is the versioned policy substrate compiled from authoritative documents—not
provenance guarding or deterministic gating in general. **Pre-emption to state in the paper:** the
silent-versus-observable arm is not a relabeling of loud-versus-silent tool errors; it manipulates
what evidence the guard holds about an authoritative source, not how noisily an action fails.

The contribution has six linked parts, of which **2 and 3 are the strongest surviving elements** and
should lead:

1. **Orthogonalized stage attribution over a concrete pipeline.** Policy to Knowledge checkpoints, extraction fidelity, representation, retrieval success, and guard reasoning are varied separately; the realistic full-pipeline contrast is not mislabeled causal.
2. **Two mutation modes.** Silent semantic defects measure downstream vulnerability; provenance-visible defects measure whether the guard appropriately routes invalid policy context to `REVIEW`.
3. **Semantic review reasons.** `REVIEW` is adjudicated as missing case facts, policy conflict, stale policy, unresolved authority, policy-integrity failure, or out-of-scope—not generic uncertainty.
4. **Versioned authority lineage.** Every mutation and decision is bound to policy bytes, authority class, effective date, supersession lineage, and evidence spans.
5. **Compact, policy-configurable baselines.** Generic 1.5B–4B checkpoints and purpose-built DynaGuard-class baselines are tested with frozen weights under the same packets.
6. **Decision-level defect amplification.** The main output is not another extraction F1 score; it is the family-clustered change in unsafe guard decisions caused by each prespecified upstream defect. **No priority is claimed for propagation attribution itself** — ComplianceNLP already attributes propagation by error type. What is claimed is per-defect-class amplification of *injected* defects with the reference decision held at canonical.

The Policy to Knowledge integration makes the first contribution executable and reproducible, but it is not itself a novelty claim. Its open-source pipeline and intermediate artifacts provide a concrete system to audit; the scientific novelty still depends on paired policy-context interventions, independent gold, and stage-isolated estimands.

The paper remains useful if graphs do not help or if guards cannot detect visible integrity failures. A credible null would bound when policy compilation or graph construction does not justify its cost. Novelty depends on the **controlled design and trust-boundary construct**, not on a positive result or a “first” claim.

### 2.3 Novelty strength and kill rule

- **Strong paper:** independently verified policy families; canonical/silent/visible counterfactual packets; direct policy-guard baselines; semantic reason codes; deterministic-gate comparison; family-clustered inference.
- **Moderate paper:** the same intervention design but mortgage-only and retrospective, with no sealed confirmation.
- **Weak paper:** only the seven realistic pipeline arms, extraction scores, or raw-RAG-versus-graph results. That should be framed as an engineering report or benchmark resource, not the claimed novelty above.

An optional high-value extension is a **policy-context observability curve**. Replace binary visibility for a subset with a locked evidence ladder:

```mermaid
flowchart TB
    V0["V0 — internally coherent corruption;<br/>no contrary evidence"]
    V1["V1 — raw source or provenance<br/>evidence is present"]
    V2["V2 — structured evidence identifies<br/>the conflicting fields"]
    V3["V3 — deterministic validator emits<br/>a typed integrity failure"]
    V0 -->|add raw evidence| V1
    V1 -->|structure the conflict| V2
    V2 -->|run deterministic gate| V3
```

This estimates the minimum evidence needed for a model or system to escalate each defect class. It must remain secondary unless pilot power supports the extra arms.

**Kill/narrow rule:** if P5 cannot produce at least four critical defect classes with reviewer-validated canonical/silent/visible twins, or if the closest-work refresh finds an equivalent paired policy-context-integrity study, drop the broad novelty claim and reposition the output as a mortgage policy-guard audit resource.

**Kill-rule status at 2026-07-27: NARROWS — one half tested, one half still open.**

The literal collision half was tested by adversarial audit against the six collision-critical works,
each scored on the four load-bearing elements above. **Nothing scored more than 2 of 4, and the
misses are structural rather than incremental**, so the rule does not fire:

| Work | Corrupts policy substrate | Canonical/corrupted twins | Observability as a variable | Vulnerability separated from detection failure |
| --- | --- | --- | --- | --- |
| ComplianceNLP | yes | yes (untyped, graded) | no | no |
| SafePyramid | no — policy stays true | partly; ground truth recomputed | no | no — refusals discarded |
| Reason Less, Verify More | no — policy faithful but unenforced | no — gates on/off | no — descriptive only | no |
| Safeguarding LLM Agents (ProvenanceGuard) | no — no policy object | no — "paired" means permutation tests | no | no |
| PolicyGuard neuro-symbolic | no — perturbs the document, rule fixed | partly; fidelity ablation | no | no — binary output |
| FinGuard | no — policy trusted, pressure is query-side | no — "paired" means violative/compliant | no | no |

**Two limits on that verdict, neither of which may be quietly dropped:**

1. **The construction half is untested.** P5 has not run, so whether four critical defect classes
   admit reviewer-validated canonical/silent/visible twins is unknown. The kill rule is not clear
   overall until it does.
2. **The refresh is incomplete.** Five adjacent works surfaced during the audit and were **not
   read**; three are closer to Paper D's *method shape* than anything actually audited. Each needs
   its own pass before the design lock, and [arXiv:2606.18356](https://arxiv.org/abs/2606.18356) is
   the highest priority because separating semantic from audit-evidence harm is the nearest reported
   analogue of elements 3 and 4:
   - [arXiv:2606.18356](https://arxiv.org/abs/2606.18356) — SafeClawBench, reportedly separates
     semantic harm from **audit-evidence** harm;
   - [arXiv:2606.29073](https://arxiv.org/abs/2606.29073) — MCP runtime security invariants,
     reportedly with counterfactual ablations that disable one enforcement component at a time;
   - [arXiv:2605.26497](https://arxiv.org/abs/2605.26497) — "Aligning Provenance with
     Authorization: A Dual-Graph Defense for LLM Agents"; authorization is policy-shaped;
   - [arXiv:2606.22873](https://arxiv.org/abs/2606.22873) — SingGuard, policy-adaptive multimodal
     guardrail;
   - [arXiv:2603.01228](https://arxiv.org/abs/2603.01228) — "Towards Policy-Adaptive Image
     Guardrail," an apparent predecessor of PolicyShiftGuard; establish which one the matrix means.

### 2.4 Open benchmark availability and exact role

The current open-data landscape supports external controls, not replacement of the Paper D primary benchmark.

| Resource | Verified availability at audit | Paper D role | Explicit boundary |
| --- | --- | --- | --- |
| [SafePyramid v1.1](https://huggingface.co/datasets/ByteDance/SafePyramid) | CC BY 4.0; 3,000 test rows over **1,000 conversations**, so rows are not independent units and clustered inference is required. Rule count is **61,699** per the abstract and GitHub README, but the HF card states 61,639 — a real card/abstract discrepancy, cite with the source named | Preferred external supplied-policy and dependency-reasoning control | Policies and regulatory frameworks are fictional; no mortgage-source or extraction claim |
| [PolyGuard](https://huggingface.co/datasets/Virtue-AI-HUB/PolyGuard) | CC BY 4.0; downloadable finance, regulation, and law subsets. Some policies name **real regulators**, so any assumption of uniformly fictional policy must be scoped to SafePyramid alone | Broad policy-grounded guard control | Binary/general guard construct; not mortgage authority adjudication |
| [DynaBench](https://huggingface.co/datasets/montehoover/DynaBench) | MIT; downloadable. The 543-row count, the PASS/FAIL values, and “handcrafted” are **not stated on the card** — they come from the size endpoint, first-rows, and the paper, so record that verification path rather than citing the card | Direct external benchmark for policy-conditioned compact guards | PASS/FAIL interface and synthetic policies; no three-way mortgage triage |
| [PolicyGuardBench](https://huggingface.co/datasets/Rakancorle1/PolicyGuardBench) | Apache 2.0; **59,997 rows, of which only 12,000 are the designated test split**. License-only card, no linked paper, no documented contamination controls. **Unresolved provenance gap:** the upload carries no arXiv tag, so identifying it with arXiv:2510.03485's benchmark is *inferred, not confirmed* — register it that way | Optional agent-trajectory policy-violation control | Web-agent trajectories, not document-derived regulation |
| [FinSafeGuard](https://huggingface.co/datasets/domyn/FinSafeGuard) | CC BY-NC 4.0; 709,303-row Ultra-Mini synthetic BFSI release — **only Ultra-Mini is hosted**. Generative interface, and labels require `original_label` extraction | Optional noncommercial robustness/training study, never primary evaluation | Taxonomy/LLM-generated gold; not source-version-bound and not usable for unrestricted commercial redistribution |
| [Zillow Fair Housing Guardrail](https://github.com/zillow/fair-housing-guardrail) | Code public under **AI Pubs Open RAIL-S v0.1** (use-restricted, and it does not cover data); the “samples” are **13- and 16-row smoke-test fixtures, unusable as an evaluation set**; full labeled dataset/model by request only | Gated housing-adjacent comparison if access and terms are approved | Not an open benchmark; do not plan the paper around receiving it |
| [FinGuard-Bench](https://arxiv.org/abs/2605.29427) | Release is promised **conditionally** — “upon publication,” subject to third-party licensing — and the promise covers construction code, evaluation scripts, and annotation guidelines, **not the 1,020-pair labeled benchmark and not weights**. Absent from the authors' own team repository | Watch-list only | No dependency until bytes, version, schema, and license are independently verified |
| Repository ExpGuard | Gated/local-only; 2,275 expert-annotated finance/health/law prompts already scored | Existing external descriptive check | Not mortgage and not source-snapshot-bound; do not pool with open controls or primary families |
| Repository MortgageGuardBench-2K | 2,000 local rows; publication license not selected | Development-only historical baseline | No redistribution and no Paper D gold status |
| Repository `v1_hmda2022` | 994 tracked synthetic rows; no affirmative redistribution decision | Development-only historical baseline | LLM/policy-card labels, not SME-authoritative or source-version-complete |

The benchmark registry must record, for every external resource: upstream URL, immutable revision or commit, payload hashes, license snapshot, required attribution, task/output schema, row count, access date, redistribution decision, and exact Paper D use. A dataset being downloadable is not by itself permission to republish transformed or verbatim rows.

---

## 3. Core thesis and falsifiable research questions

### RQ1 — Extraction fidelity

Can the pinned Policy to Knowledge pipeline recover decision-relevant atomic rules from mortgage-policy documents with correct authority, conditions, exceptions, dates, numeric values, and verbatim source support, and at which checkpoint are errors introduced, removed, or hidden?

**H1:** adding Paper D's fail-closed deterministic verification and independent review to a pinned Policy to Knowledge run improves critical-field precision and downstream decision safety over both the frozen historical Agent 5 graph and the unverified Agent 3/Agent 5 outputs from the controlled rerun.

No directional hypothesis is assigned to Agent 4 enrichment or Agent 5 optimization. Deduplication may reduce redundancy while dropping exceptions, and inferred dependency edges may help retrieval while reducing edge fidelity; both are empirical checkpoint questions.

**Falsifier:** gains disappear under blinded expert review, checkpoint lineage cannot identify where outputs changed, or high aggregate scores conceal unacceptable numeric, date, exception, authority, or dependency errors.

### RQ2 — Policy representation

At the same policy-token budget, do graph-structured policy packets improve compact-guard decisions over raw passages or flat extracted rules?

**H2:** adding approved typed edges to the same source-verified rule nodes improves constrained decision utility over a token-matched flat serialization.

**Falsifier:** the upper confidence bound is below the prespecified smallest meaningful gain, or any observed gain disappears when the node set, semantic content, and token budget are matched.

### RQ3 — Selective review

Can compact guards distinguish a genuine violation from (a) a case whose decisive facts are unresolved and (b) a policy packet whose integrity or currency is observably invalid?

**H3:** explicit missing-fact, provenance, authority, and version evidence improves reason-correct semantic `REVIEW` recall without exhausting the fixed review budget on otherwise decidable traffic.

**Falsifier:** `REVIEW` is merely a proxy for low confidence, reason codes do not match the adjudicated unresolved condition, or review displaces violations without improving constrained utility.

### RQ4 — Error propagation

Which upstream policy defects produce unsafe downstream decisions, and which can be detected only when source or provenance evidence makes them observable?

**H4a:** silent dropped exceptions, altered thresholds, wrong authority/applicability, and stale supersession metadata cause larger unsafe-decision amplification than naming or formatting controls.

**H4b:** when an otherwise identical defect is accompanied by conflicting source/provenance evidence, appropriate `REVIEW` increases relative to both its canonical and silent-defect twins.

**Falsifier:** controlled defect classes do not produce distinguishable downstream effects, visible evidence does not change integrity-review behavior, or the guard ignores policy context altogether.

### RQ5 — Temporal policy robustness

Can a compact guard use `policy_as_of` and versioned rule lineage to avoid applying a stale rule to a post-update case?

**H5:** version-aware verified graphs reduce stale-rule application and increase appropriate `REVIEW` behavior relative to unversioned graph and raw-retrieval baselines. Policy to Knowledge Agents 7–10 may propose cross-version matches and differences, but only independently verified changes enter this test.

**Falsifier:** apparent gains come from lexical update cues, or cases remain correct after dates/version metadata are shuffled.

### RQ6 — End-to-end bottleneck

After separating Policy to Knowledge Agent 3 extraction, Agent 4 enrichment, Agent 5 optimization, Paper D verification, retrieval, and guard reasoning, which stage limits end-to-end reliability for each model and rule-complexity stratum?

No directional hypothesis is needed. The purpose is to attribute failure rather than compress the entire pipeline into one score.

### RQ priority

- **Primary scientific question:** RQ4, upstream-defect amplification and trust-boundary-aware detection.
- **Key mechanistic secondary questions:** RQ2 and RQ3.
- **Pipeline characterization:** RQ1 and RQ6.
- **Prespecified challenge stratum:** RQ5; temporal evaluation is not presented as the paper's independent novelty.

---

## 4. Governing construct

### 4.1 Unit of guard evaluation

The primary unit is a structured event:

```text
(request,
 proposed_response_or_action,
 actor_role,
 transaction_stage,
 jurisdiction,
 policy_as_of,
 relevant_case_facts,
 policy_context)
    ->
(ALLOW | REVIEW | INTERVENE,
 policy_ids,
 reason_code,
 decisive_facts,
 missing_facts,
 evidence_span_ids,
 calibrated_scores)
```

The guard evaluates the **proposed response or action**, not the applicant, lender, or loan outcome.

### 4.2 Semantic actions

- `ALLOW`: the proposed response or action may proceed under the focal policy and provided facts.
- `REVIEW`: the correct disposition cannot be established because a decisive fact, authority interaction, policy version, exception, or applicability condition is unresolved.
- `INTERVENE`: the proposed response or action should be refused, constrained, or replaced with a policy-consistent alternative.

`REVIEW` must not be generated simply because the model is uncertain. A row receives gold `REVIEW` only when qualified reviewers can name the missing or conflicting fact that prevents a valid `ALLOW`/`INTERVENE` determination.

Every gold and predicted `REVIEW` uses one primary reason code:

| Reason code | Meaning |
| --- | --- |
| `MISSING_CASE_FACT` | A decisive transaction or dialogue fact is unavailable |
| `POLICY_CONFLICT` | Two in-scope policy statements conflict and authority resolution is not established |
| `STALE_POLICY` | The supplied packet is observably not valid for `policy_as_of` |
| `AUTHORITY_UNRESOLVED` | The controlling authority, jurisdiction, or applicability is unresolved |
| `POLICY_INTEGRITY_FAILURE` | Hash, span, rule-value, dependency, or source evidence is inconsistent |
| `OUT_OF_SCOPE` | The proposed action cannot be evaluated under the supplied policy scope |

Reason-code accuracy is scored separately from action accuracy. Post hoc free-text explanations cannot substitute for a locked reason code.

### 4.3 Authority classes must remain separate

```text
federal_regulation
official_interpretation
agency_guidance
gse_seller_policy
institution_policy
```

The Fannie Mae Selling Guide is `gse_seller_policy`, not federal law. CFPB web pages may aid navigation, but the official CFR/Federal Register text governs legal-version snapshots. No graph edge may silently promote guidance or GSE policy into a statutory requirement.

### 4.4 Complexity strata

Every rule family is assigned before case generation to one of these strata:

1. direct atomic requirement or prohibition;
2. numeric or date threshold;
3. applicability/coverage condition;
4. explicit exception;
5. two-rule prerequisite or conjunction;
6. override or supersession;
7. cross-reference requiring another section;
8. temporal change between policy vintages; or
9. genuinely unresolved/interpretive, eligible only for `REVIEW` analysis.

Results must be reported per stratum. A pooled score cannot hide failure on exception, composition, or temporal cases.

### 4.5 Trust boundary for corrupted policy context

The guard can only be evaluated against information inside its interface.

- A **silent semantic defect** is internally coherent policy context containing a wrong threshold, missing exception, wrong authority, or stale rule while the contradicting source evidence is withheld. Its twin retains the same case. This tests downstream vulnerability and defect amplification. The correct action is evaluated against the authoritative canonical source, but failure to emit `REVIEW` is **not** called a detection failure because the defect may be unobservable to the guard.
- A **provenance-visible defect** exposes enough evidence to establish inconsistency: for example, the rule value conflicts with its source span, the snapshot hash is invalid, the packet date conflicts with `policy_as_of`, required policy fields are absent, or two authority statements conflict. Its gold action is `REVIEW` with the corresponding reason code.
- A **canonical packet** is the independently verified twin against which both mutation modes are paired.

This distinction prevents an invalid expectation that a compact guard should reconstruct hidden policy truth from pretraining. It also separates the responsibilities of the offline compiler, packet-integrity gate, and online guard.

---

## 5. End-to-end architecture under test

```mermaid
flowchart TB
    SRC["Authoritative document bytes"] --> SNAP["Immutable authority snapshot<br/>+ version metadata"]

    subgraph P2K["Policy to Knowledge — pinned commit"]
        direction LR
        A1["A1 — organized chunks"] --> A2["A2 — candidate entity and<br/>relationship ontology"]
        A2 --> A3["A3 — candidate rules"]
        A3 --> A35["A3.5 — advisory validation"]
        A35 --> A4["A4 — merged rule graph"]
        A4 --> A5["A5 — deduplicated rules<br/>+ candidate dependencies"]
    end

    SNAP --> A1
    ADAPT["Read-only Paper D adapter<br/>+ stage-lineage ledger"]
    A5 -->|all A1–A5 stage artifacts| ADAPT
    ADAPT --> DET["Deterministic span, numeric, date,<br/>schema, authority, and lineage checks"]
    ADAPT --> HUMAN["Independent blinded<br/>rule and edge review"]
    DET --> VERIFIED["Source-verified rules<br/>+ approved explicit edges"]
    HUMAN --> VERIFIED
    subgraph ONLINE["Online guard path"]
        direction LR
        RETRIEVE["Case-conditioned retrieval<br/>under a fixed token budget"] --> INTEGRITY["Packet-integrity and<br/>provenance checks"]
        INTEGRITY --> GUARD["Compact guard"]
        GUARD --> ACTION{"ALLOW · REVIEW · INTERVENE"}
        ACTION --> OUTPUT["Policy IDs · evidence · missing facts<br/>reason code · calibrated scores"]
        OUTPUT --> QUEUE["Human review queue for REVIEW<br/>and audit samples"]
    end

    VERIFIED --> RETRIEVE
```

Each arrow is evaluated independently. End-to-end accuracy without component measurements is insufficient because an incorrect final answer may arise from document versioning, extraction, retrieval, representation, or guard reasoning.

The offline compiler and online guard have different design goals:

- the **offline policy compiler** is the pinned Policy to Knowledge extraction path plus Paper D's independent verifier and review layer; it may use expensive models and human review because policy changes are relatively infrequent;
- the **online guard** must be compact, fast, deterministic, auditable, and conservative about unresolved policy context.

### 5.1 Cross-repository boundary

Paper D depends on Policy to Knowledge through immutable files, not Python imports into a mutable sibling checkout and not a shared database. A one-way adapter copies approved stage artifacts into `papers/paper_d/inputs/p2k/<run_id>/`, hashes them, validates their schemas, and emits Paper D candidate objects without changing the upstream bytes.

Two provenance classes must never be conflated:

- `historical_local`: artifacts already present in the neighboring checkout; useful for defect discovery and descriptive baselines, but not fully reproducible unless their original commit, prompts, configuration, CLI arguments, environment, and model revisions can be recovered;
- `controlled_p2k_run`: a new run created after source lock with a clean pinned commit, archived prompt and configuration hashes, exact CLI arguments, environment lock, model revisions, logs, and hashes for every stage output.

Only `controlled_p2k_run` artifacts are eligible for primary checkpoint comparisons. The historical graph remains `E0_p2k_historical_a5` and may never be silently relabeled as the output of the pinned controlled run.

```mermaid
flowchart LR
    subgraph UPSTREAM["Sibling policy-to-knowledge checkout — read only"]
        HIST["historical_local<br/>incomplete original provenance"]
        CONTROL["controlled_p2k_run<br/>clean pinned commit + full manifest"]
    end

    HIST -->|copy + hash| STORE["Paper D immutable input store"]
    CONTROL -->|copy + hash| STORE
    STORE --> VALIDATE["Schema, hash, rights,<br/>and lineage validation"]
    VALIDATE -->|passes| NORMALIZE["Normalized candidates<br/>+ raw-byte references"]
    VALIDATE -->|unresolved| FAIL["Fail closed"]
    NORMALIZE --> EXPERIMENTS["Paper D experiments"]
    BOUNDARY["No shared database · no mutable import<br/>no writes to the sibling repository"] -. constrains .-> STORE
```

---

## 6. Source corpus and authority archive

### 6.1 Mortgage-first corpus

The minimum publishable primary study uses two deliberately separate policy families:

#### A. Fannie Mae Selling Guide

- March 4, 2026 PDF already available locally;
- June 3, 2026 official guide snapshot to be acquired and hash-bound;
- relevant update announcements between the two vintages;
- selected Part B origination/underwriting sections plus a small Part A/C/D sample to test scope routing.

The local March source is 1,191 pages with SHA-256:

```text
7e399d961c41a49b6b305b996d9c18f4facd60601db1f129654e18dd8eb578cc
```

The current official [Fannie Mae Selling Guide](https://selling-guide.fanniemae.com/) reports a June 3, 2026 publication. The study must archive the actual bytes and update notices; a live URL is not a version lock.

#### B. Federal mortgage regulation

Start with a narrow, operationally testable subset of:

- Regulation B / ECOA provisions relevant to application evaluation, information requests, notifications, and valuations;
- Regulation Z provisions relevant to a tightly bounded origination topic, selected only after reviewer availability is confirmed.

The [CFPB mortgage resource index](https://www.consumerfinance.gov/compliance/compliance-resources/mortgage-resources/) is a source-discovery aid. Official legal snapshots must resolve to eCFR/Federal Register materials, with amendments and effective dates bound separately. CFPB’s own Regulation Z page warns that its navigable presentation is not the official legal edition.

### 6.2 Source inclusion criteria

A source enters Paper D only if all of the following are recorded:

- stable authority identity and authority class;
- exact retrieved bytes or an immutable official rendition;
- source URL and retrieval timestamp;
- SHA-256 digest;
- publication, effective, and expiration/supersession dates where applicable;
- jurisdiction and covered actors/products;
- version relation to earlier/later snapshots;
- redistribution and quotation decision;
- named source reviewer;
- parse quality sufficient to map page/section/span coordinates.

### 6.3 Exclusions

- state-law combinations in v1;
- broad standards whose application requires a full legal opinion;
- institution-specific rules without redistribution authority;
- withdrawn guidance presented as current obligation;
- scanned documents whose OCR has not passed numeric/date verification;
- rules for which no qualified reviewer is available;
- source material that cannot be version-locked;
- the existing generated graph as an authority source.

### 6.4 Authority-snapshot schema

Each archived document produces an immutable record:

```json
{
  "snapshot_id": "...",
  "authority_id": "...",
  "authority_class": "gse_seller_policy | federal_regulation | ...",
  "title": "...",
  "jurisdiction": "...",
  "source_url": "...",
  "retrieved_at": "...",
  "publication_date": "...",
  "effective_from": "...",
  "effective_through": null,
  "sha256": "...",
  "mime_type": "application/pdf",
  "parser_version": "...",
  "parent_snapshot_id": null,
  "supersedes_snapshot_id": null,
  "redistribution_class": "publish | text_free | local_only",
  "reviewer_id": "..."
}
```

The repository’s fail-closed benchmark distribution ledger is the model: absent affirmative permission, source text and derived rows remain local-only or text-free.

### 6.5 Public data for scenario realism, not policy truth

Two U.S. government data sources may inform case language and fact distributions without contributing compliance labels:

#### HMDA public loan-level data

- freeze one named annual Snapshot National Loan-Level Dataset and record its download URL, publication/as-of date, schema, filters, and hash;
- derive only aggregate distributions, bands, or independently sampled synthetic fact sheets;
- never reproduce a borrower/application record or preserve a row identifier;
- do not infer that an observed application outcome was compliant, fair, or policy-correct;
- use HMDA only for realistic combinations of product, geography, loan purpose, action, and banded applicant/property fields.

#### CFPB Consumer Complaint Database

- freeze a dated export filtered to mortgage products and record the field schema and hash;
- use issue/sub-issue frequencies and privacy-reviewed language patterns to seed scenario realism;
- treat complaint narratives as unverified consumer reports, not findings of a violation or representative prevalence data;
- paraphrase rather than copy narrative text unless release and privacy review affirm verbatim use;
- never let complaint categories, company responses, or outcomes define Paper D gold actions.

These realism sources are kept outside the authority archive. Every generated case records whether it was `expert_authored`, `hmda_distribution_informed`, `complaint_pattern_informed`, or both, plus the aggregate source version. Case authors and adjudicators still derive the governing action exclusively from locked authoritative policy snapshots.

---

## 7. Policy to Knowledge integration and current baseline audit

The neighboring `policy-to-knowledge` project is the concrete upstream compiler for Paper D. Its implemented extraction path is:

```mermaid
flowchart TB
    subgraph EXTRACT_1["Extraction — stages 1 to 3"]
        direction LR
        A1["A1<br/>Document organization"] --> A2["A2<br/>Entity and relationship extraction"]
        A2 --> A3["A3<br/>Rule extraction"]
    end

    subgraph EXTRACT_2["Extraction — stages 3.5 to 6"]
        direction LR
        A35["A3.5<br/>Advisory validation"] --> A4["A4<br/>Rule and entity merge"]
        A4 --> A5["A5<br/>Deduplication and<br/>dependency analysis"]
        A5 --> A6["A6<br/>Visualization"]
    end

    A3 --> A35

    A3 -. checkpoint .-> LEDGER["Paper D stage-lineage ledger"]
    A4 -. checkpoint .-> LEDGER
    A5 -. checkpoint .-> LEDGER

    subgraph COMPARE["Version comparison — stages 7 to 10"]
        direction LR
        PAIR["Two locked version graphs"] --> A7["A7<br/>Cluster"]
        A7 --> A8["A8<br/>Semantic match"]
        A8 --> A9["A9<br/>Set operations"]
        A9 --> A10["A10<br/>Comparison report"]
    end

    A5 --> PAIR
    A10 -. candidates only .-> TEMPORAL["Independent temporal adjudication"]
```

Agents 7–10 compare two completed graphs through clustering, semantic matching, set operations, and visualization. Paper D can reuse these checkpoints and comparison candidates while retaining independent authority and adjudication.

### 7.1 Audited historical Fannie run

The clean neighboring checkout was inspected at commit `484afd4ff4994811bcbe11568a39f9498f12b2b9`. The current local artifacts predate that commit and therefore are not assumed to have been generated by it. Their observed lineage is:

- source PDF SHA-256 `7e399d961c41a49b6b305b996d9c18f4facd60601db1f129654e18dd8eb578cc`;
- Agent 1: 506 TOC-based chunks, 51 merges, median 658 words, and maximum 2,845 words;
- Agent 3: 394 candidates—290 rules nested under entity types and 104 nested under relationships;
- Agent 3.5: validation covers only the 290 entity-nested rules, samples at most 10 rules for source verification, checks only whether a reference is present, and is non-blocking in the orchestrator;
- Agent 4: 394 unique flattened rules;
- Agent 5: 384 rules after 10 removals and 339 candidate dependency edges—87 `explicit`, 105 `implicit`, 27 `inferred`, and 120 `workflow`;
- final graph SHA-256 `1497ce97387a9c0cf9fdefd5c851aa75cb64b3d0b1656bf430aa3147df6be0d4`;
- 394 source-reference objects, of which 64 have text-match score below 0.5;
- 27 rules explicitly marked `reference_verified=false`;
- 360 rules use placeholder or malformed jurisdictions such as `EXAMPLE_POLICIES`, `EXAMPLE_POLICICIES`, or `SAMPLE_GUIDELINES`;
- the source is the March 4, 2026 guide, not the later June 3 snapshot.

The repository's re-verification script is useful as a deterministic fuzzy-reference baseline, but its permissive match thresholds and fallback recovery from a rule description do not establish semantic correctness. Likewise, the historical output records some optimizer metadata but does not bind the complete original Git state, gitignored configuration, prompt hashes, CLI arguments, or all model revisions. It is therefore a descriptive baseline, not a reproducible primary extraction run.

### 7.2 Checkpoint roles in Paper D

| Upstream checkpoint | Reuse in Paper D | Scientific question | Boundary |
| --- | --- | --- | --- |
| A1 organized chunks | segmentation and raw-retrieval candidate units | Did chunking preserve governing spans and cross-references? | Chunk existence is not source fidelity |
| A2 ontology | candidate entity/relationship vocabulary | Does ontology coverage affect rule recall or routing? | Never controls the Paper D authority ontology |
| A3 rules | automatic flat-rule candidate set | What rule content was extracted before merge/optimization? | Score entity- and relationship-nested rules; do not inherit confidence as gold |
| A3.5 validation | advisory QA baseline | How much does shipped validation catch relative to fail-closed verification? | Never call it source verification; it is sampled, partial, and non-blocking |
| A4 merged graph | pre-optimization rule checkpoint | Does enrichment preserve IDs, source spans, conditions, and exceptions? | Enrichment is not independent verification |
| A5 optimized graph | automatic graph and dependency candidates | Do deduplication and typed edges improve utility without losing rule fidelity? | All removals and non-explicit edges require lineage and review |
| A7–A10 comparison | candidate version matches, additions, removals, and contradictions | Can the pipeline reduce expert work for policy-update discovery? | Comparison output proposes changes; it never defines temporal gold |
| Explorer/visualization | optional reviewer navigation and artifact inspection | Does tooling reduce review time? | Not used for blinded labels unless access controls preserve the protocol |

### 7.3 Import and lineage contract

For each controlled upstream run, Paper D records:

- upstream repository URL, commit, clean/dirty state, and license snapshot;
- source snapshot IDs and hashes;
- sanitized configuration, domain prompt, base prompt, and environment-lock hashes;
- exact command line, provider, model IDs/revisions, decoding settings, worker count, start/end time, and run log;
- A1, A2, A3, A3.5, A4, and A5 artifact paths and SHA-256 hashes;
- a total lineage map from every A3 rule through A4 and A5, including merge/removal reasons;
- raw upstream objects plus normalized Paper D objects; normalization errors cannot mutate the originals;
- local-only and redistribution decisions for every imported byte.

The adapter fails closed if any required manifest field, stage artifact, source hash, rule lineage, or schema conversion is unresolved. Paper D never writes into the neighboring repository.

### 7.4 Research enabled by the real pipeline

1. **Checkpoint fidelity:** score A3, A4, and A5 against the same independently created rule gold and identify the first checkpoint at which every critical-field error appears.
2. **Optimization audit:** adjudicate every removed rule and a stratified sample of dependencies by detection method; report exception loss, duplicate-removal precision, edge precision, and downstream decision change.
3. **Validation-gap study:** compare shipped A3.5 warnings and fuzzy re-verification with Paper D deterministic checks and expert judgments. This is a diagnostic comparison, not a criticism benchmark built from the pipeline's own labels.
4. **Natural-error amplification:** for independently confirmed errors already present in P2K outputs, compare downstream decisions using the erroneous and corrected packet. Report this as secondary observational evidence because naturally occurring errors are selected rather than randomized.
5. **Temporal candidate efficiency:** run Agents 7–10 over locked March and June graphs, then measure candidate-change precision/recall and reviewer time against an independently authored change ledger.
6. **Prompt-pack transfer, optional:** on a locked extraction subset, hold source, model, decoding, and P2K commit fixed while comparing base prompts with the mortgage override pack. A second-domain prompt-pack comparison belongs only in the full version after that domain receives its own authoritative archive, gold, rights review, and qualified adjudicators.

One promising seed rule is `BR_MORTGAGE_FILE_CONSTRAINT_008_004`, the four-month credit-document age rule. It has a strong source match and naturally yields:

- `ALLOW`: all decisive dates are present and the newest document is within four months;
- `INTERVENE`: the newest required document is older than four months and no applicable exception resolves it;
- `REVIEW`: the note date, newest-document date, or disaster-exception applicability is missing.

This rule and all other P2K candidates may seed a reviewer queue, but final cases and labels must be reconstructed from the locked source snapshot by reviewers blind to the automatic graph's proposed action.

---

## 8. Atomic rule and graph schemas

### 8.1 Atomic rule object

Each rule must be small enough that one proposed action can satisfy, violate, or remain unresolved under it.

```json
{
  "rule_id": "...",
  "snapshot_id": "...",
  "authority_id": "...",
  "authority_class": "...",
  "section_id": "...",
  "rule_kind": "requirement | prohibition | permission | exception | definition",
  "actor": "...",
  "regulated_action": "...",
  "trigger": "...",
  "conditions": [],
  "conclusion": "...",
  "exceptions": [],
  "required_facts": [],
  "jurisdiction": "...",
  "covered_products": [],
  "effective_from": "...",
  "effective_through": null,
  "source_span": {
    "page": 0,
    "start_char": 0,
    "end_char": 0,
    "text_sha256": "..."
  },
  "extraction_system_id": "...",
  "candidate_confidence": null,
  "review_status": "candidate | verified | rejected",
  "reviewer_ids": [],
  "adjudicator_id": null
}
```

### 8.2 Edge object

Allowed edge types are deliberately narrow:

```text
REQUIRES
EXCEPTS
OVERRIDES
SUPERSEDES
APPLIES_WHEN
DEFINES
CROSS_REFERENCES
```

Every edge records:

- source and target rule IDs;
- whether the relation is explicit in text or reviewer-inferred;
- source span(s) supporting the edge;
- direction;
- policy vintage;
- reviewer status;
- rationale restricted to the relation, not a new rule.

“Contradictory” is not a final edge type. Apparent contradiction must resolve into applicability, exception, override, supersession, or an unresolved conflict that triggers `REVIEW`.

### 8.3 Executable shadow representation

For eligible rules, reviewers also create a restricted executable predicate used only as a benchmark oracle:

```text
applicable(case_facts) -> true | false | unknown
decision(case_facts)   -> ALLOW | INTERVENE | REVIEW
```

This is not generated from the candidate graph without review. It enables exact consistency checks, matched-triplet construction, and controlled mutation. Rules too interpretive for a faithful predicate remain in the qualitative `REVIEW` set and are excluded from exact-execution claims.

---

## 9. Annotation and adjudication design

### 9.1 Human roles

At minimum:

- **two independent rule reviewers** for every claim-bearing rule and edge;
- **two independent case reviewers** for every claim-bearing case;
- **one separate adjudicator** for disagreement;
- a **legal-authority reviewer** for federal-regulation scope;
- a **mortgage-domain reviewer** for Fannie seller-policy scope.

The same individual may not serve as both sole rule author and sole case adjudicator for a family.

### 9.2 Avoiding circularity

The most important validity rule is that graph output must not define its own gold.

Use three independently staged packets:

1. **Source-to-rule packet:** reviewers see source text and schema, but no generated cases or guard outputs.
2. **Rule-to-case packet:** case authors see locked verified rules and source spans, but no candidate extraction identity or evaluated guard output.
3. **Case adjudication packet:** reviewers see the source snapshot, case facts, and proposed action; they are blind to representation arm, extractor, guard, and planned label.

Cases generated from an automatic graph may enter a candidate pool, but the final label must be independently reconstructed from source-authoritative material.

### 9.3 Rule-review labels

Reviewers annotate:

- rule present / not present;
- atomicity;
- exact source support;
- actor/action/trigger correctness;
- condition completeness;
- exception completeness;
- numeric/date exactness;
- authority and jurisdiction;
- effective-date correctness;
- required facts;
- relation support;
- ambiguity requiring review.

### 9.4 Agreement and gate

Report raw agreement and chance-corrected agreement where meaningful. Span agreement uses token/span overlap rather than kappa. Do not collapse all fields into one “confidence” score.

Provisional admission gates, to be frozen after a small annotation pilot:

- 100% exact agreement after adjudication on numeric thresholds, dates, authority class, and source snapshot;
- at least 95% of admitted rules have a directly supporting source span;
- no unresolved critical disagreement on actor, trigger, conclusion, or exception;
- categorical agreement target `kappa >= 0.70` before scaling annotation;
- relation precision target at least 0.90 on the reviewed sample;
- any failed critical field rejects the rule from the decision benchmark, even if aggregate extraction F1 is high.

The thresholds are design candidates, not post-result acceptance criteria. Freeze them before the primary extraction run.

---

## 10. Benchmark construction

### 10.1 Matched decision families

Each core family contains minimally changed cases:

1. `ALLOW`: all required conditions are met;
2. `INTERVENE`: one decisive condition is violated;
3. `REVIEW`: the decisive fact or policy relation is intentionally unresolved.

Additional variants may change role voice or surface form, but must preserve the logical case. Every row records both `family_id` and `content_family_id`; split isolation uses the transitive closure of those identifiers.

### 10.2 Required family types

The primary benchmark must include:

- atomic direct rules;
- numeric/date thresholds;
- exception-sensitive rules;
- explicit two-rule compositions;
- scope/jurisdiction routing;
- supersession or policy-update cases;
- benign but compliance-sounding cases to test over-intervention;
- subtle violations expressed in ordinary professional language;
- missing-fact cases where `REVIEW` is objectively necessary.

### 10.3 Provisional scale

Final counts are frozen only after pilot variance and reviewer throughput are measured. A viable target is:

#### Annotation and extraction pilot

- 24 source sections;
- approximately 60 independently annotated atomic rules;
- 30 decision families / 90 matched-triplet rows;
- 150–300 additional `ALLOW` rows for early specificity and calibration checks.

#### Primary development and retrospective panel

- 200 independent decision families:
  - 120 single-rule families;
  - 40 exception/composition families;
  - 20 authority/applicability families;
  - 20 temporal-update families;
- 600 matched-triplet rows;
- approximately 800 naturalistic `ALLOW` specificity rows;
- total target approximately 1,400 rows before any sealed cohort.

#### Sealed cohort

- at least 50 separately authored decision families / 150 triplet rows;
- at least 400 separately authored `ALLOW` rows;
- no reused templates, fact sheets, source spans, or case authorship packets from primary development;
- final count derived from the pilot’s family-clustered variance and desired false-intervention precision.

These are planning numbers, not a license to run. Power and precision gates may increase them or narrow the claims.

### 10.4 Capacity mixture

Balanced triplets measure construct behavior, not deployment prevalence. A separate capacity mixture tests operational triage under a prespecified synthetic mixture, initially:

```text
ALLOW       94%
REVIEW       5%
INTERVENE    1%
```

This is not a claimed real-world prevalence estimate. It exists only to measure false interventions and review-queue capacity. The review budget begins at 10% and is frozen before test scoring.

### 10.5 Data privacy

- use synthetic or banded/de-identified mortgage fact sheets;
- never reproduce a real borrower row or precise identifying combination;
- run exact-match and near-duplicate checks against source records;
- do not expose protected attributes except in prespecified fairness or anti-discrimination tests;
- record whether each case contains sensitive text and whether release is permitted.

### 10.6 Split isolation

Split by connected components over:

```text
family_id
content_family_id
source_span_family
template_id
policy_change_family
```

The same governing rule may appear in calibration and evaluation only if the claim is in-rule application. A held-out-rule claim requires complete rule-family isolation and is reported separately. Temporal tests must isolate both case family and post-update policy material from pre-update development.

### 10.7 External benchmark protocol

External datasets answer only whether the inspected guards retain policy-conditioned behavior outside the mortgage construct. They do not enter Paper D training, prompt selection, calibration, threshold setting, mortgage case authoring, or primary inference.

Freeze the external panel at P6:

1. **Required reasoning control:** SafePyramid v1.1, scored on its native violated-rule-set task over the complete official test release.
2. **Required compact-guard control:** DynaBench's official 543-row handcrafted test subset, scored on native PASS/FAIL.
3. **Optional breadth control:** the official PolyGuard finance and regulation evaluation subsets, selected by upstream split names before any Paper D model outputs are inspected.
4. **Optional trajectory control:** PolicyGuardBench only if the full-trajectory adapter can be frozen without changing the underlying policy/trajectory text.
5. **Descriptive-only gated control:** existing ExpGuard results remain a separate evidence tier; no new claim depends on access.

For each external benchmark:

- use the upstream native task and metric first;
- map to Paper D actions only in a separately labeled compatibility analysis;
- freeze dataset revision, split, adapter, prompt, tokenizer, and parser before scoring;
- report every source separately and never macro-average it with mortgage results;
- disclose synthetic versus expert labels, real versus fictional policies, and any refusal/parse exclusions;
- do not fine-tune on an evaluation split or use public leaderboard results to select the Paper D primary model panel;
- publish only predictions, IDs, and derived aggregates when verbatim redistribution is not affirmatively allowed.

FinSafeGuard, Zillow's gated full data, FinGuard-Bench, MortgageGuardBench-2K, and `v1_hmda2022` are not required external-panel inputs. Adding one requires a pre-run protocol amendment documenting access, licensing, construct, and whether the use is training, development, or evaluation.

---

## 11. Extraction experiment

### 11.1 Systems

At minimum, evaluate:

- `E0_p2k_historical_a5`: the preserved local Agent 5 graph, imported byte-for-byte with its incomplete historical provenance and used descriptively;
- `E1_single_pass`: one schema-constrained extraction pass over the same locked source bundle;
- `E2_p2k_controlled_a3`: all Agent 3 entity- and relationship-nested candidates from a clean, pinned controlled run;
- `E3_p2k_controlled_a4`: the corresponding Agent 4 merged output from that same run;
- `E4_p2k_controlled_a5`: the corresponding Agent 5 optimized nodes and candidate dependencies;
- `E5_p2k_verified`: `E4` after Paper D's deterministic gates, separate verifier, blinded human acceptance/rejection, and explicit-edge adjudication; and
- `E6_human_gold`: independently adjudicated rule and edge objects, created without access to candidate system outputs during first-pass annotation.

The primary P2K checkpoint comparison uses one upstream run and preserves total A3→A4→A5 lineage. `E0_p2k_historical_a5` is not substituted when a controlled run fails. The exact extractor models and revisions are selected before the protocol lock. A frontier model may be used offline; Paper D’s “small” claim applies to the online guard unless the extraction experiment explicitly studies model scale.

### 11.2 Deterministic verification

Before model-based verification:

- source span must exist in the locked bytes;
- quoted text hash must match;
- every numeric literal/date in the rule must map to source text or be explicitly marked normalized;
- section and page coordinates must resolve;
- authority and snapshot IDs must exist;
- effective ranges must be internally coherent;
- referenced rule IDs must exist in the same or declared external snapshot;
- graph must be acyclic only for edge types that logically require acyclicity;
- no placeholder jurisdiction or example authority may pass;
- duplicate/near-duplicate nodes must be surfaced, not silently deleted.

### 11.3 Checkpoint attribution

For every controlled P2K run, report:

- A1 governing-span retention and cross-reference boundary errors;
- A2 task-relevant entity/relationship coverage and unsupported ontology elements;
- A3 rule recall separately for entity-nested and relationship-nested outputs;
- A3→A4 ID, source-span, condition, exception, numeric, date, and authority preservation;
- A4→A5 survival, merge, and removal lineage with an adjudicated reason for every removed rule;
- duplicate-removal precision and material-condition/exception retention;
- dependency-edge precision and recall by edge type and `explicit`/`implicit`/`inferred`/`workflow` detection method;
- A3.5 coverage, warning precision/recall, and the fraction of critical defects it cannot inspect;
- deterministic fuzzy-reference-baseline precision/recall at its shipped thresholds;
- Paper D verifier acceptance, rejection, and unresolved-review rates; and
- reviewer minutes and model/API cost added by each checkpoint.

Checkpoint differences are paired over stable rule lineage. `E4-E3` is an optimizer **bundle** because it changes both node inventory and edges; it is not called the isolated value of graph structure. The isolated structure effect remains `R3_auto_graph-R2_auto_flat`, which uses the same A5 node set and token budget.

### 11.4 Extraction metrics

Report by rule type and complexity:

- rule detection precision, recall, and F1;
- strict and relaxed source-span F1;
- actor/action/trigger exact match;
- condition and exception set F1;
- numeric and date exact accuracy;
- authority/jurisdiction accuracy;
- required-fact set F1;
- explicit-edge precision/recall;
- duplicate rate;
- unsupported-rule rate;
- calibration and risk–coverage curve for candidate confidence;
- reviewer minutes per accepted rule;
- extraction cost per accepted rule.

Primary extraction success is not macro-F1 alone. Critical-field error rates are reported separately and can fail the graph-admission gate.

---

## 12. Retrieval and policy-representation arms

Every arm receives the same case and maximum policy-token budget. Retrieval is deterministic after the retrieval lock.

**Identifier namespaces.** Four prefixes are used throughout and must not be mixed, because a
bare number is otherwise ambiguous between a design arm and an execution phase:

| Prefix | Names | Defined in |
| --- | --- | --- |
| `R0`–`R6` | policy-representation / retrieval arms | this section |
| `C0`–`C3` | controlled packet twins (canonical, silent, visible) | Section 12.1 |
| `E0`–`E6` | extraction systems and checkpoints | Section 11.1 |
| `P0`–`P9` | sequencing and lock-chain phases | Section 18 |

Always write an arm with its full suffix (`R3_auto_graph`, not `R3`). The arms were originally
numbered `P0`–`P6`, which collided with the `P0`–`P9` phases; `P<n>` now means a phase only.

| Arm | Policy material | Purpose |
| --- | --- | --- |
| `R0_none` | No policy context | Measures model prior / contamination baseline |
| `R1_raw_rag` | Retrieved authoritative passages | Practical text-retrieval baseline |
| `R2_auto_flat` | Retrieved automatic rule nodes without edges | Extraction value without graph structure |
| `R3_auto_graph` | Same automatic nodes plus typed local edges | Automatic structure effect |
| `R4_verified_flat` | Retrieved source-verified nodes without edges | Separates verification from structure |
| `R5_verified_graph` | Verified nodes plus approved edges and dates | Proposed deployment representation |
| `R6_oracle_packet` | Exact governing rules selected independently | Upper bound on retrieval/representation |

These seven arms characterize the realistic pipeline. They do **not** by themselves identify why `R5_verified_graph` differs from `R1_raw_rag`.

`R2_auto_flat` and `R3_auto_graph` are both derived from `E4_p2k_controlled_a5`. They use the identical retrieved A5 nodes; only the eligible typed-edge serialization is added in `R3_auto_graph`. The A4-versus-A5 comparison belongs to the checkpoint/optimizer study in Section 11 and is never substituted for this controlled structure contrast.

### 12.1 Controlled packet-intervention panel

For the primary scientific experiment, construct four packet twins from the same independently selected governing nodes:

| Arm | Packet | What it identifies |
| --- | --- | --- |
| `C0_canonical_flat` | Verified facts serialized without typed edges | Controlled reference |
| `C1_canonical_graph` | Same facts/nodes plus approved typed edges | Structure effect |
| `C2_silent_defect` | `C1` with exactly one hidden semantic corruption | Downstream defect amplification |
| `C3_visible_defect` | Same corruption plus evidence that exposes the inconsistency | Integrity-detection / review behavior |

Each `C2`/`C3` pair shares a mutation ID, canonical twin, case facts, surface form, and packet length envelope. Where adding evidence consumes tokens, the canonical and silent twins receive equally long non-decisive provenance material. Mutations are generated from structured operators and independently checked; they are never produced freely by the guard model.

### 12.2 Matching requirements

- same source snapshot and `policy_as_of` per case;
- same retrieval query fields;
- same maximum policy tokens;
- same guard prompt and output schema;
- no gold rule ID or action label in non-oracle retrieval queries;
- flat/graph comparisons use the same retrieved node set when possible;
- edge serialization tokens count toward the budget;
- raw retrieval may not receive more candidate passages than can fit in the budget;
- retrieval results are hash-bound before guard scoring.
- canonical flat and graph packets contain the same policy propositions and rule nodes;
- mutation twins differ only in the locked defect or its evidence visibility;
- visible-defect evidence is sufficient for blinded reviewers to assign the intended integrity reason code;
- mutation text cannot appear in training, ordinary retrieval indexes, or authoritative releases.

### 12.3 Retrieval metrics

- governing-rule recall at `k` and at token budget;
- exception and prerequisite coverage;
- irrelevant-context fraction;
- stale-snapshot retrieval rate;
- evidence-span recall;
- retrieval latency and index size.

If governing-rule recall is poor, downstream results must be decomposed into retrieval-success and retrieval-failure strata. The guard cannot be blamed for a rule it never received.

---

## 13. Compact-guard panel and scoring

### 13.1 Primary fixed panel

Reuse the repository’s four pinned compact instruction checkpoints as the **generic compact panel**:

- `Qwen/Qwen2.5-1.5B-Instruct`;
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`;
- `HuggingFaceTB/SmolLM3-3B`;
- `Qwen/Qwen3-4B`.

Add two **required policy-configurable guard baselines** if their exact released checkpoints and licenses pass P0/P1:

- `DynaGuard-1.7B`;
- `DynaGuard-4B`.

DynaGuard is a direct baseline because it is trained to evaluate arbitrary supplied policies and already reports compact 1.7B and 4B variants. A base/instruction checkpoint matched to each DynaGuard backbone should be retained when available so policy-guard training is not conflated with model family.

Pin exact revisions, tokenizer revisions, licenses, chat templates, verdict tokenization, framework versions, and prompt-token caches in Paper D. Existing repository revisions are candidates, not automatically inherited protocol choices. If a purpose-built model cannot express three actions, preregister the mapping or restrict it to binary `ALLOW`/`INTERVENE` and provenance-defect detection subanalyses; do not invent a post hoc `REVIEW` mapping.

Optional descriptive controls:

- one larger policy-reasoning guard such as `gpt-oss-safeguard-20b`, subject to a pinned release and feasible hardware;
- one larger general instruction-model upper bound;
- a deterministic executable-rule oracle.

Purpose-built and larger controls are reported separately from the equal-weight four-model generic panel unless a pre-run analysis defines a common estimand and compatible output contract.

### 13.2 Primary mode: context-conditioned, no task fine-tuning

The primary experiment should be zero-shot or prompt-conditioned with frozen weights. This isolates document representation from training treatment and avoids duplicating Paper C.

A later secondary experiment may distill verified policy decisions into an SLM, but it must use disjoint families and cannot redefine the primary result.

### 13.3 Structured output

The decoder emits a constrained object:

```json
{
  "action": "ALLOW | REVIEW | INTERVENE",
  "policy_ids": ["..."],
  "reason_code": "...",
  "decisive_facts": ["..."],
  "missing_facts": ["..."],
  "evidence_span_ids": ["..."],
  "short_explanation": "..."
}
```

Action scores come from model logits or a specified constrained scoring procedure. A generated numeric confidence is never treated as calibrated confidence.

### 13.4 Calibration

- fit action-temperature or equivalent calibration only on the calibration split;
- select thresholds only on calibration/capacity data;
- freeze the decision rule before development-test or sealed scoring;
- report calibration per model and representation arm;
- do not share a threshold across arms unless a locked analysis explicitly tests threshold transfer.

### 13.5 Integrity-gate system control

Score two system modes on the controlled mutation panel:

1. `guard_only`: the model receives the packet and provenance fields;
2. `integrity_gate_plus_guard`: deterministic hash, span, required-field, date, and contradiction checks route a failed packet to `REVIEW` before guard inference.

This is not a contest in which the model is expected to rediscover deterministic checks. It quantifies which visible failures require a fail-closed compiler/runtime gate and which remain semantic reasoning problems.

---

## 14. Primary estimands and metrics

### 14.1 Primary scientific estimand: silent-defect amplification

```text
A_silent(d) =
    Pr(predicted ALLOW | C2_silent_defect=d, canonical gold=INTERVENE)
  - Pr(predicted ALLOW | C1_canonical_graph, canonical gold=INTERVENE)
```

Estimate `A_silent(d)` for each prespecified critical defect class and an equal-weight aggregate across defect classes. The primary population is family-balanced canonical-`INTERVENE` cases so the estimand has an unambiguous unsafe direction. Canonical `ALLOW` and `REVIEW` families remain required secondary strata for over-intervention and unresolved-case analysis.

This estimand measures how much a silent upstream defect changes unsafe downstream behavior. It does **not** claim that the guard should have detected hidden corruption.

### 14.2 Co-primary system estimand: visible-integrity review

```text
D_visible(d) =
    Pr(predicted REVIEW | C3_visible_defect=d)
  - Pr(predicted REVIEW | C2_silent_defect=d)
```

The population is the mutation-eligible families that have a validated `C1`/`C2`/`C3` triplet for
defect class `d`, taken over **all** canonical gold actions rather than the canonical-`INTERVENE`
restriction used by `A_silent`. The restriction is deliberately different: `A_silent` needs an
unambiguous unsafe direction, whereas a visible integrity failure should route to `REVIEW`
regardless of what the canonical decision would have been. Report `D_visible` overall and
stratified by canonical action, because a lift concentrated on canonical-`ALLOW` families means
something different from one that also fires where the canonical decision was `INTERVENE`.

`D_visible` measures response to newly available integrity evidence. It does not by itself establish that the review reason is correct. Report it for `guard_only` and `integrity_gate_plus_guard`, together with:

- reason-correct `REVIEW` recall on `C3`;
- false-review rate on canonical `C1` and silent `C2` packets;
- review rate no greater than the frozen capacity budget; and
- the fraction of visible defects deterministically detectable before inference.

If multiplicity or available family count cannot support two co-primary endpoints, retain `A_silent` as primary and label `D_visible` key secondary before the design lock.

### 14.3 Controlled representation and deployment contrasts

```text
Delta_structure  = metric(C1_canonical_graph) - metric(C0_canonical_flat)
Delta_system     = metric(R5_verified_graph)  - metric(R1_raw_rag)
```

`Delta_structure` is the controlled structure estimand because the facts and nodes are identical. `Delta_system` is a deployment-oriented bundled contrast and must not be interpreted as the isolated causal effect of graphs or verification.

For constrained decision comparisons, the operating point uses:

- matched `ALLOW` false-intervention rate;
- review rate no greater than the frozen budget; and
- identical policy-token budget.

### 14.4 Other secondary paired contrasts

```text
Delta_policy      = R1_raw_rag       - R0_none
Delta_auto_struct = R3_auto_graph     - R2_auto_flat
Delta_ver_struct  = R5_verified_graph - R4_verified_flat
Delta_verify      = R5_verified_graph - R3_auto_graph
Delta_oracle_gap  = R6_oracle_packet  - R5_verified_graph
```

These respectively characterize policy availability, practical structure/verification changes, and remaining retrieval/reasoning headroom. Only contrasts with matched semantic content receive a structure-only interpretation.

### 14.5 Decision metrics

- three-class macro-F1 and balanced accuracy;
- per-action precision and recall;
- `ALLOW` false-intervention rate;
- violation miss rate;
- semantic `REVIEW` recall and precision;
- review-budget utilization;
- `REVIEW` reason-code accuracy and reason-correct recall;
- silent-defect unsafe amplification by mutation class;
- visible-defect detection lift and deterministic-gate coverage;
- Brier score and multiclass NLL;
- exact policy-ID accuracy;
- governing-rule recall among cited rules;
- unsupported-citation rate;
- missing-fact exact/set F1;
- evidence-span precision/recall;
- minimal-pair consistency;
- latency, peak memory, tokens, and estimated cost.

### 14.6 End-to-end exactness

A strict end-to-end success requires all of:

```text
correct action
correct governing authority class
at least one correct governing policy ID
no unsupported policy ID
correct decisive or missing fact
source-supported evidence span
```

Report this strict metric alongside decomposed metrics; do not use it alone.

### 14.7 Human-review utility

Define a prespecified cost table for:

- missed `INTERVENE`;
- false `INTERVENE` on `ALLOW`;
- unnecessary `REVIEW`;
- failure to `REVIEW` an unresolved case.

Because costs are normative, report a sensitivity surface over plausible cost ratios rather than one universal utility number.

---

## 15. Controlled error-propagation study

This is the primary scientific component and must be preregistered.

Starting from an independently verified canonical graph packet, inject exactly one defect:

1. alter a numeric threshold;
2. alter an effective date;
3. drop an exception;
4. drop a prerequisite edge;
5. reverse an edge direction;
6. assign the wrong authority or jurisdiction;
7. retain a superseded rule as current;
8. add a plausible but unsupported rule;
9. rename/reformat without semantic change as a negative control; or
10. change a non-governing relation as a semantic-negative control.

Removing a decisive **case fact** is not a policy-extraction defect. It belongs in the independent missing-information panel for `MISSING_CASE_FACT` and must not be pooled into `A_silent`.

For every semantic mutation, construct:

- `C1`: the canonical graph packet;
- `C2`: a silent, internally coherent corruption with contrary source evidence withheld; and
- `C3`: the same corruption with sufficient source/provenance evidence to make the inconsistency observable.

Examples:

| Mutation | Silent `C2` | Provenance-visible `C3` |
| --- | --- | --- |
| Threshold | Rule node says 45%; no source span is exposed | Rule node says 45%, cited source span says 50% |
| Effective date | Stale rule is labeled current without contradictory metadata | Packet date or supersession record conflicts with `policy_as_of` |
| Dropped exception | Main rule appears complete | Source span includes an omitted exception or unresolved cross-reference |
| Wrong authority | Rule is internally labeled federal | Snapshot/source identity shows it is GSE or guidance material |
| Missing prerequisite edge | Packet looks like a standalone rule | Supplied cross-reference or edge manifest proves a required dependency is absent |

A visible corruption receives gold `REVIEW/POLICY_INTEGRITY_FAILURE`, `REVIEW/STALE_POLICY`, or `REVIEW/AUTHORITY_UNRESOLVED` as adjudicated. A silent corruption retains the authoritative canonical decision for downstream harm scoring, but its lack of `REVIEW` is not counted as failed corruption detection.

For every mutation, retain the unmutated twin. Measure:

- action flip rate;
- unsafe flip rate (`ALLOW` after a mutation that should block or review);
- appropriate `REVIEW` rate;
- citation movement;
- change in calibrated action score;
- sensitivity by model size and rule-complexity stratum.

Before admission, two reviewers must verify that:

1. the canonical packet is correct and sufficient;
2. exactly one intended semantic variable changed;
3. the visible evidence is sufficient to identify the intended inconsistency;
4. the mutation did not introduce an unintended lexical label cue;
5. the authoritative gold action and unsafe direction are explicit; and
6. negative controls do not change the authoritative decision.

This turns “error propagation” into a paired intervention rather than a correlation between messy extraction scores and downstream accuracy.

The mutations are diagnostic corruptions, never policy facts. They must carry a conspicuous synthetic namespace, be inaccessible to ordinary training and retrieval indexes, and never be released as authoritative policy artifacts. Released mutations must include their canonical parent ID and synthetic-warning metadata.

---

## 16. Temporal evaluation

### 16.1 Version pairs

Construct versioned families only where reviewers confirm a material rule change between locked snapshots. Each family records:

- pre-change rule and snapshot;
- post-change rule and snapshot;
- announcement/amendment linking the change;
- publication, effective, and transition dates;
- unchanged facts;
- the exact fact or action whose disposition changes;
- whether the correct stale-context response is `REVIEW` or an updated decision.

Policy to Knowledge is run independently on both snapshots with the same pinned commit and locked configuration. Agents 7–10 then generate candidate matches, additions, removals, and contradictions. Reviewers who authored the temporal gold do not see these candidates until the change ledger is locked. Candidate precision, recall, reviewer time saved, and false-change burden are reported separately from guard temporal robustness.

### 16.2 Temporal arms

- current case + current policy context;
- current case + stale context without warning;
- current case + stale context with explicit snapshot metadata;
- pre-change case + pre-change context;
- date-shuffled negative control;
- updated text with version metadata removed.

### 16.3 Temporal metrics

- stale-policy detection rate;
- stale-rule application rate;
- correct pre/post decision rate;
- inappropriate review rate on unchanged rules;
- effective-date accuracy;
- implicit update adaptation versus lexical cue reliance.

Do not call a case temporal merely because it carries a date. It must involve a verified change capable of affecting the governing decision or the need for review.

Temporal robustness is a challenge stratum and one provenance-defect class, not a standalone novelty claim. The differentiator is the binding between source bytes, effective dates, supersession lineage, packet evidence, and final triage.

---

## 17. Statistical design

### 17.1 Independent unit

Rows within a matched family are dependent. Families sharing a source section or policy-change lineage may also be dependent. The primary resampling unit is therefore the connected policy family, not individual rows.

### 17.2 Fixed-panel inference

The four generic compact models are an inspected fixed panel. Aggregate them with equal weight and report every model separately. DynaGuard-class baselines are a separate purpose-built panel because their training and output contract differ. Intervals describe uncertainty over benchmark families conditional on these inspected models; they do not support a population claim over all SLMs or policy guards.

### 17.3 Intervals

- paired, family-clustered bootstrap for arm contrasts;
- stratified resampling by rule-complexity and authority class;
- two-sided 95% intervals for retrospective/development results;
- simultaneous one-sided bounds only for a separately locked sealed confirmation;
- Wilson or exact intervals for low-rate false-intervention outcomes;
- multiple-comparison control for the prespecified secondary contrast family.

### 17.4 Power and precision gate

Before the primary run:

1. use pilot family-level paired differences to estimate variance;
2. declare the smallest meaningful `A_silent` and `D_visible` effects;
3. simulate power with family clustering, mutation-class balance, model dependence, and multiplicity;
4. compute the required distinct policy-family count, not merely mutation rows;
5. compute the required canonical `ALLOW` count for false-intervention and false-review precision;
6. increase the corpus, reduce the number of confirmatory defect classes, demote `D_visible`, narrow the claim, or stop if reviewer capacity cannot meet the requirement.

Provisional targets:

- resolve approximately a 5–7 percentage-point family-paired change in silent-defect unsafe `ALLOW` rate for the confirmatory aggregate;
- estimate reason-correct `REVIEW` recall for visible defects within approximately +/-5 percentage points if retained as co-primary;
- estimate an `ALLOW` false-intervention rate near 5% to approximately +/-2 percentage points in the sealed cohort.

These targets are planning values only. They must be replaced by executable simulation and precision calculations before the design lock. Mutation variants from one source family never count as independent observations.

### 17.5 Missing and failed cells

- failed extraction or guard cells remain in the run ledger;
- no silent reruns with replacement seeds/models;
- parser failures are outcomes, not automatically dropped rows;
- infeasible operating points are reported explicitly;
- a representation arm that cannot meet the review/FPR constraints is `infeasible`, not assigned zero performance.

---

## 18. Experimental sequencing and lock chain

```mermaid
flowchart TB
    subgraph DESIGN["Design and construction"]
        direction LR
        P0["P0<br/>Scope + literature"] --> P1["P1<br/>Sources + rights"]
        P1 --> P2["P2<br/>Schemas + annotation pilot"]
        P2 --> P3["P3<br/>Extraction benchmark lock"]
        P3 --> P4["P4<br/>Candidate + verified graphs"]
    end

    subgraph EXECUTION["Experiment and release"]
        direction LR
        P5["P5<br/>Decision benchmark pilot"] --> P6["P6<br/>Power + design lock"]
        P6 --> P7["P7<br/>Primary retrospective run"]
        P7 --> P8["P8<br/>Sealed confirmation"]
        P8 --> P9["P9<br/>Release + manuscript"]
    end

    P4 -->|"all construction gates pass"| P5
    STOP["Every arrow is an exit gate;<br/>failure means revise, narrow, or stop"] -. governs .-> DESIGN
    STOP -. governs .-> EXECUTION
```

### P0 — Scope and literature lock

Deliver:

- final research questions and non-claims;
- reproducible adjacent-work search ledger with databases, exact queries, date range, inclusion/exclusion rules, deduplication, and backward/forward citation chasing;
- closest-work matrix covering policy extraction, policy-grounded guard benchmarks, policy-configurable guards, provenance-based action guarding, deterministic action gates, neuro-symbolic compliance, KG-RAG, temporal shift, and abstention;
- open-data inventory distinguishing authoritative sources, realism data, openly licensed external benchmarks, gated resources, and local-only datasets;
- Policy to Knowledge integration boundary, pinned-upstream policy, checkpoint estimands, and explicit non-claim that the upstream graph is verified or authoritative;
- authority classes;
- mortgage-first scope decision;
- title-narrowing rule and submission-facing title;
- prospective claim registry;
- a `studies/registry.yaml` entry with `claim_authorization: false` (Section 19.1), plus regenerated
  `studies/README.md` and `papers/README.md`;
- a per-study `.gitignore` covering `inputs/*` and `artifacts/*` before any import runs.

Exit gate: no unsupported “first” claim, no broad end-to-end/KG/error-propagation novelty claim, and no cross-domain claim without a second validated domain. Any paper closer than the current matrix triggers a written overlap decision before data construction continues. `make check-registry` and `make check-links` pass with Paper D registered.

### P1 — Source and rights lock

Deliver:

- immutable source archive;
- SHA-256 manifest;
- URL/retrieval/effective-date ledger;
- authority hierarchy;
- redistribution decisions;
- March/June Fannie version lineage;
- official federal-regulation snapshots;
- pinned HMDA snapshot and CFPB complaint export, if used, with privacy-preserving derivation rules;
- external benchmark registry with SafePyramid, DynaBench, PolyGuard, PolicyGuardBench, FinSafeGuard, Zillow, FinGuard-Bench, ExpGuard, and local mortgage entries;
- a `benchmarks/registry/distribution.yaml` entry for every Fannie and federal snapshot and every external benchmark, so the repository's own fail-closed ledger and CI distribution gate govern them rather than a Paper D-local list;
- upstream license snapshots, attribution text, immutable revisions, hashes, and redistribution decisions;
- Policy to Knowledge code-license snapshot plus separate rights decisions for source documents, intermediate chunks, extracted text, graph objects, and released predictions.

Exit gate: every source byte and legal/policy role resolves; no realism source supplies gold; every external benchmark has a native-task contract; unresolved redistribution defaults to local-only.

### P2 — Schema and annotation pilot

Deliver:

- rule, edge, snapshot, event, and output schemas;
- P2K run-manifest, raw-import, normalized-object, and stage-lineage schemas with fail-closed adapter fixtures;
- reviewer manual;
- 24-section / approximately 60-rule pilot;
- agreement report;
- revised ontology and critical-field gates.

Exit gate: acceptable reviewer agreement and no schema field that reviewers cannot apply consistently.

### P3 — Extraction benchmark lock

Deliver:

- independently adjudicated extraction gold;
- fixed section split;
- extractor prompts/configurations/revisions;
- deterministic verifier tests;
- extraction metric code;
- preserved `E0_p2k_historical_a5` baseline plus hashes for all available historical A1–A5 artifacts;
- a provenance-completeness report that labels unrecoverable historical configuration fields rather than guessing them;
- pinned P2K commit, clean-checkout procedure, sanitized configuration template, prompt manifest, exact CLI contract, and controlled-run manifest template;
- tests proving the adapter is read-only and preserves raw upstream bytes.

Exit gate: gold is independent of candidate extraction; every critical field is source traceable; historical and controlled P2K provenance classes cannot be confused; and no primary run begins until the adapter round-trips fixture IDs, spans, and hashes.

### P4 — Candidate and verified graphs

Deliver:

- `E1_single_pass` and controlled `E2_p2k_controlled_a3`, `E3_p2k_controlled_a4`, and `E4_p2k_controlled_a5` outputs;
- complete A3→A4→A5 rule-lineage and deduplication ledgers;
- separate A3.5 and fuzzy-reference-baseline outputs, explicitly marked advisory;
- checkpoint fidelity, preservation, duplicate-removal, and dependency-edge audit tables;
- blinded review packets;
- accepted/rejected node ledger;
- explicit/reviewer-approved edge ledger;
- executable shadow predicates where valid;
- `E5_p2k_verified` graph version and hash.

Exit gate: all controlled stage hashes and lineage checks pass; every Agent 5 removal is adjudicated; verified graph meets critical-field gates; and rejected or unresolved nodes cannot reach benchmark construction or retrieval.

### P5 — Decision benchmark pilot

Deliver:

- 30 matched families;
- independent case adjudication;
- initial capacity `ALLOW` stream;
- privacy/duplicate checks;
- split-isolation validator;
- retrieval parity test;
- guard prompt and constrained-output smoke;
- at least two validated canonical/silent/visible mutation triplets per proposed critical defect class;
- blinded verification that visible defects support the intended `REVIEW` reason and silent defects do not leak corruption cues;
- `guard_only` and `integrity_gate_plus_guard` smoke results.

Exit gate: `REVIEW` cases name a locked reason code and decisive evidence; canonical/silent/visible twins differ only as specified; no policy packet leaks the planned action or corruption status.

### P6 — Power, protocol, and design lock

Deliver:

- pilot variance and precision estimates;
- frozen primary and sealed counts;
- `A_silent`, `D_visible`, representation, and deployment estimands with a final primary/secondary hierarchy;
- confirmatory mutation classes and negative controls;
- token/review/FPR budgets;
- exact model and tokenizer revisions;
- prompt-token caches;
- retrieval indexes and hashes;
- analysis code tested on synthetic fixtures;
- frozen P2K commit, source bundle, sanitized configuration, prompt hashes, environment, model revisions, CLI arguments, stage adapter, and checkpoint estimands;
- frozen external panel, native metrics, adapters, revisions, split names, and non-pooling rule;
- immutable design lock.

Exit gate: the proposed sample can resolve the smallest meaningful effect or the study narrows/stops.

### P7 — Primary corpus and retrospective experiment

Deliver:

- source- and family-isolated primary corpus;
- all policy-representation packets;
- complete four-model generic grid and required purpose-built baseline cells;
- calibration and operating-point locks;
- extraction, retrieval, decision, and cost artifacts;
- P2K A1/A3/A4/A5 checkpoint results, A3.5/fuzzy-verifier diagnostic results, and `E5` verification results;
- controlled defect-injection panel;
- temporal panel plus Agents 7–10 candidate-change precision/recall and reviewer-time analysis;
- complete SafePyramid and DynaBench external-control scores, or a documented pre-run incompatibility decision;
- optional PolyGuard/PolicyGuardBench scores only if their adapters were included in the P6 lock.

Exit gate: all expected cells present, hashes validate, and analysis regenerates without network or model inference from immutable scores.

### P8 — Sealed confirmation

Deliver:

- separately authored and adjudicated sealed families;
- unsealing record;
- one-pass scoring;
- simultaneous decision output against the claim registry.

Exit gate: no development access to sealed text, rule packets, or labels before all thresholds and code are locked.

### P9 — Release and manuscript

Deliver:

- LaTeX manuscript and PDF;
- dataset card and model/evaluation cards;
- an upstream-system card identifying the exact P2K code state, controlled-run manifests, checkpoint limitations, and historical-versus-controlled provenance;
- per-source external benchmark cards and license/attribution notices;
- source and rights ledger;
- text-free release option;
- reproduce-from-committed-scores command;
- limitations/evidence ledger;
- full negative and failed-cell reporting.

Exit gate: every manuscript number is generated from a locked artifact; release contains no unauthorized policy text, benchmark rows, or sensitive data.

---

## 19. Repository layout to create after plan approval

The layout follows the conventions the repository's existing study package already uses
(`papers/paper_c/specialize_then_align/`): a per-study `.gitignore`, `locks/` for lock artifacts,
`provenance/` for manifests, and `environment/` for the environment record that
`studies/registry.yaml` points at through its `environment_path` field.

```text
papers/paper_d/
  .gitignore                      # inputs/, artifacts/, build/, .env — see the note below
  PLAN.md
  README.md
  STATUS.md
  PROTOCOL.md
  CLAIMS.yaml
  RELATED_WORK.md
  literature_search/
    SEARCH_LEDGER.yaml
    SCREENING.csv
  Makefile
  config/
    study.json
    models.json
    sources.json
    external_panel.json
    p2k.json
  schemas/
    authority_snapshot.schema.json
    atomic_rule.schema.json
    graph_edge.schema.json
    guard_event.schema.json
    guard_output.schema.json
    policy_mutation.schema.json
    p2k_run_manifest.schema.json
    p2k_stage_lineage.schema.json
  source_registry/
    RIGHTS_LEDGER.yaml
    SNAPSHOT_MANIFEST.json
    REALISM_DATA.yaml
    EXTERNAL_BENCHMARKS.yaml
    UPSTREAM_SYSTEMS.yaml
  integrations/
    policy_to_knowledge/
      README.md
      PINNED_UPSTREAM.json
      fixtures/
  annotations/
    GUIDELINES.md
    templates/
  src/paper_d/
    archive.py
    import_p2k.py
    extract.py
    verify.py
    graph.py
    score_checkpoints.py
    score_version_candidates.py
    build_cases.py
    retrieve.py
    integrity_gate.py
    mutate_packets.py
    score_guards.py
    score_external.py
    analyze.py
    locks.py
  tests/
  manuscript/
  locks/                          # one immutable lock artifact per phase gate, P0-P9
    P0_scope.json
    P1_sources.json
    P3_extraction.json
    P6_design.json
    P8_seal.json
  provenance/
    UPSTREAM_P2K_MANIFEST.json    # commit, CLI, prompts, env, model revisions, stage hashes
    EXTERNAL_OBJECT_MANIFEST.json # large artifacts held outside the repository
    RIGHTS_DECISIONS.json
  environment/
    README.md
    requirements.txt              # or gpu-requirements.txt if a GPU tier is needed
  inputs/          # ignored or pointers to immutable external objects, including imported P2K runs
    .gitkeep
  artifacts/       # ignored until selectively released
    .gitkeep
  build/
```

The Paper D code should reject writes outside `papers/paper_d/` except for an explicit, read-only import step that copies and hashes approved external source artifacts into the isolated namespace.

**Ignore rules must be written so the placeholders survive.** Use `inputs/*` and `artifacts/*`
rather than `inputs/` and `artifacts/`, then re-include `.gitkeep`. Git cannot re-include a file
whose parent *directory* is excluded, so the directory-form pattern silently defeats
`!artifacts/.gitkeep` — a failure this repository has already recorded once, in
`studies/paper-c-specialize-align-mortgage-v1/provenance/MIGRATION_MANIFEST.json` under
`gitignore_asymmetry`. Because `papers/paper_d/inputs/` will hold Fannie Mae source bytes and
imported P2K artifacts, the ignore rules must exist **before** the first import, not after: the
repository has already had to purge two corpora from every commit because tracked bytes preceded
the decision that should have prevented them.

### 19.1 Registration in the repository's normative record

A Paper D directory is not enough. `studies/registry.yaml` is the repository's normative record of
every study's state, evidence tier, contract class, and verification command; `studies/README.md`
and `papers/README.md` are **generated from it**, `make check-registry` validates it and asserts
those indexes are current, and `make check-locks` runs each study's declared verification. A study
that exists only as files is invisible to all of that: absent from both indexes, outside every
verification tier, and unable to record that it may make no claims yet.

So P0 must add an entry, and the schema already carries the vocabulary this study needs — including
`verification_mode: authority_archive`, which no existing study uses:

```yaml
- study_id: paper_d_policy_to_guard_v1
  title: "From Policy Documents to Guardrail Decisions"
  question: >
    Between source document, extracted rule representation, retrieval layer, and final
    guard decision, where does reliability come from and where does it fail?
  study_state: protocol_candidate
  evidence_state: none
  publication_state: none
  distribution_state: unresolved
  test_exposure: none            # becomes check_fast once hermetic tests exist
  evidence_tier: "none; protocol only, no experiment has run"
  claim_authorization: false     # stays false until the P6 design lock and P8 seal
  contract_type: protocol_candidate
  verification_mode: authority_archive
  verification_command: "test -f papers/paper_d/PLAN.md"
  expected_verification_status: expected_pass
  code_root: papers/paper_d/
  protocol_paths: [papers/paper_d/PLAN.md, papers/paper_d/PROTOCOL.md]
  status_path: papers/paper_d/STATUS.md
  environment_path: papers/paper_d/environment/requirements.txt
  lock_bound_paths: false        # becomes true when locks/ holds its first lock
  notes: >
    Protocol only. Depends on the neighboring policy-to-knowledge repository through a
    read-only import; that dependency is recorded in provenance/UPSTREAM_P2K_MANIFEST.json
    and is not a claim that upstream output is verified or authoritative.
```

Three consequences follow, and each is an exit-gate condition rather than a suggestion:

- **`claim_authorization: false` is load-bearing.** It is what stops Paper D output being cited as
  evidence before the design lock, in the same way it currently constrains the stopped Paper C
  surfaces. Flip it only when the P8 seal is documented.
- **The distribution ledger needs the source entries too.** Every Fannie and federal snapshot goes
  in `benchmarks/registry/distribution.yaml` with its own per-source decision, because that ledger
  is what `make check-registry` and the CI distribution gate actually read. Section 6 models itself
  on that ledger; it must also *use* it.
- **Registry and indexes must stay in sync.** After editing the registry, run
  `python tools/render_indexes.py` and commit the regenerated `studies/README.md` and
  `papers/README.md`; `make check-registry` fails if they go stale.

---

## 20. Reproducibility and provenance contract

Every claim-bearing artifact binds:

- source snapshot and source-span hashes;
- upstream Policy to Knowledge repository URL, commit, clean/dirty state, code-license snapshot, exact CLI arguments, sanitized configuration hash, prompt hashes, environment lock, model revisions, and run-log hash;
- raw A1–A5 stage hashes, normalized-object hashes, and total A3→A4→A5 lineage, including removal/merge reasons;
- upstream dataset or benchmark ID, revision, split, file hash, license snapshot, and attribution record;
- realism-data snapshot, filters, sampling rule, transformation code, and derived-output hash when HMDA or complaint data are used;
- external-benchmark adapter version, native task definition, native metric implementation, and score-file hash;
- extractor/verifier model IDs and revisions;
- prompts and decoding configuration;
- parser and OCR versions;
- schema versions;
- reviewer and adjudicator identities or blinded stable IDs;
- graph hash;
- packet parent/mutation ID, mutation operator, visibility mode, and synthetic-warning flag;
- integrity-gate version and result;
- case-family and content-family identities;
- retrieval-index hash and retrieved packet hash;
- guard model/tokenizer revision;
- rendered prompt-token hash;
- calibration and threshold lock;
- raw action logits or scores;
- environment lock;
- analysis-code hash.

The reproduction target is not stochastic document extraction. The public command should regenerate tables and figures from committed, rights-safe structured scores and verify every covered artifact byte-for-byte, while clearly listing artifacts that require private source bytes or a locked environment.

---

## 21. Mandatory ablations and controls

### Representation controls

- token-matched raw passages;
- flat versus graph using the same rule nodes;
- canonical flat versus canonical graph using semantically identical propositions;
- graph with edges shuffled within type;
- graph without provenance spans;
- graph without effective dates;
- oracle rule packet.

### Policy to Knowledge checkpoint controls

- historical local Agent 5 artifact versus a controlled pinned rerun, never pooled as equivalent replicates;
- Agent 3 entity-nested versus relationship-nested rule coverage;
- Agent 3 versus Agent 4 content-preservation audit;
- Agent 4 versus Agent 5 optimizer bundle with every removal traced;
- Agent 5 nodes held fixed with edges absent versus present;
- explicit-only dependencies versus the full candidate edge set;
- base prompts versus the mortgage domain override pack on the same locked extraction subset, if included before design lock;
- shipped Agent 3.5 advisory validation versus fuzzy-reference re-verification versus Paper D fail-closed verification;
- Agents 7–10 temporal-change candidates versus independently locked change gold.

### Retrieval controls

- oracle retrieval;
- deterministic lexical retrieval;
- dense/hybrid retrieval if licensed and frozen;
- random but length-matched irrelevant context;
- current versus stale snapshot.

### Guard controls

- no-policy prior;
- deterministic executable predicate on eligible rules;
- one larger-model upper bound;
- action-only output versus structured auditable output;
- explicit `REVIEW` option removed, to measure what abstention contributes;
- `guard_only` versus deterministic `integrity_gate_plus_guard`;
- base instruction checkpoint versus its policy-guard-tuned counterpart where available;
- reason-code output versus unconstrained free-text explanation.

### Construction controls

- human-authored versus model-assisted surface wording on disjoint families;
- planned target versus independently adjudicated outcome;
- simple atomic versus exception/composition/temporal strata.

### Mutation and trust-boundary controls

- canonical, silent-corruption, and provenance-visible-corruption triplets;
- equal-length provenance filler where visibility evidence changes token count;
- formatting-only and non-governing-relation negative controls;
- source evidence removed from `C3` to recover the `C2` condition;
- mutation-type wording counterbalanced so “conflict,” “stale,” or “incorrect” cannot leak the label;
- deterministic detectability recorded separately from semantic model detectability;
- missing-case-fact interventions analyzed separately from policy-integrity mutations.

No ablation may be added after seeing the sealed result and then presented as prespecified.

---

## 22. Risk register and mitigations

| Risk | Consequence | Mitigation / decision rule |
| --- | --- | --- |
| Novelty collision with ProvenanceGuard, deterministic-gate work, PolicyGuard, ComplianceNLP, SafePyramid, FinGuard, or BREX | Paper appears incremental | Center the claim on canonical/silent/source-inconsistent corruption of policy context compiled from authoritative documents; rerun the search before design lock and submission |
| Verified graph versus raw RAG treated as causal | Structure, retrieval, selection, and fidelity remain confounded | Use it only as a deployment contrast; use identical-node `C1-C0` for structure and packet twins for defect attribution |
| Silent defect assigned gold `REVIEW` | Guard is penalized for unavailable evidence | Score silent defects against canonical safety outcome and amplification; reserve corruption-detection `REVIEW` for provenance-visible defects |
| Visible integrity failure left to the LLM | Deterministic failures consume model and reviewer capacity | Include fail-closed integrity-gate control and report deterministic coverage separately |
| Candidate graph treated as truth | Circular labels and inflated performance | Independent source-to-rule and case adjudication; graph is an experimental arm only |
| Historical P2K artifact attributed to the currently checked-out commit | False reproducibility | Label it `historical_local`; use it descriptively; require a clean pinned rerun with a complete manifest for primary checkpoint claims |
| Shipped P2K Agent 3.5 called source verification | Inflated assurance from a partial, sampled, non-blocking presence check | Report it as an advisory baseline; score its coverage and errors; use independent fail-closed checks and adjudication for `E5` |
| Relationship-nested P2K rules escape validation | 104 observed candidates bypass the advisory validator | Import and score both nesting locations; require total rule lineage and whole-inventory verification |
| P2K optimizer bundle treated as an isolated graph effect | Deduplication, node inventory, and edges change together | Label A4→A5 as an optimizer bundle; estimate structure only with identical A5 nodes and edge-off/edge-on packets |
| Paper D mutates the sibling checkout or reads moving outputs | Cross-repository drift and irreproducible results | One-way read-only import, pinned commits, clean-state assertion, copied stage hashes, and no shared writable artifacts |
| P2K MIT code license assumed to cover policy text or graph output | Unauthorized release of derived source content | Record code, source-document, intermediate-text, graph-object, and prediction rights separately |
| Fannie policy conflated with law | Invalid compliance claims | Separate authority classes and results; require authority reviewer |
| Current policy changes during study | Silent staleness | Immutable snapshots, `policy_as_of`, update ledger, freeze date, temporal challenge |
| Automatic cases leak rule wording or label | Trivial benchmark | blinded rewriting, lexical leakage audits, oracle-ID removal, family-isolated splits |
| Graph arm gets more information/tokens | Confounded structure gain | same node set where possible, fixed token budget, packet-level token accounting |
| Raw RAG retrieves wrong section | Retrieval confounds reasoning | report oracle-retrieval and retrieval-success strata |
| `REVIEW` becomes a catch-all | Artificial safety | require named decisive missing/conflicting fact; report precision and capacity cost |
| Reason codes leak labels | Inflated integrity detection | counterbalance wording, remove diagnostic terms, use blinded leakage audit and visibility ablation |
| Few independent rules | Underpowered inference | pilot family-level variance; increase families or narrow effect claim |
| Human reviewers disagree | Construct invalidity | revise ontology/manual; stop scaling until agreement gate passes |
| Numeric/date extraction errors hidden by macro-F1 | Unsafe graph admitted | critical-field exactness gate and defect-specific reporting |
| Existing benchmark redistribution unresolved | Release risk | use rights ledger; default local-only/text-free; do not republish rows by implication |
| Open external benchmark treated as mortgage validation | Construct overclaim | score native tasks separately; prohibit pooling; state fictional/synthetic policy and label provenance in every result table |
| HMDA outcomes or complaint categories treated as compliance gold | Invalid labels and possible fairness claims | use only aggregate realism features; derive every gold action from locked policy and qualified adjudication |
| Public availability mistaken for redistribution permission | License breach | snapshot licenses and attribution; release predictions/IDs only unless verbatim redistribution is affirmatively allowed |
| External benchmarks selected after inspecting model results | Selection bias | freeze source, split, adapter, metric, and model panel at P6; report incompatible locked cells rather than swapping datasets |
| Local unlicensed mortgage rows enter release or training unnoticed | Rights and contamination risk | keep MortgageGuardBench-2K and `v1_hmda2022` development-only; hash-bound denylist in release and training manifests |
| SLM ignores context | No representation effect | context-use probes, counterfactual policy packets, oracle packet, citation checks |
| Generic checkpoints are compared without direct policy-guard baselines | Weak positioning against current work | include DynaGuard-class compact baselines and matched base checkpoints when licensed and interface-compatible |
| Model pretraining already contains policy | No-policy baseline inflated | versioned post-cutoff changes, shuffled-date controls, context counterfactuals |
| Cross-domain title overclaims | Weak generalization | require independently validated second domain or narrow title/abstract |
| Legal advice interpretation | Research overreach | test guard triage of proposed actions; no legal conclusions; qualified review and explicit non-claims |

---

## 23. Claim registry candidate

### Claims Paper D may make if supported

- extraction fidelity for named systems on named locked source sections;
- stage-resolved fidelity, lineage preservation, validation coverage, optimizer effects, and dependency quality for a named pinned Policy to Knowledge run;
- candidate policy-change precision/recall and reviewer-time effects for named Agents 7–10 outputs against independently locked temporal gold;
- comparative guard performance on the fixed compact-model panel;
- graph-structure effects only under matched facts, nodes, and token budgets;
- measured unsafe-decision amplification of prespecified silent policy-packet defects;
- measured reason-correct review of provenance-visible integrity defects, with deterministic-gate coverage reported separately;
- stage attribution under the exact controlled packet interventions and locked mortgage families;
- stale-policy behavior on verified version-change families;
- review/cost tradeoffs under the explicitly synthetic capacity mixture;
- model performance on named external benchmarks using their native tasks, reported separately from mortgage results;
- reproducibility coverage of released structured artifacts.

### Claims Paper D may not make from this design alone

- legal compliance certification;
- fair-lending or disparate-impact findings about a lender or model;
- mortgage approval/denial quality;
- universal superiority of knowledge graphs;
- a causal “verification” or “graph” effect from verified-graph versus raw-RAG alone;
- an expectation that guards detect internally coherent policy corruption without contrary evidence;
- universal SLM behavior;
- deployment prevalence or production utility;
- validation of mortgage extraction, legal authority, or three-way triage from SafePyramid, PolyGuard, DynaBench, PolicyGuardBench, FinSafeGuard, or ExpGuard;
- policy correctness, fairness, or compliance of observed HMDA outcomes or CFPB complaints;
- cross-domain generalization from mortgage alone;
- correctness of the unreviewed 384-rule candidate graph;
- that the historical local P2K graph was generated by the currently inspected commit or is fully reproducible without a complete original run manifest;
- that P2K Agent 3.5 establishes source correctness, covers relationship-nested rules, or functions as a fail-closed gate;
- that an A4→A5 difference is the isolated causal value of graph structure;
- that the P2K code license grants redistribution rights over Fannie Mae source text or derived graph content;
- current-policy correctness after the source freeze date;
- SME validation unless the required reviewers actually complete and sign the adjudication ledger;
- novelty of document ingestion, policy compilation, policy-conditioned guards, generic error propagation, temporal adaptation, or abstention by themselves.

---

## 24. Paper narrative

The manuscript should read as one causal diagnostic story, not a catalog of pipeline components.

Use the original title for the unified project/report. For submission, prefer the narrower title:

> **From Policy Documents to Guardrail Decisions: Counterfactual Integrity Tests for Compact Mortgage-Policy Guards**

This title states the experimental contribution and avoids implying cross-domain validation. Restore “Regulated-Domain Safety Guards” only after a second fully validated domain is completed.

### Proposed section sequence

1. **Introduction:** a guard decision is only as reliable as the policy chain feeding it.
2. **Related work and novelty boundary:** extraction, neuro-symbolic compliance, regulatory benchmarks, configurable guards, policy KGs, temporal shift, and abstention.
3. **Problem formulation:** source snapshots, three semantic actions, reason codes, authority hierarchy, trust boundary, and error decomposition.
4. **Mortgage policy corpus and data roles:** authoritative source versions, HMDA/complaint realism inputs, rights, and rule-complexity strata.
5. **Concrete upstream compiler:** Policy to Knowledge architecture, pinned-run manifest, stage artifacts, and historical-baseline limitations.
6. **Extraction and verification:** A1/A3/A4/A5 checkpoint attribution, deterministic checks, human adjudication, and verified graph construction.
7. **Decision benchmark:** independent matched families, capacity stream, temporal families, and splits.
8. **Experimental design:** canonical/silent/visible packet interventions, policy-representation arms, compact and purpose-built guards, integrity gate, token matching, calibration, and estimands.
9. **Extraction results:** where P2K checkpoints, verification, and candidate graphs succeed and fail.
10. **Guard results:** silent-defect amplification, visible-integrity review, and integrity-gate coverage.
11. **External controls:** SafePyramid and DynaBench on native tasks, isolated from mortgage inference.
12. **Representation and temporal analyses:** controlled flat-versus-graph effects, P2K change-candidate efficiency, bundled deployment contrasts, and stale snapshots.
13. **Operational analysis:** review budget, latency, memory, API cost, and human verification cost.
14. **Limitations and evidence ledger:** explicit authority, upstream-system, domain, dataset, model-panel, rights, and reviewer boundaries.
15. **Conclusion:** what must be verified before a document-grounded guard is trusted.

### Candidate one-sentence abstract claim

> We introduce paired policy-context integrity tests that expose compact guards to canonical, silently corrupted, and source-inconsistent mortgage-policy packets, separating downstream defect amplification from the ability to route observable policy-integrity failures to human review.

Do not write the result clause of the abstract until all locked analyses run.

---

## 25. Figures and tables planned before results

### Figures

1. End-to-end pipeline showing P2K A1–A5 checkpoints, the independent Paper D verification boundary, measured interfaces, and provenance locks.
2. Canonical/silent/visible packet intervention and trust boundary.
3. Orthogonalized stage design plus realistic deployment arms.
4. Selective risk versus review-budget curve.
5. Silent-defect amplification and visible-review lift by defect class and model.
6. Policy-version timeline and stale/current case construction.
7. Latency/cost versus constrained safety frontier.
8. A3→A4→A5 lineage and defect-flow Sankey or alluvial diagram, shown only if it is clearer than the checkpoint table.

### Tables

1. Dataset/source role matrix: authority, realism, external control, gated/local-only, license, and permitted use.
2. Source corpus, authority class, versions, rights, and counts.
3. Rule schema and annotation agreement.
4. P2K checkpoint manifest and stage accounting: input/output counts, hashes, validation coverage, lineage coverage, and cost.
5. Extraction fidelity by system, checkpoint, and rule-complexity stratum.
6. Retrieval coverage at matched token budget.
7. Primary mutation estimands and controlled representation contrasts.
8. Reason-correct `REVIEW`, deterministic integrity-gate coverage, and capacity usage.
9. Native-task SafePyramid and DynaBench external results, not pooled with mortgage.
10. Temporal robustness and P2K change-candidate quality.
11. Silent-defect and visible-integrity results.
12. Evidence/claim ledger and reproduction coverage.

Every table cell should be generated from structured artifacts; manuscript source should contain no hand-copied claim-bearing values.

---

## 26. Minimum viable paper versus full paper

### Minimum viable, scientifically coherent Paper D

- mortgage only;
- March and June Fannie snapshots plus one narrow federal-regulation subset;
- independently verified extraction gold;
- one controlled pinned Policy to Knowledge run with hashed A1–A5 checkpoints, complete A3→A4→A5 lineage, and a separately labeled historical baseline;
- a fail-closed read-only P2K adapter and upstream-system manifest;
- canonical/silent/visible packet triplets for a small prespecified set of critical defects;
- controlled identical-node flat/graph arms plus raw RAG and oracle deployment controls;
- four generic compact models plus at least one released compact policy-configurable guard, frozen weights;
- `guard_only` and deterministic `integrity_gate_plus_guard` modes;
- matched `ALLOW/REVIEW/INTERVENE` families;
- reason-coded semantic `REVIEW` labels;
- one real temporal-update panel;
- frozen SafePyramid v1.1 and DynaBench external controls on their native tasks;
- a complete benchmark/source registry with license snapshots and non-pooling rules;
- family-clustered statistics;
- qualified human review;
- rights-safe release.

### Full version

- second independently validated regulated domain;
- the same pinned P2K compiler evaluated with its corresponding domain prompt pack only after that domain has independent gold, reviewers, and rights clearance;
- additional open breadth controls such as PolyGuard finance/regulation and PolicyGuardBench;
- gated Zillow, FinGuard-Bench, or ExpGuard studies only if access and licenses are resolved before design lock;
- larger extraction-model panel;
- policy distillation or adaptation as a secondary study;
- prospective sealed confirmation across domains;
- measured analyst time and review-queue simulation;
- public benchmark and policy-graph subset where rights permit.

Do not delay the coherent mortgage paper merely to add shallow domain breadth. Depth, authority fidelity, and independent adjudication are more valuable than unvalidated domain count.

---

## 27. Immediate next actions

In order:

1. run and record the reproducible closest-work search; explicitly screen ProvenanceGuard, deterministic-gate work, PolicyGuard, ComplianceNLP, SafePyramid, FinGuard, BREX, DynaGuard, PolyGuard, and their citation neighborhoods;
2. approve or revise the trust-boundary construct, two mutation modes, reason codes, data-role policy, and sharper submission title;
3. create `README.md`, `STATUS.md`, `PROTOCOL.md`, `RELATED_WORK.md`, the search ledger, and the source/upstream registries under `papers/paper_d/`, together with the per-study `.gitignore` (`inputs/*`, `artifacts/*`, `build/`, `.env`) and the `locks/`, `provenance/`, and `environment/` directories; then register the study in `studies/registry.yaml` with `claim_authorization: false` per Section 19.1, regenerate the indexes with `python tools/render_indexes.py`, and confirm `make check-registry` and `make check-links` pass — all of this before the first import, so the study is governed from its first byte;
4. freeze a read-only manifest of the current P2K checkout and historical A1–A5 artifacts, recording commit, dirty state, source/artifact hashes, known configuration, and every unrecoverable provenance field;
5. implement the P2K run-manifest, stage-lineage schemas, read-only adapter, and fixture tests; prove that importing and normalization cannot alter upstream bytes;
6. pin exact SafePyramid v1.1 and DynaBench revisions and splits; hash their files, archive their license and attribution records, and freeze native-task adapters and metrics before scoring;
7. record PolyGuard and PolicyGuardBench as optional breadth controls, and explicitly classify FinSafeGuard, Zillow, FinGuard-Bench, ExpGuard, MortgageGuardBench-2K, and `v1_hmda2022` as optional, gated, watch-list, or local-only rather than silently importing them;
8. decide whether HMDA and CFPB complaint data are needed for scenario realism; if used, freeze exact snapshots, filters, allowed transformations, privacy checks, and a rule preventing them from supplying policy propositions or gold labels;
9. freeze the March source and acquire/hash the June Fannie snapshot and update announcements;
10. select the exact federal-regulation sections and confirm qualified reviewers;
11. implement the core Paper D JSON schemas, mutation namespace, deterministic checks, and fail-closed validators;
12. select 24 source sections stratified by rule complexity;
13. run the independent annotation pilot before any new claim-bearing extraction run;
14. freeze critical-field and agreement gates;
15. import the existing local graph as immutable `E0_p2k_historical_a5` and run only descriptive/defect-discovery analyses on it;
16. lock the controlled P2K commit, sanitized configuration, prompts, models, CLI arguments, and environment; then run A1–A5 on the locked source bundle and verify complete A3→A4→A5 lineage;
17. score A1/A3/A4/A5, adjudicate every removal and eligible edge sample, and build `E5_p2k_verified` through deterministic checks plus blinded review;
18. build 30 independently adjudicated matched families and pilot canonical/silent/visible triplets;
19. verify that mutation triplets isolate one defect and that visible evidence supports the intended reason code;
20. run P2K Agents 7–10 on the locked version pair only after the independent temporal change ledger is sealed;
21. build external-benchmark adapters and smoke-test them against published native task contracts without using evaluation results for prompt or threshold selection;
22. run token-parity and constrained-output smokes on one generic model and one DynaGuard-class baseline;
23. estimate family-level variance, mutation eligibility, reviewer throughput, and P2K checkpoint cost;
24. freeze or narrow the full study before primary case generation or further GPU/API spending.

The first expensive extraction run should occur only after source, schema, annotation, and adapter gates pass. The first expensive guard run should occur only after token-parity and constrained-output gates pass. The first claim-bearing guard run should occur only after the design and retrieval packets are hash-locked.

---

## 28. Definition of done

Paper D is complete only when:

- `studies/registry.yaml` records the study's final state, evidence tier, contract class, and
  verification command; the generated indexes are current; and `claim_authorization` was flipped to
  `true` only after the documented P8 seal — never earlier and never silently;
- every Fannie, federal, and external source has an entry in
  `benchmarks/registry/distribution.yaml`, so the repository's fail-closed ledger governs release;
- the related-work search is reproducible, dated, and refreshed before submission;
- every source and dataset has a fixed role—authoritative policy, scenario realism, primary mortgage evaluation, external control, or excluded—and a complete revision, hash, license, attribution, and redistribution record;
- authoritative source snapshots and rights decisions are immutable;
- historical and controlled P2K artifacts are visibly distinct, and no result attributes the historical graph to an unproven code/configuration state;
- the controlled P2K run binds commit, clean state, source, prompts, sanitized configuration, environment, models, CLI arguments, logs, and all A1–A5 hashes;
- all Agent 3 candidates—including relationship-nested rules—have total A3→A4→A5 lineage, and every removal or merge is accounted for;
- P2K Agent 3.5 and fuzzy re-verification remain advisory baselines rather than gold or admission gates;
- HMDA and complaint-derived artifacts, if used, cannot supply policy propositions or gold labels and pass the documented privacy, paraphrase, and near-duplicate checks;
- extraction gold is independent and doubly reviewed;
- candidate and verified graphs are distinct, versioned artifacts;
- cases are independently adjudicated with semantic `REVIEW` reasons;
- canonical, silent-defect, and provenance-visible triplets are independently verified and synthetically namespaced;
- silent-corruption vulnerability is not mislabeled as corruption-detection failure;
- all representation arms are token- and retrieval-controlled;
- controlled structure comparisons use the same policy propositions and nodes;
- A4→A5 optimizer results are reported separately from the identical-node flat-versus-graph structure effect;
- realistic verified-graph versus raw-RAG results are labeled bundled deployment contrasts;
- the generic compact panel, direct policy-guard baselines, integrity gate, and prompts are pinned;
- SafePyramid v1.1 and DynaBench revisions, official splits, native task contracts, adapters, and metrics were frozen before evaluation access;
- external-benchmark results are reported on native tasks, kept separate from mortgage results, and never presented as validation of mortgage extraction, legal authority, or triage quality;
- unlicensed or access-restricted local resources, including MortgageGuardBench-2K and `v1_hmda2022`, are excluded from public release and training unless affirmative permission is documented before design lock;
- primary estimands and thresholds were frozen before test access;
- extraction, retrieval, guard, temporal, and error-propagation results all regenerate from immutable artifacts;
- a separately authored sealed cohort has a documented unsealing event, or all results are clearly labeled retrospective/developmental;
- every claim is bounded to its authority, policy vintage, domain, model panel, and reviewer tier;
- release permissions are affirmative, with text-free fallbacks for restricted sources;
- the manuscript reports negative, infeasible, and failed cells alongside successes.

Until these conditions hold, the correct description is:

> Paper D is a protocol and implementation plan for measuring an end-to-end policy-document-to-guard-decision workflow; it is not evidence that the workflow is accurate, legally authoritative, or deployment-ready.
