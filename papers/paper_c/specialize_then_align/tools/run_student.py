#!/usr/bin/env python
"""Train one aligned student arm.

    student::<arm>::<backbone>::<seed>::<panel>

All five arms start from the same backbone/seed joint reference and consume the same
matched event set, so the only thing that varies between the two CM-DPO arms is which
model proposed the candidates.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_c_sta.contracts import output_path, read_json  # noqa: E402
from paper_c_sta.train import run_cell  # noqa: E402

ARM_SOURCE = {
    "specialist_cm_dpo": "category_specialist",
    "specialist_pairce": "category_specialist",
    "generalist_cm_dpo": "joint_generalist",
}


def load_pairs(panel: str, backbone: str, source: str) -> list[dict]:
    path = ROOT / f"artifacts/pairs/{panel}__{backbone}/{source}.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing pair inventory: {path}")
    return [json.loads(line) for line in path.open() if line.strip()]


def load_events(split: str) -> list[dict]:
    rows = []
    with (ROOT / "artifacts/cohort/samples.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == split:
                rows.append(row)
    return rows


def main(cell_id: str) -> int:
    import torch

    _, arm, backbone, seed_s, panel = cell_id.split("::")
    seed = int(seed_s)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_json(output_path("config/study.json"))

    reference = ROOT / f"artifacts/cells/reference__{backbone}__{seed}/step0400"
    if not reference.is_dir():
        print(f"missing reference adapter: {reference}", flush=True)
        return 3

    if arm in ARM_SOURCE:
        pair_rows = load_pairs(panel, backbone, ARM_SOURCE[arm])
        rows = []
        for record in pair_rows:
            rows.append({
                "sample_id": record["sample_id"],
                "family_id": record["family_id"],
                "category": record["category"],
                "request": record.get("request", ""),
                "proposed_response": None,
                "context": None,
                "policy_context": record.get("policy_context"),
                "gold": {"action": record["gold_action"],
                         "category": record["category"],
                         "violation_tags": [], "policy_ids": record.get("policy_ids", []),
                         "rationale": ""},
                "chosen": record["chosen"]["text"],
                "rejected": record["rejected"]["text"],
            })
    elif arm == "gold_sft":
        rows = load_events("alignment_pool")
    elif arm == "soft_distill":
        # teacher probabilities come from the specialist inventory's chosen candidate
        pair_rows = load_pairs(panel, backbone, "category_specialist")
        rows = []
        for record in pair_rows:
            rows.append({
                "sample_id": record["sample_id"],
                "family_id": record["family_id"],
                "category": record["category"],
                "request": record.get("request", ""),
                "proposed_response": None, "context": None,
                "policy_context": record.get("policy_context"),
                "gold": {"action": record["gold_action"], "category": record["category"],
                         "violation_tags": [], "policy_ids": [], "rationale": ""},
                "teacher_action_probabilities": record["chosen"]["probabilities"],
            })
    else:
        print(f"unknown arm: {arm}", flush=True)
        return 1

    if not rows:
        print(f"no training rows for {cell_id}", flush=True)
        return 2

    # The events carry no request text in the pair inventory; refill from the cohort so
    # the prompt the student sees is the same one the teacher saw.
    events = {r["sample_id"]: r for r in load_events("alignment_pool")}
    for row in rows:
        source = events.get(row["sample_id"])
        if source:
            row["request"] = source["request"]
            row["proposed_response"] = source.get("proposed_response")
            row["context"] = source.get("context")

    print(f"cell={cell_id} arm={arm} rows={len(rows)} device={device}", flush=True)
    record = run_cell(
        config, kind=arm, backbone_key=backbone, seed=seed, rows=rows,
        out_dir=f"artifacts/cells/{cell_id.replace('::', '__')}",
        reference_adapter=str(reference), device=device,
        # Pair arms hold a policy and a frozen reference and run up to six forwards per
        # step (chosen/rejected x policy/reference, plus the composite's gold anchor and
        # replay KL).  A smaller batch and shorter context keep that inside 40 GB.
        batch_size=2 if arm in ARM_SOURCE else 4,
        max_length=768 if arm in ARM_SOURCE else 1024,
    )
    print(json.dumps({k: v for k, v in record.items() if k != "loss_history"},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
