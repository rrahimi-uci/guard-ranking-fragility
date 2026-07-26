#!/usr/bin/env python
"""Generate teacher candidates and build matched preference pairs for one panel.

    python tools/run_propose.py <target_backbone> <panel>     # panel: pilot|primary

Candidates depend on the teacher backbone and teacher seeds, not on the target seed,
so one run covers every target seed in the panel.  Pilot and primary use disjoint
teacher seeds, keeping the two namespaces separate as the protocol requires.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_c_sta.contracts import output_path, read_json, write_json  # noqa: E402
from paper_c_sta.pairs import assert_sources_matched, build_pairs  # noqa: E402
from paper_c_sta.propose import SOURCES, propose_for_source  # noqa: E402

PANEL_TEACHER_SEEDS = {"pilot": (7, 8), "primary": (42, 43)}
PANEL_TARGET_SEEDS = {"pilot": (7, 8), "primary": (42, 43, 44)}


def main(target_backbone: str, panel: str) -> int:
    import torch

    if panel not in PANEL_TEACHER_SEEDS:
        print(f"panel must be pilot or primary, got {panel}")
        return 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_json(output_path("config/study.json"))

    rows = []
    with (ROOT / "artifacts/cohort/samples.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "alignment_pool":
                rows.append(row)
    print(f"alignment_pool rows: {len(rows)}  target={target_backbone} panel={panel} "
          f"device={device}", flush=True)

    teacher_seeds = PANEL_TEACHER_SEEDS[panel]
    cells_root = ROOT / "artifacts/cells"
    inventories, records = {}, {}
    for source in SOURCES:
        started = time.time()
        candidates, record = propose_for_source(
            config, rows, source=source, target_backbone_key=target_backbone,
            teacher_seeds=teacher_seeds, cells_root=cells_root, device=device,
            batch_size=16,
        )
        record["wall_seconds"] = round(time.time() - started, 1)
        print(f"  {source:20s} events_with_two_candidates="
              f"{record['events_with_two_candidates']}  {record['wall_seconds']}s", flush=True)
        built = build_pairs(
            rows, candidates, candidate_source=source,
            target_backbone_key=target_backbone,
            teacher_backbone_keys=[record["teacher_backbone_key"]],
            teacher_seeds=list(teacher_seeds),
            teacher_cells=record["teacher_cells"],
            calibration_lock_id=f"CAL-CANDIDATE-{panel}",
            adjudicator_id="llm_free_gold_adjudication_v1",
            reviewer_ids=["auto:gold_action_rank#r1", "auto:gold_action_rank#r2"],
        )
        inventories[source] = built["records"]
        records[source] = {**record, "rejected": built["rejected"],
                           "pairs": len(built["records"])}
        print(f"  {source:20s} pairs={len(built['records'])}  rejected={built['rejected']}",
              flush=True)

    # The primary contrast is only interpretable on events both sources could rank.
    left, right = inventories["category_specialist"], inventories["joint_generalist"]
    common = {r["sample_id"] for r in left} & {r["sample_id"] for r in right}
    left = [r for r in left if r["sample_id"] in common]
    right = [r for r in right if r["sample_id"] in common]
    matched = assert_sources_matched(left, right)
    print(f"\nMATCHED events={matched['matched_events']} pairs/source={matched['pairs_per_source']}",
          flush=True)
    print(f"  strata specialist={matched['strata']['left']}", flush=True)
    print(f"  strata generalist={matched['strata']['right']}", flush=True)

    out = ROOT / f"artifacts/pairs/{panel}__{target_backbone}"
    out.mkdir(parents=True, exist_ok=True)
    for source, recs in (("category_specialist", left), ("joint_generalist", right)):
        with (out / f"{source}.jsonl").open("w") as handle:
            for r in recs:
                handle.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    write_json(out / "propose_report.json", {
        "panel": panel, "target_backbone_key": target_backbone,
        "target_seeds": list(PANEL_TARGET_SEEDS[panel]),
        "teacher_seeds": list(teacher_seeds),
        "per_source": records, "matched": matched,
        "evidence_tier": "auto-adjudicated against gold; no human reviewer",
    })
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
