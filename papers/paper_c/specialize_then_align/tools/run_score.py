#!/usr/bin/env python
"""Score one trained cell's checkpoint ladder over the evaluation splits.

    score::<cell_dir_name>::<backbone>

Emits one JSONL per checkpoint into artifacts/scores/<cell>/, covering the
calibration, checkpoint_selection and sealed splits.  No threshold is applied and no
checkpoint is chosen here: that happens later, on the splits reserved for it.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_c_sta.contracts import output_path, read_json, write_json  # noqa: E402
from paper_c_sta.score import score_rows, write_scores  # noqa: E402

# The sealed cohort is the held-out confirmation set.  This study has no separately
# authored sealed stream, so the checkpoint_selection split doubles as it -- recorded
# explicitly rather than silently, because it weakens the confirmatory claim.
EVAL_SPLITS = ("calibration", "checkpoint_selection")


def load_rows(splits) -> list[dict]:
    rows = []
    with (ROOT / "artifacts/cohort/samples.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") in splits:
                rows.append(row)
    return rows


def main(cell_name: str, backbone: str) -> int:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_json(output_path("config/study.json"))
    cell = ROOT / "artifacts/cells" / cell_name
    if not cell.is_dir():
        print(f"missing cell: {cell}", flush=True)
        return 2
    checkpoints = sorted(p for p in cell.glob("step*") if p.is_dir())
    if not checkpoints:
        print(f"no checkpoints in {cell}", flush=True)
        return 2

    rows = load_rows(EVAL_SPLITS)
    print(f"scoring {cell_name}: {len(checkpoints)} checkpoints x {len(rows)} rows "
          f"on {device}", flush=True)

    out_dir = ROOT / "artifacts/scores" / cell_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for checkpoint in checkpoints:
        started = time.time()
        records = score_rows(config, rows, backbone_key=backbone,
                             adapter_path=str(checkpoint), device=device,
                             batch_size=32)
        info = write_scores(records, f"artifacts/scores/{cell_name}/{checkpoint.name}.jsonl")
        info["wall_seconds"] = round(time.time() - started, 1)
        summary[checkpoint.name] = info
        print(f"  {checkpoint.name}: {info['rows']} rows {info['by_split']} "
              f"{info['wall_seconds']}s", flush=True)

    write_json(out_dir / "score_report.json", {
        "cell": cell_name, "backbone_key": backbone,
        "splits": list(EVAL_SPLITS), "checkpoints": summary,
        "sealed_note": "no separately authored sealed stream exists; "
                       "checkpoint_selection doubles as the confirmation split, "
                       "which weakens the confirmatory claim and is reported as such",
    })
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
