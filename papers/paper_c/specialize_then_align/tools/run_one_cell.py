#!/usr/bin/env python
"""Run one named training cell from the cohort. Cell ids encode the whole spec.

    reference::<backbone>::<seed>
    specialist::<backbone>::<seed>::<category>

Used by cloud/run_cells.sh on the VM, and runnable locally for debugging.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paper_c_sta.contracts import output_path, read_json  # noqa: E402
from paper_c_sta.train import run_cell  # noqa: E402


def load_rows(split: str, category: str | None = None) -> list[dict]:
    rows = []
    with (ROOT / "artifacts/cohort/samples.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != split:
                continue
            if category and row.get("category") != category:
                continue
            rows.append(row)
    return rows


def main(cell_id: str) -> int:
    import torch

    parts = cell_id.split("::")
    kind, backbone_key, seed = parts[0], parts[1], int(parts[2])
    category = parts[3] if len(parts) > 3 else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = read_json(output_path("config/study.json"))
    rows = load_rows("specialist_train", category)

    # A specialist is not trained from the bare backbone: it continues from the joint
    # multitask reference for its own backbone and seed, so every specialist in a cell
    # family shares one starting point and the arms stay comparable.
    reference_adapter = None
    if kind == "specialist":
        ref = ROOT / f"artifacts/cells/reference__{backbone_key}__{seed}/step0400"
        if not ref.is_dir():
            print(f"missing reference adapter for {backbone_key}/{seed}: {ref}", flush=True)
            return 3
        reference_adapter = str(ref)
    if not rows:
        print(f"no rows for {cell_id}", flush=True)
        return 2

    print(f"cell={cell_id} kind={kind} backbone={backbone_key} seed={seed} "
          f"category={category} rows={len(rows)} device={device}", flush=True)
    record = run_cell(
        config, kind=kind, backbone_key=backbone_key, seed=seed, category=category,
        rows=rows, out_dir=f"artifacts/cells/{cell_id.replace('::', '__')}",
        reference_adapter=reference_adapter,
        device=device, batch_size=4, max_length=1024,
    )
    print(json.dumps({k: v for k, v in record.items() if k != "loss_history"},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
