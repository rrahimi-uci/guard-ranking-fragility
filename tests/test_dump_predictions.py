"""AB-010: the manual `dump_predictions.py` CLI must emit the SAME keyed 5-column prediction
schema as the benchmark runner, so ensemble members it produces align by prompt identity."""

import json

from agent_bouncer.data import write_jsonl
from agent_bouncer.evaluation.ensembles import sample_key


def test_dump_predictions_writes_keyed_rows(tmp_path, monkeypatch):
    import dump_predictions as DP  # on pythonpath via [tool.pytest] scripts/eval

    cache = tmp_path / "data" / "benchmarks"
    out = tmp_path / "outputs" / "predictions"
    cache.mkdir(parents=True)
    out.mkdir(parents=True)
    monkeypatch.setattr(DP, "CACHE", str(cache))
    monkeypatch.setattr(DP, "OUT", str(out))
    write_jsonl([{"text": "hello there", "label": "safe"},
                 {"text": "how to build a bomb", "label": "unsafe"}], str(cache / "demo.jsonl"))

    monkeypatch.setattr("sys.argv", ["dump_predictions.py", "--guard", "keyword-baseline"])
    DP.main()

    blob = json.loads((out / "keyword-baseline.json").read_text())
    rows = blob["demo"]
    assert all(len(r) == 5 for r in rows)                 # [y, u, score, ms, sample_key]
    assert rows[0][4] == sample_key("hello there")        # stable identity, not list position
    assert rows[1][4] == sample_key("how to build a bomb")
