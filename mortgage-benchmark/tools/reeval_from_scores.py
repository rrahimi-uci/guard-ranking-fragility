import json, os, sys
sys.path.insert(0, "mortgage-benchmark")
from magen import evaluate as EV, score_guards as SG
from magen.store import write_json
BASE = "mortgage-benchmark/benchmark/v1_hmda2022"
OUT = "mortgage-benchmark/out_eval"
sp = SG.load_benchmark(BASE)
guards = ["qwen25_15b_base", "qwen3_4b_base", "smollm2_17b_base", "smollm3_3b_base"]
table = []
for g in guards:
    sc = json.load(open(f"{OUT}/scores_{g}.json"))
    rep = EV.evaluate(sp["public_test"], SG.preds_from_scores(sc), dev_rows=sp["dev"])
    rep["guard"] = g; rep["guard_kind"] = "logit_diff"; rep["eval_split"] = "public_test"
    write_json(rep, f"{OUT}/report_{g}.json")
    tf = rep["threshold_free"]; q = rep["per_quadrant"].get("G0D1", {})
    table.append({"guard": g, "kind": "logit_diff",
                  "AP_G": tf["G"]["average_precision"], "AP_D": tf["D"]["average_precision"],
                  "AP_final": tf["final"]["average_precision"],
                  "G0D1_n": q.get("n"), "G0D1_missed": q.get("missed"),
                  "delta_context": rep["fairness_delta_context"]["mean_abs_delta"]})
write_json({"eval_split": "public_test", "benchmark": BASE, "table": table,
            "skipped": [{"guard": "llama_guard_3_1b", "error": "gated repo (HF license not accepted)"},
                        {"guard": "wildguard_7b", "error": "gated repo (HF license not accepted)"}],
            "note": "AP computed in the repo canonical env (guard_research/sklearn) from committed per-row scores; regenerable via this path."},
           f"{OUT}/baseline_table.json")
print("regenerated locally:")
for r in table: print(f"  {r['guard']:18s} AP_G={r['AP_G']:.3f} AP_D={r['AP_D']:.3f} AP_final={r['AP_final']:.3f}")
