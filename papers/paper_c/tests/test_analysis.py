from pathlib import Path
import uuid

from paper_c.analyze import factorial_contrasts, select_checkpoints, tie_aware_average_precision
from paper_c.contracts import load_config, output_path, write_jsonl


def test_tie_aware_ap_groups_equal_scores():
    assert tie_aware_average_precision([1, 0], [0.5, 0.5]) == 0.5
    assert tie_aware_average_precision([1, 0, 1, 0], [2.0, 1.0, 2.0, 0.0]) == 1.0


def _score_rows(model, seed, *, objective=None, sampler=None, step=None, perfect=True):
    rows = []
    for source in ("a", "b"):
        for gold in (0, 1):
            score = (2.0 if gold else -2.0) if perfect else (-2.0 if gold else 2.0)
            row = {
                "score_role": "stage2_dev",
                "model_key": model,
                "seed": seed,
                "source": source,
                "gold": gold,
                "score_unsafe_minus_safe": score,
            }
            if objective is not None:
                row.update({"objective": objective, "sampler": sampler, "step": step})
            rows.append(row)
    return rows


def test_checkpoint_selection_uses_only_development_and_exact_grid():
    config = load_config("config/study.json")
    work = output_path(Path("build/test-work") / uuid.uuid4().hex)
    work.mkdir(parents=True)
    baselines, candidates = [], []
    for model in config["models"]:
        for seed in config["seeds"]:
            baselines.extend(_score_rows(model, seed))
            for sampler in config["stage2"]["samplers"]:
                for objective in config["stage2"]["objectives"]:
                    for step in config["stage2"]["checkpoint_steps"]:
                        candidates.extend(_score_rows(
                            model, seed, objective=objective, sampler=sampler, step=step,
                            perfect=step >= 50,
                        ))
    baseline_path = work / "stage1.jsonl"
    candidate_path = work / "candidates.jsonl"
    output_path_value = work / "selection.jsonl"
    write_jsonl(baseline_path, baselines)
    write_jsonl(candidate_path, candidates)
    selected = select_checkpoints(
        config=config,
        stage1_scores_path=baseline_path,
        candidate_scores_path=candidate_path,
        out_path=output_path_value,
    )
    assert len(selected) == 120
    assert {row["selected_step"] for row in selected} == {50}
    assert all(row["selection_status"] == "target_reached" for row in selected)


def test_factorial_contrasts_use_equal_weight_sampler_marginal():
    rows = []
    values = {
        "uncertain": {"verdict_ce": 0.6, "pair_ce": 0.7, "dpo": 0.9},
        "matched_random": {"verdict_ce": 0.5, "pair_ce": 0.55, "dpo": 0.6},
    }
    for sampler, objectives in values.items():
        for objective, value in objectives.items():
            rows.append({
                "model_key": "m", "seed": 1, "regime": "transfer",
                "sampler": sampler, "objective": objective, "macro_ap": value,
            })
    result = factorial_contrasts(rows)
    marginal = result["factorial_marginal"][0]
    assert abs(marginal["c_ref"] - 0.125) < 1e-12
    assert abs(marginal["c_pair"] - 0.075) < 1e-12
    assert abs(marginal["selection_interaction"] - 0.2) < 1e-12

