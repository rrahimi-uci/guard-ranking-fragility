from collections import Counter, defaultdict

from paper_c.sampling import build_selections, family_partition


def rows():
    output = []
    for source in ("source_a", "source_b", "source_c"):
        for gold in (0, 1):
            for index in range(20):
                family = f"{source}-{gold}-{index}"
                if index == 0:
                    family = f"cross-{gold}"
                sample = f"{source}-{gold}-{index}"
                output.append({
                    "sample_id": sample,
                    "source": source,
                    "gold": gold,
                    "family_id": family,
                    "content_sha256": f"hash-{sample}",
                })
    return output


def references(source_rows):
    output = []
    for index, row in enumerate(source_rows):
        score = (index % 19 - 9) / 5
        output.append({
            "sample_id": row["sample_id"],
            "safe_logit": -score / 2,
            "unsafe_logit": score / 2,
            "prompt_sha256": f"prompt-{row['sample_id']}",
        })
    return output


def test_global_family_split_is_deterministic_and_never_crosses():
    first = family_partition(rows(), development_fraction=0.2, seed=20260725)
    second = family_partition(rows(), development_fraction=0.2, seed=20260725)
    assert first == second
    assignments = defaultdict(set)
    for row in first:
        assignments[row["family_id"]].add(row["stage2_partition"])
    assert all(len(value) == 1 for value in assignments.values())
    counts = Counter((row["source"], row["gold"], row["stage2_partition"]) for row in first)
    for source in ("source_a", "source_b", "source_c"):
        for gold in (0, 1):
            assert counts[(source, gold, "stage2_dev")] > 0
            assert counts[(source, gold, "stage2_update")] > 0


def test_uncertain_and_random_selections_are_disjoint_and_matched():
    source_rows = rows()
    partition = family_partition(source_rows, development_fraction=0.2, seed=20260725)
    selected = build_selections(
        partition,
        references(source_rows),
        uncertain_fraction=0.25,
        seed=20260725,
    )
    by_role = defaultdict(set)
    counts = Counter()
    for row in selected:
        by_role[row["selection_role"]].add(row["sample_id"])
        counts[(row["selection_role"], row["source"], row["gold"])] += 1
        assert row["prompt_sha256"].startswith("prompt-")
    assert by_role["uncertain"].isdisjoint(by_role["matched_random"])
    for source in ("source_a", "source_b", "source_c"):
        for gold in (0, 1):
            assert counts[("uncertain", source, gold)] == counts[("matched_random", source, gold)]

