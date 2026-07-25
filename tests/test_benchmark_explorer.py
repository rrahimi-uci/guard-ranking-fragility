"""Coverage for the self-contained benchmark explorer build."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "benchmark-explorer" / "generate.py"
SPEC = importlib.util.spec_from_file_location("benchmark_explorer_generate", GENERATOR)
assert SPEC and SPEC.loader
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


class BenchmarkExplorerTest(unittest.TestCase):
    def test_guard_sections_include_every_source_row(self) -> None:
        sections = G.build_guard()

        self.assertEqual([section["id"] for section in sections], [item[0] for item in G.GUARD])
        for section in sections:
            source = G.FULL_BENCH_DIR / f"{section['id']}.jsonl"
            self.assertEqual(len(section["samples"]), _line_count(source))
            self.assertEqual(
                [sample["sample"] for sample in section["samples"]],
                list(range(1, len(section["samples"]) + 1)),
            )

    def test_safepyramid_and_hardened_guard_include_every_row(self) -> None:
        pyramid = G.build_safepyramid()[0]
        hard = G.build_hard_guard()[0]

        self.assertEqual(len(pyramid["samples"]), _line_count(G.SAFEPYRAMID))
        self.assertEqual(len(pyramid["samples"]), 3000)
        self.assertTrue(all(sample["text"] for sample in pyramid["samples"]))
        self.assertTrue(all(sample["meta"][-1]["k"] == "Policy" for sample in pyramid["samples"]))
        self.assertEqual(len(hard["samples"]), _line_count(G.HARD_GUARD))
        self.assertEqual(len(hard["samples"]), 334)
        self.assertEqual({sample["lc"] for sample in hard["samples"]}, {"safe", "unsafe"})

    def test_mortgage_section_includes_all_task_types_and_rows(self) -> None:
        section = G.build_mortgage()[0]

        self.assertEqual(len(section["samples"]), _line_count(G.MORTGAGE))
        self.assertEqual(len(section["samples"]), 2000)
        self.assertEqual(
            {sample["lc"] for sample in section["samples"]},
            {"safe", "unsafe", "review"},
        )
        self.assertTrue(all(sample["id"] and sample["text"] for sample in section["samples"]))

    def test_committed_public_build_contains_full_data_and_pagination(self) -> None:
        html = (G.OUTDIR / "index.public.html").read_text(encoding="utf-8")
        payload = html.split('<script id="data" type="application/json">', 1)[1].split(
            "</script>", 1
        )[0]
        sections = json.loads(payload)
        expected_total = sum(
            _line_count(G.FULL_BENCH_DIR / f"{item[0]}.jsonl") for item in G.GUARD
        )
        expected_total += _line_count(G.SAFEPYRAMID)
        expected_total += _line_count(G.HARD_GUARD)
        expected_total += _line_count(G.MORTGAGE)

        self.assertEqual(len(sections), 10)
        self.assertEqual(sum(len(section["samples"]) for section in sections), expected_total)
        self.assertIn(f"const PAGE_SIZE={G.PAGE_SIZE};", html)
        self.assertIn("ss.slice(start,end)", html)
        self.assertIn("data-page-select", html)


if __name__ == "__main__":
    unittest.main()
