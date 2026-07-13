"""guard_research — canonical, auditable building blocks for Paper A.

This package holds the single source of truth for the pieces that must be
byte-identical across manifest building, training, scoring, and analysis:

- metrics.py      canonical tie-aware AP / AUROC (no custom AP loops anywhere)
- provenance.py   text normalization, content/object/file hashing, MinHash
- prompts.py      the frozen system prompt, chat-template rendering, decision tokens
- thresholds.py   the conservative Clopper-Pearson operating point (plan sec 10.5)
"""

__version__ = "0.1.0"
