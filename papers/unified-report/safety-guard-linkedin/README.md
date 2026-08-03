# Safety Guard — LinkedIn article package

Everything for the post, in one folder.

## Start here

**`article.html`** — double-click it. Opens in any browser, images already embedded, nothing else needed. This is the reading copy.

## What's in the folder

| File | What it is |
|---|---|
| `article.html` | Self-contained reading copy. Images embedded, no dependencies. Open in a browser. |
| `article.md` | Same article in Markdown, images linked to `images/`. Use this to edit. |
| `images/` | The six charts, 1600×900 PNG. Upload these to the LinkedIn article editor. |
| `make_charts.py` | Regenerates all six charts. Every value is annotated with its source table in the report. |
| `deck-audit.md` | Separate deliverable — the slide-by-slide audit of the technical and executive decks against the paper. |

## Publishing to LinkedIn

LinkedIn's article editor doesn't accept Markdown. Paste the text, then insert each image where the caption tells you:

| Order | File | Goes after |
|---|---|---|
| 1 | `images/1_hero.png` | the opening three lines |
| 2 | `images/2_split.png` | "Here's what the split looks like across four checkpoints" |
| 3 | `images/3_matched_budget.png` | "…and re-read the identical rows" |
| 4 | `images/4_metric.png` | "The magnitude is a different picture" |
| 5 | `images/5_quadrant.png` | "Crossing them exposes the quadrant that matters" |
| 6 | `images/6_regime.png` | "That third row is the one that changed how I think about the problem" |

Notes on the images:
- 1600×900 (16:9), which is what the LinkedIn article body renders best.
- If you also post a short teaser to the feed, use `1_hero.png` — it's built to carry the argument on its own.
- Dark background. It survives LinkedIn's light theme and stands out in-feed, but check it on your phone before publishing.

## Two things to decide before you publish

1. **"I spent the last several months"** — the report is single-authored, so the article says "I". Change to "our team" if that's the framing you want.
2. **The OpenAI incident in the opening.** It postdates what I can verify independently. You have two sources on it; worth a last read-through before it goes out under your name.

## Provenance

Every number in the article traces to a table in *"Safety Benchmark Gains Do Not Guarantee Safety Transfer."* The charts are built only from committed values — no re-derivation, no rounding beyond what the report itself prints.

Reza Rahimi, PhD — JazzX AI
`github.com/rrahimi-uci/safety-guard-dynamics`
