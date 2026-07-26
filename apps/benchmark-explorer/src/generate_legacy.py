#!/usr/bin/env python3
"""Generate self-contained HTML explorers for every benchmark sample.

Produces (all fully offline, data embedded):
  index.public.html  Full public guard sets + SafePyramid + mortgage benchmarks
  index.html         Public data + locally cached ExpGuard domains (LOCAL ONLY —
                     gated/licensed dataset; NOT committed; gitignored)

Usage:
    python3 benchmark-explorer/generate.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "data" / "benchmarks"
FULL_BENCH_DIR = BENCH_DIR / "full"
SAFEPYRAMID = BENCH_DIR / "safepyramid.jsonl"
HARD_GUARD = ROOT / "data" / "guard_benchmark_hard.jsonl"
MORTGAGE = ROOT / "data" / "mortgage_guard_bench_2k_v0_1_0" / "data" / "mortgage_guard_bench_full.jsonl"
OUTDIR = Path(__file__).resolve().parent

PAGE_SIZE = 100
EXPGUARD_DATASET = "6rightjade/expguardmix"
EXPGUARD_FILE = "expguardtest.parquet"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
SAFE_WORDS = {"safe", "compliant", "supported"}
UNSAFE_WORDS = {"unsafe", "noncompliant", "unsupported"}


def label_class(lbl: str) -> str:
    l = str(lbl).lower()
    if l in SAFE_WORDS:
        return "safe"
    if l in UNSAFE_WORDS:
        return "unsafe"
    return "review"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact_value(value: object) -> str:
    """Render structured benchmark fields legibly without losing their content."""
    if value is None or value == "" or value == []:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# collection 1: public guard benchmarks
# ---------------------------------------------------------------------------
GUARD = [
    ("beavertails", "BeaverTails",
     "Human-annotated QA prompts spanning many hazard categories (crime, hate, weapons, privacy…)."),
    ("jailbreak_classification", "Jailbreak Classification",
     "Prompts labelled as jailbreak attempts versus benign roleplay and ordinary requests."),
    ("jailbreakbench", "JailbreakBench",
     "Canonical harmful-behavior prompts paired against benign look-alike requests."),
    ("openai_moderation", "OpenAI Moderation",
     "Content sampled from OpenAI's moderation categories (sexual, hate, self-harm, violence)."),
    ("prompt_injections", "Prompt Injections",
     "Instruction-override / injection attacks contrasted with legitimate user instructions."),
    ("toxicchat", "ToxicChat",
     "Real-world user queries to an LLM, labelled toxic versus benign."),
    ("xstest", "XSTest",
     "Exaggerated-safety probe: safe prompts that look dangerous, plus genuinely unsafe ones."),
]


def build_guard() -> list[dict]:
    out = []
    for name, title, desc in GUARD:
        rows = load_jsonl(FULL_BENCH_DIR / f"{name}.jsonl")
        safe = [r for r in rows if r.get("label") == "safe"]
        unsafe = [r for r in rows if r.get("label") == "unsafe"]
        samples = []
        for sample_number, r in enumerate(rows, start=1):
            hz = r.get("hazard", "none") or "none"
            samples.append({
                "sample": sample_number,
                "label": r.get("label", ""),
                "lc": label_class(r.get("label", "")),
                "tags": [hz],
                "text": r.get("text", ""),
            })
        out.append({
            "id": name, "title": title, "desc": desc,
            "stat": f"{len(safe)} safe / {len(unsafe)} unsafe in full set",
            "samples": samples,
        })
    return out


# ---------------------------------------------------------------------------
# collection 2: SafePyramid policy-conversation benchmark
# ---------------------------------------------------------------------------
def build_safepyramid() -> list[dict] | None:
    if not SAFEPYRAMID.exists():
        return None
    rows = load_jsonl(SAFEPYRAMID)
    samples = []
    for sample_number, r in enumerate(rows, start=1):
        gold = r.get("gold") or []
        rule_ids = r.get("rule_ids") or []
        samples.append({
            "sample": sample_number,
            "id": r.get("id", ""),
            "label": f"{len(gold)} violation" + ("s" if len(gold) != 1 else ""),
            "lc": "unsafe" if gold else "safe",
            "tags": [t for t in (r.get("domain"), r.get("level")) if t],
            "text": r.get("conversation", ""),
            "meta": [
                {"k": "ID", "v": r.get("id", f"sample-{sample_number}")},
                {"k": "Gold", "v": compact_value(gold)},
                {"k": "Rules", "v": compact_value(rule_ids)},
                {"k": "Policy", "v": r.get("policy", "")},
            ],
        })
    return [{
        "id": "safepyramid",
        "title": "SafePyramid",
        "desc": ("Policy-conditioned, multi-turn conversations across 10 domains and three "
                 "difficulty levels. Each card includes the complete conversation, policy, "
                 "applicable rule IDs, and gold violated-rule set."),
        "stat": f"{len(rows)} policy-conversation audits",
        "samples": samples,
    }]


# ---------------------------------------------------------------------------
# collection 3: hardened mortgage guard benchmark
# ---------------------------------------------------------------------------
def build_hard_guard() -> list[dict] | None:
    if not HARD_GUARD.exists():
        return None
    rows = load_jsonl(HARD_GUARD)
    safe = sum(r.get("label_binary") == "allow" for r in rows)
    unsafe = sum(r.get("label_binary") == "flag" for r in rows)
    samples = []
    for sample_number, r in enumerate(rows, start=1):
        label = r.get("label_binary", "review")
        tags = [
            r.get("label_category"), r.get("label_subcategory"), r.get("trap_type"),
            r.get("difficulty"), r.get("persona"),
        ]
        samples.append({
            "sample": sample_number,
            "id": r.get("id", ""),
            "label": label,
            "lc": "safe" if label == "allow" else "unsafe" if label == "flag" else "review",
            "tags": [t for t in tags if t and t != "none"],
            "text": r.get("text", ""),
            "meta": [
                {"k": "ID", "v": r.get("id", f"sample-{sample_number}")},
                {"k": "Why", "v": r.get("rationale", "")},
            ],
        })
    return [{
        "id": "guard_benchmark_hard",
        "title": "Hardened Mortgage Guard Benchmark",
        "desc": ("Trap-typed mortgage/lending prompts designed to test subtle violations and "
                 "over-refusal. Includes hard negatives, minimal pairs, euphemisms, coded proxies, "
                 "buried injections, and multi-turn cases."),
        "stat": f"{safe} allow / {unsafe} flag",
        "samples": samples,
    }]


# ---------------------------------------------------------------------------
# collection 4: MortgageGuardBench-2K (synthetic)
# ---------------------------------------------------------------------------
def build_mortgage() -> list[dict] | None:
    if not MORTGAGE.exists():
        return None
    rows = load_jsonl(MORTGAGE)
    # Count the binary audit subset, but expose every task in the explorer.
    safe_side, unsafe_side = [], []
    for r in rows:
        dec = (r.get("gold_answer") or {}).get("decision")
        if not dec:
            continue
        lc = label_class(dec)
        if lc == "safe":
            safe_side.append(r)
        elif lc == "unsafe":
            unsafe_side.append(r)

    samples = []
    for sample_number, r in enumerate(rows, start=1):
        ga = r.get("gold_answer") or {}
        dec = ga.get("decision")
        raw_label = dec or r.get("answer_type") or "task"
        display_label = str(raw_label).replace("_", " ")
        tags = [r.get("domain", ""), r.get("subdomain", ""), r.get("task_type", "")]
        tags = [t for t in tags if t]
        meta = [{"k": "ID", "v": r.get("id", f"sample-{sample_number}")}]
        if r.get("candidate_response") and r.get("user_prompt"):
            meta.append({"k": "Prompt", "v": r["user_prompt"]})
        if r.get("choices"):
            choices = "\n".join(
                f"{choice.get('choice_id', '?')} — {choice.get('text', '')}"
                + (f" [{choice['action_code']}]" if choice.get("action_code") else "")
                for choice in r["choices"]
            )
            meta.append({"k": "Choices", "v": choices})
        meta.append({"k": "Gold", "v": compact_value(ga)})
        if r.get("policy_context"):
            policies = "\n".join(
                f"{card.get('card_id', 'Policy')} — {card.get('text', '')}"
                for card in r["policy_context"]
            )
            meta.append({"k": "Policy", "v": policies})
        if r.get("rationale"):
            meta.append({"k": "Why", "v": r["rationale"]})
        samples.append({
            "sample": sample_number,
            "id": r.get("id", ""),
            "label": display_label,
            "lc": label_class(raw_label),
            "tags": tags,
            "title": r.get("scenario", ""),
            "text": r.get("candidate_response") or r.get("user_prompt") or "",
            "meta": meta,
        })

    return [{
        "id": "mortgage",
        "title": "MortgageGuardBench-2K",
        "desc": ("Synthetic mortgage-compliance benchmark (10 lending domains). It is a multi-task "
                 "audit suite. All binary audits, multiple-choice controls, calculations, "
                 "needs-human-review items, and agent-action tasks are included. Binary decisions "
                 "map to safe / unsafe; other task types appear under Other."),
        "stat": f"{len(safe_side)} safe-side / {len(unsafe_side)} unsafe-side auditable rows (of {len(rows)})",
        "samples": samples,
    }]


# ---------------------------------------------------------------------------
# collection 5: ExpGuard (gated — local only)
# ---------------------------------------------------------------------------
def find_expguard_parquet() -> Path | None:
    base = Path.home() / ".cache" / "huggingface" / "hub"
    hits = sorted(base.glob(f"datasets--6rightjade--expguardmix/snapshots/*/{EXPGUARD_FILE}"))
    if hits:
        return hits[0]

    token = os.environ.get("HF_TOKEN")
    env_path = ROOT / ".env"
    if not token and env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not token:
        return None
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(
            EXPGUARD_DATASET, EXPGUARD_FILE, repo_type="dataset", token=token,
        ))
    except Exception as exc:
        print(f"skip expguard download: {type(exc).__name__}")
        return None


def build_expguard() -> list[dict] | None:
    pq = find_expguard_parquet()
    if pq is None:
        return None
    try:
        import pandas as pd
    except Exception:
        return None
    df = pd.read_parquet(pq)
    out = []
    for dom, title in (("finance", "Finance"), ("healthcare", "Healthcare"), ("law", "Law")):
        sub = df[df["domain"] == dom]
        safe_idx = list(sub.index[sub["prompt_label"] == "safe"])
        unsafe_idx = list(sub.index[sub["prompt_label"] == "unsafe"])
        samples = []
        for sample_number, i in enumerate(sub.index, start=1):
            r = df.loc[i]
            def cell(key: str) -> str:
                value = r.get(key)
                return "" if pd.isna(value) else str(value)

            tags = [cell("prompt_category"), cell("scenario")]
            tags = [t for t in tags if t]
            resp = cell("response")
            meta = [
                {"k": "ID", "v": f"expguard-{i}"},
                {"k": "Response", "v": resp},
                {"k": "Resp label", "v": cell("response_label")},
                {"k": "Resp cat", "v": cell("response_category")},
            ]
            samples.append({
                "sample": sample_number,
                "id": f"expguard-{i}",
                "label": cell("prompt_label"),
                "lc": label_class(r["prompt_label"]),
                "tags": tags,
                "text": cell("prompt"),
                "meta": meta,
            })
        out.append({
            "id": f"expguard_{dom}",
            "title": f"ExpGuard · {title}",
            "desc": f"Expert-annotated {title.lower()} prompts (safe / unsafe on the input prompt). "
                    f"Part of the gated 6rightjade/expguardmix external-validation set.",
            "stat": f"{len(safe_idx)} safe / {len(unsafe_idx)} unsafe in full {title.lower()} set",
            "samples": samples,
        })
    return out


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGETITLE__</title>
<style>
:root{
  --bg:#f6f7f9;--panel:#fff;--panel-2:#fbfcfd;--ink:#1a1d21;--ink-soft:#565d66;--ink-faint:#8a929c;
  --line:#e6e9ee;--line-strong:#d3d8e0;--accent:#4f46e5;--accent-soft:#eef0fe;
  --safe:#0f9d6b;--safe-bg:#e7f6ef;--safe-line:#bfe6d3;
  --unsafe:#d5443a;--unsafe-bg:#fdecea;--unsafe-line:#f3c6c1;
  --review:#b7791f;--review-bg:#fdf3e0;--review-line:#efd9ad;
  --shadow:0 1px 2px rgba(16,24,40,.04),0 4px 16px rgba(16,24,40,.06);
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
html[data-theme="dark"]{
  --bg:#0e1116;--panel:#161a21;--panel-2:#1b2029;--ink:#e7ebf0;--ink-soft:#a8b1bd;--ink-faint:#6f7885;
  --line:#262c36;--line-strong:#333b47;--accent:#8b85ff;--accent-soft:#23243a;
  --safe:#3fd39a;--safe-bg:#123026;--safe-line:#1e4a3a;
  --unsafe:#ff7a70;--unsafe-bg:#331a18;--unsafe-line:#5a2b28;
  --review:#e0b25a;--review-bg:#2e2410;--review-line:#4d3d1a;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 22px rgba(0,0,0,.35);
}
*{box-sizing:border-box}html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:32px 24px 80px}
.nav{display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap}
.nav a{font-size:13px;font-weight:550;text-decoration:none;color:var(--ink-soft);border:1px solid var(--line-strong);
  background:var(--panel);border-radius:999px;padding:6px 14px;transition:all .15s}
.nav a:hover{color:var(--ink);border-color:var(--accent)}
.nav a.here{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.banner{background:var(--review-bg);border:1px solid var(--review-line);color:var(--review);
  border-radius:10px;padding:11px 15px;font-size:13px;margin-bottom:20px;font-weight:500}
header.masthead{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin-bottom:8px}
.title-block h1{font-size:26px;font-weight:680;letter-spacing:-.02em;margin:0 0 6px}
.title-block p{margin:0;color:var(--ink-soft);font-size:14.5px;max-width:680px}
.theme-toggle{flex:none;border:1px solid var(--line-strong);background:var(--panel);color:var(--ink-soft);
  border-radius:9px;padding:8px 12px;cursor:pointer;font-size:13px;font-family:var(--sans);
  display:inline-flex;align-items:center;gap:7px;transition:border-color .15s,color .15s}
.theme-toggle:hover{color:var(--ink);border-color:var(--accent)}
.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:22px 0 26px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;box-shadow:var(--shadow)}
.stat .num{font-size:24px;font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.stat .lbl{font-size:12px;color:var(--ink-faint);text-transform:uppercase;letter-spacing:.06em;margin-top:2px}
.stat.safe .num{color:var(--safe)}.stat.unsafe .num{color:var(--unsafe)}.stat.review .num{color:var(--review)}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:20px;position:sticky;top:0;z-index:20;
  background:linear-gradient(var(--bg) 72%,transparent);padding:12px 0 14px}
.search{flex:1 1 260px;min-width:220px;position:relative}
.search input{width:100%;border:1px solid var(--line-strong);background:var(--panel);color:var(--ink);
  border-radius:10px;padding:10px 14px 10px 36px;font-size:14px;font-family:var(--sans);transition:border-color .15s,box-shadow .15s}
.search input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.search svg{position:absolute;left:11px;top:50%;transform:translateY(-50%);color:var(--ink-faint)}
.seg{display:inline-flex;background:var(--panel);border:1px solid var(--line-strong);border-radius:10px;padding:3px}
.seg button{border:none;background:none;color:var(--ink-soft);font-family:var(--sans);font-size:13px;
  padding:6px 14px;border-radius:7px;cursor:pointer;font-weight:550;transition:all .15s}
.seg button.on{background:var(--accent);color:#fff}
.seg button.on.safe{background:var(--safe)}.seg button.on.unsafe{background:var(--unsafe)}.seg button.on.review{background:var(--review)}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:26px}
.tab{border:1px solid var(--line-strong);background:var(--panel);color:var(--ink-soft);border-radius:999px;
  padding:7px 15px;font-size:13px;font-weight:550;cursor:pointer;font-family:var(--sans);
  display:inline-flex;align-items:center;gap:8px;transition:all .15s}
.tab:hover{border-color:var(--accent);color:var(--ink)}
.tab.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.tab .cnt{font-variant-numeric:tabular-nums;font-size:11.5px;opacity:.65}
.bench{margin-bottom:40px;scroll-margin-top:90px}
.bench-head{border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:18px}
.bench-head h2{margin:0 0 4px;font-size:19px;font-weight:640;letter-spacing:-.01em;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.bench-head .pop{font-size:12px;font-weight:500;color:var(--ink-faint);font-variant-numeric:tabular-nums}
.bench-head p{margin:0;color:var(--ink-soft);font-size:13.5px;max-width:760px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
@media (max-width:900px){.stats{grid-template-columns:repeat(3,1fr)}}
@media (max-width:720px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}header.masthead{flex-direction:column}}
.card{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--line-strong);border-radius:12px;
  padding:15px 16px;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:10px;transition:border-color .15s,transform .1s}
.card:hover{transform:translateY(-1px)}
.card.safe{border-left-color:var(--safe)}.card.unsafe{border-left-color:var(--unsafe)}.card.review{border-left-color:var(--review)}
.card .row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.sample-id{margin-left:auto;color:var(--ink-faint);font-size:11.5px;font-family:var(--mono)}
.badge{font-size:11px;font-weight:650;letter-spacing:.04em;text-transform:uppercase;padding:3px 9px;border-radius:999px;border:1px solid transparent}
.badge.safe{color:var(--safe);background:var(--safe-bg);border-color:var(--safe-line)}
.badge.unsafe{color:var(--unsafe);background:var(--unsafe-bg);border-color:var(--unsafe-line)}
.badge.review{color:var(--review);background:var(--review-bg);border-color:var(--review-line)}
.tag{font-size:11.5px;color:var(--ink-soft);background:var(--panel-2);border:1px solid var(--line);padding:3px 9px;border-radius:999px;font-family:var(--mono)}
.tag.none{color:var(--ink-faint)}
.card .ttl{font-size:13px;font-weight:600;color:var(--ink);line-height:1.4}
.card .text{font-size:13.5px;color:var(--ink);white-space:pre-wrap;word-break:break-word;font-family:var(--mono);line-height:1.55;margin:0}
.card .text.clamped{display:-webkit-box;-webkit-line-clamp:8;-webkit-box-orient:vertical;overflow:hidden}
.more{align-self:flex-start;border:none;background:none;color:var(--accent);cursor:pointer;font-size:12px;font-family:var(--sans);padding:0;font-weight:550}
.meta{display:flex;flex-direction:column;gap:6px;border-top:1px dashed var(--line);padding-top:10px;margin-top:2px}
.mrow{display:flex;gap:8px;font-size:12.5px;line-height:1.45}
.mk{flex:none;color:var(--ink-faint);text-transform:uppercase;letter-spacing:.05em;font-size:10.5px;font-weight:650;padding-top:2px;width:52px}
.mv{color:var(--ink-soft);white-space:pre-wrap;word-break:break-word}
.pager{display:flex;align-items:center;justify-content:flex-end;gap:10px;margin:18px 0;flex-wrap:wrap}
.pager.top{margin-top:0}.pager.single{justify-content:flex-start}
.pager .range{color:var(--ink-faint);font-size:12.5px;margin-right:auto;font-variant-numeric:tabular-nums}
.pager button,.pager select{border:1px solid var(--line-strong);background:var(--panel);color:var(--ink-soft);border-radius:8px;
  padding:6px 10px;font:550 12.5px var(--sans)}
.pager button{cursor:pointer}.pager button:hover:not(:disabled){color:var(--ink);border-color:var(--accent)}
.pager button:disabled{opacity:.45;cursor:not-allowed}
.pager label{color:var(--ink-soft);font-size:12.5px;display:inline-flex;align-items:center;gap:6px;font-variant-numeric:tabular-nums}
.pager select{padding:5px 8px;cursor:pointer}
.empty{color:var(--ink-faint);font-size:14px;padding:24px 0;text-align:center}
footer{margin-top:48px;padding-top:20px;border-top:1px solid var(--line);color:var(--ink-faint);font-size:12.5px}
footer code{font-family:var(--mono);color:var(--ink-soft)}
mark{background:var(--accent-soft);color:inherit;border-radius:3px;padding:0 1px}
</style>
</head>
<body>
<div class="wrap">
  __NAV__
  __BANNER__
  <header class="masthead">
    <div class="title-block">
      <h1>__H1__</h1>
      <p>__SUBTITLE__</p>
    </div>
    <button class="theme-toggle" id="themeBtn" aria-label="Toggle theme"><span id="themeIcon">&#9789;</span><span id="themeLabel">Dark</span></button>
  </header>
  <div class="stats" id="stats"></div>
  <div class="controls">
    <div class="search">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>
      <input id="search" type="search" placeholder="Search across all samples&hellip;" autocomplete="off">
    </div>
    <div class="seg" id="labelFilter">
      <button data-f="all" class="on">All</button>
      <button data-f="safe">Safe</button>
      <button data-f="unsafe">Unsafe</button>
      <button data-f="review">Other</button>
    </div>
  </div>
  <div class="tabs" id="tabs"></div>
  <div id="content"></div>
  <footer>__FOOTER__</footer>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const PAGE_SIZE=__PAGE_SIZE__;
let curTab='all',curLabel='all',curQuery='';
const pageByBench=Object.fromEntries(DATA.map(b=>[b.id,1]));
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function hl(t,q){const e=esc(t);if(!q)return e;try{const re=new RegExp('('+q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','gi');return e.replace(re,'<mark>$1</mark>');}catch(_){return e;}}
function searchable(s){return[s.id||'',s.text,s.title||'',s.label||'',...(s.tags||[]),...((s.meta||[]).map(m=>m.v))].join(' ').toLowerCase();}
(function(){const tot=DATA.reduce((a,b)=>a+b.samples.length,0);const sf=DATA.reduce((a,b)=>a+b.samples.filter(s=>s.lc==='safe').length,0);
  const us=DATA.reduce((a,b)=>a+b.samples.filter(s=>s.lc==='unsafe').length,0);
  const rv=tot-sf-us;
  document.getElementById('stats').innerHTML=[[DATA.length,'Sections',''],[tot,'Total samples',''],[sf,'Safe','safe'],[us,'Unsafe','unsafe'],[rv,'Other','review']]
    .map(t=>`<div class="stat ${t[2]}"><div class="num">${t[0]}</div><div class="lbl">${t[1]}</div></div>`).join('');})();
(function(){const el=document.getElementById('tabs');
  el.innerHTML=`<button class="tab on" data-t="all">All <span class="cnt">${DATA.length}</span></button>`+
    DATA.map(b=>`<button class="tab" data-t="${b.id}">${esc(b.title)} <span class="cnt">${b.samples.length}</span></button>`).join('');
  el.addEventListener('click',e=>{const b=e.target.closest('.tab');if(!b)return;curTab=b.dataset.t;
    el.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===b));render();});})();
document.getElementById('labelFilter').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;curLabel=b.dataset.f;
  DATA.forEach(x=>pageByBench[x.id]=1);
  document.querySelectorAll('#labelFilter button').forEach(x=>{x.className=x===b?'on '+(b.dataset.f!=='all'?b.dataset.f:''):'';});render();});
let tmr;document.getElementById('search').addEventListener('input',e=>{clearTimeout(tmr);tmr=setTimeout(()=>{curQuery=e.target.value.trim();DATA.forEach(x=>pageByBench[x.id]=1);render();},120);});
function card(s){const long=s.text.length>300;
  const tags=(s.tags||[]).map(t=>`<span class="tag${t==='none'?' none':''}">${esc(t)}</span>`).join('');
  const sampleId=s.id||`Sample ${s.sample}`;
  const title=s.title?`<div class="ttl">${hl(s.title,curQuery)}</div>`:'';
  const meta=(s.meta||[]).length?`<div class="meta">${s.meta.map(m=>`<div class="mrow"><span class="mk">${esc(m.k)}</span><span class="mv">${hl(m.v,curQuery)}</span></div>`).join('')}</div>`:'';
  return `<div class="card ${s.lc}"><div class="row"><span class="badge ${s.lc}">${esc(s.label)}</span>${tags}<span class="sample-id">${esc(sampleId)}</span></div>${title}<p class="text${long?' clamped':''}">${hl(s.text,curQuery)}</p>${long?'<button class="more">Show more</button>':''}${meta}</div>`;}
function pager(b,page,pages,start,end,total,where){
  const range=`<span class="range">Showing ${start+1}&ndash;${end} of ${total} matching samples</span>`;
  if(pages<=1)return `<div class="pager single ${where}">${range}</div>`;
  const options=Array.from({length:pages},(_,i)=>`<option value="${i+1}"${i+1===page?' selected':''}>${i+1}</option>`).join('');
  return `<nav class="pager ${where}" aria-label="${esc(b.title)} sample pages">${range}<button data-page-action="prev" data-bench="${esc(b.id)}"${page===1?' disabled':''}>Previous</button><label>Page <select data-page-select="${esc(b.id)}" aria-label="${esc(b.title)} page">${options}</select> of ${pages}</label><button data-page-action="next" data-bench="${esc(b.id)}"${page===pages?' disabled':''}>Next</button></nav>`;}
function section(b){let ss=b.samples;
  if(curLabel!=='all')ss=ss.filter(s=>s.lc===curLabel);
  if(curQuery){const q=curQuery.toLowerCase();ss=ss.filter(s=>searchable(s).includes(q));}
  if(!ss.length)return'';
  const pages=Math.ceil(ss.length/PAGE_SIZE);const page=Math.min(pageByBench[b.id]||1,pages);pageByBench[b.id]=page;
  const start=(page-1)*PAGE_SIZE,end=Math.min(start+PAGE_SIZE,ss.length);const shown=ss.slice(start,end);
  const top=pager(b,page,pages,start,end,ss.length,'top'),bottom=pages>1?pager(b,page,pages,start,end,ss.length,'bottom'):'';
  return `<section class="bench" id="b-${b.id}"><div class="bench-head"><h2>${esc(b.title)} <span class="pop">${esc(b.stat)}</span></h2><p>${esc(b.desc)}</p></div>${top}<div class="grid">${shown.map(card).join('')}</div>${bottom}</section>`;}
function render(){const list=curTab==='all'?DATA:DATA.filter(b=>b.id===curTab);
  document.getElementById('content').innerHTML=list.map(section).join('')||`<div class="empty">No samples match your filters.</div>`;}
document.getElementById('content').addEventListener('click',e=>{const b=e.target.closest('.more');if(!b)return;
  const t=b.previousElementSibling;const on=t.classList.toggle('clamped');b.textContent=on?'Show more':'Show less';});
document.getElementById('content').addEventListener('click',e=>{const btn=e.target.closest('[data-page-action]');if(!btn)return;
  const id=btn.dataset.bench;pageByBench[id]+=(btn.dataset.pageAction==='next'?1:-1);render();
  requestAnimationFrame(()=>document.getElementById(`b-${id}`)?.scrollIntoView({behavior:'smooth',block:'start'}));});
document.getElementById('content').addEventListener('change',e=>{const select=e.target.closest('[data-page-select]');if(!select)return;
  const id=select.dataset.pageSelect;pageByBench[id]=Number(select.value);render();
  requestAnimationFrame(()=>document.getElementById(`b-${id}`)?.scrollIntoView({behavior:'smooth',block:'start'}));});
(function(){const r=document.documentElement,btn=document.getElementById('themeBtn'),ic=document.getElementById('themeIcon'),lb=document.getElementById('themeLabel');
  function set(m){r.dataset.theme=m;ic.innerHTML=m==='dark'?'&#9788;':'&#9789;';lb.textContent=m==='dark'?'Light':'Dark';}
  if(window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches)set('dark');
  btn.addEventListener('click',()=>set(r.dataset.theme==='dark'?'light':'dark'));})();
render();
</script>
</body>
</html>
"""


def render_page(*, out: Path, page_title: str, h1: str, subtitle: str,
                benchmarks: list[dict], footer: str, banner: str = "") -> None:
    payload = json.dumps(benchmarks, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (PAGE
            .replace("__PAGETITLE__", page_title)
            .replace("__H1__", h1)
            .replace("__SUBTITLE__", subtitle)
            .replace("__FOOTER__", footer)
            .replace("__NAV__", "")
            .replace("__BANNER__", banner)
            .replace("__PAGE_SIZE__", str(PAGE_SIZE))
            .replace("__DATA__", payload))
    out.write_text(html, encoding="utf-8")
    n = sum(len(b["samples"]) for b in benchmarks)
    print(f"Wrote {out.relative_to(ROOT)} ({len(html):,} bytes) — {len(benchmarks)} sections, {n} samples")


def main() -> None:
    guard = build_guard()
    pyramid = build_safepyramid() or []
    hard = build_hard_guard() or []
    mort = build_mortgage() or []
    exp = build_expguard() or []

    if not pyramid:
        print("skip safepyramid: dataset not found")
    if not hard:
        print("skip hardened mortgage guard: dataset not found")
    if not mort:
        print("skip mortgage: dataset not found")

    # ---- committed, shareable page: every gated-free benchmark source ----
    public = guard + pyramid + hard + mort
    exp_note = ""
    if not exp:
        exp_note = (" A gated ExpGuard set (finance / healthcare / law) is not included here; "
                    "run <code>generate.py</code> with the dataset cached to build the local <code>index.html</code>.")
    render_page(
        out=OUTDIR / "index.public.html",
        page_title="Guard Benchmark Explorer (public)",
        h1="Guard Benchmark Explorer",
        subtitle=("Every available row from the full public guard sets, SafePyramid, the hardened "
                  "mortgage guard benchmark, and MortgageGuardBench-2K. Search and label filters "
                  "cover the complete datasets; "
                  f"large result sets are paged {PAGE_SIZE} samples at a time."),
        benchmarks=public,
        footer=("Shareable build — public + synthetic data only. Generated by "
                "<code>benchmark-explorer/generate.py</code> in source order." + exp_note),
    )

    # ---- local, unified page: everything incl. gated ExpGuard (gitignored) ----
    local = public + exp
    banner = ""
    if exp:
        banner = ("⚠ Local only — this page includes the gated / licensed ExpGuard dataset "
                  "(finance · healthcare · law) and embeds its prompt text. It is gitignored and "
                  "must NOT be committed or shared. Share <code>index.public.html</code> instead.")
    render_page(
        out=OUTDIR / "index.html",
        page_title="Guard Benchmark Explorer (all benchmarks)",
        h1="Guard Benchmark Explorer",
        subtitle=("Every available public benchmark row"
                  + (", plus all locally cached expert-annotated ExpGuard rows. " if exp else ". ")
                  + f"Large result sets are paged {PAGE_SIZE} samples at a time."),
        benchmarks=local,
        footer=("Full local build (all benchmarks). Generated by "
                "<code>benchmark-explorer/generate.py</code> in source order."),
        banner=banner,
    )
    if exp:
        print("  NOTE: index.html is the FULL LOCAL build (incl. gated ExpGuard) and is gitignored — "
              "do not commit or share it. Commit/share index.public.html instead.")
    else:
        print("skip expguard: HF cache parquet not found (gated dataset) — index.html omits ExpGuard")


if __name__ == "__main__":
    main()
