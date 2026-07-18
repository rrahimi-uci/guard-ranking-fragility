#!/usr/bin/env python3
"""Generate self-contained HTML explorers for the guard evaluation benchmarks.

Produces (all fully offline, data embedded):
  index.html      7 public safe/unsafe guard benchmarks           (committable)
  mortgage.html   MortgageGuardBench-2K, synthetic audit tasks     (committable)
  expguard.html   ExpGuard finance/healthcare/law                  (LOCAL ONLY —
                  gated/licensed dataset; NOT committed; gitignored)

Usage:
    python benchmark-explorer/generate.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "data" / "benchmarks"
MORTGAGE = ROOT / "data" / "mortgage_guard_bench_2k_v0_1_0" / "data" / "mortgage_guard_bench_full.jsonl"
OUTDIR = Path(__file__).resolve().parent

N_PER_LABEL = 10
SEED = 42

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


def clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


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
        rng = random.Random(f"{SEED}:{name}")
        rows = load_jsonl(BENCH_DIR / f"{name}.jsonl")
        safe = [r for r in rows if r.get("label") == "safe"]
        unsafe = [r for r in rows if r.get("label") == "unsafe"]
        rng.shuffle(safe)
        rng.shuffle(unsafe)
        picked = safe[:N_PER_LABEL] + unsafe[:N_PER_LABEL]
        samples = []
        for r in picked:
            hz = r.get("hazard", "none") or "none"
            samples.append({
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
# collection 2: MortgageGuardBench-2K (synthetic)
# ---------------------------------------------------------------------------
def build_mortgage() -> list[dict] | None:
    if not MORTGAGE.exists():
        return None
    rng = random.Random(f"{SEED}:mortgage")
    rows = load_jsonl(MORTGAGE)
    # keep rows whose gold decision maps to a binary safe/unsafe audit verdict
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

    def round_robin_by_domain(pool: list[dict], k: int) -> list[dict]:
        by_dom: dict[str, list[dict]] = {}
        for r in pool:
            by_dom.setdefault(r.get("domain", "?"), []).append(r)
        for d in by_dom.values():
            rng.shuffle(d)
        doms = sorted(by_dom)
        rng.shuffle(doms)
        picked, i = [], 0
        while len(picked) < k and any(by_dom.values()):
            d = doms[i % len(doms)]
            if by_dom[d]:
                picked.append(by_dom[d].pop())
            i += 1
        return picked[:k]

    picked = round_robin_by_domain(safe_side, N_PER_LABEL) + \
        round_robin_by_domain(unsafe_side, N_PER_LABEL)

    samples = []
    for r in picked:
        ga = r.get("gold_answer") or {}
        dec = ga.get("decision", "")
        risk = ga.get("risk_code") or ga.get("required_action_code") or ga.get("action_code")
        tags = [r.get("domain", ""), r.get("subdomain", ""), r.get("task_type", "")]
        tags = [t for t in tags if t]
        meta = []
        gold_bits = f"{dec}" + (f"  ·  {risk}" if risk else "")
        meta.append({"k": "Gold", "v": gold_bits})
        if r.get("rationale"):
            meta.append({"k": "Why", "v": clip(r["rationale"], 260)})
        samples.append({
            "label": dec,
            "lc": label_class(dec),
            "tags": tags,
            "title": clip(r.get("scenario", ""), 240),
            "text": r.get("candidate_response") or r.get("user_prompt") or "",
            "meta": meta,
        })

    return [{
        "id": "mortgage",
        "title": "MortgageGuardBench-2K",
        "desc": ("Synthetic mortgage-compliance benchmark (10 lending domains). It is a multi-task "
                 "audit suite — these 20 cards are the binary-auditable subset (candidate-response, "
                 "security, and RAG-grounding audits) mapped to safe / unsafe; the full set also has "
                 "multiple-choice, numeric, and needs-human-review items. Each card shows the "
                 "candidate response under audit."),
        "stat": f"{len(safe_side)} safe-side / {len(unsafe_side)} unsafe-side auditable rows (of {len(rows)})",
        "samples": samples,
    }]


# ---------------------------------------------------------------------------
# collection 3: ExpGuard (gated — local only)
# ---------------------------------------------------------------------------
def find_expguard_parquet() -> Path | None:
    base = Path.home() / ".cache" / "huggingface" / "hub"
    hits = list(base.glob("datasets--6rightjade--expguardmix/snapshots/*/expguardtest.parquet"))
    return hits[0] if hits else None


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
        rng = random.Random(f"{SEED}:expguard:{dom}")
        safe_idx = list(sub.index[sub["prompt_label"] == "safe"])
        unsafe_idx = list(sub.index[sub["prompt_label"] == "unsafe"])
        rng.shuffle(safe_idx)
        rng.shuffle(unsafe_idx)
        picked = safe_idx[:N_PER_LABEL] + unsafe_idx[:N_PER_LABEL]
        samples = []
        for i in picked:
            r = df.loc[i]
            tags = [str(r.get("prompt_category") or ""), str(r.get("scenario") or "")]
            tags = [t for t in tags if t]
            resp = str(r.get("response") or "")
            meta = [{"k": "Model response", "v": clip(resp, 260)}] if resp else []
            samples.append({
                "label": str(r["prompt_label"]),
                "lc": label_class(r["prompt_label"]),
                "tags": tags,
                "text": str(r.get("prompt") or ""),
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
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0 26px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;box-shadow:var(--shadow)}
.stat .num{font-size:24px;font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.stat .lbl{font-size:12px;color:var(--ink-faint);text-transform:uppercase;letter-spacing:.06em;margin-top:2px}
.stat.safe .num{color:var(--safe)}.stat.unsafe .num{color:var(--unsafe)}
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
.seg button.on.safe{background:var(--safe)}.seg button.on.unsafe{background:var(--unsafe)}
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
@media (max-width:720px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}header.masthead{flex-direction:column}}
.card{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--line-strong);border-radius:12px;
  padding:15px 16px;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:10px;transition:border-color .15s,transform .1s}
.card:hover{transform:translateY(-1px)}
.card.safe{border-left-color:var(--safe)}.card.unsafe{border-left-color:var(--unsafe)}.card.review{border-left-color:var(--review)}
.card .row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
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
.mv{color:var(--ink-soft)}
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
      <input id="search" type="search" placeholder="Search across all shown text&hellip;" autocomplete="off">
    </div>
    <div class="seg" id="labelFilter">
      <button data-f="all" class="on">All</button>
      <button data-f="safe">Safe</button>
      <button data-f="unsafe">Unsafe</button>
    </div>
  </div>
  <div class="tabs" id="tabs"></div>
  <div id="content"></div>
  <footer>__FOOTER__</footer>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
let curTab='all',curLabel='all',curQuery='';
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function hl(t,q){const e=esc(t);if(!q)return e;try{const re=new RegExp('('+q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','gi');return e.replace(re,'<mark>$1</mark>');}catch(_){return e;}}
function searchable(s){return[s.text,s.title||'',...(s.tags||[]),...((s.meta||[]).map(m=>m.v))].join(' ').toLowerCase();}
(function(){const tot=DATA.reduce((a,b)=>a+b.samples.length,0);const sf=DATA.reduce((a,b)=>a+b.samples.filter(s=>s.lc==='safe').length,0);
  const us=DATA.reduce((a,b)=>a+b.samples.filter(s=>s.lc==='unsafe').length,0);
  document.getElementById('stats').innerHTML=[[DATA.length,'Sections',''],[tot,'Total samples',''],[sf,'Safe','safe'],[us,'Unsafe','unsafe']]
    .map(t=>`<div class="stat ${t[2]}"><div class="num">${t[0]}</div><div class="lbl">${t[1]}</div></div>`).join('');})();
(function(){const el=document.getElementById('tabs');
  el.innerHTML=`<button class="tab on" data-t="all">All <span class="cnt">${DATA.length}</span></button>`+
    DATA.map(b=>`<button class="tab" data-t="${b.id}">${esc(b.title)} <span class="cnt">${b.samples.length}</span></button>`).join('');
  el.addEventListener('click',e=>{const b=e.target.closest('.tab');if(!b)return;curTab=b.dataset.t;
    el.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===b));render();});})();
document.getElementById('labelFilter').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;curLabel=b.dataset.f;
  document.querySelectorAll('#labelFilter button').forEach(x=>{x.className=x===b?'on '+(b.dataset.f!=='all'?b.dataset.f:''):'';});render();});
let tmr;document.getElementById('search').addEventListener('input',e=>{clearTimeout(tmr);tmr=setTimeout(()=>{curQuery=e.target.value.trim();render();},120);});
function card(s){const long=s.text.length>300;
  const tags=(s.tags||[]).map(t=>`<span class="tag${t==='none'?' none':''}">${esc(t)}</span>`).join('');
  const title=s.title?`<div class="ttl">${hl(s.title,curQuery)}</div>`:'';
  const meta=(s.meta||[]).length?`<div class="meta">${s.meta.map(m=>`<div class="mrow"><span class="mk">${esc(m.k)}</span><span class="mv">${hl(m.v,curQuery)}</span></div>`).join('')}</div>`:'';
  return `<div class="card ${s.lc}"><div class="row"><span class="badge ${s.lc}">${esc(s.label)}</span>${tags}</div>${title}<p class="text${long?' clamped':''}">${hl(s.text,curQuery)}</p>${long?'<button class="more">Show more</button>':''}${meta}</div>`;}
function section(b){let ss=b.samples;
  if(curLabel!=='all')ss=ss.filter(s=>s.lc===curLabel);
  if(curQuery){const q=curQuery.toLowerCase();ss=ss.filter(s=>searchable(s).includes(q));}
  if(!ss.length)return'';
  return `<section class="bench" id="b-${b.id}"><div class="bench-head"><h2>${esc(b.title)} <span class="pop">${esc(b.stat)}</span></h2><p>${esc(b.desc)}</p></div><div class="grid">${ss.map(card).join('')}</div></section>`;}
function render(){const list=curTab==='all'?DATA:DATA.filter(b=>b.id===curTab);
  document.getElementById('content').innerHTML=list.map(section).join('')||`<div class="empty">No samples match your filters.</div>`;}
document.getElementById('content').addEventListener('click',e=>{const b=e.target.closest('.more');if(!b)return;
  const t=b.previousElementSibling;const on=t.classList.toggle('clamped');b.textContent=on?'Show more':'Show less';});
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
    payload = json.dumps(benchmarks, ensure_ascii=False).replace("</", "<\\/")
    html = (PAGE
            .replace("__PAGETITLE__", page_title)
            .replace("__H1__", h1)
            .replace("__SUBTITLE__", subtitle)
            .replace("__FOOTER__", footer)
            .replace("__NAV__", "")
            .replace("__BANNER__", banner)
            .replace("__DATA__", payload))
    out.write_text(html, encoding="utf-8")
    n = sum(len(b["samples"]) for b in benchmarks)
    print(f"Wrote {out.relative_to(ROOT)} ({len(html):,} bytes) — {len(benchmarks)} sections, {n} samples")


def main() -> None:
    guard = build_guard()
    mort = build_mortgage() or []
    exp = build_expguard() or []

    if not mort:
        print("skip mortgage: dataset not found")

    # ---- committed, shareable page: only gated-free data (guard + mortgage) ----
    public = guard + mort
    p_total = sum(len(b["samples"]) for b in public)
    exp_note = ""
    if not exp:
        exp_note = (" A gated ExpGuard set (finance / healthcare / law) is not included here; "
                    "run <code>generate.py</code> with the dataset cached to build the local <code>index.html</code>.")
    render_page(
        out=OUTDIR / "index.public.html",
        page_title="Guard Benchmark Explorer (public)",
        h1="Guard Benchmark Explorer",
        subtitle=(f"{p_total} hand-sampled prompts — {N_PER_LABEL} safe and {N_PER_LABEL} unsafe from each "
                  f"benchmark. Each card shows a real prompt with its <strong>safe / unsafe</strong> label. "
                  "Covers the public guard benchmarks plus the synthetic MortgageGuardBench-2K."),
        benchmarks=public,
        footer=("Shareable build — public + synthetic data only. Generated by "
                "<code>benchmark-explorer/generate.py</code> · deterministic (seed 42)." + exp_note),
    )

    # ---- local, unified page: everything incl. gated ExpGuard (gitignored) ----
    local = guard + mort + exp
    l_total = sum(len(b["samples"]) for b in local)
    banner = ""
    if exp:
        banner = ("⚠ Local only — this page includes the gated / licensed ExpGuard dataset "
                  "(finance · healthcare · law) and embeds its prompt text. It is gitignored and "
                  "must NOT be committed or shared. Share <code>index.public.html</code> instead.")
    render_page(
        out=OUTDIR / "index.html",
        page_title="Guard Benchmark Explorer (all benchmarks)",
        h1="Guard Benchmark Explorer",
        subtitle=(f"{l_total} hand-sampled prompts — {N_PER_LABEL} safe and {N_PER_LABEL} unsafe from each of "
                  f"{len(local)} benchmark sections: public guard benchmarks, synthetic MortgageGuardBench-2K"
                  + (", and the expert-annotated ExpGuard domains." if exp else ".")),
        benchmarks=local,
        footer=("Full local build (all benchmarks). Generated by "
                "<code>benchmark-explorer/generate.py</code> · deterministic (seed 42)."),
        banner=banner,
    )
    if exp:
        print("  NOTE: index.html is the FULL LOCAL build (incl. gated ExpGuard) and is gitignored — "
              "do not commit or share it. Commit/share index.public.html instead.")
    else:
        print("skip expguard: HF cache parquet not found (gated dataset) — index.html omits ExpGuard")


if __name__ == "__main__":
    main()
