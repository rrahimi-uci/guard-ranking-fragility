"""Async OpenAI Responses-API runner: one global work queue, one semaphore.

Design notes that matter for an ~87k-request run:

* **One queue across every (model, effort, benchmark, row).** Fanning out per config
  would idle 200 slots whenever a config's tail is slower than the rest.
* **Resumable.** Every completed row is appended to
  ``raw/{model}__{effort}__{benchmark}.jsonl`` as it lands, and a rerun skips row ids
  already present in that file. A crash or a Ctrl-C costs only the in-flight rows.
* **Text-free output.** Prediction records carry the row id, the verdict, the score
  and token counts -- never the prompt. Most sources in the distribution ledger are
  ``local_only`` or ``text_free_only``.
* **Truncation is a failure, not a zero.** A reasoning model that spends its whole
  output budget thinking returns ``status == "incomplete"`` with no JSON. That row is
  retried at double the budget, repeatedly up to the task's ``max_output_cap``, and only
  then recorded ``ok: false`` -- excluded from the metrics with a count in the report,
  never imputed as a negative.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from datasets import Record
from tasks import TASKS

HERE = Path(__file__).resolve().parent
# Mutable on purpose: run_all repoints this at raw_mock/ for --mock runs. Mock
# predictions must never land in the same cache as real ones -- they are written with
# ok:true, so a later real run would treat them as already-paid-for and silently report
# fabricated numbers. Every path goes through pred_path(), so one switch covers all of it.
RAW = HERE / "raw"
RAW_MOCK = HERE / "raw_mock"
EFFORTS = ("low", "medium", "high")

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
# Matched against the exception's whole MRO by name; see _retryable().
_RETRYABLE_EXC_NAMES = {
    "APIConnectionError", "APITimeoutError", "APIConnectionTimeoutError",
    "RateLimitError", "InternalServerError",
    "ConnectionError", "ConnectionResetError", "TimeoutError", "OSError",
    "ReadTimeout", "ConnectTimeout", "ReadError", "WriteError", "RemoteProtocolError",
    "PoolTimeout", "ConnectError",
}
MAX_ATTEMPTS = 6


def percentile(values, q: float) -> float:
    """Linearly-interpolated percentile (numpy's default convention), no numpy import.

    ``q`` is in percent. Returns nan for an empty sample so a config with no successful
    call reports "n/a" rather than 0.0, which would read as "instant".
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def latency_stats(values) -> dict:
    """mean / p50 / p90 / p99 / max over per-request seconds, plus the sample size."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"),
                "p99": float("nan"), "max": float("nan")}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p99": percentile(vals, 99),
        "max": max(vals),
    }


def load_env(path: Path | None = None) -> None:
    """Populate os.environ from the repo .env without clobbering a real export."""
    path = path or HERE.parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def pred_path(model: str, effort: str, benchmark: str) -> Path:
    return RAW / f"{model.replace('/', '_')}__{effort}__{benchmark}.jsonl"


def read_done(path: Path) -> dict[str, dict]:
    """Existing predictions keyed by row id. A duplicated id keeps the last write."""
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a torn final line from a hard kill
                continue
            if rec.get("rid"):
                out[rec["rid"]] = rec
    return out


@dataclass
class Job:
    model: str
    effort: str
    record: Record
    queued_at: float = 0.0  # set when the queue is handed to run_all


class Usage:
    """Token and call totals, plus a live progress line."""

    def __init__(self, total_jobs: int, cached: int = 0) -> None:
        self.total = total_jobs
        self.cached = cached
        self.done = 0
        self.failed = 0
        self.retries = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.latencies: list[float] = []
        self.per_model: dict[str, dict[str, int]] = {}
        self.started = time.time()
        self._lock = asyncio.Lock()

    async def add(self, model: str, in_tok: int, out_tok: int, reas_tok: int, ok: bool,
                  latency: float | None = None) -> None:
        async with self._lock:
            self.done += 1
            if not ok:
                self.failed += 1
            if latency is not None:
                self.latencies.append(latency)
            self.input_tokens += in_tok
            self.output_tokens += out_tok
            self.reasoning_tokens += reas_tok
            slot = self.per_model.setdefault(model, {"calls": 0, "in": 0, "out": 0, "reasoning": 0})
            slot["calls"] += 1
            slot["in"] += in_tok
            slot["out"] += out_tok
            slot["reasoning"] += reas_tok

    def line(self) -> str:
        elapsed = max(time.time() - self.started, 1e-6)
        rate = self.done / elapsed
        left = (self.total - self.done) / rate if rate > 0 else float("inf")
        p50 = percentile(self.latencies, 50)
        return (
            f"{self.done}/{self.total} done ({self.failed} failed) "
            f"| {rate:.1f} rows/s | eta {left / 60:.1f} min "
            f"| p50 {p50:.1f}s "
            f"| tok in {self.input_tokens / 1e6:.2f}M out {self.output_tokens / 1e6:.2f}M "
            f"(reasoning {self.reasoning_tokens / 1e6:.2f}M) | retries {self.retries}"
        )

    def summary(self) -> dict:
        return {
            "latency_run": latency_stats(self.latencies),
            "jobs_total": self.total,
            "jobs_cached_before_run": self.cached,
            "jobs_executed": self.done,
            "jobs_failed": self.failed,
            "retries": self.retries,
            "wall_seconds": round(time.time() - self.started, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "per_model": self.per_model,
        }


# ────────────────────────────────────────────────────────────────── mock backend
def _mock_response(job: Job) -> dict:
    """Deterministic pseudo-predictions: exercises the whole path with no network.

    Seeded by (model, effort, row id) so it is reproducible, and biased toward the
    gold label so the metrics come out plausible rather than at chance -- the point is
    to validate plumbing, not to fake a result.
    """
    rec = job.record
    seed = int(hashlib.blake2b(
        f"{job.model}|{job.effort}|{rec.rid}".encode(), digest_size=8).hexdigest(), 16)
    rng = random.Random(seed)
    if rec.task == "rule_attribution":
        hit = {r for r in rec.gold_rules if rng.random() < 0.6}
        miss = {r for r in rec.candidate_rules if r not in rec.gold_rules and rng.random() < 0.05}
        return {"violated": [{"rule": r, "confidence": rng.randint(50, 99)}
                             for r in sorted(hit | miss)]}
    positive = TASKS[rec.task]["positive"]
    negative = next(v for v in TASKS[rec.task]["schema"]["properties"]["verdict"]["enum"]
                    if v != positive)
    correct = rng.random() < 0.82
    is_pos = bool(rec.label) if correct else not bool(rec.label)
    risk = rng.randint(55, 99) if is_pos else rng.randint(1, 45)
    return {"verdict": positive if is_pos else negative, "risk": risk}


# ────────────────────────────────────────────────────────────────── real backend
async def _call_once(client, job: Job, max_tokens: int):
    spec = TASKS[job.record.task]
    return await client.responses.create(
        model=job.model,
        reasoning={"effort": job.effort},
        instructions=spec["instruction"],
        input=job.record.text,
        text={"format": {"type": "json_schema", "name": "guard_verdict",
                         "schema": spec["schema"], "strict": True}},
        max_output_tokens=max_tokens,
    )


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) if resp is not None else None


def _retryable(exc: Exception) -> bool:
    code = _status_code(exc)
    if code is not None:
        return code in _RETRYABLE_STATUS
    # Transport failures arrive with no status code. Matching on class *name* rather than
    # isinstance is deliberate: the openai SDK's APIConnectionError derives from its own
    # APIError, not from OSError, so an isinstance check against the builtin socket
    # exceptions silently classifies a plain connection reset as permanent and drops the
    # row without a single retry.
    names = {type(exc).__name__} | {c.__name__ for c in type(exc).__mro__}
    return bool(names & _RETRYABLE_EXC_NAMES)


async def run_job(client, job: Job, usage: Usage, sink, sem: asyncio.Semaphore,
                  mock: bool) -> None:
    rec = job.record
    base = {"rid": rec.rid, "benchmark": rec.benchmark, "task": rec.task,
            "model": job.model, "effort": job.effort}
    if mock:
        started = time.time()
        payload = _mock_response(job)
        elapsed = time.time() - started
        out = dict(base, ok=True, raw=payload, in_tok=0, out_tok=0,
                   reasoning_tok=0, attempts=1, mock=True,
                   latency_s=round(elapsed, 4), queued_s=0.0)
        await usage.add(job.model, 0, 0, 0, True, latency=elapsed)
        await sink(job, out)
        return

    budget = TASKS[rec.task]["max_output_tokens"]
    last_error = "unknown"
    in_tok = out_tok = reas_tok = 0
    latency = None
    attempts_used = 0
    async with sem:
        # `queued_s` is time spent waiting for a concurrency slot; it is not the
        # model's latency and is kept separate so the percentiles stay meaningful.
        queued = time.time() - job.queued_at
        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts_used = attempt
            call_started = time.time()
            try:
                resp = await _call_once(client, job, budget)
                latency = time.time() - call_started
                u = getattr(resp, "usage", None)
                if u is not None:
                    in_tok = int(u.input_tokens or 0)
                    out_tok = int(u.output_tokens or 0)
                    details = getattr(u, "output_tokens_details", None)
                    reas_tok = int(getattr(details, "reasoning_tokens", 0) or 0)
                text = (resp.output_text or "").strip()
                if getattr(resp, "status", None) == "incomplete" or not text:
                    reason = getattr(getattr(resp, "incomplete_details", None), "reason", "empty")
                    last_error = f"incomplete:{reason}@{budget}"
                    cap = TASKS[rec.task]["max_output_cap"]
                    if budget < cap:  # climb the ladder, then give up on the row
                        budget = min(budget * 2, cap)
                        usage.retries += 1
                        continue
                    break
                payload = json.loads(text)
                out = dict(base, ok=True, raw=payload, in_tok=in_tok, out_tok=out_tok,
                           reasoning_tok=reas_tok, attempts=attempt,
                           latency_s=round(latency, 3), queued_s=round(queued, 3))
                await usage.add(job.model, in_tok, out_tok, reas_tok, True,
                                latency=latency)
                await sink(job, out)
                return
            except json.JSONDecodeError as exc:  # strict schema should prevent this
                last_error = f"bad_json:{exc}"
                break
            except Exception as exc:  # noqa: BLE001 - the SDK raises many types
                latency = time.time() - call_started
                last_error = f"{type(exc).__name__}:{str(exc)[:160]}"
                if attempt == MAX_ATTEMPTS or not _retryable(exc):
                    break
                usage.retries += 1
                delay = min(60.0, 1.5 * 2 ** (attempt - 1)) * (0.5 + random.random())
                await asyncio.sleep(delay)

    out = dict(base, ok=False, raw=None, error=last_error, in_tok=in_tok,
               out_tok=out_tok, reasoning_tok=reas_tok, attempts=attempts_used,
               latency_s=round(latency, 3) if latency is not None else None,
               queued_s=round(queued, 3))
    await usage.add(job.model, in_tok, out_tok, reas_tok, False)
    await sink(job, out)


async def _progress(usage: Usage, every: float = 20.0) -> None:
    while True:
        await asyncio.sleep(every)
        print(f"  … {usage.line()}", flush=True)


async def run_all(jobs: list[Job], *, concurrency: int, mock: bool,
                  cached: int = 0, timeout: float = 600.0) -> Usage:
    """Execute `jobs`, appending each result to its per-config file as it lands."""
    RAW.mkdir(parents=True, exist_ok=True)
    handles: dict[Path, object] = {}
    locks: dict[Path, asyncio.Lock] = {}

    async def sink(job: Job, record: dict) -> None:
        path = pred_path(job.model, job.effort, job.record.benchmark)
        if path not in handles:
            handles[path] = path.open("a")
            locks[path] = asyncio.Lock()
        async with locks[path]:
            handles[path].write(json.dumps(record) + "\n")
            handles[path].flush()

    queued_at = time.time()
    for job in jobs:
        job.queued_at = queued_at
    usage = Usage(len(jobs), cached=cached)
    client = None
    if not mock:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(timeout=timeout, max_retries=0)  # retries handled here
    sem = asyncio.Semaphore(concurrency)
    ticker = asyncio.create_task(_progress(usage))
    try:
        await asyncio.gather(*(run_job(client, j, usage, sink, sem, mock) for j in jobs))
    finally:
        ticker.cancel()
        for handle in handles.values():
            handle.close()
        if client is not None:
            await client.close()
    print(f"  ✓ {usage.line()}", flush=True)
    return usage
