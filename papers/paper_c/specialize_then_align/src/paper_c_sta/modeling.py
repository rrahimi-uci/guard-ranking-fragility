"""Prompt rendering, the three-way action head, and masked response log-probabilities.

Everything downstream -- SFT, distillation, PairCE, CM-DPO, calibration, scoring --
reads the model through exactly two primitives defined here, so no arm can quietly
use a different decision surface than another:

``action_logits``
    One forward pass to a fixed continuation point.  The prompt ends with the literal
    ``{"action": "`` so the very next token decides the action, and the three action
    words are read off that single position.  This is the repository's single-token
    logit head generalised from two classes to three: one pass, no autoregressive
    generation, no free-text parsing, and the model's own confidence preserved.

``response_logprob``
    Sum of response-token log-probabilities with every prompt token masked, which is
    the reduction ``config.alignment.pair_logprob_reduction`` names.  Both CM-DPO
    arms and PairCE consume this and nothing else.

The action words are verified at load time to occupy distinct first tokens under the
tokenizer in use.  If a backbone ever tokenised two of them identically the head
would silently conflate two actions, so that check raises instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json

from .contracts import ContractError

ACTIONS = ("allow", "review", "intervene")
PROMPT_TAIL = '{"action": "'

SYSTEM = (
    "You are a compliance and safety guard. You are given an event: a request, the "
    "assistant's proposed response or action, structured context, and where "
    "applicable the governing policy excerpt. Decide one action.\n"
    "  allow     - the proposed response or action may proceed unchanged.\n"
    "  review    - the decisive facts, coverage, or governing policy are genuinely "
    "unresolved and a human must decide.\n"
    "  intervene - the proposed response or action must be blocked or corrected.\n"
    "You screen for risk. You do not approve or deny credit, give legal advice, or "
    "certify compliance. Answer with JSON."
)


def render_prompt(sample: Mapping) -> str:
    """Render an event into the fixed guard prompt, ending at the action position."""
    parts = [f"REQUEST:\n{sample.get('request', '')}"]
    proposed = sample.get("proposed_response")
    if proposed:
        parts.append(f"PROPOSED RESPONSE:\n{proposed}")
    context = sample.get("context")
    if isinstance(context, Mapping):
        fields = [f"  {k}: {v}" for k, v in sorted(context.items()) if v]
        if fields:
            parts.append("CONTEXT:\n" + "\n".join(fields))
    policy = sample.get("policy_context")
    if isinstance(policy, Mapping) and policy.get("policy_text"):
        ids = ", ".join(policy.get("authority_ids") or [])
        parts.append(f"GOVERNING POLICY [{ids}]:\n{policy['policy_text']}")
    return f"{SYSTEM}\n\n" + "\n\n".join(parts) + f"\n\nVERDICT:\n{PROMPT_TAIL}"


def render_response(sample: Mapping) -> str:
    """The full structured target, continuing from PROMPT_TAIL."""
    gold = sample.get("gold") or {}
    body = {
        "category": gold.get("category") or sample.get("category"),
        "violation_tags": list(gold.get("violation_tags") or [])[:6],
        "policy_ids": list(gold.get("policy_ids") or [])[:6],
    }
    tail = json.dumps(body, separators=(", ", ": "))[1:]  # drop the leading brace
    return f'{gold.get("action", "allow")}", ' + tail.lstrip()


def action_token_ids(tokenizer) -> list[int]:
    """First-token id of each action word, verified pairwise distinct."""
    ids = []
    for action in ACTIONS:
        encoded = tokenizer.encode(action, add_special_tokens=False)
        if not encoded:
            raise ContractError(f"action word {action!r} does not tokenize")
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise ContractError(
            f"tokenizer conflates action words at the first token: {dict(zip(ACTIONS, ids))}"
        )
    return ids


def action_logits(model, tokenizer, prompts: Sequence[str], *, device, max_length: int = 1024):
    """(batch, 3) logits over the three action words at the decision position."""
    import torch

    if not prompts:
        raise ContractError("action_logits requires at least one prompt")
    ids = action_token_ids(tokenizer)
    batch = tokenizer(
        list(prompts), return_tensors="pt", padding=True, truncation=True,
        max_length=max_length, padding_side="left",
    ).to(device)
    out = model(**batch).logits           # (batch, seq, vocab)
    last = out[:, -1, :]                  # left padding => final position is real
    return last[:, ids]


def response_logprob(model, tokenizer, prompt: str, response: str, *, device,
                     max_length: int = 1024):
    """Sum of response-token log-probabilities with all prompt tokens masked."""
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if not response_ids:
        raise ContractError("empty response cannot be scored")
    ids = (prompt_ids + response_ids)[-max_length:]
    n_response = min(len(response_ids), len(ids) - 1)
    if n_response <= 0:
        raise ContractError("response was fully truncated; the pair must be rejected")
    tensor = torch.tensor([ids], device=device)
    # Only the response positions contribute, so slice before any softmax.  Building
    # log_softmax over the whole sequence costs seq_len x vocab floats -- for a 1024
    # token prompt and a 151k vocab that is ~620 MB per forward, and with four pair
    # forwards plus the composite's gold-anchor and replay passes it exhausts a 40 GB
    # A100.  Slicing to the ~40 response tokens cuts it by more than an order of
    # magnitude and changes no value.
    all_logits = model(input_ids=tensor).logits[0]          # (T, V)
    logits = all_logits[-(n_response + 1):-1, :]            # response positions only
    targets = tensor[0, -n_response:]
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
    denominator = torch.logsumexp(logits, dim=-1).float()
    return (picked - denominator).sum(), len(response_ids) > n_response


def batch_response_logprobs(model, tokenizer, pairs: Sequence[tuple[str, str]], *, device,
                            max_length: int = 1024):
    """Stack response log-probabilities; also report which items were truncated."""
    import torch

    values, truncated = [], []
    for prompt, response in pairs:
        value, was_truncated = response_logprob(
            model, tokenizer, prompt, response, device=device, max_length=max_length
        )
        values.append(value)
        truncated.append(was_truncated)
    return torch.stack(values), truncated


def load_backbone(model_id: str, revision: str, *, device: str, dtype: str = "auto",
                  adapter_path: str | None = None, trainable: bool = False):
    """Load a pinned backbone, optionally with a LoRA adapter attached."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = (
        torch.bfloat16 if dtype == "bf16"
        else torch.float32 if dtype == "fp32"
        else ("bfloat16" if device == "cuda" else torch.float32)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch_dtype
    ).to(device)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    if not trainable:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    action_token_ids(tokenizer)  # fail at load, not mid-run
    return model, tokenizer


def generate_candidates(model, tokenizer, prompts: Sequence[str], *, device,
                        max_new_tokens: int = 56, do_sample: bool = False,
                        temperature: float = 1.0, seed: int | None = None,
                        max_length: int = 768):
    """Let a teacher author its own structured verdict, not just pick an action.

    The prompt already ends at ``{"action": "`` so the continuation *is* the verdict.
    This exists because an action-only candidate is a deterministic function of
    (gold, category): with three actions and gold-based adjudication, two different
    teachers produce byte-identical pairs on ~98% of events, and the candidate source
    -- the treatment -- vanishes from the data.  Letting each teacher write its own
    tags and policy ids restores it.
    """
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    batch = tokenizer(
        list(prompts), return_tensors="pt", padding=True, truncation=True,
        max_length=max_length, padding_side="left",
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = out[:, batch["input_ids"].shape[1]:]
    return [tokenizer.decode(row, skip_special_tokens=True) for row in generated]


def parse_verdict(continuation: str, *, fallback_category: str) -> dict | None:
    """Parse a generated continuation of ``{"action": "`` into a verdict.

    Returns None when the model produced nothing usable.  Failing closed matters: a
    silently defaulted verdict would be identical across teachers and would recreate
    the very collapse this generative path exists to avoid.
    """
    text = (continuation or "").strip()
    if not text:
        return None
    action = None
    for candidate in ACTIONS:
        if text.startswith(candidate):
            action = candidate
            break
    if action is None:
        return None
    body = "{\"action\": \"" + text
    # tolerate a truncated tail: close the object at the last complete field
    for end in range(len(body), len(body) - 200, -1):
        chunk = body[:end]
        if chunk.count("{") == 0:
            break
        try:
            parsed = json.loads(chunk + "}" * (chunk.count("{") - chunk.count("}")))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        tags = parsed.get("violation_tags")
        ids = parsed.get("policy_ids")
        return {
            "action": action,
            "category": parsed.get("category") or fallback_category,
            "violation_tags": [str(x) for x in tags][:6] if isinstance(tags, list) else [],
            "policy_ids": [str(x) for x in ids][:6] if isinstance(ids, list) else [],
        }
    return {"action": action, "category": fallback_category,
            "violation_tags": [], "policy_ids": []}
