"""
LLM client wrapper — config-driven Groq / OpenRouter, backed by a multi-key
pool (Phase 2, W2.4 — see docs/BENCHMARK_READINESS.md).

`call_llm(...)` keeps its exact pre-Phase-2 signature and behavior — every
existing caller (`rag.py`, `audit.py`, `mapper.py`, `brd_parser.py`) is
untouched. Internally it now goes through `shared.llm_pool.LLMPool`, which
ranks candidate (provider, key) endpoints by remaining token-budget headroom
and fails over/retries per `shared.llm_pool.classify_llm_error` instead of
the old single-client retry loop.

`call_llm_meta(...)` is the new entry point for callers that want per-call
provenance (which endpoint/key served the call, token counts, latency,
retry history) instead of just the text. `llm_call_scope()` is a
`contextvars`-based collector: code that wraps a whole pipeline run in it
gets back every `LLMCallRecord` made in that run via `get_current_llm_calls()`
— correct even when many runs share one event loop (the eval harness's
`asyncio.gather` fan-out), since each `asyncio.Task` gets its own copy of the
context.

`LLM_MODE` (env, default "live") additionally routes calls through
`shared.llm_cassette` for record/replay — see that module's docstring.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar

from shared.config import GLOBAL_SEED, LLM_PROVIDER, GROQ_MODEL, OPENROUTER_MODEL
from shared.llm_pool import LLMCallResult, get_pool
from shared.llm_pool_config import load_pool_config
from shared.llm_cassette import cassette_key, get_cassette
from shared.models import LLMCallRecord

logger = logging.getLogger(__name__)

_current_calls: ContextVar[list | None] = ContextVar("_current_llm_calls", default=None)


@contextmanager
def llm_call_scope():
    """Collect every LLMCallRecord made by call_llm()/call_llm_meta() inside
    this block. Nesting is fine (the inner scope simply shadows the outer
    one for its own duration); un-scoped calls (no active scope) are just
    not collected anywhere, matching pre-Phase-2 behavior."""
    token = _current_calls.set([])
    try:
        yield
    finally:
        _current_calls.reset(token)


def get_current_llm_calls() -> list[LLMCallRecord]:
    return list(_current_calls.get() or [])


def _record_call(record: LLMCallRecord) -> None:
    calls = _current_calls.get()
    if calls is not None:
        calls.append(record)


def _build_messages(prompt: str, system_instruction: str | None) -> list[dict]:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    return messages


def _estimate_tokens(prompt: str, system_instruction: str | None) -> int:
    """Crude chars/4 heuristic for TPM admission — good enough to rank
    endpoints by headroom; actual usage (from the response) is what gets
    recorded into the sliding window afterwards, so an inaccurate estimate
    only affects in-flight admission ordering, never the accounting."""
    chars = len(prompt) + len(system_instruction or "")
    return max(200, chars // 4 + 300)


def _replay(pool_cfg, prompt: str, system_instruction: str | None, temperature: float,
            seed: int | None) -> LLMCallResult:
    provider = (LLM_PROVIDER or "groq").lower()
    model = GROQ_MODEL if provider == "groq" else OPENROUTER_MODEL
    cassette = get_cassette(pool_cfg.cassette_dir)
    key = cassette_key(provider, model, temperature, seed, system_instruction, prompt)
    text = cassette.get(key)
    if text is None:
        raise KeyError(
            f"LLM_MODE=replay: no cassette entry for this call (provider={provider}, model={model}). "
            f"Record it first with LLM_MODE=record, or switch to LLM_MODE=live."
        )
    return LLMCallResult(
        text=text, provider=provider, model=model, key_fingerprint="replay",
        endpoint_id=f"{provider}:replay", prompt_tokens=None, completion_tokens=None,
        total_tokens=None, latency_ms=0.0, attempts=1, retry_history=[],
    )


async def call_llm_meta(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    max_retries: int = 3,
    seed: int | None = None,
    estimated_tokens: int | None = None,
) -> LLMCallResult:
    """Like call_llm, but returns full per-call provenance instead of bare
    text. See module docstring."""
    if seed is None:
        seed = GLOBAL_SEED
    pool_cfg = load_pool_config()

    if pool_cfg.llm_mode == "replay":
        result = _replay(pool_cfg, prompt, system_instruction, temperature, seed)
    else:
        pool = get_pool()
        messages = _build_messages(prompt, system_instruction)
        est = estimated_tokens if estimated_tokens is not None else _estimate_tokens(prompt, system_instruction)
        result = await pool.call(
            messages=messages, temperature=temperature, seed=seed,
            estimated_tokens=est, max_attempts=max(max_retries, 1),
        )
        if pool_cfg.llm_mode == "record":
            cassette = get_cassette(pool_cfg.cassette_dir)
            key = cassette_key(result.provider, result.model, temperature, seed, system_instruction, prompt)
            cassette.append(key, result.provider, result.model, result.text)

    _record_call(LLMCallRecord(
        provider=result.provider, model=result.model, key_fingerprint=result.key_fingerprint,
        endpoint_id=result.endpoint_id, prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens, total_tokens=result.total_tokens,
        latency_ms=result.latency_ms, attempts=result.attempts, retry_history=result.retry_history,
    ))
    return result


async def call_llm(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    max_retries: int = 3,
    seed: int | None = None,
) -> str:
    """Send a prompt to the configured LLM provider/pool and return the text
    response. Unchanged signature and behavior from the pre-pool client —
    see call_llm_meta for per-call provenance.

    Args:
        prompt: The user prompt / question.
        system_instruction: Optional system-level instruction.
        temperature: Sampling temperature (low = more deterministic).
        max_retries: Max endpoint attempts across the whole pool (was
            "retries on one client" pre-Phase-2; now also governs failover).
        seed: Best-effort determinism hint forwarded to the provider's `seed`
            param. Defaults to config.GLOBAL_SEED (None = don't send one).

    Returns:
        The model's text response.

    Raises:
        shared.llm_pool.AllEndpointsExhausted: every endpoint failed/was
            unavailable within the attempt budget.
        KeyError: LLM_MODE=replay and no cassette entry exists for this call.
    """
    result = await call_llm_meta(prompt, system_instruction, temperature, max_retries, seed)
    return result.text
