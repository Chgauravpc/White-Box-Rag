"""
Multi-key, multi-provider LLM pool (Phase 2, W2.2/W2.3 — see
docs/BENCHMARK_READINESS.md). One `AsyncOpenAI` client per key, not one
shared client, so connection pools and auth headers never interleave.

Admission ranks candidate endpoints by **highest remaining TPM headroom**
(not naive round-robin — round-robin is what makes N keys trip their rate
limits together, since they all get hit at roughly the same cadence).
Failures are classified via typed SDK exceptions into one of four actions
(`RETRY_SAME_KEY` / `FAILOVER` / `DISABLE_ENDPOINT` / `FATAL`) instead of the
old fragile `"500" in str(e)` substring match.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field

from openai import (
    AsyncOpenAI,
    RateLimitError,
    APIStatusError,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
)

from shared.llm_pool_config import LLMPoolConfig

logger = logging.getLogger(__name__)


def key_fingerprint(key: str) -> str:
    """SHA-256 prefix — enough to tell keys apart in logs/provenance without
    ever writing the raw key anywhere."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class AllEndpointsExhausted(Exception):
    """Raised when no endpoint could complete the call. Message deliberately
    contains 'capacity exhausted' so eval/harness.py::classify_error buckets
    it as CAPACITY_EXHAUSTED rather than UNKNOWN."""


# ---------------------------------------------------------------------------
# Error classification (W2.3)
# ---------------------------------------------------------------------------

RETRY_SAME_KEY = "RETRY_SAME_KEY"
FAILOVER = "FAILOVER"
DISABLE_ENDPOINT = "DISABLE_ENDPOINT"
FATAL = "FATAL"


def classify_llm_error(exc: BaseException) -> str:
    """Typed-exception classification, replacing the pre-pool client's
    `"500" in str(e)` substring matching."""
    if isinstance(exc, AuthenticationError) or isinstance(exc, PermissionDeniedError):
        return DISABLE_ENDPOINT  # bad/revoked key, or key lacks model access
    if isinstance(exc, RateLimitError):
        return FAILOVER  # this key is capacity-exhausted right now
    if isinstance(exc, APITimeoutError):
        return RETRY_SAME_KEY
    if isinstance(exc, APIConnectionError):
        return RETRY_SAME_KEY
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status in (500, 502, 503, 504):
            return RETRY_SAME_KEY
        if status == 429:
            return FAILOVER
        if status in (401, 403):
            return DISABLE_ENDPOINT
        return FATAL
    if isinstance(exc, BadRequestError):
        return FATAL
    return FATAL


def _reconcile_headers(state: "EndpointState", headers) -> None:
    """Best-effort: some OpenAI-compatible providers echo remaining-quota
    headers (`x-ratelimit-remaining-requests` / `x-ratelimit-remaining-tokens`,
    OpenAI-style). Neither Groq nor OpenRouter guarantee these, so this is
    opportunistic — headroom falls back to the local sliding-window estimate
    whenever a header is absent or unparseable."""
    if headers is None:
        return
    try:
        rem_req = headers.get("x-ratelimit-remaining-requests")
        rem_tok = headers.get("x-ratelimit-remaining-tokens")
        if rem_req is not None:
            state._server_rpm_remaining = int(rem_req)
        if rem_tok is not None:
            state._server_tpm_remaining = int(rem_tok)
    except (TypeError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Endpoint identity & live state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMEndpoint:
    provider: str
    base_url: str
    model: str
    api_key: str
    rpm_limit: int
    tpm_limit: int
    max_concurrency: int

    @property
    def key_fingerprint(self) -> str:
        return key_fingerprint(self.api_key)

    @property
    def endpoint_id(self) -> str:
        return f"{self.provider}:{self.key_fingerprint}"


class _SlidingWindow:
    """Rolling 60s sum of recorded amounts, keyed off an injectable clock
    (`time.monotonic` by default) so tests don't need real 60s sleeps."""

    def __init__(self):
        self._entries: list[tuple[float, int]] = []

    def record(self, amount: int, now: float) -> None:
        self._entries.append((now, amount))
        self._prune(now)

    def total(self, now: float) -> int:
        self._prune(now)
        return sum(amount for _, amount in self._entries)

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.pop(0)


@dataclass
class EndpointState:
    endpoint: LLMEndpoint
    inflight: int = 0
    healthy: bool = True
    cooldown_until: float = 0.0
    total_calls: int = 0
    total_errors: int = 0
    last_error_category: str | None = None
    _rpm_window: _SlidingWindow = field(default_factory=_SlidingWindow)
    _tpm_window: _SlidingWindow = field(default_factory=_SlidingWindow)
    _reserved_tokens: int = 0
    _server_rpm_remaining: int | None = None
    _server_tpm_remaining: int | None = None

    def is_available(self, now: float) -> bool:
        return (
            self.healthy
            and now >= self.cooldown_until
            and self.inflight < self.endpoint.max_concurrency
        )

    def rpm_headroom(self, now: float) -> int:
        local = max(0, self.endpoint.rpm_limit - self._rpm_window.total(now))
        return min(local, self._server_rpm_remaining) if self._server_rpm_remaining is not None else local

    def tpm_headroom(self, now: float) -> int:
        local = max(0, self.endpoint.tpm_limit - self._tpm_window.total(now) - self._reserved_tokens)
        return min(local, self._server_tpm_remaining) if self._server_tpm_remaining is not None else local

    def reserve(self, estimated_tokens: int, now: float) -> None:
        self.inflight += 1
        self._reserved_tokens += estimated_tokens
        self._rpm_window.record(1, now)
        self.total_calls += 1

    def release(self, reserved_tokens: int, actual_tokens: int | None, now: float,
                error_category: str | None = None) -> None:
        self.inflight = max(0, self.inflight - 1)
        self._reserved_tokens = max(0, self._reserved_tokens - reserved_tokens)
        self._tpm_window.record(actual_tokens if actual_tokens is not None else reserved_tokens, now)
        if error_category:
            self.total_errors += 1
            self.last_error_category = error_category

    def snapshot(self, now: float) -> dict:
        return {
            "endpoint_id": self.endpoint.endpoint_id,
            "provider": self.endpoint.provider,
            "model": self.endpoint.model,
            "key_fingerprint": self.endpoint.key_fingerprint,
            "healthy": self.healthy,
            "inflight": self.inflight,
            "cooldown_remaining_s": max(0.0, round(self.cooldown_until - now, 1)),
            "rpm_headroom": self.rpm_headroom(now),
            "tpm_headroom": self.tpm_headroom(now),
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "last_error_category": self.last_error_category,
        }


@dataclass
class LLMCallResult:
    text: str
    provider: str
    model: str
    key_fingerprint: str
    endpoint_id: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float
    attempts: int
    retry_history: list[str]


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

class LLMPool:
    def __init__(self, config: LLMPoolConfig, clock=time.monotonic):
        self._config = config
        self._clock = clock
        endpoints = self._build_endpoints(config)
        self._states = [EndpointState(endpoint=e) for e in endpoints]
        self._clients: dict[str, AsyncOpenAI] = {
            s.endpoint.endpoint_id: AsyncOpenAI(api_key=s.endpoint.api_key, base_url=s.endpoint.base_url)
            for s in self._states
        }
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)

    @staticmethod
    def _build_endpoints(config: LLMPoolConfig) -> list[LLMEndpoint]:
        endpoints = []
        for provider_cfg in config.providers:
            for api_key in provider_cfg.api_keys:
                endpoints.append(LLMEndpoint(
                    provider=provider_cfg.provider,
                    base_url=provider_cfg.base_url,
                    model=provider_cfg.model,
                    api_key=api_key,
                    rpm_limit=provider_cfg.rpm_per_key,
                    tpm_limit=provider_cfg.tpm_per_key,
                    max_concurrency=provider_cfg.max_concurrency_per_key,
                ))
        return endpoints

    @property
    def endpoint_count(self) -> int:
        return len(self._states)

    def total_max_concurrency(self) -> int:
        """Sum of per-endpoint max_concurrency across the pool — the
        call-level capacity ceiling used by W2.6's EVAL_CONCURRENCY_AUTO."""
        return sum(s.endpoint.max_concurrency for s in self._states)

    def snapshot(self) -> list[dict]:
        now = self._clock()
        return [s.snapshot(now) for s in self._states]

    def _candidates(self, now: float, required_provider: str | None) -> list[EndpointState]:
        order = [required_provider] if required_provider else list(self._config.provider_order)
        for provider in order:
            pcands = [s for s in self._states if s.endpoint.provider == provider and s.is_available(now)]
            if pcands:
                return pcands
            if required_provider or not self._config.allow_cross_provider_failover:
                return []
        return []

    def _next_wake_delay(self, now: float) -> float:
        deltas = [s.cooldown_until - now for s in self._states if s.cooldown_until > now]
        return max(0.05, min(deltas)) if deltas else 1.0

    async def _select_and_reserve(self, estimated_tokens: int, required_provider: str | None,
                                   deadline: float) -> EndpointState:
        async with self._lock:
            while True:
                now = self._clock()
                candidates = self._candidates(now, required_provider)
                if candidates:
                    candidates.sort(key=lambda s: s.tpm_headroom(now), reverse=True)
                    best = candidates[0]
                    best.reserve(estimated_tokens, now)
                    return best
                remaining = deadline - now
                if remaining <= 0:
                    raise AllEndpointsExhausted(
                        "No endpoint became available within timeout (capacity exhausted)"
                    )
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=min(remaining, self._next_wake_delay(now)))
                except asyncio.TimeoutError:
                    pass

    async def _release(self, state: EndpointState, reserved_tokens: int, actual_tokens: int | None,
                        error_category: str | None = None) -> None:
        async with self._lock:
            state.release(reserved_tokens, actual_tokens, self._clock(), error_category)
            self._cond.notify_all()

    async def call(
        self,
        *,
        messages: list[dict],
        temperature: float,
        seed: int | None = None,
        estimated_tokens: int = 800,
        max_attempts: int = 4,
        selection_timeout_s: float = 30.0,
    ) -> LLMCallResult:
        if not self._states:
            raise AllEndpointsExhausted("No LLM endpoints configured (no API keys)")

        strict = self._config.strict_single_model
        required_provider: str | None = None
        retry_history: list[str] = []
        started = self._clock()
        last_exc: BaseException | None = None

        for attempt in range(1, max_attempts + 1):
            deadline = self._clock() + selection_timeout_s
            state = await self._select_and_reserve(estimated_tokens, required_provider, deadline)
            if strict and required_provider is None:
                required_provider = state.endpoint.provider
            client = self._clients[state.endpoint.endpoint_id]

            try:
                kwargs = dict(model=state.endpoint.model, messages=messages, temperature=temperature)
                if seed is not None:
                    kwargs["seed"] = seed
                raw = await client.chat.completions.with_raw_response.create(**kwargs)
                response = raw.parse()
                _reconcile_headers(state, getattr(raw, "headers", None))
                usage = getattr(response, "usage", None)
                actual_tokens = usage.total_tokens if usage else estimated_tokens
                await self._release(state, estimated_tokens, actual_tokens)
                return LLMCallResult(
                    text=response.choices[0].message.content or "",
                    provider=state.endpoint.provider,
                    model=state.endpoint.model,
                    key_fingerprint=state.endpoint.key_fingerprint,
                    endpoint_id=state.endpoint.endpoint_id,
                    prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                    completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                    total_tokens=actual_tokens,
                    latency_ms=(self._clock() - started) * 1000,
                    attempts=attempt,
                    retry_history=retry_history,
                )
            except Exception as exc:  # noqa: BLE001 - reclassified below
                last_exc = exc
                category = classify_llm_error(exc)
                await self._release(state, estimated_tokens, None, error_category=category)
                retry_history.append(f"{state.endpoint.endpoint_id}:{category}")
                logger.warning(
                    f"LLM call failed on {state.endpoint.endpoint_id} "
                    f"(attempt {attempt}/{max_attempts}, category={category}): {exc}"
                )
                if category == DISABLE_ENDPOINT:
                    state.healthy = False
                elif category == FAILOVER:
                    state.cooldown_until = self._clock() + 5.0
                elif category == RETRY_SAME_KEY:
                    state.cooldown_until = self._clock() + min(2 ** attempt, 8)
                elif category == FATAL:
                    raise

        raise AllEndpointsExhausted(
            f"All endpoints exhausted after {max_attempts} attempts (capacity exhausted): {last_exc}"
        ) from last_exc


_pool: LLMPool | None = None


def get_pool() -> LLMPool:
    global _pool
    if _pool is None:
        from shared.llm_pool_config import load_pool_config
        _pool = LLMPool(load_pool_config())
    return _pool


def reset_pool() -> None:
    """Test/ops hook — drop the singleton so the next get_pool() rebuilds
    from current env (e.g. after monkeypatching keys mid-process)."""
    global _pool
    _pool = None
