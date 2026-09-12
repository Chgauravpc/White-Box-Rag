"""
Tests for shared/llm_pool.py — the multi-key/multi-provider pool (Phase 2,
W2.2/W2.3). Uses a fake AsyncOpenAI client (monkeypatched into the module)
and an injectable clock so admission/cooldown/failover math is exercised
without real network calls or real 60s sliding windows.
"""

import asyncio
import time
import types

import httpx
import openai
import pytest

from shared.llm_pool import (
    LLMEndpoint,
    EndpointState,
    LLMPool,
    AllEndpointsExhausted,
    classify_llm_error,
    RETRY_SAME_KEY,
    FAILOVER,
    DISABLE_ENDPOINT,
    FATAL,
    key_fingerprint,
)
from shared.llm_pool_config import LLMPoolConfig, ProviderPoolConfig


def _endpoint(provider="groq", key="k1", rpm=30, tpm=1000, max_concurrency=2, model="m1"):
    return LLMEndpoint(
        provider=provider, base_url="http://fake", model=model, api_key=key,
        rpm_limit=rpm, tpm_limit=tpm, max_concurrency=max_concurrency,
    )


class TestKeyFingerprint:
    def test_deterministic_and_short(self):
        assert key_fingerprint("secret") == key_fingerprint("secret")
        assert len(key_fingerprint("secret")) == 12

    def test_different_keys_different_fingerprints(self):
        assert key_fingerprint("k1") != key_fingerprint("k2")

    def test_never_contains_raw_key(self):
        assert "secret" not in key_fingerprint("secret")


class TestEndpointState:
    def test_available_when_healthy_and_under_concurrency(self):
        state = EndpointState(endpoint=_endpoint(max_concurrency=2))
        assert state.is_available(now=0.0) is True

    def test_unavailable_when_inflight_at_max_concurrency(self):
        state = EndpointState(endpoint=_endpoint(max_concurrency=1))
        state.reserve(estimated_tokens=10, now=0.0)
        assert state.is_available(now=0.0) is False

    def test_unavailable_during_cooldown(self):
        state = EndpointState(endpoint=_endpoint())
        state.cooldown_until = 100.0
        assert state.is_available(now=50.0) is False
        assert state.is_available(now=100.0) is True

    def test_unavailable_when_unhealthy(self):
        state = EndpointState(endpoint=_endpoint())
        state.healthy = False
        assert state.is_available(now=0.0) is False

    def test_tpm_headroom_decreases_with_reservation_and_usage(self):
        state = EndpointState(endpoint=_endpoint(tpm=1000))
        assert state.tpm_headroom(now=0.0) == 1000
        state.reserve(estimated_tokens=300, now=0.0)
        assert state.tpm_headroom(now=0.0) == 700
        state.release(reserved_tokens=300, actual_tokens=250, now=1.0)
        assert state.tpm_headroom(now=1.0) == 750

    def test_sliding_window_expires_after_60s(self):
        state = EndpointState(endpoint=_endpoint(tpm=1000))
        state.reserve(estimated_tokens=500, now=0.0)
        state.release(reserved_tokens=500, actual_tokens=500, now=0.0)
        assert state.tpm_headroom(now=30.0) == 500
        assert state.tpm_headroom(now=61.0) == 1000

    def test_rpm_headroom_tracks_request_count_not_tokens(self):
        state = EndpointState(endpoint=_endpoint(rpm=2))
        state.reserve(estimated_tokens=1, now=0.0)
        assert state.rpm_headroom(now=0.0) == 1
        state.reserve(estimated_tokens=1, now=0.0)
        assert state.rpm_headroom(now=0.0) == 0

    def test_release_records_error_and_increments_error_count(self):
        state = EndpointState(endpoint=_endpoint())
        state.reserve(estimated_tokens=10, now=0.0)
        state.release(reserved_tokens=10, actual_tokens=None, now=1.0, error_category=FAILOVER)
        assert state.total_errors == 1
        assert state.last_error_category == FAILOVER

    def test_server_reported_headroom_overrides_local_when_lower(self):
        state = EndpointState(endpoint=_endpoint(tpm=10000))
        state._server_tpm_remaining = 50
        assert state.tpm_headroom(now=0.0) == 50


class TestClassifyLlmError:
    def _response(self, status):
        req = httpx.Request("POST", "http://fake/v1/chat/completions")
        return httpx.Response(status_code=status, request=req)

    def test_rate_limit_is_failover(self):
        exc = openai.RateLimitError("rate limited", response=self._response(429), body=None)
        assert classify_llm_error(exc) == FAILOVER

    def test_auth_error_is_disable_endpoint(self):
        exc = openai.AuthenticationError("bad key", response=self._response(401), body=None)
        assert classify_llm_error(exc) == DISABLE_ENDPOINT

    def test_5xx_status_is_retry_same_key(self):
        exc = openai.APIStatusError("server error", response=self._response(503), body=None)
        assert classify_llm_error(exc) == RETRY_SAME_KEY

    def test_429_status_via_api_status_error_is_failover(self):
        exc = openai.APIStatusError("too many requests", response=self._response(429), body=None)
        assert classify_llm_error(exc) == FAILOVER

    def test_400_status_is_fatal(self):
        exc = openai.APIStatusError("bad request", response=self._response(400), body=None)
        assert classify_llm_error(exc) == FATAL

    def test_timeout_is_retry_same_key(self):
        req = httpx.Request("POST", "http://fake")
        exc = openai.APITimeoutError(request=req)
        assert classify_llm_error(exc) == RETRY_SAME_KEY

    def test_connection_error_is_retry_same_key(self):
        req = httpx.Request("POST", "http://fake")
        exc = openai.APIConnectionError(request=req)
        assert classify_llm_error(exc) == RETRY_SAME_KEY

    def test_unrecognized_exception_is_fatal(self):
        assert classify_llm_error(ValueError("something weird")) == FATAL


# ---------------------------------------------------------------------------
# LLMPool — fake AsyncOpenAI client, injectable clock
# ---------------------------------------------------------------------------

class _FakeRaw:
    def __init__(self, parsed, headers=None):
        self._parsed = parsed
        self.headers = headers or {}

    def parse(self):
        return self._parsed


def _chat_response(text, total_tokens=100):
    usage = types.SimpleNamespace(prompt_tokens=total_tokens // 2,
                                   completion_tokens=total_tokens - total_tokens // 2,
                                   total_tokens=total_tokens)
    choice = types.SimpleNamespace(message=types.SimpleNamespace(content=text))
    return types.SimpleNamespace(choices=[choice], usage=usage)


class _FakeClient:
    """Stands in for AsyncOpenAI. `behavior(**kwargs)` returns a _FakeRaw or
    raises — set per-test via the `behaviors` dict keyed by api_key."""

    def __init__(self, api_key, base_url, behaviors):
        self.api_key = api_key
        self.base_url = base_url
        self._behaviors = behaviors
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(
            with_raw_response=types.SimpleNamespace(create=self._create)
        ))

    async def _create(self, **kwargs):
        behavior = self._behaviors.get(self.api_key)
        if behavior is None:
            return _FakeRaw(_chat_response("default response"))
        return behavior(**kwargs)


def _make_pool(monkeypatch, providers, provider_order=None, allow_cross_provider_failover=False,
               strict_single_model=False, behaviors=None, clock=None):
    behaviors = behaviors or {}
    monkeypatch.setattr(
        "shared.llm_pool.AsyncOpenAI",
        lambda api_key, base_url: _FakeClient(api_key, base_url, behaviors),
    )
    config = LLMPoolConfig(
        providers=tuple(providers),
        provider_order=tuple(provider_order or [p.provider for p in providers]),
        allow_cross_provider_failover=allow_cross_provider_failover,
        strict_single_model=strict_single_model,
        llm_mode="live",
        cassette_dir="eval/cassettes",
        eval_concurrency_auto=False,
    )
    # Real time.monotonic by default: LLMPool's selection-timeout/cooldown
    # logic mixes its injected clock with asyncio.wait_for's real wall-clock
    # timer, so a frozen fake clock makes any wait-for-cooldown path loop
    # forever (the deadline it computes from the fake clock is never reached
    # in real time). Pure EndpointState/_SlidingWindow math is tested
    # directly with explicit `now=` values instead (see TestEndpointState).
    return LLMPool(config, clock=clock or time.monotonic)


def _provider_cfg(provider, keys, rpm=30, tpm=6000, max_concurrency=2, model="m1"):
    return ProviderPoolConfig(
        provider=provider, base_url=f"http://fake/{provider}", model=model,
        api_keys=tuple(keys), rpm_per_key=rpm, tpm_per_key=tpm,
        max_concurrency_per_key=max_concurrency,
    )


class TestLLMPoolBasics:
    def test_endpoint_count_matches_total_keys(self, monkeypatch):
        pool = _make_pool(monkeypatch, [_provider_cfg("groq", ["k1", "k2"])])
        assert pool.endpoint_count == 2

    def test_total_max_concurrency_sums_across_endpoints(self, monkeypatch):
        pool = _make_pool(monkeypatch, [
            _provider_cfg("groq", ["k1", "k2"], max_concurrency=2),
            _provider_cfg("openrouter", ["o1"], max_concurrency=3),
        ])
        assert pool.total_max_concurrency() == 2 + 2 + 3

    def test_no_endpoints_configured_raises_on_call(self, monkeypatch):
        pool = _make_pool(monkeypatch, [])
        with pytest.raises(AllEndpointsExhausted):
            asyncio.run(pool.call(messages=[{"role": "user", "content": "hi"}], temperature=0.2))


class TestLLMPoolCall:
    """No pytest-asyncio in this project (see backend/pytest.ini) — each test
    wraps its body in asyncio.run(), same pattern the manual smoke tests use
    for run_query_pipeline() elsewhere in this repo."""

    def test_successful_call_returns_text_and_provenance(self, monkeypatch):
        async def _body():
            behaviors = {"k1": lambda **kw: _FakeRaw(_chat_response("hello world", total_tokens=42))}
            pool = _make_pool(monkeypatch, [_provider_cfg("groq", ["k1"])], behaviors=behaviors)
            result = await pool.call(messages=[{"role": "user", "content": "hi"}], temperature=0.2)
            assert result.text == "hello world"
            assert result.provider == "groq"
            assert result.total_tokens == 42
            assert result.attempts == 1
        asyncio.run(_body())

    def test_admission_prefers_highest_tpm_headroom(self, monkeypatch):
        async def _body():
            def make_behavior(name):
                return lambda **kw: _FakeRaw(_chat_response(name, total_tokens=10))

            behaviors = {"k1": make_behavior("k1"), "k2": make_behavior("k2")}
            pool = _make_pool(monkeypatch, [_provider_cfg("groq", ["k1", "k2"], tpm=1000)], behaviors=behaviors)
            # Exhaust k1's headroom first by reserving directly on its state.
            state_k1 = next(s for s in pool._states if s.endpoint.api_key == "k1")
            state_k1.reserve(estimated_tokens=900, now=pool._clock())

            result = await pool.call(messages=[{"role": "user", "content": "hi"}], temperature=0.2)
            assert result.key_fingerprint == key_fingerprint("k2")
        asyncio.run(_body())

    def test_rate_limit_fails_over_to_second_key(self, monkeypatch):
        async def _body():
            req = httpx.Request("POST", "http://fake")
            resp = httpx.Response(status_code=429, request=req)

            def failing(**kw):
                raise openai.RateLimitError("rate limited", response=resp, body=None)

            behaviors = {
                "k1": failing,
                "k2": lambda **kw: _FakeRaw(_chat_response("from k2")),
            }
            pool = _make_pool(monkeypatch, [_provider_cfg("groq", ["k1", "k2"])], behaviors=behaviors)
            result = await pool.call(messages=[{"role": "user", "content": "hi"}], temperature=0.2, max_attempts=3)
            assert result.text == "from k2"
            assert any("FAILOVER" in h for h in result.retry_history)
        asyncio.run(_body())

    def test_auth_error_disables_endpoint_permanently(self, monkeypatch):
        async def _body():
            req = httpx.Request("POST", "http://fake")
            resp = httpx.Response(status_code=401, request=req)

            def failing(**kw):
                raise openai.AuthenticationError("bad key", response=resp, body=None)

            behaviors = {
                "k1": failing,
                "k2": lambda **kw: _FakeRaw(_chat_response("from k2")),
            }
            pool = _make_pool(monkeypatch, [_provider_cfg("groq", ["k1", "k2"])], behaviors=behaviors)
            await pool.call(messages=[{"role": "user", "content": "hi"}], temperature=0.2, max_attempts=3)
            state_k1 = next(s for s in pool._states if s.endpoint.api_key == "k1")
            assert state_k1.healthy is False
        asyncio.run(_body())

    def test_fatal_error_raises_immediately_without_exhausting_attempts(self, monkeypatch):
        async def _body():
            req = httpx.Request("POST", "http://fake")
            resp = httpx.Response(status_code=400, request=req)

            def failing(**kw):
                raise openai.APIStatusError("bad request", response=resp, body=None)

            behaviors = {"k1": failing}
            pool = _make_pool(monkeypatch, [_provider_cfg("groq", ["k1"])], behaviors=behaviors)
            with pytest.raises(openai.APIStatusError):
                await pool.call(messages=[{"role": "user", "content": "hi"}], temperature=0.2, max_attempts=5)
        asyncio.run(_body())

    def test_all_endpoints_exhausted_raises_capacity_exhausted_message(self, monkeypatch):
        async def _body():
            req = httpx.Request("POST", "http://fake")
            resp = httpx.Response(status_code=429, request=req)

            def failing(**kw):
                raise openai.RateLimitError("rate limited", response=resp, body=None)

            behaviors = {"k1": failing}
            pool = _make_pool(monkeypatch, [_provider_cfg("groq", ["k1"])], behaviors=behaviors)
            with pytest.raises(AllEndpointsExhausted, match="capacity exhausted"):
                await pool.call(messages=[{"role": "user", "content": "hi"}], temperature=0.2, max_attempts=1,
                                 selection_timeout_s=0.2)
        asyncio.run(_body())

    def test_strict_single_model_pins_provider_across_retries(self, monkeypatch):
        async def _body():
            req = httpx.Request("POST", "http://fake")
            resp = httpx.Response(status_code=429, request=req)
            attempts = {"groq_k1": 0}

            def groq_failing(**kw):
                attempts["groq_k1"] += 1
                raise openai.RateLimitError("rate limited", response=resp, body=None)

            behaviors = {
                "groq_k1": groq_failing,
                "or_k1": lambda **kw: _FakeRaw(_chat_response("should never be used")),
            }
            pool = _make_pool(
                monkeypatch,
                [_provider_cfg("groq", ["groq_k1"]), _provider_cfg("openrouter", ["or_k1"])],
                provider_order=["groq", "openrouter"],
                allow_cross_provider_failover=True,
                strict_single_model=True,
                behaviors=behaviors,
            )
            with pytest.raises(AllEndpointsExhausted):
                await pool.call(messages=[{"role": "user", "content": "hi"}], temperature=0.2, max_attempts=2,
                                 selection_timeout_s=0.2)
            # groq_k1's only call attempt triggered FAILOVER (cooldown), and
            # since strict_single_model pinned the provider to "groq" after
            # that first attempt, selection refuses to consider openrouter
            # even though allow_cross_provider_failover=True — it waits out
            # the selection timeout on groq alone and gives up, rather than
            # ever calling or_k1.
            assert attempts["groq_k1"] == 1
        asyncio.run(_body())

    def test_snapshot_never_exposes_raw_key(self, monkeypatch):
        pool = _make_pool(monkeypatch, [_provider_cfg("groq", ["super-secret-key"])])
        snap = pool.snapshot()
        assert len(snap) == 1
        assert "super-secret-key" not in str(snap)
        assert snap[0]["key_fingerprint"] == key_fingerprint("super-secret-key")
