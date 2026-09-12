"""
Tests for shared/llm.py (Phase 2, W2.4) — the public call_llm/call_llm_meta
surface, replay mode (no live pool needed), and the llm_call_scope()
contextvar collector that ingestion/pipeline.py uses to attach per-call
provenance to the returned AuditReport.
"""

import asyncio

import pytest

import shared.llm as llm
import shared.llm_cassette as llm_cassette
from shared.llm_cassette import Cassette, cassette_key
from shared.llm_pool import LLMCallResult


def _fake_pool_result(text="hello", provider="groq", model="m1", key_fp="abc123456789"):
    return LLMCallResult(
        text=text, provider=provider, model=model, key_fingerprint=key_fp,
        endpoint_id=f"{provider}:{key_fp}", prompt_tokens=10, completion_tokens=5,
        total_tokens=15, latency_ms=12.3, attempts=1, retry_history=[],
    )


class TestCallLlmLiveMode(object):
    def test_call_llm_returns_pool_text(self, monkeypatch):
        async def _body():
            class _FakePool:
                async def call(self, **kwargs):
                    return _fake_pool_result(text="pool says hi")

            monkeypatch.setattr(llm, "get_pool", lambda: _FakePool())
            text = await llm.call_llm("what's up")
            assert text == "pool says hi"
        asyncio.run(_body())

    def test_call_llm_meta_returns_full_result(self, monkeypatch):
        async def _body():
            class _FakePool:
                async def call(self, **kwargs):
                    return _fake_pool_result(provider="openrouter")

            monkeypatch.setattr(llm, "get_pool", lambda: _FakePool())
            result = await llm.call_llm_meta("prompt")
            assert result.provider == "openrouter"
            assert result.total_tokens == 15
        asyncio.run(_body())

    def test_seed_defaults_to_global_seed_when_not_passed(self, monkeypatch):
        async def _body():
            seen = {}

            class _FakePool:
                async def call(self, **kwargs):
                    seen["seed"] = kwargs.get("seed")
                    return _fake_pool_result()

            monkeypatch.setattr(llm, "get_pool", lambda: _FakePool())
            monkeypatch.setattr(llm, "GLOBAL_SEED", 999)
            await llm.call_llm("prompt")
            assert seen["seed"] == 999
        asyncio.run(_body())

    def test_explicit_seed_overrides_global_seed(self, monkeypatch):
        async def _body():
            seen = {}

            class _FakePool:
                async def call(self, **kwargs):
                    seen["seed"] = kwargs.get("seed")
                    return _fake_pool_result()

            monkeypatch.setattr(llm, "get_pool", lambda: _FakePool())
            monkeypatch.setattr(llm, "GLOBAL_SEED", 999)
            await llm.call_llm("prompt", seed=5)
            assert seen["seed"] == 5
        asyncio.run(_body())


class TestLlmCallScope:
    def test_no_active_scope_returns_empty_list(self):
        assert llm.get_current_llm_calls() == []

    def test_calls_inside_scope_are_collected(self, monkeypatch):
        async def _body():
            class _FakePool:
                async def call(self, **kwargs):
                    return _fake_pool_result(text="a")

            monkeypatch.setattr(llm, "get_pool", lambda: _FakePool())
            with llm.llm_call_scope():
                await llm.call_llm("prompt 1")
                await llm.call_llm("prompt 2")
                calls = llm.get_current_llm_calls()
            assert len(calls) == 2
            assert all(c.provider == "groq" for c in calls)
        asyncio.run(_body())

    def test_scope_resets_after_exit(self, monkeypatch):
        async def _body():
            class _FakePool:
                async def call(self, **kwargs):
                    return _fake_pool_result()

            monkeypatch.setattr(llm, "get_pool", lambda: _FakePool())
            with llm.llm_call_scope():
                await llm.call_llm("prompt")
            # Outside the scope, collection stops — a call made here is
            # simply not gathered anywhere (pre-Phase-2 behavior).
            await llm.call_llm("prompt outside scope")
            assert llm.get_current_llm_calls() == []
        asyncio.run(_body())

    def test_calls_outside_any_scope_are_not_collected(self, monkeypatch):
        async def _body():
            class _FakePool:
                async def call(self, **kwargs):
                    return _fake_pool_result()

            monkeypatch.setattr(llm, "get_pool", lambda: _FakePool())
            await llm.call_llm("no scope active")
            assert llm.get_current_llm_calls() == []
        asyncio.run(_body())


class TestReplayMode:
    def test_replay_returns_recorded_text_without_touching_pool(self, monkeypatch, tmp_path):
        async def _body():
            def _pool_should_not_be_called():
                raise AssertionError("replay mode must not build/use the live pool")

            monkeypatch.setattr(llm, "get_pool", lambda: _pool_should_not_be_called())
            monkeypatch.setattr(llm, "LLM_PROVIDER", "groq")
            monkeypatch.setattr(llm, "GROQ_MODEL", "test-model")

            class _ReplayCfg:
                llm_mode = "replay"
                cassette_dir = str(tmp_path)

            monkeypatch.setattr(llm, "load_pool_config", lambda: _ReplayCfg())

            key = cassette_key("groq", "test-model", 0.2, None, None, "hello")
            cassette = Cassette(str(tmp_path / "default.jsonl"))
            cassette.append(key, "groq", "test-model", "recorded answer")
            llm_cassette.reset_cassettes()

            text = await llm.call_llm("hello")
            assert text == "recorded answer"
        asyncio.run(_body())

    def test_replay_miss_raises_keyerror_mentioning_cassette(self, monkeypatch, tmp_path):
        async def _body():
            monkeypatch.setattr(llm, "LLM_PROVIDER", "groq")
            monkeypatch.setattr(llm, "GROQ_MODEL", "test-model")

            class _ReplayCfg:
                llm_mode = "replay"
                cassette_dir = str(tmp_path)

            monkeypatch.setattr(llm, "load_pool_config", lambda: _ReplayCfg())
            llm_cassette.reset_cassettes()

            with pytest.raises(KeyError, match="cassette"):
                await llm.call_llm("never recorded")
        asyncio.run(_body())
