"""
Tests for shared/llm_pool_config.py (Phase 2, W2.1) — env parsing, the
plural-keys-with-singular-fallback contract, and provider_order defaults
that preserve pre-Phase-2 single-provider behavior unless explicitly
opted into a multi-provider pool.
"""

import shared.config as config
from shared.llm_pool_config import load_pool_config, effective_eval_concurrency


def _clear_pool_env(monkeypatch):
    for name in (
        "GROQ_API_KEYS", "OPENROUTER_API_KEYS", "GROQ_API_KEY", "OPENROUTER_API_KEY",
        "LLM_PROVIDER_ORDER", "LLM_ALLOW_CROSS_PROVIDER_FAILOVER", "LLM_STRICT_SINGLE_MODEL",
        "LLM_MODE", "LLM_CASSETTE_DIR", "EVAL_CONCURRENCY_AUTO", "LLM_PROVIDER",
        "GROQ_RPM_PER_KEY", "GROQ_TPM_PER_KEY", "OPENROUTER_RPM_PER_KEY", "OPENROUTER_TPM_PER_KEY",
        "LLM_MAX_CONCURRENCY_PER_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


class TestKeyListParsing:
    def test_singular_key_env_still_works_as_a_pool_of_one(self, monkeypatch):
        _clear_pool_env(monkeypatch)
        monkeypatch.setattr(config, "GROQ_API_KEY", "legacy-single-key")
        monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
        cfg = load_pool_config()
        groq = cfg.provider_config("groq")
        assert groq is not None
        assert groq.api_keys == ("legacy-single-key",)

    def test_plural_keys_env_takes_priority_and_splits_on_comma(self, monkeypatch):
        _clear_pool_env(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEYS", "k1, k2 ,k3")
        monkeypatch.setattr(config, "GROQ_API_KEY", "legacy-single-key")
        monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
        cfg = load_pool_config()
        assert cfg.provider_config("groq").api_keys == ("k1", "k2", "k3")

    def test_no_keys_at_all_gives_empty_provider_list(self, monkeypatch):
        _clear_pool_env(monkeypatch)
        monkeypatch.setattr(config, "GROQ_API_KEY", None)
        monkeypatch.setattr(config, "OPENROUTER_API_KEY", None)
        monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
        cfg = load_pool_config()
        assert cfg.providers == ()
        assert cfg.total_configured_keys() == 0


class TestProviderOrderDefaults:
    def test_default_order_is_single_legacy_provider_only(self, monkeypatch):
        _clear_pool_env(monkeypatch)
        monkeypatch.setattr(config, "GROQ_API_KEY", "gk")
        monkeypatch.setattr(config, "OPENROUTER_API_KEY", "ok")
        monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
        cfg = load_pool_config()
        # Both providers HAVE keys, but without an explicit LLM_PROVIDER_ORDER
        # opt-in, the order stays pinned to the single legacy LLM_PROVIDER —
        # matching the pre-Phase-2 client's one-provider-only behavior.
        assert cfg.provider_order == ("groq",)

    def test_explicit_provider_order_is_honored(self, monkeypatch):
        _clear_pool_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER_ORDER", "openrouter,groq")
        monkeypatch.setattr(config, "GROQ_API_KEY", "gk")
        monkeypatch.setattr(config, "OPENROUTER_API_KEY", "ok")
        cfg = load_pool_config()
        assert cfg.provider_order == ("openrouter", "groq")

    def test_provider_with_keys_but_missing_from_order_is_appended_not_dropped(self, monkeypatch):
        _clear_pool_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER_ORDER", "groq")
        monkeypatch.setattr(config, "GROQ_API_KEY", "gk")
        monkeypatch.setattr(config, "OPENROUTER_API_KEY", "ok")
        cfg = load_pool_config()
        assert set(cfg.provider_order) == {"groq", "openrouter"}
        assert cfg.provider_order[0] == "groq"

    def test_default_failover_and_strict_single_model_are_off(self, monkeypatch):
        _clear_pool_env(monkeypatch)
        monkeypatch.setattr(config, "GROQ_API_KEY", "gk")
        cfg = load_pool_config()
        assert cfg.allow_cross_provider_failover is False
        assert cfg.strict_single_model is False


class TestLlmModeAndCassette:
    def test_default_mode_is_live(self, monkeypatch):
        _clear_pool_env(monkeypatch)
        assert load_pool_config().llm_mode == "live"

    def test_mode_is_lowercased(self, monkeypatch):
        _clear_pool_env(monkeypatch)
        monkeypatch.setenv("LLM_MODE", "REPLAY")
        assert load_pool_config().llm_mode == "replay"

    def test_default_cassette_dir(self, monkeypatch):
        _clear_pool_env(monkeypatch)
        assert load_pool_config().cassette_dir == "eval/cassettes"


class TestEffectiveEvalConcurrency:
    def test_auto_off_returns_default_unchanged(self, monkeypatch):
        _clear_pool_env(monkeypatch)
        assert effective_eval_concurrency(3) == 3

    def test_auto_on_with_no_pool_falls_back_to_default(self, monkeypatch):
        _clear_pool_env(monkeypatch)
        monkeypatch.setenv("EVAL_CONCURRENCY_AUTO", "true")
        monkeypatch.setattr(config, "GROQ_API_KEY", None)
        monkeypatch.setattr(config, "OPENROUTER_API_KEY", None)
        assert effective_eval_concurrency(3) == 3
