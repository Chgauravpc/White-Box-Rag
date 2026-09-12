"""
Multi-key LLM pool configuration (Phase 2, W2.1 — see docs/BENCHMARK_READINESS.md).

Keys become **lists**, not singular env vars — and a single-key setup keeps
working unchanged (a pool of one). `GROQ_API_KEYS` / `OPENROUTER_API_KEYS`
accept a comma-separated list; if unset, the legacy singular `GROQ_API_KEY` /
`OPENROUTER_API_KEY` (shared/config.py) is used as a one-item fallback, so
existing .env files need no changes.

`LLM_PROVIDER_ORDER` defaults to a list containing only the legacy
`LLM_PROVIDER` value — i.e. the pool talks to exactly one provider unless you
explicitly list more than one, which is the same behavior the single-provider
client had. `LLM_ALLOW_CROSS_PROVIDER_FAILOVER` likewise defaults to False:
even if you do list two providers, a call only spills from the primary to the
secondary once you opt in.

RPM/TPM-per-key defaults below are conservative placeholders, not a promise
about your actual plan tier — this is the "open decision" the roadmap
explicitly flags (how many keys you hold and their real rate limits). Set
`GROQ_RPM_PER_KEY` / `GROQ_TPM_PER_KEY` / `OPENROUTER_RPM_PER_KEY` /
`OPENROUTER_TPM_PER_KEY` from your console before relying on the pool's
admission math for a real benchmark run — an optimistic default just means
the pool learns the true ceiling from a 429 instead of avoiding it (see
llm_pool.py's FAILOVER/cooldown handling), which still works, but wastes calls.
"""

from dataclasses import dataclass

import shared.config as config
from shared.config import _env_str, _env_int, _env_bool, _env_list

_PROVIDER_DEFAULTS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1"},
}


@dataclass(frozen=True)
class ProviderPoolConfig:
    provider: str
    base_url: str
    model: str
    api_keys: tuple[str, ...]
    rpm_per_key: int
    tpm_per_key: int
    max_concurrency_per_key: int


@dataclass(frozen=True)
class LLMPoolConfig:
    providers: tuple[ProviderPoolConfig, ...]
    provider_order: tuple[str, ...]
    allow_cross_provider_failover: bool
    strict_single_model: bool
    llm_mode: str  # "live" | "record" | "replay"
    cassette_dir: str
    eval_concurrency_auto: bool

    def provider_config(self, provider: str) -> ProviderPoolConfig | None:
        for p in self.providers:
            if p.provider == provider:
                return p
        return None

    def total_configured_keys(self) -> int:
        return sum(len(p.api_keys) for p in self.providers)


def _provider_keys(env_name_plural: str, legacy_single: str | None) -> tuple[str, ...]:
    keys = _env_list(env_name_plural)
    if keys:
        return tuple(keys)
    return (legacy_single,) if legacy_single else ()


def load_pool_config() -> LLMPoolConfig:
    """Build the pool topology from environment every call (cheap, no I/O) —
    callers that want a stable snapshot (e.g. the pool singleton) call this
    once and hold the result."""
    groq_keys = _provider_keys("GROQ_API_KEYS", config.GROQ_API_KEY)
    openrouter_keys = _provider_keys("OPENROUTER_API_KEYS", config.OPENROUTER_API_KEY)

    providers = []
    if groq_keys:
        providers.append(ProviderPoolConfig(
            provider="groq",
            base_url=_PROVIDER_DEFAULTS["groq"]["base_url"],
            model=config.GROQ_MODEL,
            api_keys=groq_keys,
            rpm_per_key=_env_int("GROQ_RPM_PER_KEY", 30),
            tpm_per_key=_env_int("GROQ_TPM_PER_KEY", 6000),
            max_concurrency_per_key=_env_int("LLM_MAX_CONCURRENCY_PER_KEY", 2),
        ))
    if openrouter_keys:
        providers.append(ProviderPoolConfig(
            provider="openrouter",
            base_url=_PROVIDER_DEFAULTS["openrouter"]["base_url"],
            model=config.OPENROUTER_MODEL,
            api_keys=openrouter_keys,
            rpm_per_key=_env_int("OPENROUTER_RPM_PER_KEY", 20),
            tpm_per_key=_env_int("OPENROUTER_TPM_PER_KEY", 60000),
            max_concurrency_per_key=_env_int("LLM_MAX_CONCURRENCY_PER_KEY", 2),
        ))

    known = {p.provider for p in providers}
    explicit_order = _env_list("LLM_PROVIDER_ORDER")
    if explicit_order:
        # Explicit opt-in: honor it, but a provider we DO have keys for is
        # never silently dropped just because it's missing from the list
        # (append it at the end rather than ignore it).
        provider_order = tuple(p for p in explicit_order if p in known) + tuple(
            p for p in known if p not in explicit_order
        )
    else:
        # No opt-in: stay pinned to the single legacy LLM_PROVIDER, even if
        # keys exist for another provider too — this is the exact behavior
        # the pre-Phase-2 single-client `LLM_PROVIDER` had, so an existing
        # .env with both keys set doesn't silently start using both.
        provider_order = tuple(p for p in ([config.LLM_PROVIDER] if config.LLM_PROVIDER else []) if p in known)

    return LLMPoolConfig(
        providers=tuple(providers),
        provider_order=provider_order,
        allow_cross_provider_failover=_env_bool("LLM_ALLOW_CROSS_PROVIDER_FAILOVER", False),
        strict_single_model=_env_bool("LLM_STRICT_SINGLE_MODEL", False),
        llm_mode=_env_str("LLM_MODE", "live").lower(),
        cassette_dir=_env_str("LLM_CASSETTE_DIR", "eval/cassettes"),
        eval_concurrency_auto=_env_bool("EVAL_CONCURRENCY_AUTO", False),
    )


def effective_eval_concurrency(default: int) -> int:
    """W2.6: item-level eval concurrency as a function of pool size. An item
    holds one call slot at a time, so the real ceiling is the pool's
    call-level capacity (sum of each endpoint's max_concurrency) once more
    than one key exists — not the static EVAL_CONCURRENCY default sized for
    a single key. Opt-in via EVAL_CONCURRENCY_AUTO=true; falls back to
    `default` when auto-scaling is off, no pool is configured (e.g. the test
    suite, which has no live keys), or the pool has zero capacity."""
    cfg = load_pool_config()
    if not cfg.eval_concurrency_auto:
        return default
    try:
        from shared.llm_pool import get_pool
        total = get_pool().total_max_concurrency()
        return total if total > 0 else default
    except Exception:
        return default
