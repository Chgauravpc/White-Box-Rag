"""
Record/replay cassettes (Phase 2, W2.5 — see docs/BENCHMARK_READINESS.md).

A cassette maps `hash(provider|model|temperature|seed|system|prompt) ->
response text`. This is load-bearing for three separate requirements:
it's the only way to make end-to-end eval runs genuinely bit-reproducible
(provider `seed` is documented as best-effort by both Groq and OpenRouter,
never a guarantee); it lets CI run the full pipeline with zero API keys and
zero flakiness; and it lets ablations (Phase 6) differ *only* in the
governance layer rather than being confounded by fresh LLM resampling.

`LLM_MODE` (shared/llm_pool_config.py) selects the behavior:
  - "live"   (default): call the pool, never touch a cassette. Unchanged
    from pre-Phase-2 behavior.
  - "record": call the pool AND append the (key -> response) pair to the
    active cassette file.
  - "replay": look the key up in the active cassette; a miss is a hard
    KeyError (no silent fallback to a live call — a replay run must be
    either fully offline-reproducible or fail loudly, never quietly mix
    recorded and fresh responses).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CASSETTE_NAME = "default"
_active_cassette_name: ContextVar[str] = ContextVar("_active_cassette_name", default=_DEFAULT_CASSETTE_NAME)


def cassette_key(provider: str, model: str, temperature: float, seed: int | None,
                  system_instruction: str | None, prompt: str) -> str:
    """Deterministic key for one (provider, model, sampling params, prompt)
    tuple. Any change to the prompt/system text is a cache miss by design —
    a cassette records exact calls, not fuzzy-matched ones."""
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "temperature": round(temperature, 6),
            "seed": seed,
            "system": system_instruction or "",
            "prompt": prompt,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@contextmanager
def cassette_scope(name: str):
    """Select which cassette file subsequent call_llm() invocations in this
    context (and any thread/task it forks) record to / replay from — e.g.
    the eval harness names it after the dataset so each benchmark has its
    own recording."""
    token = _active_cassette_name.set(name)
    try:
        yield
    finally:
        _active_cassette_name.reset(token)


def active_cassette_name() -> str:
    return _active_cassette_name.get()


class Cassette:
    """One JSONL file: `{"key": ..., "provider":..., "model":..., "response": ...}`
    per line. Loaded once into an in-memory dict; later lines for a repeated
    key win (so re-recording a call overwrites the old response on next load)."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._index: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._index[record["key"]] = record["response"]
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"{self.path}:{line_no}: skipping malformed cassette line ({e})")

    def get(self, key: str) -> str | None:
        return self._index.get(key)

    def append(self, key: str, provider: str, model: str, response: str) -> None:
        self._index[key] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "provider": provider, "model": model, "response": response}) + "\n")


_cassettes: dict[str, Cassette] = {}


def get_cassette(cassette_dir: str, name: str | None = None) -> Cassette:
    name = name or active_cassette_name()
    path = os.path.join(cassette_dir, f"{name}.jsonl")
    if path not in _cassettes:
        _cassettes[path] = Cassette(path)
    return _cassettes[path]


def reset_cassettes() -> None:
    """Test hook — drop cached cassette objects so a fresh get_cassette()
    re-reads from disk."""
    _cassettes.clear()
