"""
runconfig.py — a frozen snapshot of the resolved configuration for one run.

Two runs whose numbers differ should be attributable to a specific cause
(code, model, corpus, provider, threshold) — not an indistinguishable "it
changed." RunConfig.to_dict() is what a later run-provenance record (the
eval_runs table) is meant to stamp onto every persisted run so comparison is
possible at all — this module only captures the snapshot; persisting it is
eval/harness.py's job.

Pure stdlib — no ML, no DB, no network beyond an optional `git rev-parse`
subprocess call that fails silently (e.g. a packaged/deployed environment
without a .git directory).
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional

from shared import config


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except Exception:
        return ""


def _parser_version() -> str:
    """PDF parser identity. Imported lazily: `shared` must not import
    `ingestion` at module scope (ingestion already imports shared)."""
    try:
        from ingestion.pdf_parser import PARSER_VERSION
        return PARSER_VERSION
    except Exception:
        return ""


@dataclass(frozen=True)
class RunConfig:
    """Everything that can make two runs produce different numbers, captured
    at the moment a run starts."""
    llm_provider: str
    llm_model: str
    generation_temperature: float
    compliance_temperature: float
    stability_temperature: float
    global_seed: Optional[int]
    premise_normalizer: str
    domain_profile: str
    dense_top_k: int
    sparse_top_k: int
    final_top_k: int
    rrf_k: int
    min_attribution_score: float
    ambiguity_gap_threshold: float
    weak_attribution_score: float
    strip_entailment_floor: float
    abstention_mean_penalty_ceil: float
    conflict_prefilter_cosine: float
    conflict_contradiction_threshold: float
    xai_embedding_model: str
    xai_embedding_model_revision: Optional[str]
    nli_model: str
    nli_model_revision: Optional[str]
    chroma_embedding_model: str
    spacy_model: str
    # Chunking determines chunk identity itself: change either value and every
    # chunk_key in the corpus moves, invalidating every labeled
    # relevant_chunk_key. They were missing here, so content_hash() was blind
    # to the single change most able to invalidate a retrieval label.
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    parser_version: str
    git_commit: str

    def content_hash(self) -> str:
        """Deterministic hash of the resolved config, independent of
        git_commit (metadata about WHERE it ran, not a config value)."""
        payload = {k: v for k, v in asdict(self).items() if k != "git_commit"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["content_hash"] = self.content_hash()
        return d


def capture_run_config() -> RunConfig:
    """Snapshot the currently-active shared/config.py values into a RunConfig."""
    provider = (config.LLM_PROVIDER or "groq").lower()
    model = config.GROQ_MODEL if provider == "groq" else config.OPENROUTER_MODEL
    return RunConfig(
        llm_provider=provider,
        llm_model=model,
        generation_temperature=config.GENERATION_TEMPERATURE,
        compliance_temperature=config.COMPLIANCE_TEMPERATURE,
        stability_temperature=config.STABILITY_TEMPERATURE,
        global_seed=config.GLOBAL_SEED,
        premise_normalizer=config.PREMISE_NORMALIZER,
        domain_profile=config.DOMAIN_PROFILE,
        dense_top_k=config.DENSE_TOP_K,
        sparse_top_k=config.SPARSE_TOP_K,
        final_top_k=config.FINAL_TOP_K,
        rrf_k=config.RRF_K,
        min_attribution_score=config.MIN_ATTRIBUTION_SCORE,
        ambiguity_gap_threshold=config.AMBIGUITY_GAP_THRESHOLD,
        weak_attribution_score=config.WEAK_ATTRIBUTION_SCORE,
        strip_entailment_floor=config.STRIP_ENTAILMENT_FLOOR,
        abstention_mean_penalty_ceil=config.ABSTENTION_MEAN_PENALTY_CEIL,
        conflict_prefilter_cosine=config.CONFLICT_PREFILTER_COSINE,
        conflict_contradiction_threshold=config.CONFLICT_CONTRADICTION_THRESHOLD,
        xai_embedding_model=config.XAI_EMBEDDING_MODEL,
        xai_embedding_model_revision=config.XAI_EMBEDDING_MODEL_REVISION,
        nli_model=config.NLI_MODEL,
        nli_model_revision=config.NLI_MODEL_REVISION,
        chroma_embedding_model=config.CHROMA_EMBEDDING_MODEL,
        spacy_model=config.SPACY_MODEL,
        chunk_max_tokens=config.CHUNK_MAX_TOKENS,
        chunk_overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
        parser_version=_parser_version(),
        git_commit=_git_commit(),
    )
