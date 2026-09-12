"""
Centralised configuration for the White Box RAG governance framework.
Loads settings from .env file at the project root.

Every decision threshold, model identity, and sampling parameter used
anywhere in the pipeline lives here, env-overridable, with today's original
value as the default — so behavior is unchanged unless a run opts in.
Before this, the same thresholds were re-declared as bare module-level
literals in several files (e.g. WEAK_ATTRIBUTION_SCORE = 0.65 existed
independently in three different modules), which is a real drift risk: a
tuning change in one copy silently wouldn't reach the others. Centralizing
also makes a run's configuration snapshot-able (see shared/runconfig.py) and
is the seam Phase 6 ablations (mitigation_off, conformal_off, ...) build on.
"""

import os
from dotenv import load_dotenv

# Load .env from project root (two levels up from backend/shared/)
load_dotenv()


def _env_str(name: str, default):
    val = os.getenv(name)
    return val if val is not None else default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


def _env_optional_int(name: str):
    """Like _env_int, but the *absence* of the env var means None (not a
    numeric default) — used for GLOBAL_SEED, where 'unset' and '0' are
    different things (0 is a valid seed)."""
    val = os.getenv(name)
    return int(val) if val is not None else None


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: list | None = None) -> list:
    """Comma-separated env var -> list[str], trimmed, empties dropped.
    Absence of the var returns `default` (or [])."""
    val = os.getenv(name)
    if val is None:
        return list(default) if default else []
    return [v.strip() for v in val.split(",") if v.strip()]


# ---------- LLM Provider (Groq / OpenRouter — both OpenAI-compatible APIs) ----------
LLM_PROVIDER = _env_str("LLM_PROVIDER", "groq")  # "groq" | "openrouter"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = _env_str("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = _env_str("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

# ---------- Storage Paths ----------
CHROMA_PATH = _env_str("CHROMA_PATH", "./data/chromadb")
SQLITE_PATH = _env_str("SQLITE_PATH", "./data/metadata.db")
# Frozen corpus manifests (W3.1). A data directory, deliberately not
# inside the backend package — it is generated artefact, not code.
CORPORA_PATH = _env_str("CORPORA_PATH", "./data/corpora")

# ---------- Domain Constants ----------
# Domain-agnostic: "publication"/"collection" is a free-text label chosen at
# ingest time, not a fixed enum. Any non-empty, reasonably-sized string is valid
# (see ingestion/routes.py::_validate_collection_label). This system is not
# restricted to any single domain (finance, legal, engineering, etc.).
MAX_COLLECTION_LABEL_LENGTH = _env_int("MAX_COLLECTION_LABEL_LENGTH", 64)

# ---------- Chunking Defaults ----------
CHUNK_MAX_TOKENS = _env_int("CHUNK_MAX_TOKENS", 512)
CHUNK_OVERLAP_TOKENS = _env_int("CHUNK_OVERLAP_TOKENS", 50)

# ---------- Retrieval Defaults ----------
DENSE_TOP_K = _env_int("DENSE_TOP_K", 20)
SPARSE_TOP_K = _env_int("SPARSE_TOP_K", 20)
FINAL_TOP_K = _env_int("FINAL_TOP_K", 10)
RRF_K = _env_int("RRF_K", 60)  # Reciprocal Rank Fusion constant

# ---------- Sentence / NLI preprocessing (shared/xai_matrices.py) ----------
MIN_SENTENCE_LEN = _env_int("MIN_SENTENCE_LEN", 20)
NLI_TOP_K = _env_int("NLI_TOP_K", 3)
NLI_PREPROCESS_AT = _env_int("NLI_PREPROCESS_AT", 60)

# ---------- Attribution thresholds (shared/xai_matrices.py) ----------
MIN_ATTRIBUTION_SCORE = _env_float("MIN_ATTRIBUTION_SCORE", 0.45)
AMBIGUITY_GAP_THRESHOLD = _env_float("AMBIGUITY_GAP_THRESHOLD", 0.02)
WEAK_ATTRIBUTION_SCORE = _env_float("WEAK_ATTRIBUTION_SCORE", 0.65)

# ---------- Trust Gate / Shapley penalty weights ----------
# verification/trust_gate.py and shared/xai_matrices.py::compute_shapley_contributions
# apply the SAME penalties so overall_score stays consistent between the two
# (see their module docstrings) — centralized here so they can't drift apart.
PENALTY_EDITION_CONFLICT = _env_float("PENALTY_EDITION_CONFLICT", 0.5)
PENALTY_CONTRADICTION = _env_float("PENALTY_CONTRADICTION", 0.30)
PENALTY_NEUTRAL = _env_float("PENALTY_NEUTRAL", 0.10)
PENALTY_LOW_CONFIDENCE = _env_float("PENALTY_LOW_CONFIDENCE", 0.20)
PENALTY_MID_CONFIDENCE = _env_float("PENALTY_MID_CONFIDENCE", 0.05)
PENALTY_AMBIGUOUS_ATTRIBUTION = _env_float("PENALTY_AMBIGUOUS_ATTRIBUTION", 0.05)
PENALTY_WEAK_ATTRIBUTION = _env_float("PENALTY_WEAK_ATTRIBUTION", 0.03)

# NLI confidence bands: entailment_score < LOW_CONFIDENCE_CEIL => low confidence;
# LOW_CONFIDENCE_CEIL <= entailment_score <= MID_CONFIDENCE_CEIL => medium confidence.
LOW_CONFIDENCE_CEIL = _env_float("LOW_CONFIDENCE_CEIL", 0.5)
MID_CONFIDENCE_CEIL = _env_float("MID_CONFIDENCE_CEIL", 0.8)

# ---------- Mitigation / Abstention (verification/mitigation.py) ----------
STRIP_ENTAILMENT_FLOOR = _env_float("STRIP_ENTAILMENT_FLOOR", 0.5)
ABSTENTION_MEAN_PENALTY_CEIL = _env_float("ABSTENTION_MEAN_PENALTY_CEIL", 0.25)

# ---------- Edition Conflict Detection (verification/edition_conflict.py) ----------
CONFLICT_PREFILTER_COSINE = _env_float("CONFLICT_PREFILTER_COSINE", 0.90)
CONFLICT_CONTRADICTION_THRESHOLD = _env_float("CONFLICT_CONTRADICTION_THRESHOLD", 0.70)
MAX_CONFLICT_CHECKS_PER_QUERY = _env_int("MAX_CONFLICT_CHECKS_PER_QUERY", 5)

# ---------- Generation Sampling ----------
GENERATION_TEMPERATURE = _env_float("GENERATION_TEMPERATURE", 0.2)   # primary RAG answer (ingestion/rag.py)
COMPLIANCE_TEMPERATURE = _env_float("COMPLIANCE_TEMPERATURE", 0.1)   # mapper/audit/brd_parser
STABILITY_TEMPERATURE = _env_float("STABILITY_TEMPERATURE", 0.7)     # paraphrase re-sample (verification/stability.py)
# Best-effort determinism hint forwarded to the LLM provider's `seed` param
# (both Groq and OpenRouter document this as best-effort, not a guarantee —
# real reproducibility still needs shared/llm_cassette.py's record/replay,
# a later phase). None (the default) means "don't send a seed at all."
GLOBAL_SEED = _env_optional_int("GLOBAL_SEED")

# ---------- Eval Harness ----------
EVAL_CONCURRENCY = _env_int("EVAL_CONCURRENCY", 3)

# ---------- Model identities ----------
# Setting a `*_REVISION` env var pins that exact HF commit/tag instead of the
# mutable default branch. Left unpinned (None) by default: guessing a
# revision hash would be worse than not pinning at all. Pin these once you've
# verified a specific revision works for you.
#
# XAI_EMBEDDING_MODEL / NLI_MODEL are the models shared/xai_matrices.py uses
# for attribution, retrieval-similarity, and NLI verification.
# CHROMA_EMBEDDING_MODEL is the (deliberately different, see finding #18 in
# the benchmark-readiness review) model ChromaDB uses to actually retrieve —
# its embedding_functions.SentenceTransformerEmbeddingFunction wrapper does
# not accept a revision pin.
XAI_EMBEDDING_MODEL = _env_str("XAI_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
XAI_EMBEDDING_MODEL_REVISION = _env_str("XAI_EMBEDDING_MODEL_REVISION", None)
NLI_MODEL = _env_str("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
NLI_MODEL_REVISION = _env_str("NLI_MODEL_REVISION", None)
CHROMA_EMBEDDING_MODEL = _env_str("CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SPACY_MODEL = _env_str("SPACY_MODEL", "en_core_web_sm")

# ---------- NLI Premise Normalization ----------
# "none" = no mutation (required for external benchmark adapters — must not
# touch the benchmark's own evidence text); "generic" = strip only PDF
# extraction artifacts (page markers), safe for any corpus; "financial_reports"
# = legacy RBI-report-tuned stripping (numeric rows, table headers, short
# all-caps lines) — opt-in only, since it deletes real evidence on other
# domains. See shared/text_normalize.py.
PREMISE_NORMALIZER = _env_str("PREMISE_NORMALIZER", "generic")

# ---------- Compliance Domain Profile ----------
# Selects the persona/category-enum/few-shot examples used by the BRD parser
# and audit-report prompts (backend/compliance/domain_profiles.py). "generic"
# matches the project's domain-agnostic ingestion design; "financial_reports"
# preserves the original RBI-specific behaviour for continuity.
DOMAIN_PROFILE = _env_str("DOMAIN_PROFILE", "generic")
