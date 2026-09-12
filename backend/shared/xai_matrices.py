"""xai_matrices.py - Pure Mathematical XAI Matrix Computations. LLM extracts, Math judges."""

import asyncio
import logging
import threading
from collections import OrderedDict
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder

from shared import config
from shared.config import PREMISE_NORMALIZER
from shared.text_normalize import normalize_premise
from shared.nli_policy import nli_penalty_flags
from shared.chunk_key import resolve_chunk_key

logger = logging.getLogger(__name__)

def _load_models():
    logger.info(f"Loading SentenceTransformer ({config.XAI_EMBEDDING_MODEL}, revision={config.XAI_EMBEDDING_MODEL_REVISION or 'default'})...")
    encoder = SentenceTransformer(config.XAI_EMBEDDING_MODEL, revision=config.XAI_EMBEDDING_MODEL_REVISION)
    logger.info(f"Loading NLI CrossEncoder ({config.NLI_MODEL}, revision={config.NLI_MODEL_REVISION or 'default'})...")
    nli = CrossEncoder(config.NLI_MODEL, revision=config.NLI_MODEL_REVISION, num_labels=3)
    logger.info("XAI matrix models ready.")
    return encoder, nli

_encoder, _nli = _load_models()
_spacy_nlp = spacy.load(config.SPACY_MODEL)

# ── NLI label order ─────────────────────────────────────────
# The 3-way output order is a property of the CHECKPOINT, not of NLI. This
# code used to hardcode cross-encoder/nli-deberta-v3-base's order
# (contradiction, entailment, neutral) in three places — but NLI_MODEL is
# env-overridable, and roberta-large-mnli (the most natural swap) emits
# (contradiction, neutral, entailment). Swapping the model would therefore
# have silently exchanged "entailment" and "neutral" for every claim in the
# system: hallucinations scored as grounded, with nothing raising.
# The order is now read from the checkpoint's own id2label.
DEFAULT_NLI_LABELS = ["contradiction", "entailment", "neutral"]


def _resolve_nli_label_order(nli) -> list[str]:
    """The checkpoint's label order, falling back to the documented default.

    Falls back silently when the config isn't a real mapping (the test suite
    mocks CrossEncoder), and loudly when it IS a mapping but doesn't describe
    a 3-way NLI head — that means the configured model cannot be used here.
    """
    try:
        id2label = nli.model.config.id2label
        if not isinstance(id2label, dict):
            return list(DEFAULT_NLI_LABELS)
        labels = [str(id2label[i]).strip().lower() for i in range(3)]
    except Exception:
        return list(DEFAULT_NLI_LABELS)

    if sorted(labels) != sorted(DEFAULT_NLI_LABELS):
        logger.error(
            "NLI model %s reports labels %s, which are not the expected "
            "three-way (contradiction/entailment/neutral) set. Falling back to %s — "
            "verdicts from this model are NOT trustworthy.",
            config.NLI_MODEL, labels, DEFAULT_NLI_LABELS,
        )
        return list(DEFAULT_NLI_LABELS)

    if labels != DEFAULT_NLI_LABELS:
        logger.warning(
            "NLI model %s uses label order %s (not the default %s). Index mapping "
            "adjusted accordingly.", config.NLI_MODEL, labels, DEFAULT_NLI_LABELS,
        )
    return labels


NLI_LABELS = _resolve_nli_label_order(_nli)
CONTRADICTION_IDX = NLI_LABELS.index("contradiction")
ENTAILMENT_IDX = NLI_LABELS.index("entailment")
NEUTRAL_IDX = NLI_LABELS.index("neutral")


def entailment_score_of(score_triplet) -> float:
    """Pull the entailment probability out of one 3-way output, whatever order
    the active checkpoint emits."""
    return float(score_triplet[ENTAILMENT_IDX])

def get_encoder(): return _encoder
def get_nli():     return _nli


def model_identity() -> dict:
    """Resolved model identity for run provenance — which encoder/NLI/spaCy
    model (and pinned revision, if any) actually produced a given run's
    numbers. See shared/runconfig.py."""
    return {
        "xai_embedding_model": config.XAI_EMBEDDING_MODEL,
        "xai_embedding_model_revision": config.XAI_EMBEDDING_MODEL_REVISION,
        "xai_embedding_dim": int(_encoder.get_sentence_embedding_dimension()),
        "nli_model": config.NLI_MODEL,
        "nli_model_revision": config.NLI_MODEL_REVISION,
        "spacy_model": config.SPACY_MODEL,
        "chroma_embedding_model": config.CHROMA_EMBEDDING_MODEL,
    }


# Re-exported from config (not re-declared as literals) so the many external
# imports of these names (e.g. verification/stability.py imports
# NLI_PREPROCESS_AT from here) keep working unchanged.
MIN_SENTENCE_LEN      = config.MIN_SENTENCE_LEN
NLI_TOP_K              = config.NLI_TOP_K
NLI_PREPROCESS_AT      = config.NLI_PREPROCESS_AT
MIN_ATTRIBUTION_SCORE  = config.MIN_ATTRIBUTION_SCORE
AMBIGUITY_GAP_THRESHOLD = config.AMBIGUITY_GAP_THRESHOLD
WEAK_ATTRIBUTION_SCORE  = config.WEAK_ATTRIBUTION_SCORE

# Sentence-splitting and embedding a premise is the expensive half of premise
# preparation: a spaCy parse plus one bge-large forward pass per sentence. The
# same premise is very often prepared many times in a row — every claim from
# one generated answer shares its source chunk, and every sentence of one
# benchmark response shares its evidence. On a RAGTruth sample that was 1,502
# preparations over 175 distinct premises, i.e. ~8.6x of the work was
# recomputation, and holding all of it at once is what exhausted memory.
#
# Bounded LRU rather than an unbounded dict: this is a long-lived process and
# an unbounded cache keyed on document text is a memory leak with extra steps.
_SENTENCE_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_SENTENCE_CACHE_LOCK = threading.Lock()


def _split_and_embed(clean_passage: str):
    """(sentences, unit-normalized embeddings) for a premise, memoized."""
    with _SENTENCE_CACHE_LOCK:
        hit = _SENTENCE_CACHE.get(clean_passage)
        if hit is not None:
            _SENTENCE_CACHE.move_to_end(clean_passage)
            return hit

    doc = _spacy_nlp(clean_passage)
    sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > MIN_SENTENCE_LEN]
    if len(sentences) > NLI_TOP_K:
        embs = _encoder.encode(sentences)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    else:
        embs = None  # no ranking needed; don't pay for embeddings
    value = (sentences, embs)

    with _SENTENCE_CACHE_LOCK:
        _SENTENCE_CACHE[clean_passage] = value
        while len(_SENTENCE_CACHE) > config.PREMISE_CACHE_SIZE:
            _SENTENCE_CACHE.popitem(last=False)
    return value


def clear_premise_cache():
    """Drop the memoized premise sentences/embeddings (tests, and any caller
    that has just changed the active profile or embedding model)."""
    with _SENTENCE_CACHE_LOCK:
        _SENTENCE_CACHE.clear()


def extract_relevant_sentences(claim, passage, profile=None):
    """Extract top-K prose sentences most similar to the claim.

    Normalizes the passage per `profile` first (default: config.PREMISE_NORMALIZER).
    Idempotent on an already-normalized passage, so callers that pre-normalize
    (verify_claims_batch) and callers that don't (build_entailment_matrix) both
    get consistent behavior.
    """
    clean_passage, _ = normalize_premise(passage, profile or PREMISE_NORMALIZER)
    sentences, sent_norm = _split_and_embed(clean_passage)
    if not sentences:
        return clean_passage[:800] if clean_passage else passage[:800]
    if len(sentences) <= NLI_TOP_K:
        return " ".join(sentences)
    claim_emb  = _encoder.encode([claim])
    claim_norm = claim_emb / np.linalg.norm(claim_emb)
    scores     = (claim_norm @ sent_norm.T).squeeze()
    if np.ndim(scores) == 0:
        return sentences[0]
    top_indices = scores.argsort()[-NLI_TOP_K:][::-1]
    return " ".join([sentences[i] for i in sorted(top_indices)])

def build_retrieval_similarity_matrix(query, chunks):
    if not chunks:
        return [], []
    q_emb  = _encoder.encode([query])
    k_emb  = _encoder.encode([c["chunk_text"] for c in chunks])
    q_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    k_norm = k_emb / np.linalg.norm(k_emb, axis=1, keepdims=True)
    S = (q_norm @ k_norm.T).squeeze(axis=0)
    if np.ndim(S) == 0:
        S = np.array([float(S)])
    # Canonical chunk keys, NOT bare section_id — two documents may share a
    # section_id (both have a "1.1"), which would otherwise collide here.
    chunk_ids = [
        resolve_chunk_key(c["publication_name"], c["edition_date"], c["section_id"], c["chunk_text"], c.get("chunk_key", ""))
        for c in chunks
    ]
    return chunk_ids, [float(x) for x in S]

def prepare_premise(claim, raw_passage, profile=None):
    """Turn a raw passage into the exact text the NLI model will see.

    THE single place premise preparation happens. `verify_claims_batch` and
    `build_entailment_matrix` used to do this independently and disagreed four
    ways for the same (claim, passage) pair:

      * the matrix applied the `NLI_PREPROCESS_AT` word-count threshold to the
        RAW passage, while verification applied it to the NORMALIZED one, so a
        passage could be sentence-focused on one path and not the other;
      * below that threshold the matrix fed the RAW passage and verification
        fed the NORMALIZED one;
      * the matrix never received the caller's profile, so it always used the
        config default even when the caller asked for "none" (which is what
        the detector plane passes for external benchmarks);
      * the matrix had no empty-premise guard, so it still sent ("", claim) to
        the model.

    The visible consequence: the entailment matrix rendered in the UI could
    contradict the verdict displayed next to it, because the two numbers came
    from different premise text.

    Returns (focused_text, deleted_lines, evidence_status), where
    evidence_status is "ok" / "no_premise" / "normalizer_deleted_all".
    """
    profile = profile or PREMISE_NORMALIZER
    stripped, deleted = normalize_premise(raw_passage, profile) if raw_passage else ("", [])

    if stripped and len(stripped.split()) > NLI_PREPROCESS_AT:
        focused = extract_relevant_sentences(claim, stripped, profile=profile)
    else:
        focused = stripped  # already short enough; no sentence selection needed

    if focused.strip():
        return focused, deleted, "ok"

    if raw_passage and raw_passage.strip():
        # A passage existed and normalization consumed all of it — a normalizer
        # misconfiguration (e.g. the financial_reports profile on a
        # non-financial corpus), not a property of the claim.
        logger.warning(
            "PREMISE_NORMALIZER '%s' deleted the entire premise for claim %r "
            "(%d lines removed) — claim cannot be verified.",
            profile, str(claim)[:80], len(deleted),
        )
        return focused, deleted, "normalizer_deleted_all"
    return focused, deleted, "no_premise"


def build_entailment_matrix(claims, passages, profile=None):
    """The (claims x passages x 3) probability cube the XAI view renders.

    Premise preparation goes through `prepare_premise`, the same function
    `verify_claims_batch` uses, so a cell of this matrix and the verdict shown
    beside it are computed from identical premise text. Cells whose premise is
    empty are left as zeros rather than being scored — an empty premise
    produces a well-formed but meaningless softmax.
    """
    if not claims or not passages:
        return np.array([]), list(NLI_LABELS)

    pairs, positions = [], []
    scores = np.zeros((len(claims), len(passages), 3), dtype="float32")
    for i, claim in enumerate(claims):
        for j, passage in enumerate(passages):
            focused, _deleted, status = prepare_premise(claim, passage, profile=profile)
            if status == "ok":
                pairs.append((focused, claim))
                positions.append((i, j))

    if pairs:
        raw = _nli.predict(pairs, apply_softmax=True)
        for (i, j), triplet in zip(positions, raw):
            scores[i, j] = triplet
    return scores, list(NLI_LABELS)

def verify_claims_batch(claims_with_passages, profile=None):
    if not claims_with_passages:
        return []
    profile = profile or PREMISE_NORMALIZER
    focused_pairs     = []
    focused_passages  = []
    premise_deletions = []
    evidence_statuses = []
    for claim, raw_passage in claims_with_passages:
        # Premise preparation — including the empty-premise guard — lives in
        # prepare_premise(), which build_entailment_matrix also calls, so the
        # verdict and the XAI matrix cell for the same (claim, passage) can no
        # longer be computed from different text.
        focused, deleted, status = prepare_premise(claim, raw_passage, profile=profile)
        evidence_statuses.append(status)
        focused_passages.append(focused)
        premise_deletions.append(deleted)
        if status == "ok":
            focused_pairs.append((focused, claim))

    # Only claims with a real premise are scored. An all-empty batch must not
    # call predict([]) — some backends raise, others return an empty array that
    # would silently misalign the zip below.
    scores = _nli.predict(focused_pairs, apply_softmax=True) if focused_pairs else []
    score_iter = iter(scores)

    results = []
    for (claim, _), focused, deleted, status in zip(
        claims_with_passages, focused_passages, premise_deletions, evidence_statuses
    ):
        if status == "ok":
            score_triplet = next(score_iter)
            verdict = NLI_LABELS[int(score_triplet.argmax())].upper()
            entail, contra, neutral = (
                float(score_triplet[ENTAILMENT_IDX]),
                float(score_triplet[CONTRADICTION_IDX]),
                float(score_triplet[NEUTRAL_IDX]),
            )
        else:
            # NOT_ENOUGH_INFO is the honest verdict for "no evidence was
            # examined". The zeros are structural, not measurements — which is
            # exactly why `evidence_status` travels alongside them.
            verdict, entail, contra, neutral = "NOT_ENOUGH_INFO", 0.0, 0.0, 0.0
        results.append({
            "claim_text":          claim,
            "verdict":             verdict,
            "entailment_score":    entail,
            "contradiction_score": contra,
            "neutral_score":       neutral,
            "focused_passage":     focused,
            "premise_deletions":   deleted,
            "evidence_status":     status,
        })
    return results

def build_attribution_matrix(answer_sentences, chunks):
    if not answer_sentences or not chunks:
        return np.array([])
    sent_emb  = _encoder.encode(answer_sentences)
    chunk_emb = _encoder.encode([c["chunk_text"] for c in chunks])
    sent_norm  = sent_emb  / np.linalg.norm(sent_emb,  axis=1, keepdims=True)
    chunk_norm = chunk_emb / np.linalg.norm(chunk_emb, axis=1, keepdims=True)
    return sent_norm @ chunk_norm.T

def compute_answer_relevancy(query, answer):
    """cosine(encode(query), encode(answer)) — simplified pure-math proxy for
    RAGAS's Answer Relevancy metric (which uses LLM-generated pseudo-questions;
    out of scope here since we don't use an LLM as judge)."""
    if not query or not answer:
        return 0.0
    q_emb = _encoder.encode([query])
    a_emb = _encoder.encode([answer])
    q_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    a_norm = a_emb / np.linalg.norm(a_emb, axis=1, keepdims=True)
    return float((q_norm @ a_norm.T).squeeze())


def compute_context_utilization(primary_attributions, chunk_ids):
    """Fraction of retrieved chunks actually cited as the primary source of >=1 claim."""
    if not chunk_ids:
        return 0.0
    cited = {a["primary_chunk_id"] for a in primary_attributions if a.get("primary_chunk_id")}
    return round(len(cited) / len(chunk_ids), 6)


def compute_context_diversity(chunks):
    """1 - mean pairwise cosine similarity of retrieved chunk embeddings.
    Flags retrieval that returned several near-duplicate chunks (wasted context)."""
    if not chunks or len(chunks) < 2:
        return 1.0
    emb = _encoder.encode([c["chunk_text"] for c in chunks])
    norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim_matrix = norm @ norm.T
    iu = np.triu_indices(len(chunks), k=1)
    if len(iu[0]) == 0:
        return 1.0
    mean_pairwise = float(sim_matrix[iu].mean())
    return round(max(0.0, 1.0 - mean_pairwise), 6)


def attribute_sentence(A_row, chunks):
    top_j     = int(A_row.argmax())
    max_score = float(A_row[top_j])
    if max_score < MIN_ATTRIBUTION_SCORE:
        return None
    return {
        "section_id":        chunks[top_j]["section_id"],
        "publication_name":  chunks[top_j]["publication_name"],
        "edition_date":      chunks[top_j]["edition_date"],
        "source_passage":    chunks[top_j]["chunk_text"],
        "attribution_score": max_score,
    }

def build_conflict_matrix(old_chunks, new_chunks):
    if not old_chunks or not new_chunks:
        return np.array([])
    pairs  = [(old["chunk_text"], new["chunk_text"]) for old in old_chunks for new in new_chunks]
    scores = _nli.predict(pairs, apply_softmax=True)
    return scores[:, 0].reshape(len(old_chunks), len(new_chunks))

def detect_conflicts(old_chunks, new_chunks, threshold=0.7):
    C = build_conflict_matrix(old_chunks, new_chunks)
    if C.size == 0:
        return []
    conflicts = []
    for i, j in zip(*np.where(C > threshold)):
        conflicts.append({
            "old_section":         old_chunks[i]["section_id"],
            "new_section":         new_chunks[j]["section_id"],
            "contradiction_score": float(C[i][j]),
            "superseding_edition": new_chunks[j]["edition_date"],
        })
    return conflicts

def compute_shapley_contributions(verifications, primary_attributions=None):
    """Compute Shapley penalty vector across NLI verdicts AND attribution quality.

    primary_attributions[i] corresponds to verifications[i].
    Including attribution penalties ensures shapley.overall_score matches
    trust_gate.overall_score (both deduct the same set of penalties — both
    read the SAME config.PENALTY_* constants, so they can't drift apart).
    """
    if primary_attributions is None:
        primary_attributions = []

    nli_penalties  = {
        "contradiction": config.PENALTY_CONTRADICTION,
        "neutral": config.PENALTY_NEUTRAL,
        "low_confidence": config.PENALTY_LOW_CONFIDENCE,
        "mid_confidence": config.PENALTY_MID_CONFIDENCE,
    }
    attr_penalties = {
        "ambiguous": config.PENALTY_AMBIGUOUS_ATTRIBUTION,
        "weak": config.PENALTY_WEAK_ATTRIBUTION,
    }

    contributions = []
    for i, v in enumerate(verifications):
        phi          = 0.0
        reasons      = []
        verdict      = v.get("verdict", "").upper()
        entail_score = v.get("entailment_score", 1.0)

        # NLI penalties — decided by shared/nli_policy.py, the same function
        # verification/trust_gate.py calls. These used to be two independent
        # copies of the rule, matching bare string literals here and the
        # NLIVerdict enum (including its aliases) there, so a CONTRADICTED /
        # NOT_ENOUGH_INFO verdict scored differently in the two computations.
        flags = nli_penalty_flags(verdict, entail_score)
        if flags["contradiction"]:
            phi += nli_penalties["contradiction"]; reasons.append("contradiction")
        if flags["neutral"]:
            phi += nli_penalties["neutral"]; reasons.append("neutral")
        if flags["low_confidence"]:
            phi += nli_penalties["low_confidence"]; reasons.append("low NLI confidence ({:.2f})".format(entail_score))
        if flags["mid_confidence"]:
            phi += nli_penalties["mid_confidence"]; reasons.append("mid NLI confidence ({:.2f})".format(entail_score))

        # Attribution penalties (mirrors trust gate logic exactly)
        if i < len(primary_attributions):
            attr = primary_attributions[i]
            if attr.get("ambiguous"):
                phi += attr_penalties["ambiguous"]
                reasons.append("ambiguous attribution (gap={:.4f})".format(attr.get("confidence_gap", 0)))
            elif attr.get("attribution_score", 1.0) < WEAK_ATTRIBUTION_SCORE:
                phi += attr_penalties["weak"]
                reasons.append("weak attribution (score={:.4f})".format(attr.get("attribution_score", 0)))

        contributions.append({
            "claim_text":      v.get("claim_text", ""),
            "shapley_value":   round(phi, 4),
            "penalty_reasons": reasons,
        })

    contributions.sort(key=lambda x: x["shapley_value"], reverse=True)
    overall_score = max(0.0, 1.0 - sum(c["shapley_value"] for c in contributions))
    return {
        "overall_score":   round(overall_score, 4),
        "claim_texts":     [c["claim_text"]    for c in contributions],
        "shapley_values":  [c["shapley_value"] for c in contributions],
        "penalty_reasons": [c["penalty_reasons"] for c in contributions],
    }

def compute_primary_attributions(A, chunk_ids):
    """Per-sentence: primary chunk, runner-up, confidence gap, and calibrated ambiguity flag."""
    if A.size == 0 or not chunk_ids:
        return []
    results = []
    for i in range(A.shape[0]):
        row        = A[i]
        sorted_idx = row.argsort()[::-1]
        top_idx    = int(sorted_idx[0])
        top_score  = float(row[top_idx])
        ru_idx     = int(sorted_idx[1]) if len(sorted_idx) > 1 else top_idx
        ru_score   = float(row[ru_idx]) if len(sorted_idx) > 1 else top_score
        gap = round(top_score - ru_score, 4)
        results.append({
            "sentence_index":     i,
            "primary_chunk_id":   chunk_ids[top_idx] if top_idx < len(chunk_ids) else "unknown",
            "attribution_score":  round(top_score, 4),
            "runner_up_chunk_id": chunk_ids[ru_idx]  if ru_idx  < len(chunk_ids) else "unknown",
            "runner_up_score":    round(ru_score, 4),
            "confidence_gap":     gap,
            "ambiguous":          gap < AMBIGUITY_GAP_THRESHOLD,
        })
    return results

def find_related_queries(query, stored_embeddings, top_k=3, threshold=0.70):
    """Find related past queries via pure cosine similarity. No LLM.
    Caller should filter cosine_similarity >= 0.99 (exact re-runs) after calling.
    """
    if not stored_embeddings:
        return []
    q_emb        = _encoder.encode([query])
    q_norm       = q_emb / np.linalg.norm(q_emb)
    expected_dim = q_emb.shape[1]
    results      = []
    for record in stored_embeddings:
        raw = record.get("embedding")
        if not raw:
            continue
        stored_emb = np.array(raw, dtype=np.float32)
        if stored_emb.shape[0] != expected_dim:
            continue
        stored_norm = stored_emb / (np.linalg.norm(stored_emb) + 1e-9)
        sim = float(q_norm @ stored_norm)
        if sim >= threshold:
            results.append({
                "id":                record["id"],
                "query":             record["query"],
                "timestamp":         record.get("timestamp", ""),
                "cosine_similarity": round(sim, 4),
                "trust_status":      record.get("trust_status", "unknown"),
            })
    results.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    return results[:top_k]


# ── Async wrappers ──────────────────────────────────────────
# Every model call in this module is synchronous, and they are invoked from
# `async def` functions (nli_engine, stability, the pipeline). A CrossEncoder
# forward pass therefore blocked the event loop: under EVAL_CONCURRENCY each
# item's NLI batch stalled every other in-flight item's LLM HTTP calls, so
# raising concurrency bought far less than it looked like it should.
#
# The lock is not optional. `_encoder` and `_nli` are single shared
# module-level torch modules; running predict() on one from several threads at
# once multiplies peak memory by the thread count and can exhaust a machine
# that comfortably runs one. An RLock (not Lock) because the guarded functions
# call each other — verify_claims_batch -> extract_relevant_sentences both
# touch the models, and a plain Lock would deadlock on the re-entry.
#
# Deliberately a threading primitive rather than an asyncio one: an
# asyncio.Semaphore binds to the event loop that first awaits it, which breaks
# the moment a second loop appears (every `asyncio.run` in the test suite).
_MODEL_LOCK = threading.RLock()


def _locked(fn, *args, **kwargs):
    with _MODEL_LOCK:
        return fn(*args, **kwargs)


async def averify_claims_batch(claims_with_passages, profile=None):
    """`verify_claims_batch` off the event loop."""
    return await asyncio.to_thread(_locked, verify_claims_batch, claims_with_passages, profile)


async def abuild_entailment_matrix(claims, passages, profile=None):
    """`build_entailment_matrix` off the event loop."""
    return await asyncio.to_thread(_locked, build_entailment_matrix, claims, passages, profile)


async def anli_predict(pairs, **kwargs):
    """Raw NLI scoring off the event loop, for callers that build their own
    pairs (verification/stability.py)."""
    return await asyncio.to_thread(_locked, _nli.predict, pairs, **kwargs)
