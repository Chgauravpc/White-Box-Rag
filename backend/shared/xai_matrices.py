"""xai_matrices.py - Pure Mathematical XAI Matrix Computations. LLM extracts, Math judges."""

import logging
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder

from shared import config
from shared.config import PREMISE_NORMALIZER
from shared.text_normalize import normalize_premise
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

def extract_relevant_sentences(claim, passage, profile=None):
    """Extract top-K prose sentences most similar to the claim.

    Normalizes the passage per `profile` first (default: config.PREMISE_NORMALIZER).
    Idempotent on an already-normalized passage, so callers that pre-normalize
    (verify_claims_batch) and callers that don't (build_entailment_matrix) both
    get consistent behavior.
    """
    clean_passage, _ = normalize_premise(passage, profile or PREMISE_NORMALIZER)
    doc = _spacy_nlp(clean_passage)
    sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > MIN_SENTENCE_LEN]
    if not sentences:
        return clean_passage[:800] if clean_passage else passage[:800]
    if len(sentences) <= NLI_TOP_K:
        return " ".join(sentences)
    claim_emb  = _encoder.encode([claim])
    sent_embs  = _encoder.encode(sentences)
    claim_norm = claim_emb  / np.linalg.norm(claim_emb)
    sent_norm  = sent_embs  / np.linalg.norm(sent_embs, axis=1, keepdims=True)
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

def build_entailment_matrix(claims, passages):
    if not claims or not passages:
        return np.array([]), ["contradiction", "entailment", "neutral"]
    pairs = [
        (extract_relevant_sentences(claim, passage) if len(passage.split()) > NLI_PREPROCESS_AT else passage, claim)
        for claim in claims for passage in passages
    ]
    raw_scores = _nli.predict(pairs, apply_softmax=True)
    return raw_scores.reshape(len(claims), len(passages), 3), ["contradiction", "entailment", "neutral"]

def verify_claims_batch(claims_with_passages, profile=None):
    if not claims_with_passages:
        return []
    profile = profile or PREMISE_NORMALIZER
    focused_pairs     = []
    focused_passages  = []
    premise_deletions = []
    for claim, raw_passage in claims_with_passages:
        # Always normalize first — even short passages may contain PDF artifacts
        # (or, under "financial_reports", table rows). Deletions are recorded per
        # claim so premise mutation is auditable rather than invisible.
        stripped, deleted = normalize_premise(raw_passage, profile) if raw_passage else ("", [])
        if stripped and len(stripped.split()) > NLI_PREPROCESS_AT:
            focused = extract_relevant_sentences(claim, stripped, profile=profile)
        else:
            focused = stripped  # already clean; no sentence selection needed
        focused_pairs.append((focused, claim))
        focused_passages.append(focused)
        premise_deletions.append(deleted)
    scores = _nli.predict(focused_pairs, apply_softmax=True)
    labels = ["contradiction", "entailment", "neutral"]
    results = []
    for (claim, _), score_triplet, focused, deleted in zip(claims_with_passages, scores, focused_passages, premise_deletions):
        top_idx = int(score_triplet.argmax())
        results.append({
            "claim_text":          claim,
            "verdict":             labels[top_idx].upper(),
            "entailment_score":    float(score_triplet[1]),
            "contradiction_score": float(score_triplet[0]),
            "neutral_score":       float(score_triplet[2]),
            "focused_passage":     focused,
            "premise_deletions":   deleted,
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

        # NLI penalties
        if verdict == "CONTRADICTION":
            phi += nli_penalties["contradiction"]; reasons.append("contradiction")
        if verdict == "NEUTRAL":
            phi += nli_penalties["neutral"]; reasons.append("neutral")
        if verdict != "CONTRADICTION":
            if entail_score < config.LOW_CONFIDENCE_CEIL:
                phi += nli_penalties["low_confidence"]; reasons.append("low NLI confidence ({:.2f})".format(entail_score))
            elif entail_score <= config.MID_CONFIDENCE_CEIL:
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
