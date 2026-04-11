"""xai_matrices.py - Pure Mathematical XAI Matrix Computations. LLM extracts, Math judges."""

import re
import logging
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)

def _load_models():
    logger.info("Loading SentenceTransformer (BAAI/bge-large-en-v1.5)...")
    encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")
    logger.info("Loading NLI CrossEncoder (cross-encoder/nli-deberta-v3-base)...")
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-base", num_labels=3)
    logger.info("XAI matrix models ready.")
    return encoder, nli

_encoder, _nli = _load_models()
_spacy_nlp = spacy.load("en_core_web_sm")

def get_encoder(): return _encoder
def get_nli():     return _nli

MIN_SENTENCE_LEN  = 20
NLI_TOP_K         = 3
NLI_PREPROCESS_AT = 60

_RE_NUMERIC_ROW    = re.compile(r"^[\d\s\(\),\.\-]+$")
_RE_PAGE_MARKER    = re.compile(r"^\d+\s*\|\s*P\s*a\s*g\s*e$", re.IGNORECASE)
_RE_TABLE_HEADER   = re.compile(r"^(Month\s*End|FCA|Gold|SDR|RTP|Forex\s*Reserves|USD\s*Million|Rs\.?\s*Crore|Table\s*\d+|Chart\s*\d+).*$", re.IGNORECASE)
_RE_ALL_CAPS_SHORT = re.compile(r"^[A-Z\s\-\.]{1,30}$")
_RE_DATE_NUM_ROW  = re.compile(r"^[A-Za-z]+-\d{2,4}\s+[\d\s\(\),\.\-]+$")

def strip_table_noise(text):
    """Remove numeric table rows, page markers, and column headers from PDF-extracted text."""
    lines = text.split("\n")
    clean = []
    for line in lines:
        s = line.strip()
        if not s:                               continue
        if _RE_NUMERIC_ROW.match(s):            continue
        if _RE_PAGE_MARKER.match(s):            continue
        if _RE_TABLE_HEADER.match(s):           continue
        if _RE_ALL_CAPS_SHORT.match(s) and len(s) < 25: continue
        if _RE_DATE_NUM_ROW.match(s):               continue   # e.g. "September-24  617075..."
        clean.append(s)
    return " ".join(clean)

def extract_relevant_sentences(claim, passage):
    """Extract top-K prose sentences most similar to the claim. Strips table noise first."""
    clean_passage = strip_table_noise(passage)
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
    return [c["section_id"] for c in chunks], [float(x) for x in S]

def build_entailment_matrix(claims, passages):
    if not claims or not passages:
        return np.array([]), ["contradiction", "entailment", "neutral"]
    pairs = [
        (extract_relevant_sentences(claim, passage) if len(passage.split()) > NLI_PREPROCESS_AT else passage, claim)
        for claim in claims for passage in passages
    ]
    raw_scores = _nli.predict(pairs, apply_softmax=True)
    return raw_scores.reshape(len(claims), len(passages), 3), ["contradiction", "entailment", "neutral"]

def verify_claims_batch(claims_with_passages):
    if not claims_with_passages:
        return []
    focused_pairs    = []
    focused_passages = []
    for claim, raw_passage in claims_with_passages:
        # Always strip table noise first — even short passages may contain table rows
        stripped = strip_table_noise(raw_passage) if raw_passage else ""
        if stripped and len(stripped.split()) > NLI_PREPROCESS_AT:
            focused = extract_relevant_sentences(claim, stripped)
        else:
            focused = stripped  # already clean; no sentence selection needed
        focused_pairs.append((focused, claim))
        focused_passages.append(focused)
    scores = _nli.predict(focused_pairs, apply_softmax=True)
    labels = ["contradiction", "entailment", "neutral"]
    results = []
    for (claim, _), score_triplet, focused in zip(claims_with_passages, scores, focused_passages):
        top_idx = int(score_triplet.argmax())
        results.append({
            "claim_text":          claim,
            "verdict":             labels[top_idx].upper(),
            "entailment_score":    float(score_triplet[1]),
            "contradiction_score": float(score_triplet[0]),
            "neutral_score":       float(score_triplet[2]),
            "focused_passage":     focused,
        })
    return results

MIN_ATTRIBUTION_SCORE = 0.45

def build_attribution_matrix(answer_sentences, chunks):
    if not answer_sentences or not chunks:
        return np.array([])
    sent_emb  = _encoder.encode(answer_sentences)
    chunk_emb = _encoder.encode([c["chunk_text"] for c in chunks])
    sent_norm  = sent_emb  / np.linalg.norm(sent_emb,  axis=1, keepdims=True)
    chunk_norm = chunk_emb / np.linalg.norm(chunk_emb, axis=1, keepdims=True)
    return sent_norm @ chunk_norm.T

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

AMBIGUITY_GAP_THRESHOLD = 0.02   # recalibrated from 0.05
WEAK_ATTRIBUTION_SCORE  = 0.65

def compute_shapley_contributions(verifications, primary_attributions=None):
    """Compute Shapley penalty vector across NLI verdicts AND attribution quality.

    primary_attributions[i] corresponds to verifications[i].
    Including attribution penalties ensures shapley.overall_score matches
    trust_gate.overall_score (both deduct the same set of penalties).
    """
    if primary_attributions is None:
        primary_attributions = []

    nli_penalties  = {"contradiction": 0.30, "neutral": 0.10, "low_confidence": 0.20, "mid_confidence": 0.05}
    attr_penalties = {"ambiguous": 0.05, "weak": 0.03}

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
            if entail_score < 0.5:
                phi += nli_penalties["low_confidence"]; reasons.append("low NLI confidence ({:.2f})".format(entail_score))
            elif entail_score <= 0.8:
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
