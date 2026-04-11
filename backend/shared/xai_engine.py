"""
xai_engine.py — Pure Mathematical XAI Computation Layer

Implements all matrix computations described in the XAI spec:
  - Retrieval Similarity Matrix:  S = q·Kᵀ / (‖q‖·‖k‖)
  - Faithfulness Score:           |ENTAILMENT| / |total claims|
  - Trust Score Breakdown:        Shapley-style penalty decomposition
  - Related Query Lookup:         cosine(q_current, q_past) via stored embeddings
  - Edition Conflict Detection:   pure cosine comparison, NO LLM judge
"""

import json
import math
import logging
from datetime import datetime
from typing import List, Tuple, Optional

from shared.models import (
    Claim, VerificationResult, TrustGate, EditionConflict,
    RetrievalScore, TrustScoreBreakdown, XAIArtifacts, RelatedQuery,
    NLIVerdict, TrustStatus
)
from shared.database import get_sqlite_conn

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# 1. Cosine Similarity (pure math, no libs)
# ─────────────────────────────────────────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
    Returns value ∈ [-1, 1]. Returns 0.0 if either vector is zero.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def l2_norm(v: List[float]) -> float:
    """‖v‖ = √(Σ vᵢ²)"""
    return math.sqrt(sum(x * x for x in v))


# ─────────────────────────────────────────
# 2. Build Retrieval Similarity Matrix
# ─────────────────────────────────────────

def build_retrieval_similarity_matrix(
    chunks_with_scores: List[dict]
) -> List[RetrievalScore]:
    """
    Materialises the Similarity Matrix S for every retrieved chunk.
    Input: list of dicts with keys:
        chunk_id, section_id, publication, edition,
        dense_score (cosine), bm25_score, rrf_score, rank
    """
    matrix = []
    for i, item in enumerate(chunks_with_scores):
        matrix.append(RetrievalScore(
            chunk_id=item.get("chunk_id", f"chunk_{i}"),
            section_id=item.get("section_id", ""),
            publication=item.get("publication", ""),
            edition=item.get("edition", ""),
            dense_score=round(item.get("dense_score", 0.0), 6),
            bm25_score=round(item.get("bm25_score", 0.0), 6),
            rrf_score=round(item.get("rrf_score", 0.0), 6),
            rank=item.get("rank", i + 1),
        ))
    return matrix


# ─────────────────────────────────────────
# 3. Faithfulness & Citation Precision
# ─────────────────────────────────────────

def compute_faithfulness(verifications: List[VerificationResult]) -> float:
    """
    Faithfulness = |{cᵢ : verdict = ENTAILMENT or SUPPORTED}| / n
    Returns 1.0 if no claims to verify.
    """
    n = len(verifications)
    if n == 0:
        return 1.0
    entailed = sum(
        1 for v in verifications
        if v.verdict in (NLIVerdict.ENTAILMENT, NLIVerdict.SUPPORTED)
    )
    return round(entailed / n, 6)


def compute_citation_precision(claims: List[Claim]) -> float:
    """
    Citation Precision = |{cᵢ : source_section_id ≠ ''}| / n
    Measures what fraction of claims have verifiable source attribution.
    """
    n = len(claims)
    if n == 0:
        return 1.0
    cited = sum(1 for c in claims if c.source_section_id.strip() != "")
    return round(cited / n, 6)


# ─────────────────────────────────────────
# 4. Shapley Trust Score Breakdown
# ─────────────────────────────────────────

def compute_trust_score_breakdown(
    verifications: List[VerificationResult],
    edition_conflicts: List[EditionConflict]
) -> TrustScoreBreakdown:
    """
    Shapley-style additive decomposition:
      Overall_Score = 1.0
        - 0.5  × [unresolved conflicts > 0]
        - Σᵢ 0.3  × [verdict = CONTRADICTION]
        - Σᵢ 0.1  × [verdict = NEUTRAL]
        - Σᵢ 0.2  × [entailment_score < 0.5]
        - Σᵢ 0.05 × [0.5 ≤ entailment_score ≤ 0.8]
    φᵢ (per-claim Shapley contribution) = the marginal penalty from claim i.
    """
    initial = 1.0
    contradiction_pen = 0.0
    neutral_pen = 0.0
    low_conf_pen = 0.0
    med_conf_pen = 0.0
    conflict_pen = 0.0
    per_claim = []

    unresolved = [c for c in edition_conflicts if c.has_conflict]
    if unresolved:
        conflict_pen = 0.5
    
    for v in verifications:
        phi = 0.0
        reason = []

        if v.verdict in (NLIVerdict.CONTRADICTION, NLIVerdict.CONTRADICTED):
            contradiction_pen += 0.3
            phi += 0.3
            reason.append("CONTRADICTION penalty: -0.3")
        elif v.verdict in (NLIVerdict.NEUTRAL, NLIVerdict.NOT_ENOUGH_INFO):
            neutral_pen += 0.1
            phi += 0.1
            reason.append("NEUTRAL penalty: -0.1")

        if v.entailment_score < 0.5:
            low_conf_pen += 0.2
            phi += 0.2
            reason.append(f"Low confidence ({v.entailment_score:.3f} < 0.5): -0.2")
        elif 0.5 <= v.entailment_score <= 0.8:
            med_conf_pen += 0.05
            phi += 0.05
            reason.append(f"Medium confidence ({v.entailment_score:.3f}): -0.05")

        per_claim.append({
            "claim_text": v.claim_text[:80] + "..." if len(v.claim_text) > 80 else v.claim_text,
            "verdict": v.verdict,
            "entailment_score": v.entailment_score,
            "shapley_phi": round(phi, 4),
            "penalty_reasons": reason,
        })

    total_penalty = contradiction_pen + neutral_pen + low_conf_pen + med_conf_pen + conflict_pen
    final_score = max(0.0, min(1.0, initial - total_penalty))

    return TrustScoreBreakdown(
        initial_score=initial,
        contradiction_penalty=round(contradiction_pen, 4),
        neutral_penalty=round(neutral_pen, 4),
        low_confidence_penalty=round(low_conf_pen, 4),
        medium_confidence_penalty=round(med_conf_pen, 4),
        edition_conflict_penalty=round(conflict_pen, 4),
        final_score=round(final_score, 6),
        per_claim_contributions=per_claim,
    )


# ─────────────────────────────────────────
# 5. Edition Conflict (Pure Cosine — NO LLM)
# ─────────────────────────────────────────

CONFLICT_COSINE_THRESHOLD = 0.75  # sections with sim < 0.75 are considered conflicting

def detect_edition_conflict_pure_math(
    publication: str,
    section_id: str,
    older_edition: str,
    older_embedding: List[float],
    newer_edition: str,
    newer_embedding: List[float],
) -> dict:
    """
    Deterministic edition conflict detection using ONLY cosine similarity.
    No LLM judge involved.

    If cos(older_section, newer_section) < THRESHOLD, the sections diverge
    → conflict detected.
    The NEWER edition is always considered authoritative (recency heuristic).

    Authority_Score = cosine_sim × recency_factor
    where recency_factor = 1.0 for newer, 0.5 for older.
    """
    sim = cosine_similarity(older_embedding, newer_embedding)
    has_conflict = sim < CONFLICT_COSINE_THRESHOLD

    result = {
        "publication": publication,
        "section_id": section_id,
        "older_edition": older_edition,
        "newer_edition": newer_edition,
        "cosine_similarity": round(sim, 6),
        "conflict_threshold": CONFLICT_COSINE_THRESHOLD,
        "has_conflict": has_conflict,
        "authoritative_edition": newer_edition,  # recency rule
        "authority_score_newer": round(sim * 1.0, 6),
        "authority_score_older": round(sim * 0.5, 6),
        "resolution_method": "cosine_recency_heuristic",
        "reasoning": (
            f"cos({older_edition}, {newer_edition}) = {sim:.4f} "
            + (f"< {CONFLICT_COSINE_THRESHOLD} → CONFLICT. {newer_edition} is authoritative by recency."
               if has_conflict
               else f"≥ {CONFLICT_COSINE_THRESHOLD} → CONSISTENT. No divergence detected.")
        ),
    }

    # Cache in SQLite for audit trail
    try:
        conn = get_sqlite_conn()
        conn.execute(
            "INSERT OR REPLACE INTO edition_conflict_cache "
            "(publication, section_id, edition_a, edition_b, cosine_sim, has_conflict, authoritative, cached_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (publication, section_id, older_edition, newer_edition,
             sim, int(has_conflict), newer_edition, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not cache conflict result: {e}")

    return result


# ─────────────────────────────────────────
# 6. Related Query Lookup (Cosine over stored embeddings)
# ─────────────────────────────────────────

RELATED_QUERY_THRESHOLD = 0.70  # queries with sim > 0.70 are "related"
TOP_K_RELATED = 3

def find_related_queries(
    current_embedding: List[float],
    current_id: Optional[int] = None
) -> List[RelatedQuery]:
    """
    For the current query embedding q, retrieve all past query embeddings
    from SQLite and compute cos(q, qᵢ) for each.

    Returns the top-K past queries with cosine_sim > RELATED_QUERY_THRESHOLD.
    This allows the audit trail to show: "This query is semantically related to
    previous queries #3 and #7, which had trust_status=Safe."
    """
    try:
        conn = get_sqlite_conn()
        rows = conn.execute(
            "SELECT id, timestamp, query, query_embedding, trust_gate_status FROM audit_logs "
            "WHERE query_embedding IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception as e:
        logger.warning(f"Related query lookup failed: {e}")
        return []

    scored: List[Tuple[float, dict]] = []
    for row in rows:
        if current_id and row["id"] == current_id:
            continue  # skip itself
        try:
            past_emb = json.loads(row["query_embedding"])
            sim = cosine_similarity(current_embedding, past_emb)
            if sim >= RELATED_QUERY_THRESHOLD:
                scored.append((sim, dict(row)))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        RelatedQuery(
            id=item["id"],
            timestamp=item["timestamp"],
            query=item["query"],
            cosine_similarity=round(sim, 6),
            trust_status=item.get("trust_gate_status", ""),
        )
        for sim, item in scored[:TOP_K_RELATED]
    ]


def store_query_embedding(log_id: int, embedding: List[float]) -> None:
    """Persist the query embedding in the audit log row for future lookups."""
    try:
        conn = get_sqlite_conn()
        conn.execute(
            "UPDATE audit_logs SET query_embedding = ? WHERE id = ?",
            (json.dumps(embedding), log_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to store query embedding: {e}")
