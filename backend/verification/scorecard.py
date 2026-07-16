from typing import List, Optional

from shared.models import VerificationResult, EditionConflict, TrustScorecard, Claim, NLIVerdict


def generate_scorecard(
    query: str,
    response: str,
    verifications: List[VerificationResult],
    conflicts: List[EditionConflict],
    claims: List[Claim],
    retrieval_scores: Optional[List[float]] = None,
    paraphrase_stability: float = 1.0,
    answer_relevancy: float = 0.0,
    context_utilization: float = 0.0,
    context_diversity: float = 0.0,
) -> TrustScorecard:
    """Generates the RAGAS-style Trust Scorecard from precomputed pure-math inputs.

    A pure aggregator — every metric here is either computed inline from
    `verifications`/`claims`/`retrieval_scores`, or passed in already-computed
    from the caller (paraphrase_stability, answer_relevancy, context_utilization,
    context_diversity all require the encoder/NLI models or a second Gemini
    sample, which the caller — ingestion/pipeline.py — already has in scope).

    `retrieval_scores`/`context_*` args are optional because the standalone
    `/verify/` sandbox endpoint has no retrieval step at all — those metrics
    are reported as 0.0 there (the frontend shows "N/A (sandbox mode)").
    """
    retrieval_scores = retrieval_scores or []

    # 1. Faithfulness: % of claims with ENTAILMENT verdict
    total_claims = len(verifications)
    if total_claims == 0:
        faithfulness = 1.0  # Default if no claims to verify
    else:
        entailed_claims = sum(1 for v in verifications if v.verdict in (NLIVerdict.ENTAILMENT, NLIVerdict.SUPPORTED))
        faithfulness = entailed_claims / total_claims

    # 2. Citation Precision: fraction of claims with a real source_section_id
    if not claims:
        citation_precision = 1.0
    else:
        cited = sum(1 for c in claims if c.source_section_id.strip() != "")
        citation_precision = cited / len(claims)

    # 3. Edition-Conflict Risk: Binary — any unresolved conflicts?
    edition_conflict_risk = any(c.has_conflict for c in conflicts)

    # 4. Context Relevance: mean cosine(query, retrieved chunk) — real S_scores,
    #    not a hardcoded constant. Tends to be high/low-variance since retrieval
    #    already selects top-k; context_diversity is a more discriminative companion.
    context_relevance = (sum(retrieval_scores) / len(retrieval_scores)) if retrieval_scores else 0.0

    return TrustScorecard(
        context_relevance=round(context_relevance, 6),
        faithfulness=round(faithfulness, 6),
        citation_precision=round(citation_precision, 6),
        edition_conflict_risk=edition_conflict_risk,
        paraphrase_stability=paraphrase_stability,
        answer_relevancy=answer_relevancy,
        context_utilization=context_utilization,
        context_diversity=context_diversity,
    )
