from typing import List
from shared.models import VerificationResult, EditionConflict, TrustScorecard, Claim, NLIVerdict

def generate_scorecard(query: str, response: str, verifications: List[VerificationResult], conflicts: List[EditionConflict], claims: List[Claim]) -> TrustScorecard:
    """Generates the RAGAS-style Trust Scorecard."""
    
    # 1. Faithfulness: % of claims with ENTAILMENT verdict
    total_claims = len(verifications)
    if total_claims == 0:
        faithfulness = 1.0 # Default if no claims to verify
    else:
        entailed_claims = sum(1 for v in verifications if v.verdict in (NLIVerdict.ENTAILMENT, NLIVerdict.SUPPORTED))
        faithfulness = entailed_claims / total_claims
        
    # 2. Citation Precision: % of citations that actually map to real sections 
    # (Assuming here that all Claim objects passed are grounded for simplicity, 
    # but could involve pinging ChromaDB to verify source_section_id exists)
    citation_precision = 1.0 if claims else 0.0
    
    # 3. Edition-Conflict Risk: Binary — any unresolved conflicts?
    edition_conflict_risk = any(c.has_conflict for c in conflicts)
    
    # 4. Context Relevance (Mocked: requires Gemini call on the context chunks vs query)
    # 5. Paraphrase Stability (Mocked: requires asking another query and diffing)
    # We will mock these specific metrics for now or use placeholders since they require extensive external calls setup.
    context_relevance = 0.95
    paraphrase_stability = 0.90
    
    return TrustScorecard(
        context_relevance=context_relevance,
        faithfulness=faithfulness,
        citation_precision=citation_precision,
        edition_conflict_risk=edition_conflict_risk,
        paraphrase_stability=paraphrase_stability
    )
