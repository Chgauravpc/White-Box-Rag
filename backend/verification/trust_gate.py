from typing import List
from shared.models import VerificationResult, EditionConflict, TrustGate, TrustStatus, NLIVerdict

def compute_trust_gate(verifications: List[VerificationResult], conflicts: List[EditionConflict]) -> TrustGate:
    """
    Computes the Trust Gate decision based on NLI results and edition conflicts.
    
    Rules:
    - NON_COMPLIANT: Any CONTRADICTION, or confidence < 0.5, or unresolved conflicts.
    - NEEDS_HUMAN_REVIEW: Any NEUTRAL claims, or confidence 0.5-0.8.
    - SAFE: All claims ENTAILMENT, confidence > 0.8, no edition conflicts.
    """
    
    overall_score = 1.0 # Start max
    reasons = []
    
    # 1. Check for Conflicts
    unresolved_conflicts = [c for c in conflicts if c.has_conflict]
    if unresolved_conflicts:
         reasons.append("Unresolved edition conflicts detected.")
         status = "Non_Compliant"
         # Early return for critical failures could happen here, but we'll collect all reasons
         overall_score -= 0.5
    
    # 2. Check Verifications
    has_contradiction = False
    has_low_confidence = False
    has_neutral = False
    has_medium_confidence = False
    
    for v in verifications:
        if v.verdict in (NLIVerdict.CONTRADICTION, NLIVerdict.CONTRADICTED):
            has_contradiction = True
            reasons.append(f"Contradiction found in claim: '{v.claim_text}'")
            overall_score -= 0.3
        elif v.verdict in (NLIVerdict.NEUTRAL, NLIVerdict.NOT_ENOUGH_INFO):
            has_neutral = True
            reasons.append(f"Neutral verdict for claim: '{v.claim_text}'")
            overall_score -= 0.1
            
        if v.entailment_score < 0.5:
            has_low_confidence = True
            reasons.append(f"Low confidence ({v.entailment_score}) for claim: '{v.claim_text}'")
            overall_score -= 0.2
        elif 0.5 <= v.entailment_score <= 0.8:
            has_medium_confidence = True
            reasons.append(f"Medium confidence ({v.entailment_score}) for claim: '{v.claim_text}'")
            overall_score -= 0.05
            
    # Apply threshold logic
    if has_contradiction or has_low_confidence or unresolved_conflicts:
        status = TrustStatus.NON_COMPLIANT
    elif has_neutral or has_medium_confidence:
        status = TrustStatus.NEEDS_HUMAN_REVIEW
    else:
        status = TrustStatus.SAFE
        reasons.append("All claims strongly supported by sources without conflicts.")
        
    return TrustGate(
        status=status,
        reasoning=" | ".join(reasons),
        overall_score=max(0.0, min(1.0, overall_score))
    )
