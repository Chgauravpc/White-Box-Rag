from typing import List
from shared import config
from shared.models import VerificationResult, EditionConflict, TrustGate, TrustStatus, NLIVerdict

# Re-exported from config (not re-declared as a literal) — this used to be
# an independent copy of the same value declared separately in
# shared/xai_matrices.py and verification/mitigation.py, a real drift risk.
WEAK_ATTRIBUTION_SCORE = config.WEAK_ATTRIBUTION_SCORE


def compute_trust_gate(
    verifications: List[VerificationResult],
    conflicts: List[EditionConflict],
    primary_attributions: list = None,
) -> TrustGate:
    """
    Computes the Trust Gate using NLI verdicts, edition conflicts, and attribution quality.
    Attribution penalties mirror Shapley values exactly so overall_score is consistent.

    Rules:
    - NON_COMPLIANT: CONTRADICTION, entailment_score < 0.5, or unresolved conflicts.
    - NEEDS_HUMAN_REVIEW: NEUTRAL, mid-confidence (0.5-0.8), or ambiguous/weak attribution.
    - SAFE: All ENTAILMENT > 0.8, no conflicts, all attributions clear.
    """
    if primary_attributions is None:
        primary_attributions = []

    # No claims, no conflicts, nothing to attribute — this is "nothing to
    # verify," not "everything verified." Guard it explicitly rather than
    # falling through to SAFE with a misleading "all claims supported" message.
    if not verifications and not conflicts and not primary_attributions:
        return TrustGate(
            status=TrustStatus.SAFE,
            reasoning="No claims were extracted to verify.",
            overall_score=1.0,
        )

    overall_score        = 1.0
    reasons              = []
    has_weak_attribution = False

    # 1. Edition conflicts
    unresolved = [c for c in conflicts if c.has_conflict]
    if unresolved:
        reasons.append("Unresolved edition conflicts detected.")
        overall_score -= config.PENALTY_EDITION_CONFLICT

    # 2. NLI verdicts
    has_contradiction     = False
    has_low_confidence    = False
    has_neutral           = False
    has_medium_confidence = False

    for v in verifications:
        is_contradiction = v.verdict in (NLIVerdict.CONTRADICTION, NLIVerdict.CONTRADICTED)
        if is_contradiction:
            has_contradiction = True
            reasons.append(f"Contradiction: '{v.claim_text[:60]}...'")
            overall_score -= config.PENALTY_CONTRADICTION
        elif v.verdict in (NLIVerdict.NEUTRAL, NLIVerdict.NOT_ENOUGH_INFO):
            has_neutral = True
            reasons.append(f"Neutral verdict: '{v.claim_text[:60]}...'")
            overall_score -= config.PENALTY_NEUTRAL

        # Confidence-band penalty is skipped once a claim is already a
        # CONTRADICTION — mirrors xai_matrices.py::compute_shapley_contributions
        # exactly (same guard there). Without this guard a CONTRADICTION with
        # entailment_score < LOW_CONFIDENCE_CEIL is double-penalized here but
        # only single-penalized in the Shapley computation, so overall_score
        # silently diverges from shapley["overall_score"] — exactly the
        # consistency the docstring above promises.
        if not is_contradiction:
            if v.entailment_score < config.LOW_CONFIDENCE_CEIL:
                has_low_confidence = True
                reasons.append(f"Low NLI confidence ({v.entailment_score:.2f})")
                overall_score -= config.PENALTY_LOW_CONFIDENCE
            elif config.LOW_CONFIDENCE_CEIL <= v.entailment_score <= config.MID_CONFIDENCE_CEIL:
                has_medium_confidence = True
                reasons.append(f"Medium NLI confidence ({v.entailment_score:.2f})")
                overall_score -= config.PENALTY_MID_CONFIDENCE

    # 3. Attribution quality — same penalty scale as Shapley for score consistency
    for a in primary_attributions:
        if a.get("ambiguous"):
            overall_score -= config.PENALTY_AMBIGUOUS_ATTRIBUTION
            has_weak_attribution = True
            reasons.append(
                "Ambiguous attribution (gap={:.4f}) sentence {}.".format(
                    a.get("confidence_gap", 0), a.get("sentence_index", "?")
                )
            )
        elif a.get("attribution_score", 1.0) < WEAK_ATTRIBUTION_SCORE:
            overall_score -= config.PENALTY_WEAK_ATTRIBUTION
            has_weak_attribution = True
            reasons.append(
                "Weak attribution (score={:.4f}) sentence {}.".format(
                    a.get("attribution_score", 0), a.get("sentence_index", "?")
                )
            )

    # 4. Final gate decision
    if has_contradiction or has_low_confidence or unresolved:
        status = TrustStatus.NON_COMPLIANT
    elif has_neutral or has_medium_confidence or has_weak_attribution:
        status = TrustStatus.NEEDS_HUMAN_REVIEW
    else:
        status = TrustStatus.SAFE
        reasons.append("All claims strongly supported. No conflicts. Attribution clear.")

    return TrustGate(
        status=status,
        reasoning=" | ".join(reasons),
        overall_score=max(0.0, min(1.0, overall_score)),
    )
