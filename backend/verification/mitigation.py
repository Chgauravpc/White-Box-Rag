"""
mitigation.py — Claim-level filtering and abstention.

The Trust Gate (trust_gate.py) computes a verdict but historically nothing
acted on it: the same Gemini answer was always returned to the user. This
module is where the pipeline actually intervenes — stripping ungrounded
sentences and, when necessary, abstaining entirely — while preserving the
raw answer and per-claim reasons for full audit traceability.

All thresholds are pure-math (NLI probabilities / attribution cosine scores),
no LLM-as-judge.
"""

from typing import List

from shared.models import Claim, VerificationResult, TrustGate, NLIVerdict, TrustStatus
from verification.trust_gate import compute_trust_gate

# Mirrors trust_gate.py's own thresholds — a claim is only STRIPPED when it
# fails a genuine NLI-grounded signal (contradiction or low entailment).
# Weak/ambiguous *attribution* alone is not grounds to strip: a sentence can
# legitimately have no single best-matching source chunk (discourse glue like
# "In summary, the following requirements apply:") without being ungrounded.
STRIP_ENTAILMENT_FLOOR = 0.5
WEAK_ATTRIBUTION_SCORE = 0.65

ABSTENTION_MEAN_PENALTY_CEIL = 0.25


def _should_strip(verification: VerificationResult) -> bool:
    if verification.verdict in (NLIVerdict.CONTRADICTION, NLIVerdict.CONTRADICTED):
        return True
    if verification.entailment_score < STRIP_ENTAILMENT_FLOOR:
        return True
    return False


def _flag_reason(verification: VerificationResult, attribution: dict) -> str:
    """Reason string for a claim that is retained but flagged as weak."""
    if verification.verdict in (NLIVerdict.NEUTRAL, NLIVerdict.NOT_ENOUGH_INFO):
        return "flagged:neutral"
    if 0.5 <= verification.entailment_score <= 0.8:
        return "flagged:medium_confidence"
    if not attribution:
        return "flagged:unattributed"
    if attribution.get("ambiguous"):
        return "flagged:ambiguous_attribution"
    if attribution.get("attribution_score", 1.0) < WEAK_ATTRIBUTION_SCORE:
        return "flagged:weak_attribution"
    return ""


def filter_claims(
    claims: List[Claim],
    verifications: List[VerificationResult],
    primary_attributions: List[dict],
) -> tuple[list[str], list[bool], list[str]]:
    """Decide which claims are grounded enough to keep.

    Returns (retained_claim_texts, retained_flags, filter_reasons) — the
    latter two are index-aligned with `claims`/`verifications`. Claims and
    verifications are already produced in matching sentence order elsewhere
    in the pipeline, so no separate matching step is needed here.
    """
    retained_flags: list[bool] = []
    filter_reasons: list[str] = []
    retained_texts: list[str] = []

    for i, (claim, verification) in enumerate(zip(claims, verifications)):
        attr = primary_attributions[i] if i < len(primary_attributions) else {}

        if _should_strip(verification):
            reason = (
                "contradiction"
                if verification.verdict in (NLIVerdict.CONTRADICTION, NLIVerdict.CONTRADICTED)
                else "low_confidence"
            )
            retained_flags.append(False)
            filter_reasons.append(reason)
            continue

        reason = _flag_reason(verification, attr)
        retained_flags.append(True)
        filter_reasons.append(reason)
        retained_texts.append(claim.text)

    return retained_texts, retained_flags, filter_reasons


def should_abstain(
    claims: List[Claim],
    verifications: List[VerificationResult],
    primary_attributions: List[dict],
    retained_flags: List[bool],
    conflicts: list,
) -> tuple[bool, str, float]:
    """Decide whether the mitigated answer is still too risky to serve.

    Computed on the RETAINED set (post-filtering), not the raw trust gate —
    otherwise stripping bad claims would have no effect on this decision.
    Uses mean per-claim penalty (not cumulative sum) so answer length doesn't
    bias the outcome.

    Returns (abstained, reason, retained_trust_score).
    """
    if not claims:
        # "No documents retrieved" path — an honest non-answer already, not
        # a mitigation-triggered abstention.
        return False, "", 1.0

    retained_verifications = [v for v, keep in zip(verifications, retained_flags) if keep]
    retained_attrs = [a for a, keep in zip(primary_attributions, retained_flags) if keep] if primary_attributions else []

    if not retained_verifications:
        return True, "Every extracted claim was contradicted by or unsupported by the source documents.", 0.0

    retained_gate: TrustGate = compute_trust_gate(retained_verifications, conflicts, retained_attrs)
    mean_penalty = (1.0 - retained_gate.overall_score) / max(1, len(retained_verifications))

    if retained_gate.status == TrustStatus.NON_COMPLIANT:
        return True, f"Remaining claims still fail trust gating: {retained_gate.reasoning}", retained_gate.overall_score

    if mean_penalty > ABSTENTION_MEAN_PENALTY_CEIL:
        return True, (
            f"Aggregate risk across remaining claims is too high "
            f"(mean penalty {mean_penalty:.2f} > {ABSTENTION_MEAN_PENALTY_CEIL})."
        ), retained_gate.overall_score

    return False, "", retained_gate.overall_score


ABSTENTION_MESSAGE_TEMPLATE = (
    "I don't have sufficiently grounded information in the ingested sources to answer this "
    "confidently. {reason} Please consult the source documents directly or rephrase your query."
)
