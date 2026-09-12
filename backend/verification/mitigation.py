"""
mitigation.py — Claim-level filtering and abstention.

The Trust Gate (trust_gate.py) computes a verdict but historically nothing
acted on it: the same LLM answer was always returned to the user. This
module is where the pipeline actually intervenes — stripping ungrounded
sentences and, when necessary, abstaining entirely — while preserving the
raw answer and per-claim reasons for full audit traceability.

All thresholds are pure-math (NLI probabilities / attribution cosine scores),
no LLM-as-judge.
"""

from typing import List, Optional, Tuple

from shared import config
from shared.nli_policy import is_contradiction
from shared.models import Claim, VerificationResult, TrustGate, NLIVerdict, TrustStatus
from verification.trust_gate import compute_trust_gate
from verification.conformal import load_active_threshold

# Re-exported from config (not re-declared as literals) — mirrors
# trust_gate.py's own thresholds so a claim is only STRIPPED when it fails a
# genuine NLI-grounded signal (contradiction or low entailment). Weak/ambiguous
# *attribution* alone is not grounds to strip: a sentence can legitimately
# have no single best-matching source chunk (discourse glue like "In summary,
# the following requirements apply:") without being ungrounded.
STRIP_ENTAILMENT_FLOOR = config.STRIP_ENTAILMENT_FLOOR
WEAK_ATTRIBUTION_SCORE = config.WEAK_ATTRIBUTION_SCORE

ABSTENTION_MEAN_PENALTY_CEIL = config.ABSTENTION_MEAN_PENALTY_CEIL


def _should_strip(verification: VerificationResult) -> bool:
    if is_contradiction(verification.verdict):
        return True
    # A claim with no premise is unverifiable, not weakly verified. It is still
    # stripped — retaining an unverifiable sentence in a mitigated answer would
    # be a governance regression, and this module cannot tell legitimate
    # discourse glue ("In summary, ...") from an ungrounded assertion without
    # attribution context it isn't given. What changes is that the decision is
    # now explicit rather than falling out of a 0.0 < STRIP_ENTAILMENT_FLOOR
    # comparison against a score that was never measured.
    if verification.evidence_status != "ok":
        return True
    if verification.entailment_score < STRIP_ENTAILMENT_FLOOR:
        return True
    return False


def _flag_reason(verification: VerificationResult, attribution: dict) -> str:
    """Reason string for a claim that is retained but flagged as weak."""
    if verification.verdict in (NLIVerdict.NEUTRAL, NLIVerdict.NOT_ENOUGH_INFO):
        return "flagged:neutral"
    if config.LOW_CONFIDENCE_CEIL <= verification.entailment_score <= config.MID_CONFIDENCE_CEIL:
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
            # The reason must name what actually happened. A claim with no
            # premise is stripped for having no evidence, NOT because the NLI
            # model scored it poorly — the model was never run on it, and its
            # entailment_score is a structural 0.0. Reporting that as
            # "low_confidence" attributed a model judgement to a model that
            # never saw the claim.
            if is_contradiction(verification.verdict):
                reason = "contradiction"
            elif verification.evidence_status != "ok":
                reason = f"no_evidence:{verification.evidence_status}"
            else:
                reason = "low_confidence"
            retained_flags.append(False)
            filter_reasons.append(reason)
            continue

        reason = _flag_reason(verification, attr)
        retained_flags.append(True)
        filter_reasons.append(reason)
        retained_texts.append(claim.text)

    return retained_texts, retained_flags, filter_reasons


def nonconformity_score(
    claims: List[Claim],
    verifications: List[VerificationResult],
    primary_attributions: List[dict],
    retained_flags: List[bool],
    conflicts: list,
) -> Tuple[Optional[float], Optional[TrustGate]]:
    """The mean-penalty nonconformity score `should_abstain` compares against
    the abstention threshold — factored out so the conformal calibration
    pipeline (eval/harness.py) computes the IDENTICAL quantity should_abstain
    actually uses, instead of reconstructing an approximation from
    claims_total/claims_stripped that silently diverges on two degenerate
    paths: empty retrieval (no claims at all) and every claim stripped. Both
    previously got a fabricated 0.0/1.0 score injected into calibration; both
    now correctly return None so the caller EXCLUDES them instead.

    Returns (score, retained_gate) — score is None when there is no
    nonconformity score for this item (no retained claims to compute one
    over), in which case retained_gate is also None.
    """
    retained_verifications = [v for v, keep in zip(verifications, retained_flags) if keep]
    retained_attrs = [a for a, keep in zip(primary_attributions, retained_flags) if keep] if primary_attributions else []

    if not retained_verifications:
        return None, None

    retained_gate: TrustGate = compute_trust_gate(retained_verifications, conflicts, retained_attrs)
    mean_penalty = (1.0 - retained_gate.overall_score) / max(1, len(retained_verifications))
    return mean_penalty, retained_gate


def should_abstain(
    claims: List[Claim],
    verifications: List[VerificationResult],
    primary_attributions: List[dict],
    retained_flags: List[bool],
    conflicts: list,
    threshold: Optional[float] = None,
) -> tuple[bool, str, float]:
    """Decide whether the mitigated answer is still too risky to serve.

    Computed on the RETAINED set (post-filtering), not the raw trust gate —
    otherwise stripping bad claims would have no effect on this decision.
    Uses mean per-claim penalty (not cumulative sum) so answer length doesn't
    bias the outcome.

    `threshold`: an explicit ceiling override. When omitted (None), the
    active conformal threshold is read from disk on every call — fine for a
    single live query, but a long eval run should read it ONCE and pass it
    in here, so a concurrent recalibration can't change abstention behavior
    mid-run.

    Returns (abstained, reason, retained_trust_score).
    """
    if not claims:
        # "No documents retrieved" path — an honest non-answer already, not
        # a mitigation-triggered abstention.
        return False, "", 1.0

    mean_penalty, retained_gate = nonconformity_score(
        claims, verifications, primary_attributions, retained_flags, conflicts
    )
    if mean_penalty is None:
        return True, "Every extracted claim was contradicted by or unsupported by the source documents.", 0.0

    if retained_gate.status == TrustStatus.NON_COMPLIANT:
        return True, f"Remaining claims still fail trust gating: {retained_gate.reasoning}", retained_gate.overall_score

    # Prefer the conformally-calibrated threshold when one exists; otherwise fall
    # back to the fixed ceiling (identical behaviour to before calibration).
    ceil = threshold if threshold is not None else load_active_threshold()
    if ceil is None:
        ceil = ABSTENTION_MEAN_PENALTY_CEIL
    if mean_penalty > ceil:
        ceil_str = "∞" if ceil == float("inf") else f"{ceil:.2f}"
        return True, (
            f"Aggregate risk across remaining claims is too high "
            f"(mean penalty {mean_penalty:.2f} > {ceil_str})."
        ), retained_gate.overall_score

    return False, "", retained_gate.overall_score


ABSTENTION_MESSAGE_TEMPLATE = (
    "I don't have sufficiently grounded information in the ingested sources to answer this "
    "confidently. {reason} Please consult the source documents directly or rephrase your query."
)
