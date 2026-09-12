"""
nli_policy.py — the single source of truth for "what does this verdict cost?".

`verification/trust_gate.py` and `shared/xai_matrices.py::compute_shapley_contributions`
both turn (verdict, entailment_score) into penalties, and both docstrings promise
they mirror each other exactly. They have now drifted three separate times:

* a missing contradiction guard (fixed in W0.2) double-penalized a
  CONTRADICTION whose entailment_score was also below the floor;
* the Shapley side matches the bare string literals `"CONTRADICTION"` and
  `"NEUTRAL"`, while the trust gate matches the `NLIVerdict` enum *including*
  its canonical aliases (`CONTRADICTED`, `NOT_ENOUGH_INFO`). So a
  `CONTRADICTED` verdict is a contradiction to the gate and a
  not-a-contradiction to Shapley — which then also applies the confidence-band
  penalty the gate skipped, and the two overall_scores silently diverge.

Two copies of a rule drift; one copy cannot. Both modules now call
`nli_penalty_flags()` and act on the flags it returns, so a change to the
policy is a change in exactly one place.

Alias note: `NLIVerdict` (shared/models.py) carries canonical values
(SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO) *and* BP2 aliases
(ENTAILMENT / CONTRADICTION / NEUTRAL). The pipeline emits the aliases today;
the canonical spellings arrive via the `/verify` API and external benchmark
adapters. Every predicate here accepts both spellings, and either a raw string
or an `NLIVerdict`.
"""

from shared import config

# Both spellings of each verdict class, compared case-insensitively.
CONTRADICTION_VERDICTS = frozenset({"CONTRADICTION", "CONTRADICTED"})
NEUTRAL_VERDICTS = frozenset({"NEUTRAL", "NOT_ENOUGH_INFO"})
ENTAILMENT_VERDICTS = frozenset({"ENTAILMENT", "SUPPORTED"})


def normalize_verdict(verdict) -> str:
    """Accept an NLIVerdict, a raw string, or None → an upper-case string.

    `NLIVerdict` is a `str` Enum, so `.value` is the bare spelling; plain
    `str()` on an enum member would give "NLIVerdict.NEUTRAL" on some Python
    versions, which would silently match nothing.
    """
    if verdict is None:
        return ""
    value = getattr(verdict, "value", verdict)
    return str(value).upper().strip()


def is_contradiction(verdict) -> bool:
    return normalize_verdict(verdict) in CONTRADICTION_VERDICTS


def is_neutral(verdict) -> bool:
    return normalize_verdict(verdict) in NEUTRAL_VERDICTS


def is_entailment(verdict) -> bool:
    return normalize_verdict(verdict) in ENTAILMENT_VERDICTS


def nli_penalty_flags(verdict, entailment_score: float) -> dict:
    """Which NLI penalties apply to one claim.

    Returns a dict of boolean flags plus the total penalty. Callers apply the
    flags; they must not re-derive them.

    The confidence band is skipped entirely for a contradiction: a
    CONTRADICTION is already charged the (larger) contradiction penalty, and
    charging it again for the low entailment_score that *accompanies* every
    contradiction would penalize one fact twice. Neutral verdicts are not
    exempt — a neutral claim with a low entailment score is two distinct
    weaknesses (the model picked "neutral", and it was not confident), and
    both computations have always charged both.
    """
    contradiction = is_contradiction(verdict)
    neutral = is_neutral(verdict)

    low_confidence = False
    mid_confidence = False
    if not contradiction:
        if entailment_score < config.LOW_CONFIDENCE_CEIL:
            low_confidence = True
        elif entailment_score <= config.MID_CONFIDENCE_CEIL:
            mid_confidence = True

    penalty = 0.0
    if contradiction:
        penalty += config.PENALTY_CONTRADICTION
    if neutral:
        penalty += config.PENALTY_NEUTRAL
    if low_confidence:
        penalty += config.PENALTY_LOW_CONFIDENCE
    if mid_confidence:
        penalty += config.PENALTY_MID_CONFIDENCE

    return {
        "contradiction": contradiction,
        "neutral": neutral,
        "low_confidence": low_confidence,
        "mid_confidence": mid_confidence,
        "penalty": penalty,
    }
