"""
Tests for counterfactual explanations (Feature 4).

Pure math over synthetic VerificationResults (conftest mocks the ML deps that
xai_matrices imports). Verifies phi computation, exact leave-one-out status,
ordering, primary-driver flagging, and empty output on a clean answer.
"""

from shared.models import VerificationResult, NLIVerdict
from verification.trust_gate import compute_trust_gate
from verification.counterfactual import compute_counterfactuals


def _v(text, verdict, score):
    return VerificationResult(claim_text=text, verdict=verdict, entailment_score=score)


def test_all_supported_yields_no_counterfactuals():
    verifs = [
        _v("a", NLIVerdict.ENTAILMENT, 0.97),
        _v("b", NLIVerdict.ENTAILMENT, 0.91),
    ]
    gate = compute_trust_gate(verifs, [], [])
    cfs = compute_counterfactuals(verifs, [], [], gate)
    assert cfs == []  # nothing penalising → nothing to contrast


def test_single_contradiction_flips_to_safe_when_removed():
    verifs = [
        _v("good claim", NLIVerdict.ENTAILMENT, 0.95),
        _v("bad claim", NLIVerdict.CONTRADICTION, 0.9),
    ]
    gate = compute_trust_gate(verifs, [], [])
    assert gate.status.value == "Non_Compliant"

    cfs = compute_counterfactuals(verifs, [], [], gate)
    assert len(cfs) == 1
    cf = cfs[0]
    assert cf["claim_text"] == "bad claim"
    assert cf["phi"] == 0.3  # contradiction penalty
    assert "contradiction" in cf["penalty_reasons"]
    assert cf["flips_status"] is True
    assert cf["status_if_removed"] == "Safe"
    assert cf["score_if_removed"] > gate.overall_score
    assert cf["primary_driver"] is True


def test_ordering_by_phi_desc_and_single_primary_driver():
    verifs = [
        _v("contradiction", NLIVerdict.CONTRADICTION, 0.9),  # phi 0.3
        _v("neutral", NLIVerdict.NEUTRAL, 0.9),              # phi 0.1
        _v("clean", NLIVerdict.ENTAILMENT, 0.95),            # phi 0 -> excluded
    ]
    gate = compute_trust_gate(verifs, [], [])
    cfs = compute_counterfactuals(verifs, [], [], gate)

    assert [c["claim_text"] for c in cfs] == ["contradiction", "neutral"]
    assert cfs[0]["phi"] >= cfs[1]["phi"]
    assert sum(1 for c in cfs if c["primary_driver"]) == 1
    assert cfs[0]["primary_driver"] is True


def test_low_confidence_penalty_included():
    verifs = [
        _v("solid", NLIVerdict.ENTAILMENT, 0.95),
        _v("shaky", NLIVerdict.ENTAILMENT, 0.3),  # entailment but low score -> low_confidence 0.2
    ]
    gate = compute_trust_gate(verifs, [], [])
    cfs = compute_counterfactuals(verifs, [], [], gate)
    shaky = [c for c in cfs if c["claim_text"] == "shaky"]
    assert shaky and shaky[0]["phi"] == 0.2


def test_conflict_keeps_noncompliant_even_after_removal():
    from shared.models import EditionConflict
    conflict = EditionConflict(
        publication="P", section_id="1.1", older_edition="2023", newer_edition="2024",
        has_conflict=True,
    )
    verifs = [_v("bad", NLIVerdict.CONTRADICTION, 0.9)]
    gate = compute_trust_gate(verifs, [conflict], [])
    cfs = compute_counterfactuals(verifs, [], [conflict], gate)
    # Removing the claim doesn't resolve the conflict, which alone forces Non_Compliant.
    assert len(cfs) == 1
    assert cfs[0]["status_if_removed"] == "Non_Compliant"
    assert cfs[0]["flips_status"] is False


def test_empty_input_is_safe():
    gate = compute_trust_gate([], [], [])
    assert compute_counterfactuals([], [], [], gate) == []
