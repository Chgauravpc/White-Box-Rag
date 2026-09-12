"""
Tests for eval/threshold.py.

The property that matters most: the sweep must replicate the SHIPPED strip
rule with only the entailment floor varied. `mitigation._should_strip` also
fires on a contradiction verdict and on missing evidence, and those branches
produce false positives too — sweeping `entailment_score < t` alone would
describe a detector this repo does not ship, and would flatter it.
"""

import pytest

from eval.threshold import _flagged_at, select_threshold, sweep_thresholds
from shared import config
from shared.models import NLIVerdict, VerificationResult
from verification.mitigation import _should_strip


def _r(label, verdict, score, evidence_status="ok"):
    return {"label": label, "verdict": verdict, "entailment_score": score,
            "evidence_status": evidence_status}


class TestMatchesShippedRule:
    @pytest.mark.parametrize("verdict,score,status", [
        ("ENTAILMENT", 0.95, "ok"),
        ("ENTAILMENT", 0.10, "ok"),
        ("NEUTRAL", 0.40, "ok"),
        ("CONTRADICTION", 0.99, "ok"),
        ("NOT_ENOUGH_INFO", 0.00, "no_premise"),
        ("NOT_ENOUGH_INFO", 0.00, "normalizer_deleted_all"),
    ])
    def test_sweep_at_the_shipped_floor_equals_should_strip(self, verdict, score, status):
        """At the configured floor the sweep and the production rule must agree
        on every case, or the curve describes a different detector."""
        result = _r("SUPPORTED", verdict, score, status)
        verification = VerificationResult(
            claim_text="c", verdict=NLIVerdict(verdict),
            entailment_score=score, evidence_status=status,
        )
        assert _flagged_at(result, config.STRIP_ENTAILMENT_FLOOR) == _should_strip(verification)

    def test_contradiction_is_flagged_at_every_threshold(self):
        """Lowering the floor must not make a contradiction survive — the
        contradiction branch is independent of the score."""
        res = [_r("REFUTED", "CONTRADICTION", 0.99)]
        assert all(p["tp"] == 1 for p in sweep_thresholds(res, grid=[0.0, 0.5, 1.0]))

    def test_missing_evidence_is_flagged_at_every_threshold(self):
        res = [_r("NEI", "NOT_ENOUGH_INFO", 0.0, "no_premise")]
        assert all(p["tp"] == 1 for p in sweep_thresholds(res, grid=[0.0, 0.5, 1.0]))


class TestSweep:
    def _results(self):
        # Grounded claims score high; hallucinated ones score low.
        return ([_r("SUPPORTED", "ENTAILMENT", 0.9)] * 10
                + [_r("REFUTED", "NEUTRAL", 0.2)] * 10)

    def test_floor_of_zero_flags_only_the_non_score_branches(self):
        point = sweep_thresholds(self._results(), grid=[0.0])[0]
        assert point["tp"] == 0 and point["fp"] == 0

    def test_floor_of_one_flags_everything(self):
        point = sweep_thresholds(self._results(), grid=[1.0])[0]
        assert point["tp"] == 10 and point["fp"] == 10

    def test_a_separating_threshold_is_perfect(self):
        point = sweep_thresholds(self._results(), grid=[0.5])[0]
        assert point["precision"] == 1.0 and point["recall"] == 1.0

    def test_curve_covers_the_whole_grid(self):
        assert len(sweep_thresholds(self._results(), grid=[0.1, 0.2, 0.3])) == 3


class TestSelection:
    def _separable(self):
        return ([_r("SUPPORTED", "ENTAILMENT", 0.9)] * 20
                + [_r("REFUTED", "NEUTRAL", 0.1)] * 5)

    def test_max_f1_finds_the_separating_point(self):
        out = select_threshold(self._separable(), criterion="max_f1")
        assert out["selected"]["f1"] == 1.0
        assert 0.1 < out["selected"]["threshold"] <= 0.9

    def test_target_recall_is_met(self):
        out = select_threshold(self._separable(), criterion="target_recall", target=1.0)
        assert out["selected"]["recall"] == 1.0

    def test_unreachable_target_is_reported_not_silently_missed(self):
        """A threshold that cannot reach the requested recall must say so
        rather than return a point that quietly misses it."""
        res = [_r("REFUTED", "ENTAILMENT", 1.0)]  # never flagged at any floor < 1.0
        out = select_threshold(res, criterion="target_recall", target=1.0,
                               grid=[0.0, 0.25, 0.5])
        assert "no threshold reached recall" in out["reason"]

    def test_current_operating_point_is_always_reported(self):
        """A recommendation without the status quo beside it is not
        actionable."""
        out = select_threshold(self._separable())
        assert out["current"]["threshold"] == config.STRIP_ENTAILMENT_FLOOR

    def test_unknown_criterion_raises(self):
        with pytest.raises(ValueError):
            select_threshold(self._separable(), criterion="vibes")

    def test_empty_results_do_not_crash(self):
        out = select_threshold([])
        assert out["selected"] is None
