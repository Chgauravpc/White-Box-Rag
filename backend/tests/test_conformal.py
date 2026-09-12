"""
Tests for conformal-prediction abstention calibration (Feature 3).

Pure math + JSON/SQLite persistence — no pipeline, no LLM. The live
calibration run (run_calibration) needs a real key + corpus and is not
exercised here.
"""

import math

from verification.conformal import (
    calibrate_threshold,
    save_calibration,
    load_active_calibration,
    load_active_threshold,
    min_n_for_alpha,
    recommended_n_for_alpha,
    record_calibration_history,
    CalibrationStatus,
)


def test_exact_quantile():
    # n=9, alpha=0.1 -> k = ceil(10 * 0.9) = 9 -> 9th smallest = 0.9
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result = calibrate_threshold(scores, alpha=0.1)
    assert result["threshold"] == 0.9
    assert result["n"] == 9
    assert result["alpha"] == 0.1
    assert result["status"] == CalibrationStatus.CALIBRATED


def test_quantile_is_order_independent():
    scores = [0.9, 0.1, 0.5, 0.3, 0.7]  # n=5, alpha=0.2 -> k = ceil(6*0.8)=5 -> max
    result = calibrate_threshold(scores, alpha=0.2)
    assert result["threshold"] == 0.9


def test_small_n_is_insufficient_and_the_fixed_ceiling_fallback_still_applies(tmp_path):
    """This is the fail-open regression test. Before the fix, a too-small
    calibration set produced threshold=+inf, and because '> +inf' is never
    True, that SILENTLY DISABLED abstention entirely — the 'calibrated'
    system was strictly more permissive than before calibration. The correct
    behavior: status is INSUFFICIENT_N, and load_active_threshold must return
    None (triggering the fixed-ceiling fallback in mitigation.should_abstain),
    regardless of the stored +inf threshold value.
    """
    # n=3, alpha=0.1 -> k = ceil(4 * 0.9) = 4 > 3 -> mathematically +inf
    result = calibrate_threshold([0.2, 0.4, 0.6], alpha=0.1)
    assert result["status"] == CalibrationStatus.INSUFFICIENT_N
    assert result["threshold"] == math.inf  # the mathematical fact, kept for transparency
    assert "recommended" in result["coverage_note"] or "Add more" in result["coverage_note"]

    # The behavioral fix: saving this must NOT make should_abstain more permissive.
    path = str(tmp_path / "calib.json")
    save_calibration(result, path)
    assert load_active_threshold(path) is None


def test_empty_scores_returns_none():
    result = calibrate_threshold([], alpha=0.1)
    assert result["threshold"] is None
    assert result["n"] == 0
    assert result["status"] == CalibrationStatus.NO_DATA


def test_lower_alpha_gives_higher_threshold():
    scores = [i / 100 for i in range(1, 101)]  # 0.01 .. 1.00
    lax = calibrate_threshold(scores, alpha=0.3)["threshold"]
    strict = calibrate_threshold(scores, alpha=0.05)["threshold"]
    # Tolerating fewer wrongful abstentions (smaller alpha) => more permissive (higher) threshold
    assert strict >= lax


class TestMinNForAlpha:
    def test_matches_known_values(self):
        # Hand-verified: ceil(1/alpha) - 1
        assert min_n_for_alpha(0.1) == 9
        assert min_n_for_alpha(0.2) == 4
        assert min_n_for_alpha(0.05) == 19
        assert min_n_for_alpha(0.01) == 99

    def test_calibrate_threshold_reports_the_same_min_n(self):
        result = calibrate_threshold([0.1, 0.2, 0.3], alpha=0.1)
        assert result["min_n"] == min_n_for_alpha(0.1)

    def test_recommended_n_is_a_multiple_of_min_n(self):
        assert recommended_n_for_alpha(0.1) == 3 * min_n_for_alpha(0.1)


class TestLoadActiveThresholdFailOpenFix:
    """load_active_threshold must return None — not a stored numeric
    threshold — whenever the calibration's status isn't CALIBRATED."""

    def test_insufficient_n_calibration_returns_none_not_infinity(self, tmp_path):
        path = str(tmp_path / "calib.json")
        calib = calibrate_threshold([0.2, 0.4], alpha=0.1)  # -> INSUFFICIENT_N, threshold=inf
        assert calib["status"] == CalibrationStatus.INSUFFICIENT_N
        save_calibration(calib, path)

        # The file DOES contain threshold="inf" — but load_active_threshold
        # must not trust it, because status != CALIBRATED.
        raw = load_active_calibration(path)
        assert raw["threshold"] == math.inf
        assert load_active_threshold(path) is None

    def test_no_data_calibration_returns_none(self, tmp_path):
        path = str(tmp_path / "calib.json")
        calib = calibrate_threshold([], alpha=0.1)
        save_calibration(calib, path)
        assert load_active_threshold(path) is None

    def test_calibrated_status_returns_the_real_threshold(self, tmp_path):
        path = str(tmp_path / "calib.json")
        calib = calibrate_threshold([0.1, 0.2, 0.3, 0.4, 0.5], alpha=0.2)
        assert calib["status"] == CalibrationStatus.CALIBRATED
        save_calibration(calib, path)
        assert load_active_threshold(path) == calib["threshold"]

    def test_legacy_file_with_no_status_field_falls_back_safely(self, tmp_path):
        """A calibration saved by pre-fix code has no 'status' key at all —
        must be treated as untrusted, not as an implicit CALIBRATED."""
        import json
        path = str(tmp_path / "legacy_calib.json")
        with open(path, "w") as f:
            json.dump({"threshold": 0.42, "alpha": 0.1, "n": 20}, f)  # no "status"
        assert load_active_threshold(path) is None

    def test_absent_file_returns_none(self, tmp_path):
        path = str(tmp_path / "does_not_exist.json")
        assert load_active_calibration(path) is None
        assert load_active_threshold(path) is None


def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "calib.json")
    assert load_active_calibration(path) is None
    assert load_active_threshold(path) is None  # absent => fallback signal

    calib = calibrate_threshold([0.1, 0.2, 0.3, 0.4, 0.5], alpha=0.2)
    save_calibration(calib, path)

    loaded = load_active_calibration(path)
    assert loaded is not None
    assert loaded["threshold"] == calib["threshold"]
    assert load_active_threshold(path) == calib["threshold"]


def test_infinite_threshold_survives_roundtrip(tmp_path):
    path = str(tmp_path / "calib_inf.json")
    calib = calibrate_threshold([0.2, 0.4], alpha=0.1)  # -> INSUFFICIENT_N, inf
    assert calib["threshold"] == math.inf
    save_calibration(calib, path)
    # The raw stored value round-trips as +inf (transparency)...
    assert load_active_calibration(path)["threshold"] == math.inf
    # ...but it is never treated as active (the actual fail-open fix).
    assert load_active_threshold(path) is None


class TestCalibrationHistoryChain:
    """save_calibration/record_calibration_history append to a tamper-evident
    SQLite chain (shared/database.py) — this used to be the one governance
    artifact with no history and no way to prove it wasn't quietly edited."""

    def test_record_calibration_history_is_chained_and_verifiable(self):
        from shared.database import list_calibration_records, iter_calibration_chain
        from shared.audit_chain import verify_chain

        before = len(list_calibration_records())
        c1 = calibrate_threshold([0.1, 0.2, 0.3], alpha=0.1)  # INSUFFICIENT_N
        c2 = calibrate_threshold([i / 20 for i in range(1, 21)], alpha=0.1)  # CALIBRATED
        record_calibration_history(c1)
        record_calibration_history(c2)

        records = list_calibration_records()
        assert len(records) == before + 2
        assert records[-2]["status"] == CalibrationStatus.INSUFFICIENT_N
        assert records[-1]["status"] == CalibrationStatus.CALIBRATED

        result = verify_chain(iter_calibration_chain())
        assert result["intact"] is True

    def test_history_records_every_attempt_even_when_not_applied(self, tmp_path):
        """Mirrors eval/harness.py::run_calibration's gate: an INSUFFICIENT_N
        result is recorded in history but must not silently become active."""
        from shared.database import list_calibration_records

        path = str(tmp_path / "calib.json")
        before = len(list_calibration_records())
        insufficient = calibrate_threshold([0.1, 0.2], alpha=0.1)
        record_calibration_history(insufficient)  # recorded...
        # ...but NOT saved as active (this is run_calibration's job, not
        # record_calibration_history's) — the JSON pointer is untouched.
        assert load_active_calibration(path) is None
        assert len(list_calibration_records()) == before + 1
