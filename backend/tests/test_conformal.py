"""
Tests for conformal-prediction abstention calibration (Feature 3).

Pure math + JSON persistence — no pipeline, no Gemini. The live calibration run
(run_calibration) needs a real key + corpus and is not exercised here.
"""

import math

from verification.conformal import (
    calibrate_threshold,
    save_calibration,
    load_active_calibration,
    load_active_threshold,
)


def test_exact_quantile():
    # n=9, alpha=0.1 -> k = ceil(10 * 0.9) = 9 -> 9th smallest = 0.9
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result = calibrate_threshold(scores, alpha=0.1)
    assert result["threshold"] == 0.9
    assert result["n"] == 9
    assert result["alpha"] == 0.1


def test_quantile_is_order_independent():
    scores = [0.9, 0.1, 0.5, 0.3, 0.7]  # n=5, alpha=0.2 -> k = ceil(6*0.8)=5 -> max
    result = calibrate_threshold(scores, alpha=0.2)
    assert result["threshold"] == 0.9


def test_small_n_gives_infinite_threshold():
    # n=3, alpha=0.1 -> k = ceil(4 * 0.9) = 4 > 3 -> +inf (never abstain on this rule)
    result = calibrate_threshold([0.2, 0.4, 0.6], alpha=0.1)
    assert result["threshold"] == math.inf
    assert "never abstain" in result["coverage_note"]


def test_empty_scores_returns_none():
    result = calibrate_threshold([], alpha=0.1)
    assert result["threshold"] is None
    assert result["n"] == 0


def test_lower_alpha_gives_higher_threshold():
    scores = [i / 100 for i in range(1, 101)]  # 0.01 .. 1.00
    lax = calibrate_threshold(scores, alpha=0.3)["threshold"]
    strict = calibrate_threshold(scores, alpha=0.05)["threshold"]
    # Tolerating fewer wrongful abstentions (smaller alpha) => more permissive (higher) threshold
    assert strict >= lax


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
    calib = calibrate_threshold([0.2, 0.4], alpha=0.1)  # -> inf
    assert calib["threshold"] == math.inf
    save_calibration(calib, path)
    assert load_active_threshold(path) == math.inf
