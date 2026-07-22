"""
conformal.py — split-conformal calibration of the abstention threshold.

Replaces the hand-tuned ABSTENTION_MEAN_PENALTY_CEIL with a threshold calibrated
to a statistical risk target. Nonconformity score = the retained-set mean penalty
(mitigation.should_abstain). Calibrating on items labelled *answerable*
(expected_abstain=False), the ⌈(n+1)(1-α)⌉-th smallest score is the threshold, so
the system wrongly abstains on at most ~α of truly-answerable queries (marginal
coverage on the answerable set).

This is split conformal for a binary answer/abstain decision — NOT a per-token or
per-claim guarantee. Pure math + a small JSON persistence file; no ML, no Gemini,
so calibrate_threshold is trivially unit-testable. Producing the calibration
scores, however, requires running the live pipeline over a labelled set (see
eval/harness.run_calibration) — an operator action, not a CI step.
"""

import json
import math
import os
from datetime import datetime
from typing import Optional

from shared.config import SQLITE_PATH

# Persist alongside the SQLite DB (both are local run state under data/).
DEFAULT_CALIBRATION_PATH = os.path.join(os.path.dirname(SQLITE_PATH) or ".", "conformal_calibration.json")


def calibrate_threshold(scores: list[float], alpha: float) -> dict:
    """Split-conformal threshold from answerable-item nonconformity scores.

    Returns {threshold, alpha, n, coverage_note}. threshold is None when there is
    no calibration data, and +inf when n is too small for the requested α (the
    conservative "never abstain on this basis" case).
    """
    n = len(scores)
    if n == 0:
        return {"threshold": None, "alpha": alpha, "n": 0,
                "coverage_note": "No calibration data — abstention falls back to the fixed ceiling."}

    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        threshold = math.inf
        note = (f"n={n} too small for α={alpha}: threshold is +∞ (never abstain on the mean-penalty rule). "
                f"Add more answerable calibration items to tighten it.")
    else:
        threshold = sorted(scores)[k - 1]
        note = (f"Calibrated on {n} answerable items: at most ~{alpha:.0%} of truly-answerable "
                f"queries will be wrongly abstained (marginal coverage).")

    return {"threshold": threshold, "alpha": alpha, "n": n, "coverage_note": note}


def save_calibration(calibration: dict, path: Optional[str] = None) -> None:
    """Persist the active calibration. +inf is stored as the string 'inf' for JSON."""
    path = path or DEFAULT_CALIBRATION_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    to_store = dict(calibration)
    thr = to_store.get("threshold")
    if thr == math.inf:
        to_store["threshold"] = "inf"
    to_store.setdefault("created_at", datetime.utcnow().isoformat())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_store, f)


def load_active_calibration(path: Optional[str] = None) -> Optional[dict]:
    """Load the persisted calibration dict, or None if none exists."""
    path = path or DEFAULT_CALIBRATION_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("threshold") == "inf":
        data["threshold"] = math.inf
    return data


def load_active_threshold(path: Optional[str] = None) -> Optional[float]:
    """The active calibrated abstention threshold, or None to signal fallback."""
    calib = load_active_calibration(path)
    if calib is None:
        return None
    return calib.get("threshold")
