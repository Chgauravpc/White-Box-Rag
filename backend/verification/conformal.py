"""
conformal.py — split-conformal calibration of the abstention threshold.

Replaces the hand-tuned ABSTENTION_MEAN_PENALTY_CEIL with a threshold calibrated
to a statistical risk target. Nonconformity score = the retained-set mean penalty
(mitigation.nonconformity_score). Calibrating on items labelled *answerable*
(expected_abstain=False), the ⌈(n+1)(1-α)⌉-th smallest score is the threshold, so
the system wrongly abstains on at most ~α of truly-answerable queries (marginal
coverage on the answerable set) — WHEN there is enough calibration data.

Critical correctness rule: an INSUFFICIENT_N (or NO_DATA) calibration must
never make the system MORE permissive than the fixed fallback ceiling. Before
this module tracked status explicitly, a too-small calibration set produced
threshold=+inf, and because "mean_penalty > +inf" is always False, that
silently DISABLED the abstention rule entirely — the "calibrated" system was
strictly more permissive than before calibration. `load_active_threshold()`
now returns None (triggering the fixed-ceiling fallback) whenever the active
calibration's status isn't CALIBRATED, regardless of what the stored
`threshold` field says.

This is split conformal for a binary answer/abstain decision — NOT a per-token or
per-claim guarantee. Pure math + a small JSON persistence file for the hot
read path (`load_active_threshold`, called on every `should_abstain`), plus a
hash-chained SQLite history (shared/database.py) so this piece of governance
state — previously the one artifact outside the tamper-evident chain — has an
auditable record of every calibration attempt, not just the current value.
Producing the calibration scores requires running the live pipeline over a
labelled set (see eval/harness.run_calibration) — an operator action, not a
CI step.
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional

from shared.config import SQLITE_PATH

logger = logging.getLogger(__name__)

# Persist alongside the SQLite DB (both are local run state under data/).
DEFAULT_CALIBRATION_PATH = os.path.join(os.path.dirname(SQLITE_PATH) or ".", "conformal_calibration.json")


class CalibrationStatus:
    """String constants (not an Enum) so they serialize into JSON/SQLite
    without a conversion step."""
    NO_DATA = "NO_DATA"
    INSUFFICIENT_N = "INSUFFICIENT_N"
    CALIBRATED = "CALIBRATED"


def min_n_for_alpha(alpha: float) -> int:
    """Minimum answerable-item count for calibrate_threshold to produce a real
    (non-+inf) threshold at this alpha: the smallest n with
    ceil((n+1)(1-alpha)) <= n, which solves to ceil(1/alpha) - 1."""
    return math.ceil(1.0 / alpha) - 1


def recommended_n_for_alpha(alpha: float) -> int:
    """A rule-of-thumb 'comfortable' n (~3x the bare minimum). min_n_for_alpha
    only guarantees the threshold isn't the degenerate case; a materially
    larger calibration set is what actually gives the quantile estimate low
    variance. This is a practical floor for UIs/CLIs to warn against
    'technically calibrated but barely' — not a formal statistical bound."""
    return 3 * min_n_for_alpha(alpha)


def calibrate_threshold(scores: list[float], alpha: float) -> dict:
    """Split-conformal threshold from answerable-item nonconformity scores.

    Returns {status, threshold, alpha, n, min_n, recommended_n, coverage_note}.
    `threshold` is kept as the mathematical fact for transparency/debugging
    (None when there's no data, +inf when n is too small, a real number when
    calibrated) — but callers MUST gate on `status`, not on `threshold` being
    finite, before treating a calibration as usable. See module docstring.
    """
    n = len(scores)
    min_n = min_n_for_alpha(alpha)
    recommended_n = recommended_n_for_alpha(alpha)

    if n == 0:
        return {
            "status": CalibrationStatus.NO_DATA,
            "threshold": None, "alpha": alpha, "n": 0,
            "min_n": min_n, "recommended_n": recommended_n,
            "coverage_note": "No calibration data — abstention falls back to the fixed ceiling.",
        }

    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return {
            "status": CalibrationStatus.INSUFFICIENT_N,
            "threshold": math.inf, "alpha": alpha, "n": n,
            "min_n": min_n, "recommended_n": recommended_n,
            "coverage_note": (
                f"n={n} is below the minimum of {min_n} answerable items needed for α={alpha}. "
                f"The mathematical quantile is +∞, but abstention falls back to the fixed ceiling "
                f"instead of being disabled. Add more answerable calibration items "
                f"(recommended: {recommended_n}+) to actually calibrate at this α."
            ),
        }

    threshold = sorted(scores)[k - 1]
    note = (
        f"Calibrated on {n} answerable items: at most ~{alpha:.0%} of truly-answerable "
        f"queries will be wrongly abstained (marginal coverage)."
    )
    if n < recommended_n:
        note += f" n is above the bare minimum ({min_n}) but below the recommended {recommended_n} — the quantile estimate may be noisy."

    return {
        "status": CalibrationStatus.CALIBRATED,
        "threshold": threshold, "alpha": alpha, "n": n,
        "min_n": min_n, "recommended_n": recommended_n,
        "coverage_note": note,
    }


def record_calibration_history(calibration: dict) -> Optional[dict]:
    """Append a calibration attempt (of ANY status) to the tamper-evident
    SQLite chain — a permanent, auditable record that a calibration was
    attempted, independent of whether it became the active threshold.
    Failure here is logged, not raised: the active JSON pointer (the hot
    read path) must not depend on the history write succeeding.
    """
    try:
        from shared.database import insert_calibration_record
        return insert_calibration_record(calibration)
    except Exception as e:
        logger.warning(f"Failed to append calibration to the tamper-evident chain: {e}")
        return None


def save_calibration(calibration: dict, path: Optional[str] = None) -> None:
    """Persist `calibration` as the ACTIVE threshold (the JSON pointer
    should_abstain reads on every call) and append it to the tamper-evident
    history. Callers decide whether a given calibration is fit to become
    active (see eval/harness.py::run_calibration's CALIBRATED-or-force gate)
    BEFORE calling this — save_calibration itself always applies what it's
    given. +inf is stored as the string 'inf' for JSON.
    """
    path = path or DEFAULT_CALIBRATION_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    to_store = dict(calibration)
    thr = to_store.get("threshold")
    if thr == math.inf:
        to_store["threshold"] = "inf"
    to_store.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_store, f)

    record_calibration_history(calibration)


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
    """The active calibrated abstention threshold, or None to signal the
    fixed-ceiling fallback.

    Returns None whenever the persisted calibration's status is not
    CALIBRATED (including calibrations saved before `status` existed, where
    `.get("status")` is None) — this is the fail-open fix: an
    INSUFFICIENT_N/NO_DATA calibration, or a legacy file with no status at
    all, must never be trusted as a real threshold just because a numeric
    `threshold` field is present.
    """
    calib = load_active_calibration(path)
    if calib is None:
        return None
    if calib.get("status") != CalibrationStatus.CALIBRATED:
        return None
    return calib.get("threshold")
