"""
threshold.py — fit the claim-strip threshold to labeled data.

`mitigation._should_strip` flags a claim when any of three things hold: the
verdict is a contradiction, there was no usable premise, or the entailment
score falls below `STRIP_ENTAILMENT_FLOOR`. That floor is the system's single
most consequential operating parameter and it was chosen by hand as 0.5. On
RAGTruth it flags **half of all grounded sentences** — 710 false positives for
81 true catches — while the underlying score ranks well (AUROC 0.727). A model
that discriminates but is read at the wrong cut-off is a calibration problem,
not a modelling one, and calibration is exactly the kind of thing that should
be fitted rather than guessed.

Two properties this module insists on:

* **The sweep replicates the shipped rule.** Only the floor varies; the
  contradiction and missing-evidence branches fire exactly as they do in
  production. Sweeping `entailment_score < t` alone would describe a detector
  this repo does not ship, and would flatter it — those branches contribute
  false positives too.

* **Selection states its criterion.** "The best threshold" is meaningless
  without saying best at what. A governance system usually wants a bounded
  miss rate rather than a maximal F1: missing a hallucination and flagging a
  good sentence are not symmetric costs. So `target_recall` is offered
  alongside `max_f1`, and whichever is used is recorded in the result.

This module does NOT change any default. A threshold fitted on one corpus is
not automatically right for another — RAGTruth's premises are long news
articles and Wikipedia-style passages, while this system's own premises are
PDF chunks — so the value is reported for a human to set via
`STRIP_ENTAILMENT_FLOOR`, not silently applied.
"""

from shared import config
from shared.nli_policy import is_contradiction

# Fine enough to find a useful operating point, coarse enough that the curve
# stays readable when printed.
DEFAULT_GRID = [round(i / 100, 2) for i in range(0, 101, 1)]


def _flagged_at(result: dict, floor: float) -> bool:
    """`mitigation._should_strip` with the entailment floor as a parameter.

    Kept deliberately in lockstep with that function: contradiction first,
    then missing evidence, then the floor.
    """
    if is_contradiction(result.get("verdict")):
        return True
    if (result.get("evidence_status") or "ok") != "ok":
        return True
    return float(result.get("entailment_score", 0.0)) < floor


def _counts(results, floor):
    tp = fp = fn = tn = 0
    for r in results:
        positive = r.get("label") != "SUPPORTED"
        flagged = _flagged_at(r, floor)
        if positive and flagged:
            tp += 1
        elif positive:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _point(floor, tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else 0.0)
    return {
        "threshold": floor, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": None if precision is None else round(precision, 6),
        "recall": None if recall is None else round(recall, 6),
        "specificity": None if specificity is None else round(specificity, 6),
        "f1": round(f1, 6),
    }


def sweep_thresholds(results, grid=None) -> list[dict]:
    """Metrics for the shipped strip rule at each candidate entailment floor.

    `results` are per-item dicts with `label`, `verdict`, `entailment_score`
    and `evidence_status` — the shape `eval/detector.py` already emits.
    """
    grid = DEFAULT_GRID if grid is None else sorted(set(grid))
    return [_point(f, *_counts(results, f)) for f in grid]


def select_threshold(results, criterion: str = "max_f1", target: float = 0.8,
                     grid=None) -> dict:
    """Choose an entailment floor and say why.

    `max_f1` — the point with the highest F1.
    `target_recall` — the highest-precision point whose recall is at least
        `target`. This is usually the right criterion for a governance gate:
        set the miss rate you can defend, then pay as little precision as
        possible for it. Falls back to the max-recall point when no threshold
        reaches the target, and says so.
    """
    # Guard on the RESULTS, not the curve: a grid always yields points, so an
    # empty dataset would otherwise "select" a threshold from a curve of zeros.
    if not results:
        return {"criterion": criterion, "selected": None, "reason": "no results",
                "n": 0, "positives": 0, "curve": []}
    curve = sweep_thresholds(results, grid)

    if criterion == "max_f1":
        best = max(curve, key=lambda p: (p["f1"], p["precision"] or 0.0))
        reason = "highest F1"
    elif criterion == "target_recall":
        eligible = [p for p in curve if (p["recall"] or 0.0) >= target]
        if eligible:
            best = max(eligible, key=lambda p: (p["precision"] or 0.0, p["f1"]))
            reason = f"highest precision with recall >= {target}"
        else:
            best = max(curve, key=lambda p: (p["recall"] or 0.0, p["precision"] or 0.0))
            reason = (f"no threshold reached recall {target}; using the "
                      f"max-recall point ({best['recall']})")
    else:
        raise ValueError(f"unknown criterion {criterion!r} — expected max_f1 or target_recall")

    current = _point(config.STRIP_ENTAILMENT_FLOOR,
                     *_counts(results, config.STRIP_ENTAILMENT_FLOOR))
    return {
        "criterion": criterion,
        "target": target if criterion == "target_recall" else None,
        "selected": best,
        "reason": reason,
        # Always reported next to the selection: a recommendation without the
        # status quo beside it is not actionable.
        "current": dict(current, threshold=config.STRIP_ENTAILMENT_FLOOR),
        "n": len(results),
        "positives": sum(1 for r in results if r.get("label") != "SUPPORTED"),
        "curve": curve,
    }
