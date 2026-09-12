"""
scoring.py — label-vs-prediction scoring for the eval harness.

Before this module, nothing in the repo compared a prediction to a label:
`expected_abstain` was loaded by harness.py and never scored against the
actual `abstained` outcome, and every "metric" the harness produced
(abstention_rate, mean_faithfulness_post, trust_status_distribution, ...)
was an unlabeled rate or mean — a number that can't distinguish "the
detector is working" from "the detector is trigger-happy" or "the detector
is asleep."

Pure functions, no I/O, no ML models — trivially unit-testable (see
tests/test_scoring.py) and safe to call from CI. Every metric returns a
{value, n, stddev, ci_low, ci_high, method} dict so uncertainty is part of
the contract, not a later retrofit (finding #13). Wilson intervals are used
for simple proportions (precision/recall/specificity); a seeded percentile
bootstrap is used for anything else (F1, MCC, balanced accuracy, nDCG, MAP,
paired deltas) — seeded so results are reproducible run-to-run, matching the
project's determinism goals (finding #5/#19).
"""

import math
import warnings
from typing import Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from scipy.stats import binomtest

DEFAULT_SEED = 1234
DEFAULT_N_RESAMPLES = 2000
DEFAULT_ALPHA = 0.05  # for 95% CIs


# ──────────────────────────────────────────────
#  Metric contract
# ──────────────────────────────────────────────

def _round(x, ndigits=6):
    return None if x is None else round(float(x), ndigits)


def _metric(value, n, stddev=None, ci=None, method="point") -> dict:
    """The uniform metric shape every function in this module returns."""
    lo, hi = ci if ci else (None, None)
    return {
        "value": _round(value),
        "n": n,
        "stddev": _round(stddev),
        "ci_low": _round(lo),
        "ci_high": _round(hi),
        "method": method,
    }


def _empty_metric(reason: str) -> dict:
    return _metric(None, 0, method=reason)


# ──────────────────────────────────────────────
#  Interval estimators
# ──────────────────────────────────────────────

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion — well-behaved at the
    extremes (p near 0 or 1, small n) unlike the naive normal-approximation
    interval."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half_width = (z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def bootstrap_ci(
    values: list,
    statistic_fn,
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float, float]:
    """Seeded percentile bootstrap. Returns (point_estimate, ci_low, ci_high).
    `statistic_fn` takes an array of (resampled) indices' worth of `values`
    and returns a scalar. Seeded so the same input always gives the same CI —
    a bootstrap without a fixed seed would itself be a source of the
    nondeterminism this project is trying to eliminate (finding #5).
    """
    arr = np.asarray(values)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = statistic_fn(arr)
    if n == 1:
        return (float(point), float(point), float(point))
    rng = np.random.default_rng(seed)
    resampled = np.empty(n_resamples)
    idx = np.arange(n)
    for i in range(n_resamples):
        sample = rng.choice(idx, size=n, replace=True)
        resampled[i] = statistic_fn(arr[sample])
    lo, hi = np.percentile(resampled, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(point), float(lo), float(hi))


# ──────────────────────────────────────────────
#  Binary classification — the shared core
# ──────────────────────────────────────────────

def binary_classification(y_true: list[bool], y_pred: list[bool]) -> dict:
    """TP/FP/FN/TN + precision/recall/F1/specificity/balanced-accuracy/MCC.

    `y_true[i]` is whether item i is a genuine positive (e.g. "should have
    abstained" / "is actually hallucinated"); `y_pred[i]` is what the system
    decided. Every derived metric degrades gracefully (returns a
    self-describing empty metric, never a ZeroDivisionError or NaN) when its
    denominator is zero.
    """
    n = len(y_true)
    if n == 0 or len(y_pred) != n:
        empty = _empty_metric("no_data")
        return {
            "n": 0,
            "confusion": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
            "precision": empty, "recall": empty, "specificity": empty,
            "f1": empty, "balanced_accuracy": empty, "mcc": empty,
        }

    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    tn = int(np.sum(~yt & ~yp))

    precision = (
        _metric(tp / (tp + fp), tp + fp, ci=wilson_interval(tp, tp + fp), method="wilson")
        if (tp + fp) > 0 else _empty_metric("no_predicted_positives")
    )
    recall = (
        _metric(tp / (tp + fn), tp + fn, ci=wilson_interval(tp, tp + fn), method="wilson")
        if (tp + fn) > 0 else _empty_metric("no_true_positives")
    )
    specificity = (
        _metric(tn / (tn + fp), tn + fp, ci=wilson_interval(tn, tn + fp), method="wilson")
        if (tn + fp) > 0 else _empty_metric("no_true_negatives")
    )

    if precision["value"] is not None and recall["value"] is not None and (precision["value"] + recall["value"]) > 0:
        def _f1(sample_idx):
            s_tp = int(np.sum(yt[sample_idx] & yp[sample_idx]))
            s_fp = int(np.sum(~yt[sample_idx] & yp[sample_idx]))
            s_fn = int(np.sum(yt[sample_idx] & ~yp[sample_idx]))
            denom = 2 * s_tp + s_fp + s_fn
            return (2 * s_tp / denom) if denom > 0 else 0.0
        point, lo, hi = bootstrap_ci(np.arange(n), lambda idx: _f1(idx.astype(int)))
        f1 = _metric(point, n, ci=(lo, hi), method="paired_bootstrap")
    else:
        f1 = _empty_metric("undefined_precision_or_recall")

    if recall["value"] is not None and specificity["value"] is not None:
        balanced_accuracy = _metric(
            (recall["value"] + specificity["value"]) / 2, n, method="mean_of_recall_and_specificity"
        )
    else:
        balanced_accuracy = _empty_metric("undefined_recall_or_specificity")

    if len(set(yt.tolist())) > 1 and len(set(yp.tolist())) > 1:
        def _mcc(sample_idx):
            # A bootstrap resample can legitimately draw a single-class subset
            # (small/imbalanced n) — sklearn correctly returns 0.0 for that
            # degenerate resample and warns about it; the warning is expected
            # noise here, not a bug, so it's suppressed within this scope only.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return matthews_corrcoef(yt[sample_idx], yp[sample_idx])
        point, lo, hi = bootstrap_ci(np.arange(n), lambda idx: _mcc(idx.astype(int)))
        mcc = _metric(point, n, ci=(lo, hi), method="paired_bootstrap")
    else:
        mcc = _empty_metric("degenerate_labels_or_predictions")

    return {
        "n": n,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }


# ──────────────────────────────────────────────
#  Abstention accuracy
# ──────────────────────────────────────────────

def abstention_metrics(expected_abstain: list[bool], actual_abstain: list[bool]) -> dict:
    """Precision/recall/F1 of the abstain decision against `expected_abstain`,
    plus the wrongful-abstention rate on truly-answerable items — the exact
    quantity a conformal alpha target is supposed to bound (see
    verification/conformal.py). This is what should be measured empirically
    to check whether the calibration's coverage guarantee actually holds,
    rather than just asserted.
    """
    base = binary_classification(expected_abstain, actual_abstain)
    fp = base["confusion"]["fp"]  # abstained but was answerable
    tn = base["confusion"]["tn"]  # answered and was answerable
    n_answerable = fp + tn
    if n_answerable == 0:
        base["wrongful_abstention_rate"] = _empty_metric("no_answerable_items")
    else:
        base["wrongful_abstention_rate"] = _metric(
            fp / n_answerable, n_answerable, ci=wilson_interval(fp, n_answerable), method="wilson"
        )
    return base


# ──────────────────────────────────────────────
#  Hallucination detector accuracy
# ──────────────────────────────────────────────

def detector_metrics(
    y_true_hallucinated: list[bool],
    y_pred_flagged: list[bool],
    hallucination_scores: Optional[list[float]] = None,
    error_types: Optional[list[str]] = None,
) -> dict:
    """Claim-level hallucination-detection accuracy.

    `y_true_hallucinated[i]`: was claim i actually hallucinated (per label).
    `y_pred_flagged[i]`: did the system flag/strip it.
    `hallucination_scores[i]` (optional): a continuous "how hallucinated"
    score — e.g. 1 - entailment_score, or the raw contradiction_score —
    oriented so HIGHER means more likely hallucinated. Enables a
    threshold-free AUROC/AUPRC, which is a more credible headline number than
    one tuned operating point.
    `error_types[i]` (optional, from a controlled-perturbation probe):
    enables a per-error-type recall breakdown — "catches 94% of negations,
    41% of numeric swaps" is far more actionable than one aggregate F1.
    """
    base = binary_classification(y_true_hallucinated, y_pred_flagged)

    if hallucination_scores is not None and len(hallucination_scores) == len(y_true_hallucinated):
        yt = np.asarray(y_true_hallucinated, dtype=bool)
        scores = np.asarray(hallucination_scores, dtype=float)
        if len(set(yt.tolist())) > 1:
            base["auroc"] = _metric(roc_auc_score(yt, scores), len(yt), method="point")
            base["auprc"] = _metric(average_precision_score(yt, scores), len(yt), method="point")
        else:
            base["auroc"] = _empty_metric("only_one_class_present")
            base["auprc"] = _empty_metric("only_one_class_present")
    else:
        base["auroc"] = _empty_metric("no_scores_provided")
        base["auprc"] = _empty_metric("no_scores_provided")

    if error_types is not None and len(error_types) == len(y_true_hallucinated):
        by_type: dict[str, dict] = {}
        for true, pred, etype in zip(y_true_hallucinated, y_pred_flagged, error_types):
            if not true or not etype:
                continue  # recall breakdown is only defined over genuine positives
            slot = by_type.setdefault(etype, {"caught": 0, "total": 0})
            slot["total"] += 1
            if pred:
                slot["caught"] += 1
        base["recall_by_error_type"] = {
            etype: _metric(
                v["caught"] / v["total"], v["total"],
                ci=wilson_interval(v["caught"], v["total"]), method="wilson",
            )
            for etype, v in by_type.items()
        }
    else:
        base["recall_by_error_type"] = {}

    return base


# ──────────────────────────────────────────────
#  Retrieval accuracy
# ──────────────────────────────────────────────

def _dcg(grades: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def _ndcg_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    top_k = retrieved[:k]
    grades = [relevance.get(cid, 0) for cid in top_k]
    dcg = _dcg(grades)
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    idcg = _dcg(ideal_grades)
    return (dcg / idcg) if idcg > 0 else 0.0


def _average_precision(retrieved: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    hits = 0
    precisions = []
    for i, cid in enumerate(retrieved):
        if cid in relevant_ids:
            hits += 1
            precisions.append(hits / (i + 1))
    return sum(precisions) / len(relevant_ids) if precisions else 0.0


def retrieval_metrics(per_query: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10, 20)) -> dict:
    """Hit@k / recall@k / precision@k / MRR / nDCG@k / MAP, each bootstrapped
    over QUERIES (not over individual retrieved items — resampling items
    within a query would break the ranking structure the metrics depend on).

    `per_query`: list of {"retrieved": [chunk_key, ...] ranked desc,
    "relevant": {chunk_key: grade}} — only queries with ground truth (a
    non-empty `relevant`) should be passed in; callers filter beforehand
    (matching the existing harness.py convention for `ground_truth_items`).
    Chunk identity must be the canonical chunk_key (shared/chunk_key.py), not
    a bare section_id, or cross-document collisions will corrupt every
    number here.
    """
    queries = [q for q in per_query if q.get("relevant")]
    n = len(queries)
    if n == 0:
        empty = _empty_metric("no_ground_truth_items")
        result = {"n": 0, "mrr": empty, "map": empty}
        for k in ks:
            result[f"hit_at_{k}"] = empty
            result[f"recall_at_{k}"] = empty
            result[f"precision_at_{k}"] = empty
            result[f"ndcg_at_{k}"] = empty
        return result

    def _rank_of_first_relevant(retrieved, relevant_ids):
        for i, cid in enumerate(retrieved):
            if cid in relevant_ids:
                return i + 1
        return None

    reciprocal_ranks = []
    average_precisions = []
    per_k_hits = {k: [] for k in ks}
    per_k_recalls = {k: [] for k in ks}
    per_k_precisions = {k: [] for k in ks}
    per_k_ndcgs = {k: [] for k in ks}

    for q in queries:
        retrieved = q.get("retrieved", [])
        relevance = q["relevant"]
        relevant_ids = set(relevance.keys())

        rank = _rank_of_first_relevant(retrieved, relevant_ids)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        average_precisions.append(_average_precision(retrieved, relevant_ids))

        for k in ks:
            top_k = set(retrieved[:k])
            found = top_k & relevant_ids
            per_k_hits[k].append(1.0 if found else 0.0)
            per_k_recalls[k].append(len(found) / len(relevant_ids) if relevant_ids else 0.0)
            per_k_precisions[k].append(len(found) / k)
            per_k_ndcgs[k].append(_ndcg_at_k(retrieved, relevance, k))

    def _mean_metric(values, method="paired_bootstrap"):
        point, lo, hi = bootstrap_ci(values, np.mean)
        return _metric(point, len(values), stddev=float(np.std(values)), ci=(lo, hi), method=method)

    result = {
        "n": n,
        "mrr": _mean_metric(reciprocal_ranks),
        "map": _mean_metric(average_precisions),
    }
    for k in ks:
        result[f"hit_at_{k}"] = _metric(
            float(np.mean(per_k_hits[k])), n,
            ci=wilson_interval(int(sum(per_k_hits[k])), n), method="wilson",
        )
        result[f"recall_at_{k}"] = _mean_metric(per_k_recalls[k])
        result[f"precision_at_{k}"] = _mean_metric(per_k_precisions[k])
        result[f"ndcg_at_{k}"] = _mean_metric(per_k_ndcgs[k])
    return result


# ──────────────────────────────────────────────
#  Trust Gate accuracy
# ──────────────────────────────────────────────

def trust_gate_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: tuple[str, ...] = ("Safe", "Needs_Human_Review", "Non_Compliant"),
) -> dict:
    """3-class confusion matrix of the Trust Gate's SAFE/NEEDS_HUMAN_REVIEW/
    NON_COMPLIANT decision against a human-labeled severity, plus Cohen's
    kappa (chance-corrected agreement — plain accuracy overstates agreement
    when one class dominates, which SAFE typically does)."""
    n = len(y_true)
    if n == 0 or len(y_pred) != n:
        return {
            "n": 0, "confusion_matrix": {}, "labels": list(labels),
            "accuracy": _empty_metric("no_data"), "cohen_kappa": _empty_metric("no_data"),
        }

    confusion = {t: {p: 0 for p in labels} for t in labels}
    correct = 0
    for t, p in zip(y_true, y_pred):
        if t in confusion and p in confusion[t]:
            confusion[t][p] += 1
        if t == p:
            correct += 1

    kappa = cohen_kappa_score(y_true, y_pred, labels=list(labels)) if len(set(y_true)) > 1 else None

    return {
        "n": n,
        "labels": list(labels),
        "confusion_matrix": confusion,
        "accuracy": _metric(correct / n, n, ci=wilson_interval(correct, n), method="wilson"),
        "cohen_kappa": (
            _metric(kappa, n, method="point") if kappa is not None
            else _empty_metric("only_one_true_class_present")
        ),
    }


# ──────────────────────────────────────────────
#  Score calibration (is entailment_score itself trustworthy?)
# ──────────────────────────────────────────────

def calibration_metrics(scores: list[float], outcomes: list[bool], n_bins: int = 10) -> dict:
    """Expected Calibration Error + reliability bins for a probability-like
    score (e.g. entailment_score) against a binary outcome (e.g. "was this
    claim actually supported"). A well-calibrated score's mean value within
    a bin should match the bin's actual positive rate; ECE is the
    count-weighted mean absolute gap between the two.
    """
    n = len(scores)
    if n == 0 or len(outcomes) != n:
        return {"n": 0, "ece": _empty_metric("no_data"), "bins": []}

    scores_arr = np.asarray(scores, dtype=float)
    outcomes_arr = np.asarray(outcomes, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (scores_arr >= lo) & (scores_arr <= hi if i == n_bins - 1 else scores_arr < hi)
        count = int(np.sum(in_bin))
        if count == 0:
            bins.append({"range": [round(float(lo), 4), round(float(hi), 4)], "count": 0, "mean_score": None, "mean_outcome": None})
            continue
        mean_score = float(np.mean(scores_arr[in_bin]))
        mean_outcome = float(np.mean(outcomes_arr[in_bin]))
        bins.append({
            "range": [round(float(lo), 4), round(float(hi), 4)],
            "count": count,
            "mean_score": round(mean_score, 6),
            "mean_outcome": round(mean_outcome, 6),
        })
        ece += (count / n) * abs(mean_score - mean_outcome)

    return {"n": n, "ece": _metric(ece, n, method="expected_calibration_error"), "bins": bins}


# ──────────────────────────────────────────────
#  Paired comparison (for ablations — Phase 6)
# ──────────────────────────────────────────────

def paired_delta(
    a: list[float],
    b: list[float],
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
) -> dict:
    """Paired bootstrap CI on mean(a) - mean(b), for comparing two pipeline
    profiles run over the SAME items (e.g. `full` vs `plain_rag`). Resamples
    query indices jointly so the pairing is preserved — this is what makes a
    delta attributable to the profile difference rather than to which items
    happened to be easy or hard.

    If both arrays are boolean-valued decision outcomes (0/1), also runs an
    exact McNemar test on the discordant pairs.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    n = len(a_arr)
    if n == 0 or len(b_arr) != n:
        return {"delta": _empty_metric("no_data")}

    def _delta(idx):
        return float(np.mean(a_arr[idx]) - np.mean(b_arr[idx]))

    point, lo, hi = bootstrap_ci(np.arange(n), lambda idx: _delta(idx.astype(int)), seed=seed, n_resamples=n_resamples)
    result = {"delta": _metric(point, n, ci=(lo, hi), method="paired_bootstrap")}

    is_boolean = set(np.unique(a_arr)).issubset({0.0, 1.0}) and set(np.unique(b_arr)).issubset({0.0, 1.0})
    if is_boolean and n > 0:
        a_bool = a_arr.astype(bool)
        b_bool = b_arr.astype(bool)
        a_only = int(np.sum(a_bool & ~b_bool))
        b_only = int(np.sum(~a_bool & b_bool))
        discordant = a_only + b_only
        if discordant > 0:
            p_value = binomtest(min(a_only, b_only), discordant, p=0.5).pvalue
        else:
            p_value = 1.0
        result["mcnemar"] = {"a_only": a_only, "b_only": b_only, "discordant": discordant, "p_value": _round(p_value)}
    else:
        result["mcnemar"] = None

    return result
