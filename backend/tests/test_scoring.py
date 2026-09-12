"""
Tests for eval/scoring.py — label-vs-prediction accuracy metrics.

Golden-vector tests with hand-computed expected values wherever a metric has
a closed form; bootstrapped fields (F1, MCC, MRR, MAP, nDCG, paired deltas)
are checked for their exact POINT ESTIMATE (computed on the real data, not a
resample — see bootstrap_ci) plus CI sanity (bracket the point, reproducible
across repeated calls with the fixed default seed) rather than an exact CI
number, since only the point estimate is deterministic by construction.
"""

import math

import pytest

from eval.scoring import (
    binary_classification,
    abstention_metrics,
    detector_metrics,
    retrieval_metrics,
    trust_gate_metrics,
    calibration_metrics,
    paired_delta,
    wilson_interval,
    bootstrap_ci,
)


# ──────────────────────────────────────────────
#  Wilson interval
# ──────────────────────────────────────────────

class TestWilsonInterval:
    def test_n_zero_returns_full_range(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_known_value_8_of_10(self):
        # Hand-computed: p=0.8, z=1.96 -> center~=0.7167, half-width~=0.2266
        lo, hi = wilson_interval(8, 10)
        assert lo == pytest.approx(0.4902, abs=0.005)
        assert hi == pytest.approx(0.9433, abs=0.005)

    def test_interval_always_brackets_the_point_estimate(self):
        lo, hi = wilson_interval(7, 20)
        assert lo <= 7 / 20 <= hi
        assert 0.0 <= lo and hi <= 1.0


# ──────────────────────────────────────────────
#  Bootstrap CI
# ──────────────────────────────────────────────

class TestBootstrapCI:
    def test_single_value_has_zero_width_ci(self):
        point, lo, hi = bootstrap_ci([5.0], lambda arr: float(arr.mean()))
        assert point == lo == hi == 5.0

    def test_empty_input_is_nan(self):
        point, lo, hi = bootstrap_ci([], lambda arr: float(arr.mean()))
        assert math.isnan(point) and math.isnan(lo) and math.isnan(hi)

    def test_reproducible_with_fixed_seed(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        r1 = bootstrap_ci(values, lambda arr: float(arr.mean()), seed=42)
        r2 = bootstrap_ci(values, lambda arr: float(arr.mean()), seed=42)
        assert r1 == r2

    def test_point_estimate_is_exact_not_resampled(self):
        point, _, _ = bootstrap_ci([1.0, 2.0, 3.0], lambda arr: float(arr.mean()))
        assert point == pytest.approx(2.0)


# ──────────────────────────────────────────────
#  Binary classification
# ──────────────────────────────────────────────

class TestBinaryClassification:
    def test_hand_computed_confusion_and_metrics(self):
        # idx: 0=TP, 1=FN, 2=TN, 3=FP
        y_true = [True, True, False, False]
        y_pred = [True, False, False, True]
        result = binary_classification(y_true, y_pred)

        assert result["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
        assert result["precision"]["value"] == pytest.approx(0.5)
        assert result["recall"]["value"] == pytest.approx(0.5)
        assert result["specificity"]["value"] == pytest.approx(0.5)
        assert result["f1"]["value"] == pytest.approx(0.5)
        assert result["balanced_accuracy"]["value"] == pytest.approx(0.5)
        # f1's CI must bracket its own point estimate
        assert result["f1"]["ci_low"] <= result["f1"]["value"] <= result["f1"]["ci_high"]

    def test_perfect_classifier(self):
        y = [True, False, True, False, True]
        result = binary_classification(y, y)
        assert result["precision"]["value"] == 1.0
        assert result["recall"]["value"] == 1.0
        assert result["f1"]["value"] == 1.0
        assert result["mcc"]["value"] == pytest.approx(1.0)

    def test_empty_input_degrades_gracefully(self):
        result = binary_classification([], [])
        assert result["n"] == 0
        assert result["precision"]["value"] is None
        assert result["precision"]["method"] == "no_data"

    def test_no_true_positives_leaves_recall_undefined_but_precision_defined(self):
        # Nobody should abstain, system never abstains either except once (a false positive).
        result = binary_classification([False, False, False, False], [True, False, False, False])
        assert result["recall"]["value"] is None  # no true positives to compute recall over
        assert result["precision"]["value"] == pytest.approx(0.0)
        assert result["specificity"]["value"] == pytest.approx(0.75)
        # balanced_accuracy needs recall — must degrade, not raise (regression test for a real bug)
        assert result["balanced_accuracy"]["value"] is None
        assert result["balanced_accuracy"]["method"] == "undefined_recall_or_specificity"

    def test_no_true_negatives_leaves_specificity_undefined(self):
        result = binary_classification([True, True, True], [True, False, True])
        assert result["specificity"]["value"] is None
        assert result["recall"]["value"] == pytest.approx(2 / 3)
        assert result["balanced_accuracy"]["value"] is None

    def test_mismatched_lengths_degrades_gracefully(self):
        result = binary_classification([True, False], [True])
        assert result["n"] == 0
        assert result["precision"]["method"] == "no_data"


# ──────────────────────────────────────────────
#  Abstention metrics
# ──────────────────────────────────────────────

class TestAbstentionMetrics:
    def test_hand_computed(self):
        expected_abstain = [True, False, False, True, False]
        actual_abstain   = [True, False, True, False, False]
        # tp=1 (idx0), fp=1 (idx2), fn=1 (idx3), tn=2 (idx1, idx4)
        result = abstention_metrics(expected_abstain, actual_abstain)
        assert result["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 2}
        assert result["precision"]["value"] == pytest.approx(0.5)
        assert result["recall"]["value"] == pytest.approx(0.5)
        # wrongful abstention rate = fp / (fp + tn) = 1/3 — the coverage quantity
        # a conformal alpha target is supposed to bound.
        assert result["wrongful_abstention_rate"]["value"] == pytest.approx(1 / 3)
        assert result["wrongful_abstention_rate"]["n"] == 3

    def test_no_answerable_items(self):
        result = abstention_metrics([True, True], [True, True])
        assert result["wrongful_abstention_rate"]["value"] is None
        assert result["wrongful_abstention_rate"]["method"] == "no_answerable_items"

    def test_never_abstains_never_wrongly_abstains(self):
        result = abstention_metrics([False, False, True], [False, False, False])
        assert result["wrongful_abstention_rate"]["value"] == pytest.approx(0.0)


# ──────────────────────────────────────────────
#  Detector metrics
# ──────────────────────────────────────────────

class TestDetectorMetrics:
    def test_hand_computed_auroc(self):
        # positives (hallucinated)=idx0,idx1 with scores 0.9,0.4; negatives=idx2,idx3 with scores 0.3,0.8
        y_true = [True, True, False, False]
        y_pred = [True, False, False, True]
        scores = [0.9, 0.4, 0.3, 0.8]
        result = detector_metrics(y_true, y_pred, hallucination_scores=scores)
        # AUROC = P(score_pos > score_neg): pairs (0.9,0.8)=1,(0.9,0.3)=1,(0.4,0.8)=0,(0.4,0.3)=1 -> 3/4
        assert result["auroc"]["value"] == pytest.approx(0.75)
        assert result["auprc"]["value"] is not None

    def test_no_scores_degrades_gracefully(self):
        result = detector_metrics([True, False], [True, False])
        assert result["auroc"]["method"] == "no_scores_provided"

    def test_recall_by_error_type(self):
        y_true = [True, True, True, True]
        y_pred = [True, False, True, False]
        error_types = ["negation", "negation", "numeric_swap", "numeric_swap"]
        result = detector_metrics(y_true, y_pred, error_types=error_types)
        assert result["recall_by_error_type"]["negation"]["value"] == pytest.approx(0.5)
        assert result["recall_by_error_type"]["numeric_swap"]["value"] == pytest.approx(0.5)

    def test_only_one_class_present_degrades_gracefully(self):
        result = detector_metrics([True, True], [True, True], hallucination_scores=[0.9, 0.8])
        assert result["auroc"]["method"] == "only_one_class_present"


# ──────────────────────────────────────────────
#  Retrieval metrics
# ──────────────────────────────────────────────

class TestRetrievalMetrics:
    def test_hand_computed_single_query(self):
        per_query = [{
            "retrieved": ["A", "B", "C", "D"],
            "relevant": {"B": 2, "D": 1},
        }]
        result = retrieval_metrics(per_query, ks=(1, 3, 5))

        assert result["n"] == 1
        assert result["hit_at_1"]["value"] == pytest.approx(0.0)
        assert result["hit_at_3"]["value"] == pytest.approx(1.0)
        assert result["recall_at_3"]["value"] == pytest.approx(0.5)
        assert result["precision_at_3"]["value"] == pytest.approx(1 / 3)
        # MRR: first relevant (B) at rank 2 -> RR = 1/2
        assert result["mrr"]["value"] == pytest.approx(0.5)
        # MAP: hits at rank2 (p=1/2) and rank4 (p=2/4=1/2) over 2 relevant -> (0.5+0.5)/2
        assert result["map"]["value"] == pytest.approx(0.5)
        # nDCG@3: dcg = 0/log2(2) + 2/log2(3) + 0/log2(4); idcg = 2/log2(2) + 1/log2(3)
        dcg = 0 / math.log2(2) + 2 / math.log2(3) + 0 / math.log2(4)
        idcg = 2 / math.log2(2) + 1 / math.log2(3)
        assert result["ndcg_at_3"]["value"] == pytest.approx(dcg / idcg)

    def test_no_ground_truth_items(self):
        result = retrieval_metrics([{"retrieved": ["A"], "relevant": {}}], ks=(1,))
        assert result["n"] == 0
        assert result["mrr"]["value"] is None
        assert result["hit_at_1"]["value"] is None

    def test_queries_without_relevant_are_excluded_not_zeroed(self):
        per_query = [
            {"retrieved": ["A", "B"], "relevant": {"B": 1}},
            {"retrieved": ["X"], "relevant": {}},  # no ground truth — excluded
        ]
        result = retrieval_metrics(per_query, ks=(1,))
        assert result["n"] == 1  # only the item with ground truth counts

    def test_perfect_retrieval(self):
        per_query = [{"retrieved": ["A"], "relevant": {"A": 3}}]
        result = retrieval_metrics(per_query, ks=(1,))
        assert result["hit_at_1"]["value"] == 1.0
        assert result["mrr"]["value"] == 1.0
        assert result["ndcg_at_1"]["value"] == pytest.approx(1.0)


# ──────────────────────────────────────────────
#  Trust Gate metrics
# ──────────────────────────────────────────────

class TestTrustGateMetrics:
    def test_hand_computed(self):
        y_true = ["Safe", "Safe", "Needs_Human_Review", "Non_Compliant"]
        y_pred = ["Safe", "Needs_Human_Review", "Needs_Human_Review", "Non_Compliant"]
        result = trust_gate_metrics(y_true, y_pred)

        assert result["n"] == 4
        assert result["accuracy"]["value"] == pytest.approx(0.75)
        # Hand-computed Cohen's kappa: Po=0.75, Pe=(2*1 + 1*2 + 1*1)/16=5/16=0.3125
        # kappa = (Po - Pe) / (1 - Pe) = 0.4375 / 0.6875
        assert result["cohen_kappa"]["value"] == pytest.approx(0.4375 / 0.6875, abs=1e-4)
        assert result["confusion_matrix"]["Safe"]["Safe"] == 1
        assert result["confusion_matrix"]["Safe"]["Needs_Human_Review"] == 1

    def test_only_one_true_class_degrades_gracefully(self):
        result = trust_gate_metrics(["Safe", "Safe"], ["Safe", "Needs_Human_Review"])
        assert result["cohen_kappa"]["method"] == "only_one_true_class_present"

    def test_empty_input(self):
        result = trust_gate_metrics([], [])
        assert result["n"] == 0
        assert result["accuracy"]["value"] is None


# ──────────────────────────────────────────────
#  Calibration metrics
# ──────────────────────────────────────────────

class TestCalibrationMetrics:
    def test_hand_computed_ece(self):
        scores = [0.1, 0.9, 0.5, 0.5]
        outcomes = [False, True, True, False]
        result = calibration_metrics(scores, outcomes, n_bins=2)
        # bin0 [0,0.5): {0.1} -> mean_score=0.1, mean_outcome=0.0, weight=1/4
        # bin1 [0.5,1.0]: {0.9,0.5,0.5} -> mean_score=0.6333, mean_outcome=0.6667, weight=3/4
        # ECE = 0.25*0.1 + 0.75*|0.6333-0.6667| = 0.025 + 0.025 = 0.05
        assert result["ece"]["value"] == pytest.approx(0.05, abs=1e-4)
        assert len(result["bins"]) == 2
        assert result["bins"][0]["count"] == 1
        assert result["bins"][1]["count"] == 3

    def test_empty_bins_reported_with_none_not_error(self):
        result = calibration_metrics([0.9, 0.95], [True, True], n_bins=10)
        assert result["n"] == 2
        empty_bins = [b for b in result["bins"] if b["count"] == 0]
        assert len(empty_bins) > 0
        assert all(b["mean_score"] is None for b in empty_bins)

    def test_no_data(self):
        result = calibration_metrics([], [], n_bins=5)
        assert result["n"] == 0
        assert result["ece"]["value"] is None


# ──────────────────────────────────────────────
#  Paired delta (ablation comparison)
# ──────────────────────────────────────────────

class TestPairedDelta:
    def test_hand_computed_delta_and_mcnemar(self):
        a = [1, 1, 0, 0]  # mean = 0.5
        b = [1, 0, 0, 0]  # mean = 0.25
        result = paired_delta(a, b)
        assert result["delta"]["value"] == pytest.approx(0.25)
        assert result["delta"]["ci_low"] <= 0.25 <= result["delta"]["ci_high"]
        # Discordant pairs: idx1 (a=1,b=0) is the only disagreement.
        assert result["mcnemar"]["a_only"] == 1
        assert result["mcnemar"]["b_only"] == 0
        assert result["mcnemar"]["discordant"] == 1
        assert result["mcnemar"]["p_value"] == pytest.approx(1.0)

    def test_non_boolean_arrays_skip_mcnemar(self):
        result = paired_delta([0.9, 0.8, 0.7], [0.5, 0.4, 0.3])
        assert result["delta"]["value"] == pytest.approx(0.4)
        assert result["mcnemar"] is None

    def test_identical_arrays_zero_delta_no_discordance(self):
        a = [1, 0, 1, 0]
        result = paired_delta(a, a)
        assert result["delta"]["value"] == pytest.approx(0.0)
        assert result["mcnemar"]["discordant"] == 0
        assert result["mcnemar"]["p_value"] == 1.0

    def test_mismatched_lengths_degrades_gracefully(self):
        result = paired_delta([1, 0], [1])
        assert result["delta"]["value"] is None
