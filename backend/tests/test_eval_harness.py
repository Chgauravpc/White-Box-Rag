"""
Tests for eval/harness.py — the pure `_aggregate()` function only.

The full `run_eval()` execution path calls the real pipeline (real Gemini
calls via shared.gemini.call_gemini, which conftest.py does NOT mock — only
sentence_transformers/spacy are mocked) so it cannot run headless here.
`_aggregate()` is pure aggregation math over synthetic per-query dicts and is
what's actually worth unit-testing.
"""

from eval.harness import _aggregate


def _result(id_, expected=None, retrieved=None, faithfulness_post=1.0, trust_status="Safe",
            abstained=False, claims_total=2, claims_stripped=0, gemini_call_count=3,
            latency_ms=None, error=None):
    if error:
        return {"id": id_, "query": f"q{id_}", "expected_section_ids": expected or [], "error": error}
    return {
        "id": id_,
        "query": f"q{id_}",
        "expected_section_ids": expected or [],
        "retrieved_chunk_ids": retrieved or [],
        "faithfulness_raw": faithfulness_post,
        "faithfulness_post": faithfulness_post,
        "context_relevance": 0.8,
        "context_diversity": 0.7,
        "citation_precision": 0.9,
        "answer_relevancy": 0.75,
        "context_utilization": 0.6,
        "paraphrase_stability": 1.0,
        "trust_status": trust_status,
        "abstained": abstained,
        "claims_total": claims_total,
        "claims_stripped": claims_stripped,
        "latency_ms": latency_ms or {"retrieval_ms": 10.0, "generation_ms": 500.0},
        "gemini_call_count": gemini_call_count,
    }


class TestAggregate:
    def test_empty_results(self):
        agg = _aggregate([])
        assert agg["num_queries"] == 0
        assert agg["retrieval_hit_rate"] is None
        assert agg["abstention_rate"] == 0.0

    def test_hit_rate_and_mrr_computed_only_over_ground_truth_items(self):
        results = [
            _result(1, expected=["1.1"], retrieved=["2.1", "1.1", "3.1"]),  # rank 2 -> RR 0.5
            _result(2, expected=["4.4"], retrieved=["4.4"]),  # rank 1 -> RR 1.0
            _result(3, expected=[], retrieved=["9.9"]),  # no ground truth — excluded
            _result(4, expected=["5.5"], retrieved=["1.1", "2.2"]),  # not found -> RR 0.0
        ]
        agg = _aggregate(results)

        assert agg["num_ground_truth_items"] == 3
        assert agg["retrieval_hit_rate"] == round(2 / 3, 6)
        assert agg["retrieval_mrr"] == round((0.5 + 1.0 + 0.0) / 3, 6)

    def test_no_ground_truth_items_gives_none_not_zero(self):
        results = [_result(1, expected=[]), _result(2, expected=[])]
        agg = _aggregate(results)
        assert agg["retrieval_hit_rate"] is None
        assert agg["retrieval_mrr"] is None

    def test_trust_status_distribution(self):
        results = [
            _result(1, trust_status="Safe"),
            _result(2, trust_status="Safe"),
            _result(3, trust_status="Needs_Human_Review"),
            _result(4, trust_status="Non_Compliant"),
        ]
        agg = _aggregate(results)
        assert agg["trust_status_distribution"] == {"Safe": 2, "Needs_Human_Review": 1, "Non_Compliant": 1}

    def test_abstention_rate(self):
        results = [
            _result(1, abstained=True),
            _result(2, abstained=False),
            _result(3, abstained=False),
            _result(4, abstained=False),
        ]
        agg = _aggregate(results)
        assert agg["abstention_rate"] == 0.25

    def test_mean_claims_stripped_ratio(self):
        results = [
            _result(1, claims_total=4, claims_stripped=2),  # 0.5
            _result(2, claims_total=2, claims_stripped=0),  # 0.0
            _result(3, claims_total=0, claims_stripped=0),  # excluded (no claims)
        ]
        agg = _aggregate(results)
        assert agg["mean_claims_stripped_ratio"] == round((0.5 + 0.0) / 2, 6)

    def test_errors_excluded_from_metric_means_but_counted(self):
        results = [
            _result(1, faithfulness_post=0.9),
            _result(2, error="pipeline blew up"),
        ]
        agg = _aggregate(results)
        assert agg["num_queries"] == 2
        assert agg["num_errors"] == 1
        assert agg["mean_faithfulness_post"] == 0.9

    def test_latency_summary_mean_and_p95(self):
        results = [
            _result(1, latency_ms={"generation_ms": 100.0}),
            _result(2, latency_ms={"generation_ms": 200.0}),
            _result(3, latency_ms={"generation_ms": 300.0}),
        ]
        agg = _aggregate(results)
        assert agg["latency_summary"]["generation_ms"]["mean_ms"] == 200.0
        assert agg["latency_summary"]["generation_ms"]["p95_ms"] in (300.0, 290.0)  # small-n p95 approximation

    def test_mean_gemini_calls_per_query(self):
        results = [_result(1, gemini_call_count=3), _result(2, gemini_call_count=5)]
        agg = _aggregate(results)
        assert agg["mean_gemini_calls_per_query"] == 4.0
