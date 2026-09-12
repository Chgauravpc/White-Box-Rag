"""
Tests for eval/harness.py — the pure `_aggregate()` function only.

The full `run_eval()` execution path calls the real pipeline (real LLM
calls via shared.llm.call_llm, which conftest.py does NOT mock — only
sentence_transformers/spacy are mocked) so it cannot run headless here.
`_aggregate()` is pure aggregation math over synthetic per-query dicts and is
what's actually worth unit-testing.
"""

from eval.harness import _aggregate, classify_error, _normalize_gathered, _load_dataset


def _result(id_, expected=None, retrieved=None, faithfulness_post=1.0, trust_status="Safe",
            abstained=False, claims_total=2, claims_stripped=0, llm_call_count=3,
            latency_ms=None, error=None, expected_abstain=False, relevant_chunk_keys=None,
            error_category=None):
    if error:
        return {
            "id": id_, "query": f"q{id_}", "expected_section_ids": expected or [],
            "error": error, "error_category": error_category,
        }
    return {
        "id": id_,
        "query": f"q{id_}",
        "expected_section_ids": expected or [],
        "relevant_chunk_keys": relevant_chunk_keys or [],
        "expected_abstain": expected_abstain,
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
        "llm_call_count": llm_call_count,
    }


class TestAggregate:
    def test_empty_results(self):
        agg = _aggregate([])
        assert agg["num_queries"] == 0
        assert agg["abstention_rate"] == 0.0

    def test_legacy_section_id_hit_rate_is_gone(self):
        """The old top-level retrieval_hit_rate/retrieval_mrr compared bare
        section ids against canonical chunk keys, so it was structurally 0.0
        whenever ground truth existed. It must not come back — retrieval is
        scored only under accuracy.retrieval now."""
        agg = _aggregate([_result(1, expected=["1.1"], retrieved=["2.1", "1.1"])])
        assert "retrieval_hit_rate" not in agg
        assert "retrieval_mrr" not in agg

    def test_retrieval_unscored_without_graded_labels(self):
        """expected_section_ids alone is not ground truth for retrieval —
        without relevant_chunk_keys the scorer must report "no ground truth",
        not a fabricated zero."""
        agg = _aggregate([_result(1, expected=["1.1"], retrieved=["FSR|2024|1.1|0"])])
        assert agg["num_ground_truth_items"] == 0
        assert agg["accuracy"]["retrieval"]["n"] == 0
        assert agg["accuracy"]["retrieval"]["ndcg_at_5"]["value"] is None

    def test_retrieval_scored_from_graded_chunk_keys(self):
        """With real graded relevant_chunk_keys on canonical chunk keys, the
        retrieval plane scores for real."""
        r = _result(1, expected=[], retrieved=["FSR|2024|1.1|0", "FSR|2024|2.1|0"])
        r["relevant_chunk_keys"] = [{"key": "FSR|2024|1.1|0", "grade": 3}]
        agg = _aggregate([r])
        assert agg["num_ground_truth_items"] == 1
        assert agg["accuracy"]["retrieval"]["n"] == 1
        assert agg["accuracy"]["retrieval"]["hit_at_5"]["value"] == 1.0

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

    def test_mean_llm_calls_per_query(self):
        results = [_result(1, llm_call_count=3), _result(2, llm_call_count=5)]
        agg = _aggregate(results)
        assert agg["mean_llm_calls_per_query"] == 4.0


class TestAccuracyBlock:
    """Integration test for the eval/scoring.py wiring — the block that
    turns expected_abstain from a loaded-but-discarded field into a real
    scored precision/recall/F1 against the actual abstention decision.
    Metric math itself is covered by tests/test_scoring.py.
    """

    def test_abstention_accuracy_is_actually_scored(self):
        results = [
            _result(1, expected_abstain=True, abstained=True),    # correct abstain (TP)
            _result(2, expected_abstain=False, abstained=False),  # correct answer (TN)
            _result(3, expected_abstain=False, abstained=True),   # wrongful abstain (FP)
        ]
        agg = _aggregate(results)
        accuracy = agg["accuracy"]["abstention"]
        assert accuracy["confusion"] == {"tp": 1, "fp": 1, "fn": 0, "tn": 1}
        assert accuracy["precision"]["value"] == 0.5
        assert accuracy["wrongful_abstention_rate"]["value"] == 0.5  # 1 of 2 answerable items

    def test_retrieval_accuracy_uses_relevant_chunk_keys_not_bare_expected_section_ids(self):
        results = [
            _result(1, retrieved=["PUB_A|2024|1.1|0", "PUB_A|2024|2.1|0"],
                    relevant_chunk_keys=[{"key": "PUB_A|2024|1.1|0", "grade": 2}]),
        ]
        agg = _aggregate(results)
        retrieval = agg["accuracy"]["retrieval"]
        assert retrieval["n"] == 1
        assert retrieval["hit_at_1"]["value"] == 1.0

    def test_accuracy_block_present_even_with_no_labels(self):
        """Today's shipped datasets mostly have no relevant_chunk_keys — this
        must report 'no ground truth', not crash or silently omit the block."""
        agg = _aggregate([_result(1), _result(2)])
        assert agg["accuracy"]["retrieval"]["n"] == 0
        assert agg["accuracy"]["abstention"]["n"] == 2


class TestClassifyError:
    """Error taxonomy (finding #8) — turns 'num_errors: 12' into a
    diagnosable breakdown. String-matched against the exception message,
    same style as shared/llm.py's retry classification."""

    def test_rate_limit(self):
        assert classify_error(Exception("429 Too Many Requests")) == "RATE_LIMIT"
        assert classify_error(Exception("Rate limit exceeded")) == "RATE_LIMIT"

    def test_json_parse(self):
        assert classify_error(ValueError("Failed to parse audit JSON: Expecting value")) == "JSON_PARSE"

    def test_empty_corpus(self):
        assert classify_error(Exception("No relevant documents found")) == "EMPTY_CORPUS"

    def test_timeout(self):
        assert classify_error(Exception("Request timed out after 30s")) == "TIMEOUT"

    def test_connection_error_instance(self):
        assert classify_error(ConnectionError("connection reset")) == "LLM_API"

    def test_unrecognized_message_is_unknown(self):
        assert classify_error(Exception("something bizarre happened")) == "UNKNOWN"


class TestNormalizeGathered:
    """asyncio.gather(..., return_exceptions=True) can hand back a raw
    exception instead of _run_one's own error dict (e.g. task cancellation
    outside _run_one's try/except) — _normalize_gathered must produce the
    same shape either way, so _aggregate never has to special-case it."""

    def test_dict_passes_through_unchanged(self):
        item = {"id": "q1", "query": "x"}
        result = {"id": "q1", "faithfulness_post": 0.9}
        assert _normalize_gathered(item, result) is result

    def test_exception_is_normalized_to_an_error_dict(self):
        item = {"id": "q1", "query": "hello", "expected_abstain": True}
        normalized = _normalize_gathered(item, TimeoutError("timed out"))
        assert normalized["id"] == "q1"
        assert normalized["query"] == "hello"
        assert normalized["expected_abstain"] is True
        assert normalized["error"] == "timed out"
        assert normalized["error_category"] == "TIMEOUT"


class TestErrorBreakdown:
    def test_aggregate_reports_error_categories(self):
        results = [
            _result(1),
            _result(2, error="429", error_category="RATE_LIMIT"),
            _result(3, error="bad json", error_category="JSON_PARSE"),
            _result(4, error="429 again", error_category="RATE_LIMIT"),
        ]
        agg = _aggregate(results)
        assert agg["num_errors"] == 3
        assert agg["error_breakdown"] == {"RATE_LIMIT": 2, "JSON_PARSE": 1}

    def test_missing_category_falls_back_to_unknown(self):
        results = [_result(1, error="mystery failure")]
        agg = _aggregate(results)
        assert agg["error_breakdown"] == {"UNKNOWN": 1}


class TestLoadDatasetPerLineGuard:
    """One malformed JSONL line must not kill the whole run before a single
    query executes (finding #8) — the dataset load is guarded per-line."""

    def test_malformed_line_is_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            '{"id": "q1", "query": "a", "expected_abstain": false}\n'
            'not valid json at all\n'
            '{"id": "q2", "query": "b", "expected_abstain": false}\n',
            encoding="utf-8",
        )
        items = _load_dataset(str(path))
        assert [i["id"] for i in items] == ["q1", "q2"]

    def test_all_valid_lines_still_load_normally(self, tmp_path):
        path = tmp_path / "clean.jsonl"
        path.write_text(
            '{"id": "q1", "query": "a", "expected_abstain": false}\n'
            '{"id": "q2", "query": "b", "expected_abstain": true}\n',
            encoding="utf-8",
        )
        items = _load_dataset(str(path))
        assert len(items) == 2

    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text(
            '{"id": "q1", "query": "a", "expected_abstain": false}\n'
            '\n'
            '   \n'
            '{"id": "q2", "query": "b", "expected_abstain": false}\n',
            encoding="utf-8",
        )
        items = _load_dataset(str(path))
        assert len(items) == 2
