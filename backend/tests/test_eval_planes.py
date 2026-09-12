"""
Tests for plane-aware persistence (Phase 4).

The three evaluation planes report incompatible metric shapes — end-to-end has
faithfulness and trust statuses, detector has AUROC over claim labels,
retrieval has nDCG over chunk labels. Persisting them into one untagged table
would make incomparable runs indistinguishable rows, and would let
`resume_from_run_id` continue one plane's run under another plane's harness,
merging two different measurements into a single aggregate that describes
neither.
"""

import asyncio
import json

import pytest

from eval.routes import (
    DetectorEvalRequest, RetrievalEvalRequest,
    trigger_detector_eval, trigger_retrieval_eval,
)
from shared.database import get_eval_run, insert_eval_run_started, list_eval_runs


class TestPlaneColumn:
    def test_default_plane_is_end_to_end(self):
        run_id = insert_eval_run_started("t", "d.jsonl", "2026-01-01T00:00:00+00:00")
        assert get_eval_run(run_id)["plane"] == "end_to_end"

    def test_plane_is_recorded(self):
        run_id = insert_eval_run_started(
            "t", "d.jsonl", "2026-01-01T00:00:00+00:00", plane="detector"
        )
        assert get_eval_run(run_id)["plane"] == "detector"

    def test_runs_are_distinguishable_in_the_listing(self):
        insert_eval_run_started("a", "d", "2026-01-01T00:00:00+00:00", plane="detector")
        insert_eval_run_started("b", "d", "2026-01-01T00:00:00+00:00", plane="retrieval")
        planes = {r["plane"] for r in list_eval_runs()}
        assert {"detector", "retrieval"} <= planes


class TestResumeGuard:
    def test_cannot_resume_a_detector_run_as_end_to_end(self, tmp_path):
        """The bug this prevents: merging a detector run's claim-level results
        into an end-to-end aggregate."""
        from eval.harness import run_eval

        run_id = insert_eval_run_started(
            "det", "d.jsonl", "2026-01-01T00:00:00+00:00", plane="detector"
        )
        dataset = tmp_path / "ds.jsonl"
        dataset.write_text(
            json.dumps({"id": "q1", "query": "q", "expected_abstain": False}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="detector"):
            asyncio.run(run_eval(str(dataset), resume_from_run_id=run_id))

    def test_resuming_a_missing_run_is_a_clear_error(self, tmp_path):
        from eval.harness import run_eval

        dataset = tmp_path / "ds.jsonl"
        dataset.write_text(
            json.dumps({"id": "q1", "query": "q", "expected_abstain": False}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="no such run"):
            asyncio.run(run_eval(str(dataset), resume_from_run_id=999999))


class TestPlaneRunsArePersisted:
    def test_detector_run_is_recorded_with_its_plane(self):
        result = asyncio.run(trigger_detector_eval(DetectorEvalRequest()))
        assert result["plane"] == "detector"
        row = get_eval_run(result["run_id"])
        assert row["plane"] == "detector"
        assert row["status"] == "complete"
        assert row["num_queries"] == result["n"]

    def test_detector_metrics_are_persisted_without_the_per_item_blob(self):
        """metrics_json is the summary; per-item detail belongs in its own
        column, not duplicated into the metrics payload."""
        result = asyncio.run(trigger_detector_eval(DetectorEvalRequest()))
        row = get_eval_run(result["run_id"])
        metrics = json.loads(row["metrics_json"])
        assert "metrics" in metrics and "per_item" not in metrics

    def test_failed_plane_run_leaves_a_diagnosable_row(self):
        """A crash must leave a 'failed' row with the error, not silence."""
        with pytest.raises(Exception):
            asyncio.run(trigger_detector_eval(
                DetectorEvalRequest(dataset_path="does-not-exist.jsonl")
            ))
        # A missing dataset is rejected before a row is created; a failure
        # DURING the run is the case that must be recorded.
        runs_before = len(list_eval_runs())
        with pytest.raises(Exception):
            asyncio.run(trigger_retrieval_eval(
                RetrievalEvalRequest(corpus_id="no-such-frozen-corpus")
            ))
        runs = list_eval_runs()
        assert len(runs) == runs_before + 1
        failed = runs[0]
        assert failed["plane"] == "retrieval"
        assert failed["status"] == "failed"
        assert "no-such-frozen-corpus" in (failed["error"] or "")
