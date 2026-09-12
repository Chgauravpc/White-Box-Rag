"""
Tests for shared/database.py's eval-run provenance/durability helpers (W1.2):
insert_eval_run_started / finalize_eval_run and the eval_items table.
"""

from shared.database import (
    insert_eval_run_started,
    finalize_eval_run,
    list_eval_runs,
    get_eval_run,
    insert_eval_item,
    list_eval_items,
    get_completed_item_ids,
)


class TestEvalRunLifecycle:
    def test_started_run_has_running_status(self):
        run_id = insert_eval_run_started(
            run_label="test", dataset_path="x.jsonl", started_at="2026-01-01T00:00:00+00:00",
        )
        run = get_eval_run(run_id)
        assert run["status"] == "running"
        assert run["finished_at"] is None

    def test_provenance_fields_round_trip(self):
        run_id = insert_eval_run_started(
            run_label="prov-test", dataset_path="x.jsonl", started_at="2026-01-01T00:00:00+00:00",
            run_config_json='{"llm_provider": "groq"}', git_commit="abc123",
            corpus_manifest_json="[]", dataset_hash="deadbeef",
            active_calibration_json=None, model_identities_json='{"nli_model": "x"}',
            eval_mode=True,
        )
        run = get_eval_run(run_id)
        assert run["git_commit"] == "abc123"
        assert run["dataset_hash"] == "deadbeef"
        assert run["run_config_json"] == '{"llm_provider": "groq"}'
        assert run["eval_mode"] == 1

    def test_finalize_complete(self):
        run_id = insert_eval_run_started(run_label="", dataset_path="x.jsonl", started_at="2026-01-01T00:00:00+00:00")
        finalize_eval_run(run_id, status="complete", finished_at="2026-01-01T00:05:00+00:00",
                           num_queries=3, metrics_json='{"num_queries": 3}', per_query_json="[]")
        run = get_eval_run(run_id)
        assert run["status"] == "complete"
        assert run["num_queries"] == 3
        assert run["finished_at"] == "2026-01-01T00:05:00+00:00"

    def test_finalize_failed_records_error_not_metrics(self):
        """A crashed run must leave a diagnosable row, not silently vanish."""
        run_id = insert_eval_run_started(run_label="", dataset_path="x.jsonl", started_at="2026-01-01T00:00:00+00:00")
        finalize_eval_run(run_id, status="failed", finished_at="2026-01-01T00:01:00+00:00", error="pipeline blew up")
        run = get_eval_run(run_id)
        assert run["status"] == "failed"
        assert run["error"] == "pipeline blew up"

    def test_list_eval_runs_includes_status_and_git_commit(self):
        run_id = insert_eval_run_started(
            run_label="list-test", dataset_path="x.jsonl", started_at="2026-01-01T00:00:00+00:00",
            git_commit="commit1",
        )
        runs = list_eval_runs()
        this_run = next(r for r in runs if r["id"] == run_id)
        assert this_run["status"] == "running"
        assert this_run["git_commit"] == "commit1"


class TestEvalItemsDurability:
    def test_items_persist_independently_of_run_finalization(self):
        """The whole point of eval_items: durable per-item writes that don't
        depend on the run ever reaching its final aggregate step."""
        run_id = insert_eval_run_started(run_label="", dataset_path="x.jsonl", started_at="2026-01-01T00:00:00+00:00")
        insert_eval_item(run_id, "q1", "ok", None, None, '{"id": "q1", "faithfulness_post": 0.9}')
        insert_eval_item(run_id, "q2", "error", "RATE_LIMIT", "429 Too Many Requests", '{"id": "q2", "error": "429"}')
        # Run is never finalized — simulating a crash — but items still exist.
        items = list_eval_items(run_id)
        assert len(items) == 2
        statuses = {i["item_id"]: i["status"] for i in items}
        assert statuses == {"q1": "ok", "q2": "error"}

    def test_get_completed_item_ids_for_resume(self):
        run_id = insert_eval_run_started(run_label="", dataset_path="x.jsonl", started_at="2026-01-01T00:00:00+00:00")
        insert_eval_item(run_id, "q1", "ok", None, None, "{}")
        insert_eval_item(run_id, "q2", "ok", None, None, "{}")
        completed = get_completed_item_ids(run_id)
        assert completed == {"q1", "q2"}

    def test_completed_ids_are_scoped_to_their_own_run(self):
        run_a = insert_eval_run_started(run_label="", dataset_path="x.jsonl", started_at="2026-01-01T00:00:00+00:00")
        run_b = insert_eval_run_started(run_label="", dataset_path="x.jsonl", started_at="2026-01-01T00:00:00+00:00")
        insert_eval_item(run_a, "q1", "ok", None, None, "{}")
        assert get_completed_item_ids(run_a) == {"q1"}
        assert get_completed_item_ids(run_b) == set()

    def test_error_detail_is_queryable(self):
        run_id = insert_eval_run_started(run_label="", dataset_path="x.jsonl", started_at="2026-01-01T00:00:00+00:00")
        insert_eval_item(run_id, "q1", "error", "JSON_PARSE", "Expecting value: line 1 column 1", "{}")
        items = list_eval_items(run_id)
        assert items[0]["error_category"] == "JSON_PARSE"
        assert "Expecting value" in items[0]["error_detail"]
