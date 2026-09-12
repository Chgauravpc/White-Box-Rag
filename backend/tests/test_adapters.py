"""
Tests for eval/adapters.py — external benchmark → detector items.

The property worth the most here is the FEVER NEI trap. FEVER's NOT ENOUGH
INFO claims carry no gold evidence by construction, so converting them into
items with an empty premise produces claims that the empty-premise guard
flags automatically — 100% recall on the NEI class, measuring nothing, with a
number that looks like a result. These tests pin that such rows are dropped
by default and counted in the report.
"""

import json

import pytest

from eval.adapters import (
    dataset_validity_report, from_fever, from_generic, from_halueval,
    write_detector_dataset,
)
from eval.detector import run_detector_eval, validate_detector_item


class TestHaluEval:
    def _rows(self):
        return [{
            "id": "q1",
            "knowledge": "The Eiffel Tower is located in Paris, France.",
            "question": "Where is the Eiffel Tower?",
            "right_answer": "The Eiffel Tower is in Paris.",
            "hallucinated_answer": "The Eiffel Tower is in Berlin.",
        }]

    def test_each_row_yields_a_supported_and_a_refuted_item(self):
        """The pairing controls for premise difficulty: both claims are scored
        against the same evidence, so a score difference is attributable to
        the claim."""
        items, report = from_halueval(self._rows())
        assert report["n_items"] == 2
        assert sorted(i["label"] for i in items) == ["REFUTED", "SUPPORTED"]
        assert len({i["premise"] for i in items}) == 1

    def test_items_are_valid_detector_items(self):
        items, _ = from_halueval(self._rows())
        assert all(validate_detector_item(i) == [] for i in items)

    def test_ids_are_unique_and_traceable(self):
        items, _ = from_halueval(self._rows())
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("halueval-qa-q1") for i in ids)

    def test_row_without_premise_is_skipped_and_counted(self):
        items, report = from_halueval([{"id": "x", "knowledge": "   ",
                                        "right_answer": "a", "hallucinated_answer": "b"}])
        assert items == [] and report["skipped"]["no_premise"] == 1

    def test_summarization_task_uses_document_field(self):
        items, _ = from_halueval(
            [{"id": "s1", "document": "A long source document.",
              "right_summary": "A summary.", "hallucinated_summary": "A false summary."}],
            task="summarization",
        )
        assert len(items) == 2 and items[0]["premise"] == "A long source document."

    def test_unknown_task_raises_rather_than_guessing(self):
        with pytest.raises(ValueError):
            from_halueval([], task="not-a-task")


class TestFever:
    WIKI = {("Paris", 0): "Paris is the capital of France."}

    def _resolve(self, page, sent_id):
        return self.WIKI.get((page, sent_id), "")

    def _row(self, label, evidence, claim="Paris is in France.", id_=1):
        return {"id": id_, "label": label, "claim": claim, "evidence": evidence}

    def test_supports_row_gets_resolved_evidence_as_premise(self):
        items, report = from_fever(
            [self._row("SUPPORTS", [[[1, 1, "Paris", 0]]])], resolve_evidence=self._resolve
        )
        assert report["n_items"] == 1
        assert items[0]["premise"] == "Paris is the capital of France."
        assert items[0]["label"] == "SUPPORTS"

    def test_nei_without_evidence_is_skipped_by_default(self):
        """The trap: an empty premise would be auto-flagged by the
        empty-premise guard, scoring as a correct catch while measuring
        nothing."""
        items, report = from_fever(
            [self._row("NOT ENOUGH INFO", [[[1, 1, None, None]]])],
            resolve_evidence=self._resolve,
        )
        assert items == []
        assert report["skipped_nei_without_evidence"] == 1

    def test_nei_can_be_kept_deliberately_and_is_marked(self):
        items, report = from_fever(
            [self._row("NOT ENOUGH INFO", [[[1, 1, None, None]]])],
            resolve_evidence=self._resolve, keep_unresolvable_nei=True,
        )
        assert report["kept_nei_without_evidence"] == 1
        assert items[0]["premise_source"] == "none"

    def test_skipping_nei_prevents_a_fake_perfect_recall(self):
        """End-to-end demonstration of why the default matters: kept NEI rows
        are 'caught' without the model ever running."""
        rows = [self._row("NOT ENOUGH INFO", [[[1, 1, None, None]]], id_=i) for i in range(3)]
        kept, _ = from_fever(rows, resolve_evidence=self._resolve, keep_unresolvable_nei=True)
        out = run_detector_eval(kept)
        assert out["metrics"]["recall"]["value"] == 1.0
        assert all(i["evidence_status"] == "no_premise" for i in out["per_item"])
        # ...which is exactly the number the default refuses to produce.
        skipped, _ = from_fever(rows, resolve_evidence=self._resolve)
        assert skipped == []

    def test_supports_row_with_unresolvable_evidence_is_dropped(self):
        """A SUPPORTS label asserts a relationship to text we do not have."""
        items, report = from_fever(
            [self._row("SUPPORTS", [[[1, 1, "MissingPage", 3]]])], resolve_evidence=self._resolve
        )
        assert items == [] and report["skipped_unresolvable_evidence"] == 1

    def test_multi_sentence_evidence_is_joined(self):
        wiki = {("A", 0): "First sentence.", ("B", 1): "Second sentence."}
        items, _ = from_fever(
            [self._row("SUPPORTS", [[[1, 1, "A", 0], [1, 2, "B", 1]]])],
            resolve_evidence=lambda p, s: wiki.get((p, s), ""),
        )
        assert items[0]["premise"] == "First sentence. Second sentence."

    def test_duplicate_evidence_pointers_are_deduplicated(self):
        items, _ = from_fever(
            [self._row("SUPPORTS", [[[1, 1, "Paris", 0]], [[2, 2, "Paris", 0]]])],
            resolve_evidence=self._resolve,
        )
        assert items[0]["premise"] == "Paris is the capital of France."

    def test_bad_label_is_counted_not_coerced(self):
        items, report = from_fever(
            [self._row("MAYBE", [[[1, 1, "Paris", 0]]])], resolve_evidence=self._resolve
        )
        assert items == [] and report["skipped_bad_label"] == 1

    def test_evidence_lookup_failure_does_not_kill_the_conversion(self):
        def _boom(page, sent_id):
            raise RuntimeError("wiki dump unavailable")

        items, report = from_fever(
            [self._row("SUPPORTS", [[[1, 1, "Paris", 0]]])], resolve_evidence=_boom
        )
        assert items == [] and report["skipped_unresolvable_evidence"] == 1


class TestGeneric:
    def test_field_mapping(self):
        items, report = from_generic(
            [{"c": "a claim", "p": "a premise", "l": "SUPPORTS", "key": "r1"}],
            claim_field="c", premise_field="p", label_field="l", id_field="key",
        )
        assert report["n_items"] == 1
        assert items[0]["id"] == "generic-r1" and items[0]["claim"] == "a claim"

    def test_unknown_label_is_skipped_not_coerced(self):
        items, report = from_generic(
            [{"c": "x", "p": "y", "l": "PROBABLY"}],
            claim_field="c", premise_field="p", label_field="l",
        )
        assert items == [] and report["skipped"]["unknown_label"] == 1


class TestWriteDataset:
    def test_round_trips_into_the_detector(self, tmp_path):
        items, _ = from_halueval([{
            "id": "q1", "knowledge": "The sky is blue.",
            "right_answer": "The sky is blue.", "hallucinated_answer": "The sky is green.",
        }])
        path = str(tmp_path / "out.jsonl")
        write_detector_dataset(items, path)
        loaded = [json.loads(line) for line in open(path, encoding="utf-8")]
        assert len(loaded) == 2
        from eval.detector import load_detector_dataset
        reloaded, errors = load_detector_dataset(path)
        assert errors == {} and len(reloaded) == 2

    def test_refuses_to_write_invalid_items(self, tmp_path):
        with pytest.raises(ValueError):
            write_detector_dataset([{"id": "x", "claim": "", "premise": "p", "label": "SUPPORTS"}],
                                   str(tmp_path / "bad.jsonl"))


class TestValidityReport:
    """Guards against shortcuts that make a score describe the dataset's shape
    rather than the detector. Written after HaluEval QA's supported class came
    out at a median of 2 words against the refuted class's 10 — separable by
    length alone, with no reference to the evidence."""

    def _items(self, supported_claims, refuted_claims):
        out = []
        for i, c in enumerate(supported_claims):
            out.append({"id": f"s{i}", "claim": c, "premise": "some evidence", "label": "SUPPORTED"})
        for i, c in enumerate(refuted_claims):
            out.append({"id": f"r{i}", "claim": c, "premise": "some evidence", "label": "REFUTED"})
        return out

    def test_balanced_sentence_lengths_pass(self):
        items = self._items(
            ["The revenue rose by four percent last year."] * 5,
            ["The revenue fell by nine percent last year."] * 5,
        )
        assert dataset_validity_report(items)["warnings"] == []

    def test_length_shortcut_is_flagged(self):
        """The real HaluEval QA failure: 2-word answers vs 10-word sentences."""
        items = self._items(
            ["Sidney Lumet"] * 5,
            ["The film was directed by somebody else entirely, not him."] * 5,
        )
        warnings = dataset_validity_report(items)["warnings"]
        assert any("predictable from length alone" in w for w in warnings)

    def test_entity_fragments_are_flagged_specifically(self):
        """An NLI detector cannot entail a fragment; that deserves its own
        warning, not just the length one."""
        items = self._items(["Sidney Lumet"] * 5, ["Lake Placid"] * 5)
        warnings = dataset_validity_report(items)["warnings"]
        assert any("not propositions" in w for w in warnings)

    def test_single_class_is_flagged(self):
        items = self._items(["A full sentence about something."] * 4, [])
        assert any("fewer than two label classes" in w
                   for w in dataset_validity_report(items)["warnings"])

    def test_reports_per_label_statistics(self):
        items = self._items(["one two"] * 3, ["one two three four five six"] * 3)
        stats = dataset_validity_report(items)["claim_length_by_label"]
        assert stats["SUPPORTED"]["median_words"] == 2
        assert stats["REFUTED"]["median_words"] == 6
        assert stats["SUPPORTED"]["n"] == 3
