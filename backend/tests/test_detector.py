"""
Tests for eval/detector.py — the detector evaluation plane.

The NLI model is stubbed per-test with a deterministic fake rather than the
suite's random one (conftest returns `np.random.rand(n, 3)`), because these
tests are about whether labels, the shipped strip rule and the metric wiring
line up — none of which is assertable against random verdicts.
"""

import json

import numpy as np

from eval import detector
from eval.detector import (
    normalize_label, validate_detector_item, load_detector_dataset,
    run_detector_eval, run_detector_eval_from_file,
)

# Index order of the CrossEncoder's 3-way output, per verify_claims_batch.
CONTRA, ENTAIL, NEUTRAL = 0, 1, 2


def _stub_nli(monkeypatch, triplets):
    """Make the NLI model return `triplets` in order, for premise-bearing pairs."""
    import shared.xai_matrices as xm
    seq = iter(triplets)
    monkeypatch.setattr(
        xm._nli, "predict",
        lambda pairs, **kw: np.array([next(seq) for _ in pairs], dtype="float32"),
    )


def _entail(p=0.95):
    t = [0.0, 0.0, 0.0]; t[ENTAIL] = p; t[NEUTRAL] = 1 - p
    return t


def _contra(p=0.95):
    t = [0.0, 0.0, 0.0]; t[CONTRA] = p; t[ENTAIL] = 1 - p
    return t


def _item(id_, claim, premise, label):
    return {"id": id_, "claim": claim, "premise": premise, "label": label}


class TestLabelNormalization:
    def test_fever_spellings(self):
        assert normalize_label("SUPPORTS") == "SUPPORTED"
        assert normalize_label("REFUTES") == "REFUTED"
        assert normalize_label("NOT ENOUGH INFO") == "NEI"

    def test_case_and_whitespace_insensitive(self):
        assert normalize_label("  refutes  ") == "REFUTED"

    def test_unknown_label_is_empty_not_guessed(self):
        assert normalize_label("MAYBE") == ""
        assert normalize_label(None) == ""


class TestValidation:
    def test_valid_item_has_no_errors(self):
        assert validate_detector_item(_item("a", "c", "p", "SUPPORTS")) == []

    def test_missing_fields_reported(self):
        errs = validate_detector_item({"claim": "c"})
        assert any("id" in e for e in errs) and any("premise" in e for e in errs)

    def test_unrecognized_label_rejected_rather_than_coerced(self):
        errs = validate_detector_item(_item("a", "c", "p", "PROBABLY"))
        assert any("unrecognized label" in e for e in errs)

    def test_empty_claim_rejected(self):
        assert any("claim is empty" in e for e in validate_detector_item(_item("a", "  ", "p", "SUPPORTS")))


class TestRunDetectorEval:
    def test_perfect_detector_scores_perfectly(self, monkeypatch):
        """Supported claim entailed and kept; refuted claim contradicted and
        flagged — precision and recall should both be 1.0."""
        _stub_nli(monkeypatch, [_entail(), _contra()])
        out = run_detector_eval([
            _item("1", "Reserves rose.", "Reserves increased sharply.", "SUPPORTS"),
            _item("2", "Reserves fell.", "Reserves increased sharply.", "REFUTES"),
        ])
        assert out["n"] == 2
        assert out["metrics"]["precision"]["value"] == 1.0
        assert out["metrics"]["recall"]["value"] == 1.0

    def test_missed_hallucination_lowers_recall(self, monkeypatch):
        """A refuted claim the model entails is a false negative — exactly the
        failure the detector plane exists to surface."""
        _stub_nli(monkeypatch, [_entail(), _entail()])
        out = run_detector_eval([
            _item("1", "Reserves rose.", "Reserves increased.", "SUPPORTS"),
            _item("2", "Reserves fell.", "Reserves increased.", "REFUTES"),
        ])
        assert out["metrics"]["recall"]["value"] == 0.0
        assert out["per_item"][1]["correct"] is False

    def test_nei_counts_as_hallucinated(self):
        """NEI means the evidence is silent; asserting it anyway is exactly what
        the system must not do, so it belongs in the positive class."""
        out = run_detector_eval([_item("1", "Unrelated claim.", "", "NOT ENOUGH INFO")])
        assert out["per_item"][0]["label"] == "NEI"
        # empty premise -> guard fires -> flagged, which is correct here
        assert out["per_item"][0]["evidence_status"] == "no_premise"
        assert out["per_item"][0]["correct"] is True

    def test_recall_broken_down_by_label(self, monkeypatch):
        """One aggregate number hides whether the system catches contradictions
        but misses unsupported claims."""
        _stub_nli(monkeypatch, [_contra(), _entail()])
        out = run_detector_eval([
            _item("1", "a", "p", "REFUTES"),
            _item("2", "b", "p", "NEI"),
        ])
        by_type = out["metrics"]["recall_by_error_type"]
        assert by_type["REFUTED"]["value"] == 1.0
        assert by_type["NEI"]["value"] == 0.0

    def test_verdict_confusion_separates_model_from_decision_rule(self, monkeypatch):
        _stub_nli(monkeypatch, [_contra()])
        out = run_detector_eval([_item("1", "a", "p", "REFUTES")])
        assert out["verdict_confusion"]["REFUTED"]["CONTRADICTION"] == 1

    def test_premise_is_not_normalized_by_default(self):
        """External benchmark evidence must reach the model untouched, or the
        number is not reproducible by anyone else."""
        out = run_detector_eval([_item("1", "a", "p", "SUPPORTS")])
        assert out["premise_normalizer"] == "none"

    def test_empty_dataset_reports_no_items_not_a_fake_score(self):
        out = run_detector_eval([])
        assert out["n"] == 0
        assert out["metrics"]["precision"]["value"] is None

    def test_auroc_needs_both_classes(self, monkeypatch):
        _stub_nli(monkeypatch, [_entail(), _entail()])
        out = run_detector_eval([
            _item("1", "a", "p", "SUPPORTS"),
            _item("2", "b", "p", "SUPPORTS"),
        ])
        assert out["metrics"]["auroc"]["value"] is None

    def test_auroc_computed_when_both_classes_present(self, monkeypatch):
        _stub_nli(monkeypatch, [_entail(0.99), _contra(0.99)])
        out = run_detector_eval([
            _item("1", "a", "p", "SUPPORTS"),
            _item("2", "b", "p", "REFUTES"),
        ])
        assert out["metrics"]["auroc"]["value"] == 1.0


class TestDatasetLoading:
    def _write(self, tmp_path, lines):
        p = tmp_path / "detector.jsonl"
        p.write_text("\n".join(lines), encoding="utf-8")
        return str(p)

    def test_loads_valid_items(self, tmp_path):
        path = self._write(tmp_path, [json.dumps(_item("1", "c", "p", "SUPPORTS"))])
        items, errors = load_detector_dataset(path)
        assert len(items) == 1 and errors == {}

    def test_malformed_line_skipped_not_fatal(self, tmp_path):
        """One bad line must not kill a run before a single item is scored."""
        path = self._write(tmp_path, [
            json.dumps(_item("1", "c", "p", "SUPPORTS")),
            "{not json",
            json.dumps(_item("2", "c", "p", "REFUTES")),
        ])
        items, errors = load_detector_dataset(path)
        assert len(items) == 2
        assert "<line 2>" in errors

    def test_invalid_item_reported_and_excluded(self, tmp_path):
        path = self._write(tmp_path, [json.dumps(_item("bad", "c", "p", "WHO KNOWS"))])
        items, errors = load_detector_dataset(path)
        assert items == [] and "bad" in errors

    def test_run_from_file_reports_skipped(self, tmp_path):
        path = self._write(tmp_path, [
            json.dumps(_item("1", "c", "p", "SUPPORTS")),
            json.dumps(_item("bad", "c", "p", "NONSENSE")),
        ])
        out = run_detector_eval_from_file(path)
        assert out["n"] == 1
        assert "bad" in out["skipped_items"]


class TestShippedRuleIsWhatIsMeasured:
    def test_prediction_comes_from_mitigation_should_strip(self, monkeypatch):
        """Guards against someone reimplementing 'flagged' here and measuring a
        detector the repo does not actually ship."""
        called = []
        real = detector._should_strip
        monkeypatch.setattr(
            detector, "_should_strip",
            lambda v: (called.append(v.claim_text), real(v))[1],
        )
        run_detector_eval([_item("1", "a claim", "a premise", "SUPPORTS")])
        assert called == ["a claim"]


class TestShippedSmokeDataset:
    """The dataset that ships with the repo must stay loadable and balanced —
    it is what `POST /api/eval/detector` runs by default."""

    def _path(self):
        import os
        import eval.detector as d
        return os.path.join(os.path.dirname(os.path.dirname(d.__file__)), "eval", "detector_smoke.jsonl")

    def test_ships_and_every_item_validates(self):
        items, errors = load_detector_dataset(self._path())
        assert errors == {}, f"shipped smoke dataset has invalid items: {errors}"
        assert len(items) >= 10

    def test_all_three_labels_present(self):
        """A detector dataset with only one class makes AUROC undefined and
        precision/recall degenerate."""
        items, _ = load_detector_dataset(self._path())
        labels = {normalize_label(i["label"]) for i in items}
        assert labels == {"SUPPORTED", "REFUTED", "NEI"}

    def test_ids_are_unique(self):
        items, _ = load_detector_dataset(self._path())
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids))
