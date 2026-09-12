"""
Tests for eval/retrieval.py — the retrieval evaluation plane.

Two properties matter most here and both are easy to get silently wrong:

1. **Identity namespace.** Retrieved ids and labeled ids must be the same kind
   of string. The repo has already shipped one metric that compared bare
   section ids against canonical chunk keys and was therefore pinned at 0.0
   forever; these tests assert real hits rather than just "a number came out".

2. **Unlabeled items are excluded, not scored as misses.** An item with no
   ground truth cannot speak to retrieval quality; averaging it in as a zero
   would understate the system in proportion to how incomplete the labels are.

`retrieve_fn` is injected, so none of this needs a populated ChromaDB.
"""

import json

from eval.retrieval import (
    load_retrieval_dataset, run_retrieval_eval, run_retrieval_eval_from_file,
)
from shared.corpus_manifest import build_manifest
from shared.hashing import hash_text


class FakeChunk:
    """Shaped like ChunkMetadata for `_chunk_key_of`."""

    def __init__(self, key):
        pub, ed, sec, _ord = key.split("|")
        self.publication_name, self.edition_date, self.section_id = pub, ed, sec
        self.chunk_text, self.chunk_key = f"text of {key}", key


def _retriever(*result_keys):
    """A retriever that always returns these chunk keys, in order."""
    return lambda query, top_k=None: [FakeChunk(k) for k in result_keys]


def _item(id_, query, relevant):
    """relevant: list of (key, grade) or (key, grade, text)."""
    keys = []
    for entry in relevant:
        key, grade = entry[0], entry[1]
        rel = {"key": key, "grade": grade}
        if len(entry) > 2:
            rel["text_sha256"] = hash_text(entry[2])
        keys.append(rel)
    return {"id": id_, "query": query, "expected_abstain": False,
            "schema_version": 2, "relevant_chunk_keys": keys}


class TestScoring:
    def test_relevant_chunk_at_rank_one_is_a_hit(self):
        out = run_retrieval_eval(
            [_item("q1", "anything", [("PUB|2024|1.1|0", 3)])],
            top_k=5, retrieve_fn=_retriever("PUB|2024|1.1|0", "PUB|2024|2.1|0"),
        )
        assert out["n_scored"] == 1
        assert out["metrics"]["hit_at_5"]["value"] == 1.0
        assert out["per_item"][0]["hit"] is True

    def test_complete_miss_scores_zero_not_none(self):
        out = run_retrieval_eval(
            [_item("q1", "anything", [("PUB|2024|9.9|0", 3)])],
            top_k=5, retrieve_fn=_retriever("PUB|2024|1.1|0"),
        )
        assert out["metrics"]["hit_at_5"]["value"] == 0.0
        assert out["per_item"][0]["hit"] is False

    def test_rank_affects_ndcg(self):
        """A relevant chunk buried at rank 3 must score below the same chunk at
        rank 1 — otherwise ranking quality isn't being measured at all."""
        first = run_retrieval_eval(
            [_item("q1", "q", [("PUB|2024|1.1|0", 3)])],
            top_k=5, retrieve_fn=_retriever("PUB|2024|1.1|0", "x|1|a|0", "y|1|b|0"),
        )
        third = run_retrieval_eval(
            [_item("q1", "q", [("PUB|2024|1.1|0", 3)])],
            top_k=5, retrieve_fn=_retriever("x|1|a|0", "y|1|b|0", "PUB|2024|1.1|0"),
        )
        assert first["metrics"]["ndcg_at_5"]["value"] > third["metrics"]["ndcg_at_5"]["value"]

    def test_unlabeled_items_are_excluded_not_counted_as_misses(self):
        """The bug this guards: averaging an unlabeled item in as a zero
        understates retrieval in proportion to how incomplete the labels are."""
        items = [
            _item("labeled", "q", [("PUB|2024|1.1|0", 3)]),
            {"id": "unlabeled", "query": "q", "expected_abstain": False,
             "schema_version": 2, "relevant_chunk_keys": []},
        ]
        out = run_retrieval_eval(items, top_k=5, retrieve_fn=_retriever("PUB|2024|1.1|0"))
        assert out["n_scored"] == 1 and out["n_unlabeled"] == 1
        assert out["metrics"]["hit_at_5"]["value"] == 1.0

    def test_no_labels_at_all_reports_no_ground_truth(self):
        out = run_retrieval_eval(
            [{"id": "a", "query": "q", "expected_abstain": False,
              "schema_version": 2, "relevant_chunk_keys": []}],
            top_k=5, retrieve_fn=_retriever("PUB|2024|1.1|0"),
        )
        assert out["n_scored"] == 0
        assert out["metrics"]["ndcg_at_5"]["value"] is None

    def test_no_llm_is_involved(self, monkeypatch):
        """The entire point of the plane: scoring retrieval must not cost a
        generation call."""
        import shared.llm as llm

        def _boom(*a, **kw):
            raise AssertionError("retrieval plane must not call the LLM")

        monkeypatch.setattr(llm, "call_llm", _boom)
        monkeypatch.setattr(llm, "call_llm_meta", _boom)
        out = run_retrieval_eval(
            [_item("q1", "q", [("PUB|2024|1.1|0", 3)])],
            top_k=5, retrieve_fn=_retriever("PUB|2024|1.1|0"),
        )
        assert out["n_scored"] == 1


class FakeCollection:
    def __init__(self, rows):
        self._rows = rows

    def get(self, include=None, where=None):
        return {
            "ids": [k for k, _ in self._rows],
            "documents": [t for _, t in self._rows],
            "metadatas": [
                {"publication_name": k.split("|")[0], "edition_date": k.split("|")[1],
                 "section_id": k.split("|")[2], "chunk_key": k}
                for k, _ in self._rows
            ],
        }


def _manifest(*rows):
    docs = [{"publication_name": r[0].split("|")[0], "edition_date": r[0].split("|")[1],
             "filename": "d.pdf", "chunk_count": len(rows)} for r in rows[:1]]
    return build_manifest(collection=FakeCollection(list(rows)), documents=docs)


class TestLabelDurability:
    def test_label_following_moved_chunk_still_scores_a_hit(self):
        """A re-chunk moved the text to a new ordinal. A label carrying
        text_sha256 must follow the content, not silently become a miss."""
        manifest = _manifest(("PUB|2024|1.1|7", "the relevant passage"))
        out = run_retrieval_eval(
            [_item("q1", "q", [("PUB|2024|1.1|0", 3, "the relevant passage")])],
            top_k=5, manifest=manifest,
            retrieve_fn=_retriever("PUB|2024|1.1|7"),
        )
        assert out["metrics"]["hit_at_5"]["value"] == 1.0
        assert out["label_health"]["moved"] == 1

    def test_label_health_reports_resolution_fraction(self):
        manifest = _manifest(("PUB|2024|1.1|0", "the relevant passage"))
        out = run_retrieval_eval(
            [_item("q1", "q", [("PUB|2024|1.1|0", 3, "the relevant passage")])],
            top_k=5, manifest=manifest, retrieve_fn=_retriever("PUB|2024|1.1|0"),
        )
        assert out["label_health"]["resolved_fraction"] == 1.0
        assert out["label_health"]["corpus_hash"] == manifest["corpus_hash"]

    def test_key_only_label_is_reported_unverifiable(self):
        """A label with no text_sha256 cannot be proven to still point at the
        same text; that must be visible, not assumed fine."""
        manifest = _manifest(("PUB|2024|1.1|0", "the relevant passage"))
        out = run_retrieval_eval(
            [_item("q1", "q", [("PUB|2024|1.1|0", 3)])],
            top_k=5, manifest=manifest, retrieve_fn=_retriever("PUB|2024|1.1|0"),
        )
        assert out["label_health"]["unverifiable"] == 1
        assert out["label_health"]["resolved"] == 0

    def test_label_for_deleted_text_reported_missing(self):
        manifest = _manifest(("PUB|2024|1.1|0", "some other passage"))
        out = run_retrieval_eval(
            [_item("q1", "q", [("PUB|2024|1.1|0", 3, "text that no longer exists")])],
            top_k=5, manifest=manifest, retrieve_fn=_retriever("PUB|2024|1.1|0"),
        )
        assert out["label_health"]["missing"] == 1

    def test_without_manifest_no_label_health_claimed(self):
        out = run_retrieval_eval(
            [_item("q1", "q", [("PUB|2024|1.1|0", 3)])],
            top_k=5, retrieve_fn=_retriever("PUB|2024|1.1|0"),
        )
        assert "label_health" not in out


class TestDatasetLoading:
    def _write(self, tmp_path, lines):
        p = tmp_path / "retrieval.jsonl"
        p.write_text("\n".join(lines), encoding="utf-8")
        return str(p)

    def test_v1_dataset_still_loads(self, tmp_path):
        """v1 items carry expected_section_ids; the schema loader upgrades
        them, so the existing golden dataset keeps working."""
        path = self._write(tmp_path, [json.dumps(
            {"id": "q1", "query": "q", "expected_section_ids": ["1.1"], "expected_abstain": False}
        )])
        items, errors = load_retrieval_dataset(path)
        assert errors == {} and items[0]["schema_version"] == 2

    def test_malformed_line_skipped_not_fatal(self, tmp_path):
        path = self._write(tmp_path, [
            json.dumps(_item("q1", "q", [("PUB|2024|1.1|0", 3)])),
            "{broken",
        ])
        items, errors = load_retrieval_dataset(path)
        assert len(items) == 1 and "<line 2>" in errors

    def test_missing_manifest_raises_rather_than_scoring_silently(self, tmp_path):
        """Asking for durability checking and not getting it must be an error,
        not a quietly weaker guarantee."""
        import pytest

        path = self._write(tmp_path, [json.dumps(_item("q1", "q", [("PUB|2024|1.1|0", 3)]))])
        with pytest.raises(FileNotFoundError):
            run_retrieval_eval_from_file(path, corpus_id="does-not-exist")
