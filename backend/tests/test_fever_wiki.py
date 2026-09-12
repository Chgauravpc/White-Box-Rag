"""
Tests for eval/fever_wiki.py — the on-disk FEVER wiki-pages index.

The point of this module is that it does NOT hold the dump in memory, so the
tests focus on the properties that make it a safe substitute for the dict it
replaced: it parses the dump's actual tab-delimited `lines` format, survives
the malformed rows a real dump contains, is reusable across runs, and hands
`eval/adapters.py::from_fever` a resolver with the exact signature it expects.
"""

import json
import os

import pytest

from eval.adapters import from_fever
from eval.fever_wiki import WikiSentenceIndex


def _page(page_id, sentences):
    """A wiki-pages row: `lines` is "0\\tFirst.\\t...\\n1\\tSecond.\\t..."."""
    lines = "\n".join(f"{i}\t{text}\tsome\tannotation" for i, text in enumerate(sentences))
    return {"id": page_id, "text": " ".join(sentences), "lines": lines}


@pytest.fixture()
def dump(tmp_path):
    d = tmp_path / "wiki-pages"
    d.mkdir()
    (d / "wiki-001.jsonl").write_text(
        "\n".join(json.dumps(p) for p in [
            _page("Paris", ["Paris is the capital of France.", "It is on the Seine."]),
            _page("Berlin", ["Berlin is the capital of Germany."]),
        ]),
        encoding="utf-8",
    )
    (d / "wiki-002.jsonl").write_text(
        json.dumps(_page("Rome", ["Rome is the capital of Italy."])), encoding="utf-8"
    )
    return str(d)


class TestBuild:
    def test_indexes_every_sentence(self, dump, tmp_path):
        db = str(tmp_path / "wiki.sqlite")
        report = WikiSentenceIndex.build(dump, db)
        assert report["status"] == "built"
        assert report["sentences"] == 4
        assert report["files"] == 2

    def test_resolves_page_and_sentence_id(self, dump, tmp_path):
        db = str(tmp_path / "wiki.sqlite")
        WikiSentenceIndex.build(dump, db)
        with WikiSentenceIndex(db) as idx:
            assert idx.resolve("Paris", 0) == "Paris is the capital of France."
            assert idx.resolve("Paris", 1) == "It is on the Seine."
            assert idx.resolve("Rome", 0) == "Rome is the capital of Italy."

    def test_unknown_lookup_returns_empty_not_error(self, dump, tmp_path):
        db = str(tmp_path / "wiki.sqlite")
        WikiSentenceIndex.build(dump, db)
        with WikiSentenceIndex(db) as idx:
            assert idx.resolve("Atlantis", 0) == ""
            assert idx.resolve("Paris", 99) == ""

    def test_non_integer_sentence_id_is_handled(self, dump, tmp_path):
        """FEVER NEI rows carry null pointers; they must not raise."""
        db = str(tmp_path / "wiki.sqlite")
        WikiSentenceIndex.build(dump, db)
        with WikiSentenceIndex(db) as idx:
            assert idx.resolve("Paris", None) == ""
            assert idx.resolve("Paris", "not-a-number") == ""

    def test_existing_index_is_reused_not_rebuilt(self, dump, tmp_path):
        db = str(tmp_path / "wiki.sqlite")
        WikiSentenceIndex.build(dump, db)
        again = WikiSentenceIndex.build(dump, db)
        assert again["status"] == "reused"
        assert again["sentences"] == 4

    def test_force_rebuilds(self, dump, tmp_path):
        db = str(tmp_path / "wiki.sqlite")
        WikiSentenceIndex.build(dump, db)
        rebuilt = WikiSentenceIndex.build(dump, db, force=True)
        assert rebuilt["status"] == "built"

    def test_malformed_lines_are_skipped_not_fatal(self, tmp_path):
        """A real dump contains blank lines, rows without ids, and line entries
        that are not "<int>\\t<text>"."""
        d = tmp_path / "wiki"
        d.mkdir()
        (d / "a.jsonl").write_text("\n".join([
            json.dumps(_page("Good", ["A real sentence."])),
            "{not json at all",
            json.dumps({"text": "no id field", "lines": "0\torphan\t"}),
            "",
            json.dumps({"id": "Weird", "lines": "notanumber\tignored\t\n0\tKept.\t"}),
        ]), encoding="utf-8")
        db = str(tmp_path / "w.sqlite")
        report = WikiSentenceIndex.build(str(d), db)
        assert report["skipped_lines"] == 2
        with WikiSentenceIndex(db) as idx:
            assert idx.resolve("Good", 0) == "A real sentence."
            assert idx.resolve("Weird", 0) == "Kept."

    def test_duplicate_page_does_not_abort_the_build(self, tmp_path):
        """Dumps repeat pages; a duplicate primary key must not kill indexing."""
        d = tmp_path / "wiki"
        d.mkdir()
        (d / "a.jsonl").write_text("\n".join([
            json.dumps(_page("Dup", ["First version."])),
            json.dumps(_page("Dup", ["Second version."])),
        ]), encoding="utf-8")
        db = str(tmp_path / "w.sqlite")
        report = WikiSentenceIndex.build(str(d), db)
        assert report["sentences"] == 1
        with WikiSentenceIndex(db) as idx:
            assert idx.resolve("Dup", 0) == "First version."

    def test_empty_directory_raises_clearly(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            WikiSentenceIndex.build(str(d), str(tmp_path / "w.sqlite"))

    def test_index_file_is_created_on_disk(self, dump, tmp_path):
        db = str(tmp_path / "nested" / "wiki.sqlite")
        WikiSentenceIndex.build(dump, db)
        assert os.path.exists(db) and os.path.getsize(db) > 0


class TestIntegrationWithAdapter:
    def test_resolver_plugs_into_from_fever(self, dump, tmp_path):
        """The whole reason the index exists: it must satisfy from_fever's
        resolve_evidence(page, sentence_id) contract exactly."""
        db = str(tmp_path / "wiki.sqlite")
        WikiSentenceIndex.build(dump, db)
        rows = [{
            "id": 1, "label": "SUPPORTS", "claim": "Paris is in France.",
            "evidence": [[[1, 1, "Paris", 0]]],
        }]
        with WikiSentenceIndex(db) as idx:
            items, report = from_fever(rows, resolve_evidence=idx.as_resolver())
        assert report["n_items"] == 1
        assert items[0]["premise"] == "Paris is the capital of France."

    def test_multi_sentence_evidence_joins_across_pages(self, dump, tmp_path):
        db = str(tmp_path / "wiki.sqlite")
        WikiSentenceIndex.build(dump, db)
        rows = [{
            "id": 2, "label": "REFUTES", "claim": "Paris and Berlin are the same city.",
            "evidence": [[[1, 1, "Paris", 0], [1, 2, "Berlin", 0]]],
        }]
        with WikiSentenceIndex(db) as idx:
            items, _ = from_fever(rows, resolve_evidence=idx.as_resolver())
        assert items[0]["premise"] == (
            "Paris is the capital of France. Berlin is the capital of Germany."
        )
