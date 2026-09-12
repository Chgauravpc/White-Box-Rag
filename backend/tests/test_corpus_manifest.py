"""
Tests for shared/hashing.py and shared/corpus_manifest.py (W3.1).

The manifest's whole job is to make corpus drift *detectable*, so these tests
are mostly about the drift cases a labeled dataset actually dies from: a chunk
whose text moved to a new key (recoverable) versus a chunk whose key survived
but whose text changed underneath it (silent invalidation).

No ChromaDB and no embedding model: `build_manifest` takes an injectable
collection, so a dict-shaped fake is enough.
"""

from shared import corpus_manifest as cm
from shared.hashing import hash_text, normalize_for_hash, hash_file


class FakeCollection:
    """Minimal stand-in for a Chroma collection's `.get()` contract."""

    def __init__(self, rows):
        self._rows = rows

    def get(self, include=None, where=None):
        return {
            "ids": [r["chunk_key"] for r in self._rows],
            "documents": [r["text"] for r in self._rows],
            "metadatas": [
                {
                    "publication_name": r.get("publication_name", "PUB"),
                    "edition_date": r.get("edition_date", "2024"),
                    "section_id": r.get("section_id", "1.1"),
                    "chunk_key": r["chunk_key"],
                }
                for r in self._rows
            ],
        }


def _rows(*pairs, publication="PUB", edition="2024"):
    return [
        {"chunk_key": key, "text": text, "publication_name": publication, "edition_date": edition}
        for key, text in pairs
    ]


def _docs(chunk_count, publication="PUB", edition="2024", **extra):
    return [dict({
        "publication_name": publication, "edition_date": edition,
        "filename": "doc.pdf", "chunk_count": chunk_count,
        "source_sha256": "abc", "parser_version": "1.0", "structured": 1,
    }, **extra)]


class TestHashing:
    def test_normalization_absorbs_whitespace_differences(self):
        # The exact artifact class a PyMuPDF version bump produces.
        assert hash_text("Capital  adequacy\n\nratio") == hash_text("Capital adequacy ratio")

    def test_normalization_is_not_lossy_about_content(self):
        assert hash_text("ratio of 8%") != hash_text("ratio of 9%")

    def test_normalize_collapses_and_strips(self):
        assert normalize_for_hash("  a \t b \n ") == "a b"

    def test_hash_file_missing_file_returns_empty_not_raises(self):
        assert hash_file("does-not-exist-anywhere.pdf") == ""

    def test_hash_file_matches_content(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"hello")
        q = tmp_path / "b.bin"
        q.write_bytes(b"hello")
        assert hash_file(str(p)) == hash_file(str(q))


class TestBuildManifest:
    def test_empty_corpus_hash_is_sentinel_not_a_digest(self):
        """An empty corpus must not produce a real-looking hash — two empty
        machines would otherwise 'agree' on a corpus."""
        m = cm.build_manifest(collection=FakeCollection([]), documents=[])
        assert m["corpus_hash"] == cm.EMPTY_CORPUS_HASH
        assert m["chunk_count"] == 0

    def test_manifest_records_text_hash_per_chunk(self):
        m = cm.build_manifest(
            collection=FakeCollection(_rows(("PUB|2024|1.1|0", "alpha"))),
            documents=_docs(1),
        )
        chunk = m["documents"][0]["chunks"][0]
        assert chunk["chunk_key"] == "PUB|2024|1.1|0"
        assert chunk["text_sha256"] == hash_text("alpha")

    def test_corpus_hash_is_order_independent(self):
        a = cm.build_manifest(
            collection=FakeCollection(_rows(("PUB|2024|1.1|0", "alpha"), ("PUB|2024|1.1|1", "beta"))),
            documents=_docs(2))
        b = cm.build_manifest(
            collection=FakeCollection(_rows(("PUB|2024|1.1|1", "beta"), ("PUB|2024|1.1|0", "alpha"))),
            documents=_docs(2))
        assert a["corpus_hash"] == b["corpus_hash"]

    def test_corpus_hash_changes_when_text_changes(self):
        a = cm.build_manifest(collection=FakeCollection(_rows(("k|0", "alpha"))), documents=_docs(1))
        b = cm.build_manifest(collection=FakeCollection(_rows(("k|0", "CHANGED"))), documents=_docs(1))
        assert a["corpus_hash"] != b["corpus_hash"]

    def test_chunk_count_mismatch_is_reported(self):
        """SQLite is derived metadata and can drift from Chroma; the
        disagreement must surface rather than be silently resolved."""
        m = cm.build_manifest(
            collection=FakeCollection(_rows(("PUB|2024|1.1|0", "alpha"))),
            documents=_docs(99),
        )
        assert [i["type"] for i in m["inconsistencies"]] == ["chunk_count_mismatch"]

    def test_chunks_without_document_row_reported(self):
        m = cm.build_manifest(collection=FakeCollection(_rows(("PUB|2024|1.1|0", "a"))), documents=[])
        assert m["inconsistencies"][0]["type"] == "chunks_without_document"

    def test_document_without_chunks_reported(self):
        m = cm.build_manifest(collection=FakeCollection([]), documents=_docs(3))
        assert m["inconsistencies"][0]["type"] == "document_without_chunks"


class TestPersistence:
    def test_save_then_load_round_trips(self, tmp_path):
        m = cm.build_manifest(collection=FakeCollection(_rows(("k|0", "alpha"))), documents=_docs(1))
        cm.save_manifest(m, "corpus-v1", corpora_dir=str(tmp_path))
        loaded = cm.load_manifest("corpus-v1", corpora_dir=str(tmp_path))
        assert loaded["corpus_hash"] == m["corpus_hash"]
        assert loaded["corpus_id"] == "corpus-v1"

    def test_load_missing_manifest_is_none_not_error(self, tmp_path):
        assert cm.load_manifest("never-frozen", corpora_dir=str(tmp_path)) is None


class TestDiff:
    def test_identical_corpora_report_no_drift(self):
        rows = _rows(("PUB|2024|1.1|0", "alpha"))
        a = cm.build_manifest(collection=FakeCollection(rows), documents=_docs(1))
        b = cm.build_manifest(collection=FakeCollection(rows), documents=_docs(1))
        d = cm.diff_manifest(a, b)
        assert d["identical"] is True
        assert d["summary"] == {"content_changed": 0, "moved": 0, "removed": 0, "added": 0}

    def test_reordered_chunk_is_moved_not_lost(self):
        """A re-chunk that shifts ordinals must be reported as `moved` — the
        text still exists, so labels carrying text_sha256 are recoverable."""
        frozen = cm.build_manifest(collection=FakeCollection(_rows(("PUB|2024|1.1|0", "alpha"))), documents=_docs(1))
        live = cm.build_manifest(collection=FakeCollection(_rows(("PUB|2024|1.1|7", "alpha"))), documents=_docs(1))
        d = cm.diff_manifest(frozen, live)
        assert d["summary"]["moved"] == 1
        assert d["summary"]["removed"] == 0
        assert d["moved"][0]["now_at"] == ["PUB|2024|1.1|7"]
        # the new key must not also be counted as an addition
        assert d["added"] == []

    def test_same_key_different_text_is_content_changed(self):
        """The silent-invalidation case: a label still resolves and is wrong."""
        frozen = cm.build_manifest(collection=FakeCollection(_rows(("PUB|2024|1.1|0", "alpha"))), documents=_docs(1))
        live = cm.build_manifest(collection=FakeCollection(_rows(("PUB|2024|1.1|0", "different"))), documents=_docs(1))
        d = cm.diff_manifest(frozen, live)
        assert d["summary"]["content_changed"] == 1
        assert d["summary"]["moved"] == 0

    def test_deleted_chunk_is_removed(self):
        frozen = cm.build_manifest(
            collection=FakeCollection(_rows(("PUB|2024|1.1|0", "alpha"), ("PUB|2024|1.1|1", "beta"))),
            documents=_docs(2))
        live = cm.build_manifest(collection=FakeCollection(_rows(("PUB|2024|1.1|0", "alpha"))), documents=_docs(1))
        d = cm.diff_manifest(frozen, live)
        assert d["summary"]["removed"] == 1

    def test_added_chunk_is_reported(self):
        frozen = cm.build_manifest(collection=FakeCollection(_rows(("PUB|2024|1.1|0", "alpha"))), documents=_docs(1))
        live = cm.build_manifest(
            collection=FakeCollection(_rows(("PUB|2024|1.1|0", "alpha"), ("PUB|2024|1.1|1", "beta"))),
            documents=_docs(2))
        d = cm.diff_manifest(frozen, live)
        assert d["added"] == ["PUB|2024|1.1|1"]


class TestResolveLabels:
    def _manifest(self, *pairs):
        return cm.build_manifest(collection=FakeCollection(_rows(*pairs)), documents=_docs(len(pairs)))

    def test_label_with_matching_hash_resolves(self):
        m = self._manifest(("PUB|2024|1.1|0", "alpha"))
        r = cm.resolve_labels(m, [{"key": "PUB|2024|1.1|0", "text_sha256": hash_text("alpha"), "grade": 3}])
        assert r["resolved"] == ["PUB|2024|1.1|0"]
        assert r["resolved_fraction"] == 1.0

    def test_label_follows_text_when_key_moved(self):
        m = self._manifest(("PUB|2024|1.1|9", "alpha"))
        r = cm.resolve_labels(m, [{"key": "PUB|2024|1.1|0", "text_sha256": hash_text("alpha")}])
        assert r["moved"] == [{"labeled_key": "PUB|2024|1.1|0", "now_at": ["PUB|2024|1.1|9"]}]
        assert r["resolved"] == []

    def test_label_without_text_hash_is_unverifiable_not_resolved(self):
        """A key-only label cannot be proven to still point at the same text —
        it must be reported as unverifiable rather than assumed correct."""
        m = self._manifest(("PUB|2024|1.1|0", "alpha"))
        r = cm.resolve_labels(m, [{"key": "PUB|2024|1.1|0"}])
        assert r["unverifiable"] == ["PUB|2024|1.1|0"]
        assert r["resolved"] == []

    def test_label_for_deleted_text_is_missing(self):
        m = self._manifest(("PUB|2024|1.1|0", "alpha"))
        r = cm.resolve_labels(m, [{"key": "PUB|2024|9.9|0", "text_sha256": hash_text("gone")}])
        assert r["missing"] == ["PUB|2024|9.9|0"]

    def test_no_labels_gives_none_fraction_not_zero(self):
        m = self._manifest(("PUB|2024|1.1|0", "alpha"))
        r = cm.resolve_labels(m, [])
        assert r["total"] == 0 and r["resolved_fraction"] is None
