"""
Tests for ingestion/pipeline.py::_deduplicate_chunks — the chunk-identity
dedup fix.

Two documents that happen to share a bare section_id (e.g. both have a "1.1")
must not collide: each document's chunk must survive into context, keyed by
the compound chunk identity rather than the ambiguous section_id alone.
"""

from ingestion.pipeline import _deduplicate_chunks


def _chunk(pub, edition, section_id, text, chunk_key=""):
    return {
        "chunk_text": text,
        "section_id": section_id,
        "publication_name": pub,
        "edition_date": edition,
        "chunk_key": chunk_key,
    }


class TestDeduplicateChunks:
    def test_same_section_id_different_documents_both_survive(self):
        chunks = [
            _chunk("DOC_A", "2024", "1.1", "Document A's introduction.", "DOC_A|2024|1.1|0"),
            _chunk("DOC_B", "2024", "1.1", "Document B's introduction.", "DOC_B|2024|1.1|0"),
        ]
        deduped = _deduplicate_chunks(chunks)
        assert len(deduped) == 2
        texts = {c["chunk_text"] for c in deduped}
        assert texts == {"Document A's introduction.", "Document B's introduction."}

    def test_true_duplicate_is_still_collapsed(self):
        chunks = [
            _chunk("DOC_A", "2024", "1.1", "Same chunk returned twice.", "DOC_A|2024|1.1|0"),
            _chunk("DOC_A", "2024", "1.1", "Same chunk returned twice.", "DOC_A|2024|1.1|0"),
        ]
        deduped = _deduplicate_chunks(chunks)
        assert len(deduped) == 1

    def test_first_occurrence_wins_highest_rrf(self):
        chunks = [
            _chunk("DOC_A", "2024", "1.1", "kept", "DOC_A|2024|1.1|0"),
            _chunk("DOC_A", "2024", "1.1", "dropped", "DOC_A|2024|1.1|0"),
        ]
        deduped = _deduplicate_chunks(chunks)
        assert deduped[0]["chunk_text"] == "kept"

    def test_legacy_chunks_without_chunk_key_still_dedup_correctly(self):
        """Falls back to a content-addressed key when chunk_key is empty
        (data ingested before this field existed)."""
        chunks = [
            _chunk("DOC_A", "2024", "1.1", "Document A's introduction."),
            _chunk("DOC_B", "2024", "1.1", "Document B's introduction."),
        ]
        deduped = _deduplicate_chunks(chunks)
        assert len(deduped) == 2
