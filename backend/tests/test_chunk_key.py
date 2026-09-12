"""
Tests for shared/chunk_key.py — canonical, collision-safe chunk identity.

Pure functions over strings; no ML models needed.
"""

from shared.chunk_key import build_chunk_key, parse_chunk_key, content_key, resolve_chunk_key


class TestBuildAndParseChunkKey:
    def test_round_trip(self):
        key = build_chunk_key("FSR", "June 2024", "1.1", 3)
        assert parse_chunk_key(key) == {
            "publication_name": "FSR",
            "edition_date": "June 2024",
            "section_id": "1.1",
            "ordinal": 3,
        }

    def test_two_documents_sharing_a_section_id_get_distinct_keys(self):
        key_a = build_chunk_key("DOC_A", "2024", "1.1", 0)
        key_b = build_chunk_key("DOC_B", "2024", "1.1", 0)
        assert key_a != key_b

    def test_parse_rejects_malformed_key(self):
        assert parse_chunk_key("not-a-chunk-key") == {}
        assert parse_chunk_key("A|B|C|not-a-number") == {}


class TestContentKey:
    def test_two_documents_sharing_a_section_id_get_distinct_content_keys(self):
        key_a = content_key("DOC_A", "2024", "1.1", "Document A's text.")
        key_b = content_key("DOC_B", "2024", "1.1", "Document B's text.")
        assert key_a != key_b

    def test_same_document_same_text_gets_the_same_key(self):
        assert content_key("DOC_A", "2024", "1.1", "same text") == content_key("DOC_A", "2024", "1.1", "same text")


class TestResolveChunkKey:
    def test_prefers_stored_chunk_key_when_present(self):
        resolved = resolve_chunk_key("DOC_A", "2024", "1.1", "text", chunk_key="DOC_A|2024|1.1|0")
        assert resolved == "DOC_A|2024|1.1|0"

    def test_falls_back_to_content_key_for_legacy_data(self):
        resolved = resolve_chunk_key("DOC_A", "2024", "1.1", "text", chunk_key="")
        assert resolved == content_key("DOC_A", "2024", "1.1", "text")

    def test_legacy_fallback_still_disambiguates_across_documents(self):
        """The exact bug this module fixes: two documents sharing a bare
        section_id must not collide even without a stored chunk_key."""
        resolved_a = resolve_chunk_key("DOC_A", "2024", "1.1", "Document A's introduction.")
        resolved_b = resolve_chunk_key("DOC_B", "2024", "1.1", "Document B's introduction.")
        assert resolved_a != resolved_b
