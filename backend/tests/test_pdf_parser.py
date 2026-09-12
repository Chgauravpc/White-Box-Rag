"""
Tests for pdf_parser.py — section extraction and chunking logic.
"""

import pytest
from unittest.mock import MagicMock, patch

from ingestion.pdf_parser import chunk_section, detect_sections, ingest_pdf


# ──────────────────────────────────────────────
#  Section Detection Tests
# ──────────────────────────────────────────────

class TestDetectSections:
    """Test section boundary detection from page text."""

    def test_numbered_sections(self):
        """Numbered headers like '1.1 Overview' are detected."""
        pages = [
            {
                "page_number": 1,
                "text": (
                    "1.1 Overview of Financial Stability\n"
                    "The Indian banking sector continued to demonstrate resilience.\n"
                    "Key indicators remained within acceptable ranges.\n"
                    "\n"
                    "1.2 Credit Growth and Asset Quality\n"
                    "Credit growth moderated during the period under review.\n"
                    "Non-performing assets showed improvement."
                ),
            }
        ]

        sections, structured = detect_sections(pages)

        assert len(sections) == 2
        assert sections[0]["section_id"] == "1.1"
        assert sections[0]["section_title"] == "Overview of Financial Stability"
        assert "resilience" in sections[0]["text"]

        assert sections[1]["section_id"] == "1.2"
        assert sections[1]["section_title"] == "Credit Growth and Asset Quality"
        assert "moderated" in sections[1]["text"]

    def test_roman_sections(self):
        """Roman numeral headers like 'Chapter II: ...' are detected."""
        pages = [
            {
                "page_number": 1,
                "text": (
                    "Chapter I: Introduction\n"
                    "This report covers the period April to September 2024.\n"
                    "\n"
                    "Chapter II: Banking Sector Performance\n"
                    "Banks have shown improved profitability."
                ),
            }
        ]

        sections, structured = detect_sections(pages)

        assert len(sections) == 2
        assert sections[0]["section_id"] == "Chapter_I"
        assert sections[1]["section_id"] == "Chapter_II"

    def test_empty_pages_skipped(self):
        """Pages with no text are silently skipped."""
        pages = [
            {"page_number": 1, "text": ""},
            {"page_number": 2, "text": "1.1 Some Section\nSome content here."},
        ]

        sections, structured = detect_sections(pages)
        assert len(sections) == 1
        assert sections[0]["section_id"] == "1.1"

    def test_no_sections_fallback(self):
        """If no section headers found, entire doc becomes one section."""
        pages = [
            {
                "page_number": 1,
                "text": "This is a document without any recognizable section headers.",
            }
        ]

        sections, structured = detect_sections(pages)
        assert len(sections) == 1
        # Updated: fallback now uses UNSTRUCTURED-p{n} instead of 0.0
        assert sections[0]["section_id"].startswith(("UNSTRUCTURED", "0.0"))
        assert structured is False

    def test_generic_numbered_fallback(self):
        """Single-level numbered headers (e.g. contracts) are picked up by the
        generic fallback pass when no financial-report-style headers exist."""
        pages = [
            {
                "page_number": 1,
                "text": (
                    "1. Definitions\n"
                    "In this agreement, the following terms apply.\n"
                    "\n"
                    "2. Term and Termination\n"
                    "This agreement remains in effect until terminated."
                ),
            }
        ]

        sections, structured = detect_sections(pages)
        assert len(sections) == 2
        assert sections[0]["section_id"] == "1"
        assert sections[0]["section_title"] == "Definitions"
        assert sections[1]["section_id"] == "2"
        assert structured is True


# ──────────────────────────────────────────────
#  Chunking Tests
# ──────────────────────────────────────────────

class TestChunkSection:
    """Test the overlapping chunking logic."""

    def test_short_section_single_chunk(self):
        """Sections shorter than max_tokens become one chunk."""
        section = {
            "section_id": "1.1",
            "section_title": "Test",
            "text": "This is a short section with only a few words.",
            "start_page": 1,
        }

        chunks = chunk_section(section, "FSR", "June 2024", max_tokens=512, overlap_tokens=50)

        assert len(chunks) == 1
        assert chunks[0].section_id == "1.1"
        assert chunks[0].publication_name == "FSR"
        assert chunks[0].edition_date == "June 2024"

    def test_long_section_multiple_chunks(self):
        """Long sections are split into overlapping chunks."""
        # Create a section with 100 words
        words = [f"word{i}" for i in range(100)]
        section = {
            "section_id": "2.1",
            "section_title": "Long Section",
            "text": " ".join(words),
            "start_page": 3,
        }

        chunks = chunk_section(section, "MPR", "Dec 2024", max_tokens=30, overlap_tokens=5)

        assert len(chunks) > 1

        # Each chunk should have at most 30 words
        for chunk in chunks:
            word_count = len(chunk.chunk_text.split())
            assert word_count <= 30

        # All chunks should inherit metadata
        for chunk in chunks:
            assert chunk.publication_name == "MPR"
            assert chunk.edition_date == "Dec 2024"
            assert chunk.section_id == "2.1"

    def test_empty_section(self):
        """Empty sections produce no chunks."""
        section = {
            "section_id": "0.0",
            "section_title": "Empty",
            "text": "",
            "start_page": 1,
        }

        chunks = chunk_section(section, "FSR", "June 2024")
        assert len(chunks) == 0


# ──────────────────────────────────────────────
#  Chunk Identity Tests (ingest_pdf)
# ──────────────────────────────────────────────

class TestIngestPdfChunkIdentity:
    """ingest_pdf must emit a globally-unique chunk_key (not a bare section_id)
    as both the ChromaDB id and metadata field — the fix for cross-document
    section_id collisions. I/O (PDF parsing, ChromaDB, SQLite) is mocked;
    section detection and chunking run for real."""

    def _ingest_with_mocks(self, publication: str, edition: str, page_text: str):
        fake_pages = [{"page_number": 1, "text": page_text}]
        fake_collection = MagicMock()
        with patch("ingestion.pdf_parser.extract_pages", return_value=fake_pages), \
             patch("ingestion.pdf_parser.get_chroma_collection", return_value=fake_collection), \
             patch("ingestion.pdf_parser.upsert_document", return_value=1):
            ingest_pdf("dummy.pdf", publication, edition)
        return fake_collection

    def test_chunk_key_is_globally_unique_and_stored_in_metadata(self):
        collection = self._ingest_with_mocks("FSR", "June 2024", "1.1 Overview\nSome section text here.")
        _, kwargs = collection.upsert.call_args
        assert kwargs["ids"] == ["FSR|June 2024|1.1|0"]
        assert kwargs["metadatas"][0]["chunk_key"] == "FSR|June 2024|1.1|0"
        assert kwargs["metadatas"][0]["section_id"] == "1.1"

    def test_ordinal_is_per_section_not_document_global(self):
        """chunk_key's ordinal must count within its own section.

        With a document-global counter, section 2's first chunk was ordinal 1
        (or 47, depending on how much text preceded it), so adding a paragraph
        to section 1 renumbered every chunk in the document and silently
        re-pointed every labeled relevant_chunk_key at different text.
        """
        collection = self._ingest_with_mocks(
            "PUB", "2024",
            "1.1 First\nAlpha content here.\n2.1 Second\nBeta content here.",
        )
        ids = collection.upsert.call_args.kwargs["ids"]
        # Each section restarts at ordinal 0.
        assert ids == ["PUB|2024|1.1|0", "PUB|2024|2.1|0"]

    def test_earlier_section_growth_does_not_renumber_later_sections(self):
        """The stability property labels actually depend on."""
        before = self._ingest_with_mocks(
            "PUB", "2024", "1.1 First\nAlpha.\n2.1 Second\nBeta content.")
        after = self._ingest_with_mocks(
            "PUB", "2024",
            "1.1 First\nAlpha. " + ("filler words " * 400) + "\n2.1 Second\nBeta content.")
        keys_before = set(before.upsert.call_args.kwargs["ids"])
        keys_after = set(after.upsert.call_args.kwargs["ids"])
        # Section 1.1 grew and gained chunks; 2.1's identity must be untouched.
        assert "PUB|2024|2.1|0" in keys_before
        assert "PUB|2024|2.1|0" in keys_after

    def test_reingest_deletes_prior_chunks_before_upsert(self):
        """Re-ingesting must not leave orphans: `upsert` only overwrites the
        ids it is given, so a re-parse yielding fewer chunks used to leave the
        surplus live and retrievable forever."""
        fake_pages = [{"page_number": 1, "text": "1.1 Overview\nSome text."}]
        fake_collection = MagicMock()
        fake_collection.get.return_value = {"ids": ["PUB|2024|9.9|0"]}
        with patch("ingestion.pdf_parser.extract_pages", return_value=fake_pages), \
             patch("ingestion.pdf_parser.get_chroma_collection", return_value=fake_collection), \
             patch("ingestion.pdf_parser.upsert_document", return_value=1):
            ingest_pdf("dummy.pdf", "PUB", "2024")
        fake_collection.delete.assert_called_once_with(ids=["PUB|2024|9.9|0"])

    def test_two_documents_sharing_a_section_id_get_distinct_chunk_keys(self):
        collection_a = self._ingest_with_mocks("DOC_A", "2024", "1.1 Intro\nDocument A content.")
        collection_b = self._ingest_with_mocks("DOC_B", "2024", "1.1 Intro\nDocument B content.")
        ids_a = collection_a.upsert.call_args.kwargs["ids"]
        ids_b = collection_b.upsert.call_args.kwargs["ids"]
        assert ids_a != ids_b
        assert set(ids_a).isdisjoint(set(ids_b))
