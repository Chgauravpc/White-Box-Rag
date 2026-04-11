"""
Tests for pdf_parser.py — section extraction and chunking logic.
"""

import pytest

from ingestion.pdf_parser import chunk_section, detect_sections


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

        sections = detect_sections(pages)

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

        sections = detect_sections(pages)

        assert len(sections) == 2
        assert sections[0]["section_id"] == "Chapter_I"
        assert sections[1]["section_id"] == "Chapter_II"

    def test_empty_pages_skipped(self):
        """Pages with no text are silently skipped."""
        pages = [
            {"page_number": 1, "text": ""},
            {"page_number": 2, "text": "1.1 Some Section\nSome content here."},
        ]

        sections = detect_sections(pages)
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

        sections = detect_sections(pages)
        assert len(sections) == 1
        # Updated: fallback now uses UNSTRUCTURED-p{n} instead of 0.0
        assert sections[0]["section_id"].startswith(("UNSTRUCTURED", "0.0"))


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
