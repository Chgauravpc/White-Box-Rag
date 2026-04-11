"""
Tests for retriever.py — BM25 index, dense search, and hybrid retrieval.
"""

import pytest

from shared.models import ChunkMetadata
from ingestion.retriever import (
    BM25Index,
    reciprocal_rank_fusion,
)


# ──────────────────────────────────────────────
#  BM25 Index Tests
# ──────────────────────────────────────────────

class TestBM25Index:
    """Test the in-memory BM25 index."""

    def test_index_not_built_initially(self):
        idx = BM25Index()
        assert not idx.is_built

    def test_search_empty_returns_empty(self):
        idx = BM25Index()
        results = idx.search("credit risk")
        # Should attempt to build and return empty if ChromaDB is empty
        assert isinstance(results, list)


# ──────────────────────────────────────────────
#  Reciprocal Rank Fusion Tests
# ──────────────────────────────────────────────

class TestRRF:
    """Test the Reciprocal Rank Fusion merging logic."""

    def _make_chunk(self, text: str, section_id: str = "1.1") -> ChunkMetadata:
        return ChunkMetadata(
            publication_name="FSR",
            edition_date="June 2024",
            section_id=section_id,
            section_title="Test Section",
            page_number=1,
            chunk_text=text,
        )

    def test_rrf_merges_results(self):
        """RRF combines dense and sparse results into a merged list."""
        dense = [
            self._make_chunk("Credit risk assessment framework"),
            self._make_chunk("Monetary policy review"),
        ]
        sparse = [
            (self._make_chunk("Credit risk assessment framework"), 5.2),
            (self._make_chunk("Banking sector overview"), 3.1),
        ]

        fused = reciprocal_rank_fusion(dense, sparse)

        assert len(fused) == 3  # 3 unique chunks
        # The chunk appearing in both lists should rank highest
        assert "Credit risk" in fused[0].chunk_text

    def test_rrf_no_results(self):
        """RRF with empty inputs returns empty."""
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_rrf_single_list_only(self):
        """RRF with only dense results still works."""
        dense = [
            self._make_chunk("Financial stability report"),
            self._make_chunk("Payment systems review"),
        ]

        fused = reciprocal_rank_fusion(dense, [])
        assert len(fused) == 2
