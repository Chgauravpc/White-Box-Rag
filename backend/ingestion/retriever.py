"""
Hybrid Retriever — Dense (ChromaDB) + Sparse (BM25) with Reciprocal Rank Fusion.

Provides the `hybrid_retrieve()` function used by the RAG pipeline and by BP3's
requirement mapper.
"""

import logging
from collections import defaultdict

from rank_bm25 import BM25Okapi

from shared.config import DENSE_TOP_K, SPARSE_TOP_K, FINAL_TOP_K, RRF_K
from shared.database import get_chroma_collection
from shared.models import ChunkMetadata

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  BM25 Index (in-memory, rebuilt after ingestion)
# ──────────────────────────────────────────────

class BM25Index:
    """Maintains an in-memory BM25 index over all chunks in ChromaDB.

    Designed for hackathon scale (<10 k chunks). Rebuilds the full index
    from ChromaDB on each `build()` call.
    """

    def __init__(self):
        self._corpus_tokens: list[list[str]] = []
        self._metadatas: list[dict] = []
        self._documents: list[str] = []
        self._bm25: BM25Okapi | None = None

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None and len(self._corpus_tokens) > 0

    def build(self):
        """Load all documents from ChromaDB, tokenise, and build the BM25 index."""
        collection = get_chroma_collection()
        count = collection.count()

        if count == 0:
            logger.warning("ChromaDB is empty — BM25 index cannot be built")
            self._bm25 = None
            return

        # Fetch all documents (ChromaDB returns max 10k per call by default)
        results = collection.get(include=["documents", "metadatas"])
        self._documents = results["documents"]
        self._metadatas = results["metadatas"]

        # Tokenise: simple lowercase whitespace split
        self._corpus_tokens = [doc.lower().split() for doc in self._documents]

        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info(f"BM25 index built with {len(self._documents)} documents")

    def search(
        self, query: str, top_k: int = SPARSE_TOP_K, filters: dict | None = None
    ) -> list[tuple[ChunkMetadata, float]]:
        """Search the BM25 index.

        Args:
            query: Search query.
            top_k: Number of results to return.
            filters: Optional metadata filters applied post-scoring.

        Returns:
            List of (ChunkMetadata, bm25_score) tuples, sorted by score descending.
        """
        if not self.is_built:
            self.build()
            if not self.is_built:
                return []

        tokenised_query = query.lower().split()
        scores = self._bm25.get_scores(tokenised_query)

        # Pair each document with its score
        scored = list(enumerate(scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored:
            if score <= 0:
                continue

            meta = self._metadatas[idx]

            # Apply filters post-hoc
            if filters:
                skip = False
                for key, value in filters.items():
                    if meta.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            chunk = ChunkMetadata(
                publication_name=meta.get("publication_name", ""),
                edition_date=meta.get("edition_date", ""),
                section_id=meta.get("section_id", ""),
                section_title=meta.get("section_title", ""),
                page_number=meta.get("page_number", 0),
                chunk_text=self._documents[idx],
            )
            results.append((chunk, score))

            if len(results) >= top_k:
                break

        return results


# Module-level singleton
_bm25_index = BM25Index()


def get_bm25_index() -> BM25Index:
    """Return the global BM25 index singleton."""
    return _bm25_index


def rebuild_bm25_index():
    """Rebuild the BM25 index (call after ingestion)."""
    _bm25_index.build()


# ──────────────────────────────────────────────
#  Dense Search (ChromaDB)
# ──────────────────────────────────────────────

def dense_search(
    query: str, top_k: int = DENSE_TOP_K, filters: dict | None = None
) -> list[ChunkMetadata]:
    """Query ChromaDB with dense (all-MiniLM-L6-v2) embeddings.

    Args:
        query: Search query text.
        top_k: Number of results.
        filters: Optional ChromaDB where-clause, e.g. {"publication_name": "FSR"}.

    Returns:
        List of ChunkMetadata, ordered by similarity (best first).
    """
    collection = get_chroma_collection()

    where = None
    if filters:
        # ChromaDB where clause supports simple equality:
        #   {"publication_name": "FSR"}
        # For multiple filters, use $and:
        #   {"$and": [{"publication_name": "FSR"}, {"edition_date": "June 2024"}]}
        if len(filters) == 1:
            where = filters
        else:
            where = {"$and": [{k: v} for k, v in filters.items()]}

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas"],
    )

    chunks = []
    if results and results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append(
                ChunkMetadata(
                    publication_name=meta.get("publication_name", ""),
                    edition_date=meta.get("edition_date", ""),
                    section_id=meta.get("section_id", ""),
                    section_title=meta.get("section_title", ""),
                    page_number=meta.get("page_number", 0),
                    chunk_text=doc,
                )
            )

    return chunks


# ──────────────────────────────────────────────
#  Reciprocal Rank Fusion
# ──────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: list[ChunkMetadata],
    sparse_results: list[tuple[ChunkMetadata, float]],
    k: int = RRF_K,
) -> list[ChunkMetadata]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = Σ  1 / (k + rank(d))  across all lists.

    Args:
        dense_results: Ordered list from dense search (rank = list index).
        sparse_results: Ordered list from BM25 search with scores.
        k: RRF constant (default 60).

    Returns:
        Merged list of ChunkMetadata, sorted by RRF score descending.
    """
    scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, ChunkMetadata] = {}

    # Key function: use section_id + first 80 chars of text for dedup
    def _key(chunk: ChunkMetadata) -> str:
        return f"{chunk.publication_name}|{chunk.edition_date}|{chunk.section_id}|{chunk.chunk_text[:80]}"

    # Score dense results
    for rank, chunk in enumerate(dense_results):
        key = _key(chunk)
        scores[key] += 1.0 / (k + rank + 1)
        chunk_map[key] = chunk

    # Score sparse results
    for rank, (chunk, _bm25_score) in enumerate(sparse_results):
        key = _key(chunk)
        scores[key] += 1.0 / (k + rank + 1)
        chunk_map[key] = chunk

    # Sort by fused score
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [chunk_map[key] for key in sorted_keys]


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    top_k: int = FINAL_TOP_K,
    filters: dict | None = None,
) -> list[ChunkMetadata]:
    """Hybrid retrieval: dense + BM25 + Reciprocal Rank Fusion.

    Args:
        query: The user's natural language question.
        top_k: Final number of results to return.
        filters: Optional metadata filters (publication_name, edition_date).

    Returns:
        Top-K ChunkMetadata objects, ranked by hybrid relevance.
    """
    # 1. Dense search via ChromaDB
    dense_results = dense_search(query, top_k=DENSE_TOP_K, filters=filters)

    # 2. Sparse search via BM25
    bm25 = get_bm25_index()
    sparse_results = bm25.search(query, top_k=SPARSE_TOP_K, filters=filters)

    # 3. Fuse results
    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    logger.info(
        f"Hybrid retrieve: query='{query[:50]}...', "
        f"dense={len(dense_results)}, sparse={len(sparse_results)}, "
        f"fused={len(fused)}, returning top {top_k}"
    )

    return fused[:top_k]
