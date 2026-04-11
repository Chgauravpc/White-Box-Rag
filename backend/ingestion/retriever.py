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

def dense_search_with_scores(
    query: str, top_k: int = DENSE_TOP_K, filters: dict | None = None
) -> list[tuple[ChunkMetadata, float]]:
    """Same as dense_search() but also returns the distance/similarity score."""
    collection = get_chroma_collection()

    where = None
    if filters:
        if len(filters) == 1:
            where = filters
        else:
            where = {"$and": [{k: v} for k, v in filters.items()]}

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks_with_scores = []
    if results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: dist=0 means identical, dist=2 means opposite
            # Convert to cosine similarity: sim = 1 - dist (for cosine space)
            cosine_sim = round(1.0 - dist, 6)
            chunk = ChunkMetadata(
                publication_name=meta.get("publication_name", ""),
                edition_date=meta.get("edition_date", ""),
                section_id=meta.get("section_id", ""),
                section_title=meta.get("section_title", ""),
                page_number=meta.get("page_number", 0),
                chunk_text=doc,
            )
            chunks_with_scores.append((chunk, cosine_sim))

    return chunks_with_scores


def dense_search(
    query: str, top_k: int = DENSE_TOP_K, filters: dict | None = None
) -> list[ChunkMetadata]:
    """Query ChromaDB with dense (all-MiniLM-L6-v2) embeddings."""
    return [chunk for chunk, _ in dense_search_with_scores(query, top_k, filters)]


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
    """Hybrid retrieval: dense + BM25 + Reciprocal Rank Fusion."""
    chunks, _ = hybrid_retrieve_with_scores(query, top_k, filters)
    return chunks


def hybrid_retrieve_with_scores(
    query: str,
    top_k: int = FINAL_TOP_K,
    filters: dict | None = None,
) -> tuple[list[ChunkMetadata], list[dict]]:
    """Hybrid retrieval returning chunks AND per-chunk score data for XAI matrix.

    Returns:
        (chunks, score_records) where score_records contains dense_score,
        bm25_score, rrf_score and rank for each chunk.
    """
    # 1. Dense search with scores
    dense_with_scores = dense_search_with_scores(query, top_k=DENSE_TOP_K, filters=filters)
    dense_results = [chunk for chunk, _ in dense_with_scores]
    dense_score_map = {
        f"{c.publication_name}|{c.edition_date}|{c.section_id}|{c.chunk_text[:80]}": s
        for c, s in dense_with_scores
    }

    # 2. Sparse search via BM25
    bm25 = get_bm25_index()
    sparse_results = bm25.search(query, top_k=SPARSE_TOP_K, filters=filters)
    bm25_score_map = {
        f"{c.publication_name}|{c.edition_date}|{c.section_id}|{c.chunk_text[:80]}": s
        for c, s in sparse_results
    }

    # 3. Fuse results
    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    def _key(chunk: ChunkMetadata) -> str:
        return f"{chunk.publication_name}|{chunk.edition_date}|{chunk.section_id}|{chunk.chunk_text[:80]}"

    # 4. Build RRF scores
    rrf_scores: dict[str, float] = defaultdict(float)
    for rank, chunk in enumerate(dense_results):
        rrf_scores[_key(chunk)] += 1.0 / (RRF_K + rank + 1)
    for rank, (chunk, _) in enumerate(sparse_results):
        rrf_scores[_key(chunk)] += 1.0 / (RRF_K + rank + 1)

    # 5. Build score records for the similarity matrix
    top_fused = fused[:top_k]
    score_records = []
    for rank, chunk in enumerate(top_fused):
        k = _key(chunk)
        chunk_id = f"{chunk.publication_name}_{chunk.edition_date}_{chunk.section_id}_c{rank}"
        score_records.append({
            "chunk_id": chunk_id,
            "section_id": chunk.section_id,
            "publication": chunk.publication_name,
            "edition": chunk.edition_date,
            "dense_score": dense_score_map.get(k, 0.0),
            "bm25_score": bm25_score_map.get(k, 0.0),
            "rrf_score": round(rrf_scores.get(k, 0.0), 8),
            "rank": rank + 1,
        })

    logger.info(
        f"Hybrid retrieve with scores: dense={len(dense_results)}, "
        f"sparse={len(sparse_results)}, fused={len(fused)}, top {top_k}"
    )

    return top_fused, score_records
