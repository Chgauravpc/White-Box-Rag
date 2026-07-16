import logging
import re
from datetime import datetime
from typing import List

from shared.models import Claim, EditionConflict
from shared.xai_matrices import build_conflict_matrix, get_encoder
from shared.xai_engine import cosine_similarity
from shared.database import get_chroma_collection, get_sqlite_conn

logger = logging.getLogger(__name__)

# Page-level fallback citations (see ingestion/pdf_parser.py's UNSTRUCTURED-p*)
# — comparing "page 1 of document A" against "page 1 of document B" as if they
# were the same section produces false contradiction flags on unrelated text,
# so these are never eligible for cross-edition conflict discovery.
_UNSTRUCTURED_SECTION_RE = re.compile(r"^UNSTRUCTURED-p\d+$")

# Cheap cosine pre-filter before paying for an NLI call — near-identical text
# can't meaningfully contradict.
CONFLICT_PREFILTER_COSINE = 0.90
MAX_CONFLICT_CHECKS_PER_QUERY = 5


async def detect_conflicts(publication: str, topic: str, older_date: str, older_text: str, newer_date: str, newer_text: str, section_id: str) -> EditionConflict:
    """Detects conflicts between old and new editions using CrossEncoder probabilities."""
    old_chunk = {"chunk_text": older_text, "section_id": section_id, "edition_date": older_date}
    new_chunk = {"chunk_text": newer_text, "section_id": section_id, "edition_date": newer_date}
    
    C_matrix = build_conflict_matrix([old_chunk], [new_chunk])
    if C_matrix.size == 0:
        return EditionConflict(
            publication=publication, section_id=section_id, older_edition=older_date, newer_edition=newer_date,
            has_conflict=False, conflict_description="Empty input.", superseding_edition="", details=""
        )
        
    contradiction_prob = float(C_matrix[0][0])
    has_conflict = contradiction_prob > 0.70
    
    return EditionConflict(
        publication=publication,
        section_id=section_id,
        older_edition=older_date,
        newer_edition=newer_date,
        has_conflict=has_conflict,
        conflict_description=f"Contradiction Probability: {contradiction_prob:.2%}" if has_conflict else "",
        superseding_edition=newer_date if has_conflict else "",
        details=f"CrossEncoder probability of newer text contradicting older text: {contradiction_prob:.4f}"
    )


def _cache_lookup(publication: str, section_id: str, edition_a: str, edition_b: str):
    try:
        conn = get_sqlite_conn()
        row = conn.execute(
            "SELECT cosine_sim, has_conflict, authoritative FROM edition_conflict_cache "
            "WHERE publication=? AND section_id=? AND edition_a=? AND edition_b=?",
            (publication, section_id, edition_a, edition_b),
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.warning(f"Edition conflict cache lookup failed: {e}")
        return None


def _cache_store(publication: str, section_id: str, edition_a: str, edition_b: str,
                  cosine_sim: float, has_conflict: bool, authoritative: str) -> None:
    try:
        conn = get_sqlite_conn()
        conn.execute(
            "INSERT OR REPLACE INTO edition_conflict_cache "
            "(publication, section_id, edition_a, edition_b, cosine_sim, has_conflict, authoritative, cached_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (publication, section_id, edition_a, edition_b, cosine_sim, int(has_conflict),
             authoritative, datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Edition conflict cache write failed: {e}")


async def discover_and_check_conflicts(claims: List[Claim]) -> List[EditionConflict]:
    """Discover whether any section cited by these claims has a genuinely
    conflicting older/newer edition elsewhere in the knowledge base.

    Only proceeds when the SAME section_id appears under >=2 distinct edition
    labels for the same publication (real versioning, not two different
    documents that happen to share a collection). UNSTRUCTURED-p* (page-level
    fallback) section IDs are always skipped — see module-level comment above.
    """
    cited: dict = {}
    for c in claims:
        sid = (c.source_section_id or "").strip()
        if not sid or _UNSTRUCTURED_SECTION_RE.match(sid):
            continue
        key = (c.source_publication, sid)
        cited.setdefault(key, set()).add(c.source_edition)

    if not cited:
        return []

    try:
        collection = get_chroma_collection()
    except Exception as e:
        logger.warning(f"Could not access ChromaDB for conflict discovery: {e}")
        return []

    encoder = get_encoder()
    candidates = []  # (cosine, publication, section_id, cited_edition, other_edition, cited_text, other_text)

    for (publication, section_id), cited_editions in cited.items():
        if not publication:
            continue
        try:
            results = collection.get(
                where={"$and": [{"publication_name": publication}, {"section_id": section_id}]},
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning(f"ChromaDB lookup failed for {publication}/{section_id}: {e}")
            continue

        by_edition: dict = {}
        for doc, meta in zip(results.get("documents") or [], results.get("metadatas") or []):
            ed = meta.get("edition_date", "")
            if ed:
                by_edition.setdefault(ed, []).append(doc)

        # Only genuine versioning: the SAME section under >=2 distinct editions in the KB.
        if len(by_edition) < 2:
            continue

        for cited_edition in cited_editions:
            if cited_edition not in by_edition:
                continue
            cited_text = by_edition[cited_edition][0]
            for other_edition, other_chunks in by_edition.items():
                if other_edition == cited_edition:
                    continue
                other_text = other_chunks[0]

                cache_hit = (
                    _cache_lookup(publication, section_id, cited_edition, other_edition)
                    or _cache_lookup(publication, section_id, other_edition, cited_edition)
                )
                if cache_hit is not None:
                    cosine_sim = cache_hit["cosine_sim"]
                else:
                    emb = encoder.encode([cited_text, other_text])
                    cosine_sim = cosine_similarity(list(emb[0]), list(emb[1]))

                candidates.append((cosine_sim, publication, section_id, cited_edition, other_edition, cited_text, other_text))

    if not candidates:
        return []

    # Most divergent first, capped to bound latency.
    candidates.sort(key=lambda x: x[0])
    candidates = candidates[:MAX_CONFLICT_CHECKS_PER_QUERY]

    conflicts: List[EditionConflict] = []
    for cosine_sim, publication, section_id, cited_edition, other_edition, cited_text, other_text in candidates:
        if cosine_sim >= CONFLICT_PREFILTER_COSINE:
            _cache_store(publication, section_id, cited_edition, other_edition, cosine_sim, False, "")
            continue  # too similar to meaningfully contradict — skip the expensive NLI call

        older_edition, newer_edition = sorted([cited_edition, other_edition])
        older_text = cited_text if cited_edition == older_edition else other_text
        newer_text = other_text if cited_edition == older_edition else cited_text

        conflict = await detect_conflicts(
            publication, section_id, older_edition, older_text, newer_edition, newer_text, section_id
        )
        _cache_store(
            publication, section_id, cited_edition, other_edition,
            cosine_sim, conflict.has_conflict, conflict.superseding_edition,
        )
        if conflict.has_conflict:
            conflicts.append(conflict)

    return conflicts
