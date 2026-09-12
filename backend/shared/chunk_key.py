"""
chunk_key.py — canonical chunk identity.

A chunk's identity must be stable and globally unique across documents so
that two documents sharing the same section_id (e.g. both have a "1.1")
never collide in retrieval scoring, similarity matrices, or post-fusion
deduplication. Every module that builds or consumes a chunk identifier
converges on the helpers here instead of ad-hoc f-strings.
"""


def build_chunk_key(publication_name: str, edition_date: str, section_id: str, ordinal: int = 0) -> str:
    """The canonical, human-readable chunk identity: publication|edition|section|ordinal.

    `ordinal` disambiguates multiple chunks within the same section (e.g.
    after sliding-window splitting) — it must be stable per chunk (assigned
    once at ingest time), never a retrieval rank.
    """
    return f"{publication_name}|{edition_date}|{section_id}|{ordinal}"


def parse_chunk_key(key: str) -> dict:
    """Inverse of build_chunk_key(). Returns {} if `key` isn't in the canonical
    shape (e.g. a legacy bare section_id from data ingested before this existed)."""
    parts = key.split("|")
    if len(parts) != 4:
        return {}
    publication, edition, section_id, ordinal = parts
    try:
        ordinal = int(ordinal)
    except ValueError:
        return {}
    return {
        "publication_name": publication,
        "edition_date": edition,
        "section_id": section_id,
        "ordinal": ordinal,
    }


def content_key(publication_name: str, edition_date: str, section_id: str, chunk_text: str, prefix_len: int = 80) -> str:
    """Content-addressed fallback identity for chunks that arrive without a
    stored chunk_key (legacy data ingested before this migration). NOT the
    primary identity — build_chunk_key()/a stored chunk_key is, whenever
    available. Still domain-collision-safe since it's qualified by
    publication+edition+section, matching how RRF fusion already keys chunks.
    """
    return f"{publication_name}|{edition_date}|{section_id}|{chunk_text[:prefix_len]}"


def resolve_chunk_key(publication_name: str, edition_date: str, section_id: str, chunk_text: str, chunk_key: str = "") -> str:
    """The identity to actually use: the stored chunk_key if present (fresh
    ingests), else a content-addressed fallback (legacy pre-migration data)."""
    return chunk_key or content_key(publication_name, edition_date, section_id, chunk_text)
