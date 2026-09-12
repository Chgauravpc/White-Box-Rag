"""
corpus_manifest.py — a frozen, content-hashed snapshot of the ingested corpus.

Why this exists (W3.1 in docs/BENCHMARK_READINESS.md). A labeled eval item
points at chunks by `relevant_chunk_keys`. Those keys encode a *position*
(publication|edition|section|ordinal), so they are only meaningful against one
specific corpus state: re-chunk with a different `CHUNK_MAX_TOKENS`, or reparse
with a PyMuPDF version whose section detection fires differently, and the same
key now names different text. Nothing would raise — the retrieval metric would
just quietly measure something else. Labeling hundreds of items without a way
to detect that is how a benchmark number becomes unreproducible.

So the manifest records, for every chunk, both its key and the SHA-256 of its
*normalized text* (`shared/hashing.py`). The text hash is the stability anchor:
the key says where a chunk was, the hash says whether it is still the same
chunk. `diff_manifest` reports drift in exactly those terms — a chunk that
moved (same text hash, new key) is recoverable by re-pointing a label, while a
chunk whose text changed under a stable key is a silent-invalidation event and
is reported separately.

Layering: this lives in `shared/` rather than `eval/` because it is storage
domain knowledge (it reads ChromaDB and SQLite and knows how ingestion keys
chunks), and because `ingestion/` may eventually want to validate against a
manifest at ingest time — `ingestion -> eval -> ingestion` would be a cycle,
`ingestion -> shared` is the direction everything already flows.

Authority. ChromaDB is authoritative for chunk text and chunk_key: it is what
ingestion writes and what retrieval reads back. SQLite `documents` is derived
metadata and can drift from it, so any disagreement between the two is recorded
as a first-class `inconsistencies` finding rather than silently resolved.
"""

import hashlib
import json
import os
from typing import Optional

from shared import config
from shared.hashing import TEXT_HASH_NORMALIZER, hash_text

MANIFEST_VERSION = 1

# A manifest over an empty corpus must not be mistaken for a real one: hashing
# "no documents" otherwise yields a perfectly valid-looking digest that would
# compare equal to any other empty snapshot on any machine.
EMPTY_CORPUS_HASH = "EMPTY"


# ── Building ────────────────────────────────────────────────

def _chunking_config() -> dict:
    """The settings that determine chunk identity.

    Deliberately a named block rather than the whole RunConfig: most of a run's
    config (temperatures, top_k, NLI thresholds) changes the *answer* without
    changing *what a chunk is*. Only these change chunk identity, so only these
    belong in the corpus hash.
    """
    try:
        from ingestion.pdf_parser import PARSER_VERSION
    except Exception:
        PARSER_VERSION = ""
    return {
        "chunk_max_tokens": config.CHUNK_MAX_TOKENS,
        "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
        "parser_version": PARSER_VERSION,
        # Changing the embedding model does not change chunk text, but it does
        # change what retrieval returns for the same query — and the collection
        # is bound to it at creation time, so a changed value silently reuses
        # the existing collection with stale vectors.
        "chroma_embedding_model": config.CHROMA_EMBEDDING_MODEL,
        "text_hash_normalizer": TEXT_HASH_NORMALIZER,
    }


def _document_hash(chunks: list[dict]) -> str:
    """Hash a document's chunk inventory: sorted (chunk_key, text_hash) pairs.

    Sorted so the hash is independent of the order Chroma happens to return
    rows in, which is not contractual.
    """
    payload = sorted((c["chunk_key"], c["text_sha256"]) for c in chunks)
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def build_manifest(collection=None, documents: Optional[list[dict]] = None) -> dict:
    """Snapshot the live corpus.

    Both dependencies are injectable so this is testable without a live
    ChromaDB, and so a caller can reconcile against a specific document list.
    """
    if collection is None:
        from shared.database import get_chroma_collection
        collection = get_chroma_collection()
    if documents is None:
        from shared.database import list_latest_documents
        documents = list_latest_documents()

    raw = collection.get(include=["documents", "metadatas"])
    texts = raw.get("documents", []) or []
    metas = raw.get("metadatas", []) or []
    ids = raw.get("ids", []) or []

    by_doc: dict[tuple, list[dict]] = {}
    for i, meta in enumerate(metas):
        meta = meta or {}
        text = texts[i] if i < len(texts) else ""
        chunk_key = meta.get("chunk_key") or (ids[i] if i < len(ids) else "")
        key = (meta.get("publication_name", ""), meta.get("edition_date", ""))
        by_doc.setdefault(key, []).append({
            "chunk_key": chunk_key,
            "section_id": meta.get("section_id", ""),
            "text_sha256": hash_text(text),
            "chars": len(text or ""),
        })

    sqlite_by_key = {(d.get("publication_name", ""), d.get("edition_date", "")): d for d in documents}

    doc_entries = []
    inconsistencies = []
    for key in sorted(set(by_doc) | set(sqlite_by_key)):
        publication, edition = key
        chunks = sorted(by_doc.get(key, []), key=lambda c: c["chunk_key"])
        meta = sqlite_by_key.get(key, {})

        if key not in by_doc:
            inconsistencies.append({
                "type": "document_without_chunks", "publication_name": publication,
                "edition_date": edition,
                "detail": "SQLite records this document but ChromaDB holds no chunks for it",
            })
        elif key not in sqlite_by_key:
            inconsistencies.append({
                "type": "chunks_without_document", "publication_name": publication,
                "edition_date": edition,
                "detail": f"ChromaDB holds {len(chunks)} chunks with no documents row",
            })
        elif meta.get("chunk_count") not in (None, len(chunks)):
            inconsistencies.append({
                "type": "chunk_count_mismatch", "publication_name": publication,
                "edition_date": edition,
                "detail": f"documents.chunk_count={meta.get('chunk_count')} but ChromaDB holds {len(chunks)}",
            })

        doc_entries.append({
            "publication_name": publication,
            "edition_date": edition,
            "filename": meta.get("filename", ""),
            "source_sha256": meta.get("source_sha256", "") or "",
            "parser_version": meta.get("parser_version", "") or "",
            "structured": meta.get("structured"),
            "chunk_count": len(chunks),
            "document_hash": _document_hash(chunks),
            "chunks": chunks,
        })

    chunking = _chunking_config()
    if doc_entries and any(d["chunk_count"] for d in doc_entries):
        corpus_hash = hashlib.sha256(json.dumps({
            "documents": [[d["publication_name"], d["edition_date"], d["document_hash"]] for d in doc_entries],
            "chunking_config": chunking,
        }, sort_keys=True).encode("utf-8")).hexdigest()
    else:
        corpus_hash = EMPTY_CORPUS_HASH

    return {
        "manifest_version": MANIFEST_VERSION,
        "corpus_hash": corpus_hash,
        "chunking_config": chunking,
        "document_count": len(doc_entries),
        "chunk_count": sum(d["chunk_count"] for d in doc_entries),
        "documents": doc_entries,
        "inconsistencies": inconsistencies,
    }


# ── Persistence ─────────────────────────────────────────────

def manifest_path(corpus_id: str, corpora_dir: Optional[str] = None) -> str:
    return os.path.join(corpora_dir or config.CORPORA_PATH, f"{corpus_id}.json")


def save_manifest(manifest: dict, corpus_id: str, corpora_dir: Optional[str] = None) -> str:
    path = manifest_path(corpus_id, corpora_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    stamped = dict(manifest, corpus_id=corpus_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stamped, f, indent=2, sort_keys=True)
    return path


def load_manifest(corpus_id: str, corpora_dir: Optional[str] = None) -> Optional[dict]:
    """Return the frozen manifest, or None if it doesn't exist.

    A missing manifest is an ordinary state (nothing frozen yet), not an error.
    """
    path = manifest_path(corpus_id, corpora_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Drift ───────────────────────────────────────────────────

def _chunk_index(manifest: dict) -> tuple[dict, dict]:
    """(chunk_key -> text_sha256, text_sha256 -> [chunk_key]) over the manifest."""
    by_key, by_hash = {}, {}
    for doc in manifest.get("documents", []):
        for chunk in doc.get("chunks", []):
            by_key[chunk["chunk_key"]] = chunk["text_sha256"]
            by_hash.setdefault(chunk["text_sha256"], []).append(chunk["chunk_key"])
    return by_key, by_hash


def diff_manifest(frozen: dict, live: dict) -> dict:
    """Compare a frozen manifest against a live one.

    The distinction that matters for labels:

    * `moved` — the text still exists but under a different chunk_key. Labels
      are recoverable: re-point them at the new key (that is why the text hash
      is stored alongside the key in the first place).
    * `content_changed` — the key survived but now names different text. This
      is the silent-invalidation case: a label still "resolves" and is wrong.
    * `removed` — the text is gone from the corpus entirely.
    """
    f_by_key, f_by_hash = _chunk_index(frozen)
    l_by_key, l_by_hash = _chunk_index(live)

    content_changed, moved, removed = [], [], []
    for key, f_hash in sorted(f_by_key.items()):
        l_hash = l_by_key.get(key)
        if l_hash == f_hash:
            continue
        if l_hash is None:
            new_keys = l_by_hash.get(f_hash, [])
            (moved if new_keys else removed).append(
                {"chunk_key": key, "text_sha256": f_hash, "now_at": new_keys} if new_keys
                else {"chunk_key": key, "text_sha256": f_hash}
            )
        else:
            content_changed.append({
                "chunk_key": key, "frozen_text_sha256": f_hash, "live_text_sha256": l_hash,
            })

    added = sorted(set(l_by_key) - set(f_by_key) - {k for m in moved for k in m["now_at"]})

    return {
        "identical": frozen.get("corpus_hash") == live.get("corpus_hash"),
        "frozen_corpus_hash": frozen.get("corpus_hash"),
        "live_corpus_hash": live.get("corpus_hash"),
        "chunking_config_changed": frozen.get("chunking_config") != live.get("chunking_config"),
        "content_changed": content_changed,
        "moved": moved,
        "removed": removed,
        "added": added,
        "summary": {
            "content_changed": len(content_changed), "moved": len(moved),
            "removed": len(removed), "added": len(added),
        },
    }


def resolve_labels(manifest: dict, relevant_chunk_keys: list[dict]) -> dict:
    """Resolve a dataset item's labels against a manifest.

    Hash-first: a label carrying `text_sha256` follows the text even when the
    chunk_key moved, which is the whole point of storing it. Falls back to the
    bare key for labels written before `text_sha256` existed — those cannot
    survive a re-chunk, and are reported as `unverifiable` so the gap is
    visible rather than assumed safe.
    """
    by_key, by_hash = _chunk_index(manifest)
    resolved, moved, missing, unverifiable = [], [], [], []
    for label in relevant_chunk_keys or []:
        key, text_hash = label.get("key", ""), label.get("text_sha256", "")
        if text_hash:
            if by_key.get(key) == text_hash:
                resolved.append(key)
            elif text_hash in by_hash:
                moved.append({"labeled_key": key, "now_at": by_hash[text_hash]})
            else:
                missing.append(key)
        elif key in by_key:
            unverifiable.append(key)
        else:
            missing.append(key)
    total = len(relevant_chunk_keys or [])
    return {
        "total": total, "resolved": resolved, "moved": moved,
        "missing": missing, "unverifiable": unverifiable,
        "resolved_fraction": round(len(resolved) / total, 6) if total else None,
    }
