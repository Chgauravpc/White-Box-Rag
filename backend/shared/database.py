"""
database.py - ChromaDB + SQLite initialisation.
Singletons shared across all services.
"""

import os
import json
import sqlite3
from datetime import datetime

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from shared.config import CHROMA_PATH, SQLITE_PATH


# ── ChromaDB ──────────────────────────────────────────────
_chroma_client     = None
_chroma_collection = None
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"


def get_chroma_collection():
    """Return the rbi_sections ChromaDB collection (lazy singleton)."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _chroma_client     = chromadb.PersistentClient(path=CHROMA_PATH)
        embedding_fn       = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="rbi_sections",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


# ── SQLite ─────────────────────────────────────────────────

def _init_sqlite_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            filename         TEXT NOT NULL,
            publication_name TEXT NOT NULL,
            edition_date     TEXT NOT NULL,
            chunk_count      INTEGER DEFAULT 0,
            ingested_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT NOT NULL,
            query             TEXT NOT NULL,
            response_json     TEXT,
            trust_gate_status TEXT,
            audit_data_json   TEXT,
            query_embedding   TEXT
        );

        CREATE TABLE IF NOT EXISTS edition_conflict_cache (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            publication  TEXT NOT NULL,
            section_id   TEXT NOT NULL,
            edition_a    TEXT NOT NULL,
            edition_b    TEXT NOT NULL,
            cosine_sim   REAL NOT NULL,
            has_conflict INTEGER NOT NULL,
            authoritative TEXT NOT NULL,
            cached_at    TEXT NOT NULL
        );
    """)
    conn.commit()


def get_sqlite_connection():
    """Return a new SQLite connection with row_factory set."""
    os.makedirs(os.path.dirname(SQLITE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    _init_sqlite_tables(conn)
    return conn


# ── SQLite Helper Functions ─────────────────────────────────

def insert_document(filename, publication_name, edition_date, chunk_count):
    conn = get_sqlite_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO documents (filename, publication_name, edition_date, chunk_count, ingested_at) VALUES (?, ?, ?, ?, ?)",
            (filename, publication_name, edition_date, chunk_count, datetime.now().isoformat()),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_documents():
    conn = get_sqlite_connection()
    try:
        rows = conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def list_documents_with_sections():
    collection = get_chroma_collection()
    docs       = list_documents()
    result     = []
    for doc in docs:
        pub   = doc["publication_name"]
        ed    = doc["edition_date"]
        chroma_results = collection.get(
            where={"$and": [{"publication_name": {"$eq": pub}}, {"edition_date": {"$eq": ed}}]},
            include=["metadatas"],
        )
        seen_sections = {}
        for meta in chroma_results.get("metadatas", []):
            sid = meta.get("section_id", "")
            if sid and sid not in seen_sections:
                seen_sections[sid] = {
                    "section_id":    sid,
                    "section_title": meta.get("section_title", ""),
                    "chunk_count":   0,
                    "page_number":   meta.get("page_number", 0),
                }
            if sid:
                seen_sections[sid]["chunk_count"] += 1
        doc["sections"] = list(seen_sections.values())
        result.append(doc)
    return result


def insert_audit_log(query, response_json, trust_gate_status=""):
    conn = get_sqlite_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO audit_logs (timestamp, query, response_json, trust_gate_status) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), query, response_json, trust_gate_status),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_audit_log(log_id):
    conn = get_sqlite_connection()
    try:
        row = conn.execute("SELECT * FROM audit_logs WHERE id = ?", (log_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_audit_logs():
    conn = get_sqlite_connection()
    try:
        rows = conn.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


# ── Alias for BP3 compatibility ─────────────────────────────

def get_sqlite_conn():
    return get_sqlite_connection()


# ── Query Embedding Storage (for related-query cosine lookup) ─

def store_query_embedding(log_id, embedding):
    """Persist a bge-large query embedding into the audit_logs row."""
    conn = get_sqlite_connection()
    try:
        conn.execute(
            "UPDATE audit_logs SET query_embedding = ? WHERE id = ?",
            (json.dumps(embedding), log_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_past_query_embeddings(exclude_id=None, limit=200):
    """Return past audit log entries with stored embeddings for cosine lookup.
    Includes timestamp so RelatedQuery Pydantic model is fully populated.
    """
    conn = get_sqlite_connection()
    try:
        rows = conn.execute(
            "SELECT id, query, timestamp, query_embedding, trust_gate_status "
            "FROM audit_logs WHERE query_embedding IS NOT NULL "
            "AND id != ? ORDER BY id DESC LIMIT ?",
            (exclude_id or -1, limit),
        ).fetchall()
        result = []
        for row in rows:
            try:
                emb = json.loads(row["query_embedding"])
            except Exception:
                continue
            result.append({
                "id":           row["id"],
                "query":        row["query"],
                "timestamp":    row["timestamp"] or "",
                "embedding":    emb,
                "trust_status": row["trust_gate_status"] or "unknown",
            })
        return result
    finally:
        conn.close()
