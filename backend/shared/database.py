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
from shared.audit_chain import compute_record_hash, next_link


# ── ChromaDB ──────────────────────────────────────────────
_chroma_client     = None
_chroma_collection = None
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"


def get_chroma_collection():
    """Return the document_sections ChromaDB collection (lazy singleton)."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _chroma_client     = chromadb.PersistentClient(path=CHROMA_PATH)
        embedding_fn       = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="document_sections",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


# ── SQLite ─────────────────────────────────────────────────

def _ensure_column(conn, table: str, column: str, decl: str):
    """Idempotently add a column to an existing table.

    The schema is created with CREATE TABLE IF NOT EXISTS, which never alters a
    table that already exists — so new columns on pre-existing databases need an
    explicit ALTER. Guarded by PRAGMA table_info so it is safe to call every time.
    """
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _init_sqlite_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            filename         TEXT NOT NULL,
            publication_name TEXT NOT NULL,
            edition_date     TEXT NOT NULL,
            chunk_count      INTEGER DEFAULT 0,
            ingested_at      TEXT NOT NULL,
            structured       INTEGER DEFAULT 1
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

        CREATE TABLE IF NOT EXISTS eval_runs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_label     TEXT NOT NULL,
            dataset_path  TEXT NOT NULL,
            started_at    TEXT NOT NULL,
            finished_at   TEXT,
            num_queries   INTEGER,
            metrics_json  TEXT,
            per_query_json TEXT
        );

        CREATE TABLE IF NOT EXISTS brd_validation_runs (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT NOT NULL,
            source_filename TEXT,
            requirements_json TEXT NOT NULL,
            results_json   TEXT NOT NULL,
            overall_score  REAL
        );
    """)

    # Tamper-evident audit chain columns (added via ALTER so existing DBs upgrade).
    # chain_index defines the hash-chain order independently of row id, so
    # concurrent/out-of-order finalization can't fork the chain.
    _ensure_column(conn, "audit_logs", "prev_hash", "TEXT")
    _ensure_column(conn, "audit_logs", "record_hash", "TEXT")
    _ensure_column(conn, "audit_logs", "chain_index", "INTEGER")

    conn.commit()


def get_sqlite_connection():
    """Return a new SQLite connection with row_factory set."""
    os.makedirs(os.path.dirname(SQLITE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    _init_sqlite_tables(conn)
    return conn


# ── SQLite Helper Functions ─────────────────────────────────

def insert_document(filename, publication_name, edition_date, chunk_count, structured: bool = True):
    conn = get_sqlite_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO documents (filename, publication_name, edition_date, chunk_count, ingested_at, structured) VALUES (?, ?, ?, ?, ?, ?)",
            (filename, publication_name, edition_date, chunk_count, datetime.now().isoformat(), int(structured)),
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


# ── Eval Run Helpers ─────────────────────────────────────────

def insert_eval_run(run_label, dataset_path, started_at, finished_at, num_queries, metrics_json, per_query_json):
    conn = get_sqlite_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO eval_runs (run_label, dataset_path, started_at, finished_at, num_queries, metrics_json, per_query_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_label, dataset_path, started_at, finished_at, num_queries, metrics_json, per_query_json),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_eval_runs():
    conn = get_sqlite_connection()
    try:
        rows = conn.execute(
            "SELECT id, run_label, dataset_path, started_at, finished_at, num_queries, metrics_json "
            "FROM eval_runs ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_eval_run(run_id: int):
    conn = get_sqlite_connection()
    try:
        row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── BRD Validation Run Helpers (server-side history, not localStorage) ──────

def insert_brd_validation_run(source_filename, requirements_json, results_json, overall_score):
    conn = get_sqlite_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO brd_validation_runs (timestamp, source_filename, requirements_json, results_json, overall_score) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), source_filename, requirements_json, results_json, overall_score),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_brd_validation_runs():
    conn = get_sqlite_connection()
    try:
        rows = conn.execute(
            "SELECT id, timestamp, source_filename, overall_score FROM brd_validation_runs ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_brd_validation_run(run_id: int):
    conn = get_sqlite_connection()
    try:
        row = conn.execute("SELECT * FROM brd_validation_runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


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


# ── Tamper-evident audit chain ──────────────────────────────

def finalize_audit_record(log_id: int, full_record: dict) -> tuple[str, str, int]:
    """Chain-link and persist the COMPLETE audit record.

    The initial INSERT (in compliance/audit.py) stores only a partial report
    before the pipeline appends scorecard/XAI/faithfulness. This overwrites
    audit_data_json with the full record and stamps its hash-chain link.

    Caller MUST hold the pipeline chain lock — this does read-last-then-update,
    which is only race-free when serialized. `chain_index` (not row id) defines
    chain order, so out-of-order finalization can't fork the chain.
    Returns (prev_hash, record_hash, chain_index).
    """
    conn = get_sqlite_connection()
    try:
        row = conn.execute(
            "SELECT record_hash, chain_index FROM audit_logs "
            "WHERE chain_index IS NOT NULL ORDER BY chain_index DESC LIMIT 1"
        ).fetchone()
        prev_row = {"record_hash": row["record_hash"], "chain_index": row["chain_index"]} if row else None
        prev_hash, chain_index = next_link(prev_row)
        record_hash = compute_record_hash(full_record, prev_hash)

        stored = dict(full_record)
        stored["prev_hash"] = prev_hash
        stored["record_hash"] = record_hash
        conn.execute(
            "UPDATE audit_logs SET audit_data_json=?, prev_hash=?, record_hash=?, chain_index=? WHERE id=?",
            (json.dumps(stored, default=str), prev_hash, record_hash, chain_index, log_id),
        )
        conn.commit()
        return prev_hash, record_hash, chain_index
    finally:
        conn.close()


def iter_audit_chain() -> list:
    """Return all finalized audit rows in chain order for integrity verification."""
    conn = get_sqlite_connection()
    try:
        rows = conn.execute(
            "SELECT id, chain_index, prev_hash, record_hash, audit_data_json "
            "FROM audit_logs WHERE chain_index IS NOT NULL ORDER BY chain_index ASC"
        ).fetchall()
        return [{
            "id":          r["id"],
            "chain_index": r["chain_index"],
            "prev_hash":   r["prev_hash"],
            "record_hash": r["record_hash"],
            "record_json": r["audit_data_json"],
        } for r in rows]
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
