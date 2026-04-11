"""
Database initialisation — ChromaDB (vector store) + SQLite (metadata & audit).
Provides module-level singletons so all services share the same connections.
"""

import os
import sqlite3
from datetime import datetime

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from shared.config import CHROMA_PATH, SQLITE_PATH


# ──────────────────────────────────────────────
#  ChromaDB
# ──────────────────────────────────────────────

_chroma_client = None
_chroma_collection = None

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_chroma_collection():
    """Return the 'rbi_sections' ChromaDB collection (lazy singleton).

    Uses persistent storage at CHROMA_PATH with all-MiniLM-L6-v2 embeddings.
    """
    global _chroma_client, _chroma_collection

    if _chroma_collection is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="rbi_sections",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


# ──────────────────────────────────────────────
#  SQLite
# ──────────────────────────────────────────────

def _init_sqlite_tables(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT NOT NULL,
            publication_name TEXT NOT NULL,
            edition_date TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            ingested_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            query           TEXT NOT NULL,
            response_json   TEXT,
            trust_gate_status TEXT,
            audit_data_json TEXT
        );
    """)
    conn.commit()


def get_sqlite_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set.

    Creates the DB file and tables on first call.
    """
    os.makedirs(os.path.dirname(SQLITE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    _init_sqlite_tables(conn)
    return conn


# ──────────────────────────────────────────────
#  SQLite Helper Functions
# ──────────────────────────────────────────────

def insert_document(filename: str, publication_name: str, edition_date: str, chunk_count: int) -> int:
    """Insert a document record and return its ID."""
    conn = get_sqlite_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO documents (filename, publication_name, edition_date, chunk_count, ingested_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (filename, publication_name, edition_date, chunk_count, datetime.now().isoformat()),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_documents() -> list[dict]:
    """Return all ingested documents as a list of dicts."""
    conn = get_sqlite_connection()
    try:
        rows = conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def insert_audit_log(query: str, response_json: str, trust_gate_status: str = "") -> int:
    """Insert an audit log entry and return its ID."""
    conn = get_sqlite_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO audit_logs (timestamp, query, response_json, trust_gate_status) "
            "VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), query, response_json, trust_gate_status),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_audit_log(log_id: int) -> dict | None:
    """Retrieve a single audit log by ID."""
    conn = get_sqlite_connection()
    try:
        row = conn.execute("SELECT * FROM audit_logs WHERE id = ?", (log_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_audit_logs() -> list[dict]:
    """Return all audit logs, newest first."""
    conn = get_sqlite_connection()
    try:
        rows = conn.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


# ──────────────────────────────────────────────
#  Alias for BP3 compatibility
# ──────────────────────────────────────────────

def get_sqlite_conn() -> sqlite3.Connection:
    """Alias for get_sqlite_connection() — used by BP3 compliance module."""
    return get_sqlite_connection()
