"""
Centralised configuration for the White Box RAG governance framework.
Loads settings from .env file at the project root.
"""

import os
from dotenv import load_dotenv

# Load .env from project root (two levels up from backend/shared/)
load_dotenv()

# ---------- Gemini API ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ---------- Storage Paths ----------
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chromadb")
SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/metadata.db")

# ---------- Domain Constants ----------
# Domain-agnostic: "publication"/"collection" is a free-text label chosen at
# ingest time, not a fixed enum. Any non-empty, reasonably-sized string is valid
# (see ingestion/routes.py::_validate_collection_label). This system is not
# restricted to any single domain (finance, legal, engineering, etc.).
MAX_COLLECTION_LABEL_LENGTH = 64

# ---------- Chunking Defaults ----------
CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

# ---------- Retrieval Defaults ----------
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
FINAL_TOP_K = 10
RRF_K = 60  # Reciprocal Rank Fusion constant
