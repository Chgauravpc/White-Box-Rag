"""
Centralised configuration for the XAI Governance Framework.
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
PUBLICATIONS = ["FSR", "MPR", "PSR", "FER"]

# ---------- Chunking Defaults ----------
CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

# ---------- Retrieval Defaults ----------
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
FINAL_TOP_K = 10
RRF_K = 60  # Reciprocal Rank Fusion constant
