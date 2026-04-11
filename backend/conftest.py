"""
pytest configuration — adds backend/ to sys.path so absolute imports work.
Sets storage paths to temp dirs for test isolation.
"""
import os
import sys
import tempfile

# Add backend/ to sys.path so absolute imports like `from shared.X` resolve
sys.path.insert(0, os.path.dirname(__file__))

# Redirect DB storage to a temp directory so tests don't pollute real data
_tmp = tempfile.mkdtemp()
os.environ.setdefault("CHROMA_PATH", os.path.join(_tmp, "chromadb"))
os.environ.setdefault("SQLITE_PATH", os.path.join(_tmp, "metadata.db"))
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
