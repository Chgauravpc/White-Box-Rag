"""
hashing.py — the content-hash primitives corpus identity is built on.

Two distinct jobs live here, and they hash different things on purpose:

* `hash_file` fingerprints raw *bytes* — "is this the same PDF I ingested?".
  Chunked reads, because a benchmark corpus PDF can be hundreds of megabytes
  and the previous implementation (`eval/harness.py::_hash_file`) read the
  whole file into memory.

* `hash_text` fingerprints *normalized* text — "is this the same chunk?".
  Raw PyMuPDF output carries incidental whitespace and line-join artifacts
  that differ across PyMuPDF versions, so hashing it verbatim would report
  corpus drift on a dependency bump that changed no content. Normalization is
  deliberately minimal and explicitly versioned (`TEXT_HASH_NORMALIZER`): it
  must never do the domain-specific stripping that
  `shared/text_normalize.py` does for NLI premises, because a chunk hash has
  to stay stable under a change of domain profile.
"""

import hashlib
import unicodedata

# Bump when the normalization below changes — a stored hash is only
# comparable to a new one computed by the same normalizer.
TEXT_HASH_NORMALIZER = "nfc-collapse-ws-v1"

_CHUNK_BYTES = 1024 * 1024


def hash_file(path: str) -> str:
    """SHA-256 of a file's raw bytes, read incrementally.

    Returns "" when the file can't be read — a missing source PDF is a fact
    to record in a manifest, not a reason to fail an ingest or an eval run.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(_CHUNK_BYTES), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def normalize_for_hash(text: str) -> str:
    """NFC-normalize and collapse all runs of whitespace to single spaces.

    Intentionally lossy only in ways that cannot change meaning, so that the
    same chunk text extracted by two PyMuPDF versions hashes identically.
    """
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def hash_text(text: str) -> str:
    """SHA-256 of `text` after `normalize_for_hash`. The chunk-identity anchor."""
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()
