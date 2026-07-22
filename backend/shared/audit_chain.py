"""
audit_chain.py — tamper-evident hash-chaining primitives.

Every persisted audit (and every human-review action) is linked to its
predecessor by SHA-256, blockchain-style: record_hash = sha256(prev_hash +
canonical_json(record)). Any retroactive edit, deletion, or reordering breaks
the chain and is detectable by `verify_chain`.

Pure standard-library — no ML, no Gemini, no DB. The DB layer (shared/database.py)
supplies rows; this module only does the math so it stays trivially unit-testable.
"""

import hashlib
import json
from typing import Iterable, Optional

# Genesis link for the first record in a chain.
GENESIS_HASH = "0" * 64

# Keys that are metadata *about* the chain position, not part of the hashed
# payload. Excluded from the canonical view so a record can't hash over its own
# hash, and so the stored JSON (which carries these for the frontend) still
# reproduces the original hash at verify time.
_EXCLUDED_KEYS = ("id", "prev_hash", "record_hash", "chain_index")


def hashable_view(record: dict) -> dict:
    """Return the record without chain-metadata keys — the exact payload hashed."""
    return {k: v for k, v in record.items() if k not in _EXCLUDED_KEYS}


def canonical_json(record: dict) -> str:
    """Deterministic serialization: sorted keys, no whitespace, str-coerced fallbacks.

    sort_keys makes key order irrelevant; default=str coerces enums/datetimes the
    same way at compute and verify time. Both sides feed `hashable_view` output
    here, so the text — and therefore the hash — is reproducible.
    """
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


def compute_record_hash(record: dict, prev_hash: str) -> str:
    """SHA-256 over (prev_hash + canonical payload). `record` may be raw (unstripped)."""
    payload = prev_hash + canonical_json(hashable_view(record))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_chain(rows: Iterable[dict]) -> dict:
    """Walk a chain in order and confirm it is intact.

    Each row must provide: `chain_index`, `record_hash`, `prev_hash`, and the
    hashed payload — supplied either as a parsed `record` dict or a `record_json`
    string. Rows must already be ordered by `chain_index` ascending.

    Returns {"intact": bool, "count": int, "first_break": {index, id, reason} | None}.
    Detects content tampering (recomputed hash mismatch) and broken linkage
    (a deleted/reordered predecessor).
    """
    expected_prev = GENESIS_HASH
    count = 0
    for row in rows:
        count += 1
        record = row.get("record")
        if record is None:
            record = json.loads(row["record_json"])

        prev_hash = row.get("prev_hash") or ""
        stored_hash = row.get("record_hash") or ""

        if prev_hash != expected_prev:
            return {
                "intact": False,
                "count": count,
                "first_break": {
                    "chain_index": row.get("chain_index"),
                    "id": row.get("id"),
                    "reason": "broken_linkage",
                },
            }

        recomputed = compute_record_hash(record, prev_hash)
        if recomputed != stored_hash:
            return {
                "intact": False,
                "count": count,
                "first_break": {
                    "chain_index": row.get("chain_index"),
                    "id": row.get("id"),
                    "reason": "content_tampered",
                },
            }

        expected_prev = stored_hash

    return {"intact": True, "count": count, "first_break": None}


def next_link(prev_row: Optional[dict]) -> tuple[str, int]:
    """Given the last finalized row (or None), return (prev_hash, chain_index) for the next.

    Callers hold the chain lock while using this so assignment stays sequential.
    """
    if prev_row is None:
        return GENESIS_HASH, 0
    return prev_row["record_hash"], prev_row["chain_index"] + 1
