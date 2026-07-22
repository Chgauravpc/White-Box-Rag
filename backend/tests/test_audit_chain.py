"""
Unit tests for the tamper-evident audit hash-chain (Feature 1).

Pure math over synthetic rows — no DB, no models, no Gemini. Mirrors exactly
what shared/database.py::finalize_audit_record persists and what the
/audit/verify-integrity endpoint checks.
"""

import json

from shared.audit_chain import (
    GENESIS_HASH,
    compute_record_hash,
    verify_chain,
    next_link,
    hashable_view,
)


def _build_chain(records: list[dict]) -> list[dict]:
    """Chain a list of payloads exactly as finalize_audit_record would, returning
    verify_chain-ready rows (chain_index ordered, record stored as JSON)."""
    rows = []
    prev_row = None
    for i, rec in enumerate(records):
        prev_hash, chain_index = next_link(prev_row)
        record_hash = compute_record_hash(rec, prev_hash)
        stored = dict(rec)
        stored["id"] = i + 1
        stored["prev_hash"] = prev_hash
        stored["record_hash"] = record_hash
        rows.append({
            "id": i + 1,
            "chain_index": chain_index,
            "prev_hash": prev_hash,
            "record_hash": record_hash,
            "record_json": json.dumps(stored, default=str),
        })
        prev_row = {"record_hash": record_hash, "chain_index": chain_index}
    return rows


def _sample_records():
    return [
        {"query": "What is X?", "trust_gate": {"status": "Safe", "overall_score": 0.9}},
        {"query": "What is Y?", "trust_gate": {"status": "Needs_Human_Review", "overall_score": 0.6}},
        {"query": "What is Z?", "trust_gate": {"status": "Non_Compliant", "overall_score": 0.2}},
    ]


def test_genesis_linkage():
    rows = _build_chain(_sample_records())
    assert rows[0]["prev_hash"] == GENESIS_HASH
    assert rows[0]["chain_index"] == 0
    assert rows[1]["prev_hash"] == rows[0]["record_hash"]
    assert rows[2]["prev_hash"] == rows[1]["record_hash"]


def test_intact_chain_verifies():
    rows = _build_chain(_sample_records())
    result = verify_chain(rows)
    assert result["intact"] is True
    assert result["count"] == 3
    assert result["first_break"] is None


def test_empty_chain_is_intact():
    result = verify_chain([])
    assert result["intact"] is True
    assert result["count"] == 0


def test_hashable_view_excludes_chain_metadata():
    rec = {"query": "q", "id": 5, "prev_hash": "abc", "record_hash": "def", "chain_index": 2}
    view = hashable_view(rec)
    assert view == {"query": "q"}


def test_content_tampering_detected():
    rows = _build_chain(_sample_records())
    # Mutate the middle record's stored payload without recomputing its hash.
    tampered = json.loads(rows[1]["record_json"])
    tampered["trust_gate"]["status"] = "Safe"  # attacker rewrites a Non-Compliant-adjacent verdict
    rows[1]["record_json"] = json.dumps(tampered, default=str)

    result = verify_chain(rows)
    assert result["intact"] is False
    assert result["first_break"]["reason"] == "content_tampered"
    assert result["first_break"]["id"] == 2


def test_deleted_record_breaks_linkage():
    rows = _build_chain(_sample_records())
    # Remove the middle row — row 3's prev_hash no longer matches row 1's record_hash.
    del rows[1]
    result = verify_chain(rows)
    assert result["intact"] is False
    assert result["first_break"]["reason"] == "broken_linkage"
    assert result["first_break"]["id"] == 3


def test_reordered_records_break_linkage():
    rows = _build_chain(_sample_records())
    rows[1], rows[2] = rows[2], rows[1]  # swap order
    result = verify_chain(rows)
    assert result["intact"] is False
    assert result["first_break"]["reason"] == "broken_linkage"


def test_hash_is_deterministic_regardless_of_key_order():
    a = {"query": "q", "trust_gate": {"status": "Safe", "overall_score": 0.9}}
    b = {"trust_gate": {"overall_score": 0.9, "status": "Safe"}, "query": "q"}
    assert compute_record_hash(a, GENESIS_HASH) == compute_record_hash(b, GENESIS_HASH)
