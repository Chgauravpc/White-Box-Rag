"""
Tests for the human-in-the-loop review chain (Feature 2).

Exercises the real DB helpers against conftest's temp SQLite (no Gemini, no
models). Covers: status derivation, supersession, pending-queue exclusion,
chain integrity, and tamper detection.
"""

import json

import pytest

from shared.database import (
    get_sqlite_connection,
    insert_review_action,
    list_review_actions,
    latest_review_status,
    list_pending_reviews,
    iter_review_chain,
)
from shared.audit_chain import verify_chain


def _seed_audit(query: str, status: str) -> int:
    """Insert a minimal finalized audit row and return its id."""
    conn = get_sqlite_connection()
    try:
        cur = conn.execute(
            "INSERT INTO audit_logs (timestamp, query, trust_gate_status, audit_data_json) VALUES (?,?,?,?)",
            ("2026-01-01T00:00:00", query, status, json.dumps({"query": query})),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def test_resolve_sets_derived_status():
    audit_id = _seed_audit("needs review q", "Needs_Human_Review")
    assert latest_review_status(audit_id) == "Pending"

    insert_review_action(audit_id, "alice", "approve", "looks fine")
    assert latest_review_status(audit_id) == "Approved"


def test_second_action_supersedes():
    audit_id = _seed_audit("supersede q", "Needs_Human_Review")
    insert_review_action(audit_id, "alice", "approve", "ok")
    insert_review_action(audit_id, "bob", "reject", "actually no")

    assert latest_review_status(audit_id) == "Rejected"
    history = list_review_actions(audit_id)
    assert len(history) == 2
    assert [h["action"] for h in history] == ["approve", "reject"]  # chain order preserved


def test_pending_queue_excludes_resolved():
    a_unresolved = _seed_audit("still pending", "Needs_Human_Review")
    a_resolved = _seed_audit("resolved one", "Needs_Human_Review")
    insert_review_action(a_resolved, "carol", "override", "override it")

    pending_ids = {row["id"] for row in list_pending_reviews()}
    assert a_unresolved in pending_ids
    assert a_resolved not in pending_ids


def test_safe_audits_never_in_queue():
    safe_id = _seed_audit("safe q", "Safe")
    pending_ids = {row["id"] for row in list_pending_reviews()}
    assert safe_id not in pending_ids


def test_review_chain_verifies_and_detects_tampering():
    audit_id = _seed_audit("chain q", "Needs_Human_Review")
    insert_review_action(audit_id, "alice", "approve", "one")
    insert_review_action(audit_id, "bob", "reject", "two")

    rows = iter_review_chain()
    assert verify_chain(rows)["intact"] is True

    # Tamper: rewrite a reviewer name in the stored row without re-hashing.
    conn = get_sqlite_connection()
    try:
        target = rows[0]["id"]
        conn.execute("UPDATE review_actions SET reviewer = ? WHERE id = ?", ("mallory", target))
        conn.commit()
    finally:
        conn.close()

    result = verify_chain(iter_review_chain())
    assert result["intact"] is False
    assert result["first_break"]["reason"] == "content_tampered"


def test_invalid_action_maps_to_pending():
    # latest_review_status is defensive: an unknown verb never crashes, falls to Pending.
    audit_id = _seed_audit("weird action", "Needs_Human_Review")
    insert_review_action(audit_id, "alice", "shrug", "not a real verb")
    assert latest_review_status(audit_id) == "Pending"
