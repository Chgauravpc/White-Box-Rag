"""
Governance routes — human-in-the-loop (HITL) review of flagged audits.

Completes the NEEDS_HUMAN_REVIEW verdict that trust_gate.py already emits:
- GET  /api/review/queue            — audits awaiting human resolution
- POST /api/review/{audit_id}/resolve — append a chained resolution
- GET  /api/review/{audit_id}/history — full resolution history + current status

Resolutions are append-only, hash-chained review_actions — the audit row is
never mutated (that would break its own tamper-evident chain). Reviewer identity
is supplied in the request body; there is no auth in this system (documented
limitation).
"""

import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from shared.models import ACTION_TO_STATUS, ResolveReviewRequest
from shared.database import (
    insert_review_action,
    list_review_actions,
    latest_review_status,
    list_pending_reviews,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/review", tags=["Governance"])

# Serializes review-chain appends (see the audit chain lock in pipeline.py for the
# same rationale — read-last-then-insert must not interleave or the chain forks).
_review_lock = asyncio.Lock()


@router.get("/queue")
async def review_queue():
    """Audits flagged Needs_Human_Review with no resolution yet."""
    try:
        return {"status": "success", "data": list_pending_reviews()}
    except Exception as e:
        logger.error(f"Error building review queue: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to build review queue"})


@router.post("/{audit_id}/resolve")
async def resolve_review(audit_id: int, req: ResolveReviewRequest):
    """Append a human resolution (approve | override | reject) to the review chain."""
    if req.action not in ACTION_TO_STATUS:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid action '{req.action}'. Use one of: {list(ACTION_TO_STATUS)}"},
        )
    if not req.reviewer.strip():
        return JSONResponse(status_code=400, content={"status": "error", "message": "Reviewer is required."})

    try:
        async with _review_lock:
            record = insert_review_action(audit_id, req.reviewer.strip(), req.action, req.note)
        return {
            "status": "success",
            "message": "Review resolution recorded",
            "data": record,
            "review_status": latest_review_status(audit_id),
        }
    except Exception as e:
        logger.error(f"Error resolving review for audit {audit_id}: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to record resolution"})


@router.get("/{audit_id}/history")
async def review_history(audit_id: int):
    """Full chained resolution history for one audit, plus the derived current status."""
    try:
        return {
            "status": "success",
            "data": list_review_actions(audit_id),
            "review_status": latest_review_status(audit_id),
        }
    except Exception as e:
        logger.error(f"Error fetching review history for audit {audit_id}: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to fetch review history"})
