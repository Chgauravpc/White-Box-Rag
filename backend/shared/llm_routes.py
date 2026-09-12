"""
Live pool observability (Phase 2, W2.4) — GET /api/llm/pool.

Deliberately built without ever constructing the pool as a side effect of an
unrelated import: this router calls shared.llm_pool.get_pool() only when the
endpoint is actually hit, same lazy-build discipline as call_llm() itself.
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/llm", tags=["llm"])


@router.get("/pool")
async def get_pool_status():
    """Per-endpoint health/headroom snapshot: which (provider, key) endpoints
    exist, how many calls are in flight, remaining RPM/TPM headroom, and
    error counts — never the raw key, only its fingerprint."""
    try:
        from shared.llm_pool import get_pool
        pool = get_pool()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM pool unavailable: {e}")
    return {"endpoint_count": pool.endpoint_count, "endpoints": pool.snapshot()}
