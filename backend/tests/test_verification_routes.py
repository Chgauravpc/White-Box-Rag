"""
Tests for verification/routes.py.

`POST /api/verify/` was dead: it unpacked three values from
`verify_all_claims`, which has returned a 4-tuple since the premise-deletion
workstream added `premise_deletions`. Every call raised
`ValueError: too many values to unpack`, and nothing caught it because no test
exercised the route at all.

The repo's tests are unit-level (no TestClient anywhere) and drive coroutines
with `asyncio.run` rather than depending on pytest-asyncio — both conventions
are followed here. Calling the handler directly is enough to pin the arity
contract, which is the thing that actually broke.
"""

import asyncio

import pytest

from shared.models import Claim, RAGResponse
from verification.routes import verify_rag_response


def test_verify_route_runs_end_to_end():
    """Regression: this raised ValueError on every call."""
    request = RAGResponse(
        answer="Reserves rose sharply.",
        claims=[Claim(
            text="Reserves rose sharply.",
            source_passage="Foreign exchange reserves increased from USD 668B to USD 700B.",
            source_publication="FER", source_edition="Oct 2025", source_section_id="I.2.1",
        )],
    )
    result = asyncio.run(verify_rag_response(request))
    assert result.verifications
    assert result.trust_gate is not None


def test_verify_route_rejects_empty_claims():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        asyncio.run(verify_rag_response(RAGResponse(answer="anything", claims=[])))
    assert exc.value.status_code == 400


def test_verify_route_handles_claim_without_a_source_passage():
    """A sandbox caller can post a claim with no premise; that must produce an
    honest NOT_ENOUGH_INFO rather than a fabricated NLI score."""
    request = RAGResponse(
        answer="Unsupported.",
        claims=[Claim(text="Unsupported assertion with no source.")],
    )
    result = asyncio.run(verify_rag_response(request))
    assert result.verifications[0].evidence_status == "no_premise"
    assert result.verifications[0].entailment_score == 0.0
