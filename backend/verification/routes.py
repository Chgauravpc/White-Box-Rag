from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel

from shared.models import (
    RAGResponse, 
    EditionConflict,
    VerificationResult,
    TrustGate,
    TrustScorecard
)
from .nli_engine import verify_all_claims
from .edition_conflict import detect_conflicts
from .trust_gate import compute_trust_gate
from .scorecard import generate_scorecard

# Combined response for Verify endpoints
class VerificationCompleteResponse(BaseModel):
    verifications: List[VerificationResult]
    conflicts: List[EditionConflict]
    trust_gate: TrustGate
    scorecard: TrustScorecard

router = APIRouter(prefix="/verify", tags=["Verification (BP2)"])

# Input model for conflict checking
class ConflictRequest(BaseModel):
    publication: str
    topic: str
    older_date: str
    older_text: str
    newer_date: str
    newer_text: str
    section_id: str

@router.post("/", response_model=VerificationCompleteResponse)
async def verify_rag_response(request: RAGResponse):
    """
    Takes a RAGResponse (answer + claims), and performs:
    1. NLI verification of all claims
    2. Trust Gating
    3. Scorecard generation
    (Assumes no edition conflicts passed directly here for simplicity, or could fetch internally)
    """
    if not request.claims:
        raise HTTPException(status_code=400, detail="No claims provided to verify.")
        
    # 1. Verify claims
    verifications = await verify_all_claims(request.claims)
    
    # 2. Gate (Assuming no conflicts passed in this simple flow)
    conflicts = [] 
    gate = compute_trust_gate(verifications, conflicts)
    
    # 3. Scorecard
    scorecard = generate_scorecard(
        query="Mocked Query", 
        response=request.answer, 
        verifications=verifications, 
        conflicts=conflicts,
        claims=request.claims
    )
    
    return VerificationCompleteResponse(
        verifications=verifications,
        conflicts=conflicts,
        trust_gate=gate,
        scorecard=scorecard
    )

@router.post("/check-conflicts", response_model=EditionConflict)
async def check_edition_conflicts(req: ConflictRequest):
    """
    Checks for contradictions between an older and newer edition of a publication.
    """
    conflict = await detect_conflicts(
        req.publication, req.topic, 
        req.older_date, req.older_text, 
        req.newer_date, req.newer_text, 
        req.section_id
    )
    return conflict
