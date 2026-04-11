"""
Pydantic models — the team-wide data contract.
All backend services (BP1, BP2, BP3) and the frontend (FP1) code against these schemas.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────

class NLIVerdict(str, Enum):
    """Natural Language Inference verdict for a claim."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class TrustStatus(str, Enum):
    """Trust gate classification for a response."""
    SAFE = "Safe"
    NEEDS_HUMAN_REVIEW = "Needs_Human_Review"
    NON_COMPLIANT = "Non_Compliant"


class RiskLevel(str, Enum):
    """Risk level for BRD requirement compliance."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ──────────────────────────────────────────────
#  BP1: Ingestion & RAG Models
# ──────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """A single chunk of text from an RBI publication with full provenance."""
    model_config = ConfigDict(from_attributes=True)

    publication_name: str = Field(..., description="Publication code: FSR, MPR, PSR, or FER")
    edition_date: str = Field(..., description="Edition date, e.g. 'June 2024'")
    section_id: str = Field(..., description="Section identifier, e.g. '1.1', '2.3.1'")
    section_title: str = Field(default="", description="Section heading text")
    page_number: int = Field(default=0, description="Source page in the PDF")
    chunk_text: str = Field(..., description="The actual text content of this chunk")


class Claim(BaseModel):
    """A single claim extracted from a RAG response, with source attribution."""
    text: str = Field(..., description="The claim sentence from the generated answer")
    source_publication: str = Field(default="", description="Publication code of the source")
    source_edition: str = Field(default="", description="Edition date of the source")
    source_section_id: str = Field(default="", description="Section ID cited for this claim")
    source_passage: str = Field(default="", description="The actual source passage backing this claim")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score (BP2 overwrites)")


class RAGResponse(BaseModel):
    """Response from the RAG pipeline — answer text plus structured claims."""
    answer: str = Field(..., description="The full generated answer text")
    claims: list[Claim] = Field(default_factory=list, description="Individual claims with attributions")


class DocumentInfo(BaseModel):
    """Metadata about an ingested document."""
    id: int = Field(..., description="Database row ID")
    filename: str
    publication_name: str
    edition_date: str
    chunk_count: int = Field(default=0)
    ingested_at: str = Field(default="")


class SectionInfo(BaseModel):
    """Summary info about a section within a publication edition."""
    section_id: str
    section_title: str
    chunk_count: int = Field(default=0)
    page_number: int = Field(default=0)


# ──────────────────────────────────────────────
#  BP2: Verification & Trust Models
# ──────────────────────────────────────────────

class VerificationResult(BaseModel):
    """NLI-based verification result for a single claim."""
    claim_text: str
    verdict: NLIVerdict
    entailment_score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(default="")


class EditionConflict(BaseModel):
    """Detected conflict between different editions of a publication."""
    publication: str
    section_id: str
    older_edition: str
    newer_edition: str
    has_conflict: bool = Field(default=False)
    conflict_description: str = Field(default="")
    superseding_edition: str = Field(default="")
    details: str = Field(default="")


class TrustGate(BaseModel):
    """Trust gate decision for a RAG response."""
    status: TrustStatus
    reasoning: str = Field(default="")
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)


class TrustScorecard(BaseModel):
    """RAGAS-style scorecard with 5 trust metrics."""
    context_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    faithfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    edition_conflict_risk: bool = Field(default=False)
    paraphrase_stability: float = Field(default=0.0, ge=0.0, le=1.0)


# ──────────────────────────────────────────────
#  BP3: Compliance & Audit Models
# ──────────────────────────────────────────────

class BRDRequirement(BaseModel):
    """A single requirement extracted from a BRD document."""
    id: str = Field(..., description="Requirement ID, e.g. REQ-001")
    text: str = Field(..., description="Requirement text")
    category: str = Field(default="", description="e.g. 'payment processing', 'KYC'")
    regulatory_relevance: str = Field(default="", description="RBI domain: FSR/MPR/PSR/FER")
    mapped_sections: list[str] = Field(default_factory=list)
    alignment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    gaps: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    remediation: str = Field(default="")


class AuditReport(BaseModel):
    """Full audit trail for a governance interaction."""
    id: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    query: str = Field(default="")
    response: str = Field(default="")
    claims: list[Claim] = Field(default_factory=list)
    verifications: list[VerificationResult] = Field(default_factory=list)
    trust_gate: Optional[TrustGate] = None
    edition_conflicts: list[EditionConflict] = Field(default_factory=list)


# ──────────────────────────────────────────────
#  API Request Models
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the /api/query endpoint."""
    query: str = Field(..., min_length=1, description="The user's question")
    filters: Optional[dict] = Field(default=None, description="Optional filters: publication_name, edition_date")
