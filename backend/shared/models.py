"""
Pydantic models — the team-wide data contract.
All backend services (BP1, BP2, BP3) and the frontend (FP1) code against these schemas.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────

class NLIVerdict(str, Enum):
    """Natural Language Inference verdict for a claim.
    
    Canonical values: SUPPORTED, CONTRADICTED, NOT_ENOUGH_INFO
    Aliases (BP2 compatibility): ENTAILMENT, CONTRADICTION, NEUTRAL
    """
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"
    # BP2 aliases
    ENTAILMENT = "ENTAILMENT"
    CONTRADICTION = "CONTRADICTION"
    NEUTRAL = "NEUTRAL"


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


class ReviewStatus(str, Enum):
    """Human-in-the-loop resolution state for an audit flagged Needs_Human_Review."""
    PENDING = "Pending"
    APPROVED = "Approved"
    OVERRIDDEN = "Overridden"
    REJECTED = "Rejected"


# Reviewer-supplied action verb → resulting ReviewStatus (the set of valid actions).
ACTION_TO_STATUS = {
    "approve": ReviewStatus.APPROVED,
    "override": ReviewStatus.OVERRIDDEN,
    "reject": ReviewStatus.REJECTED,
}


# ──────────────────────────────────────────────
#  BP1: Ingestion & RAG Models
# ──────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """A single chunk of text from an ingested document with full provenance."""
    model_config = ConfigDict(from_attributes=True)

    publication_name: str = Field(..., description="Free-text collection/document label")
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
    source_passage: str = Field(default="", description="Full source chunk — for human reading")
    focused_passage: str = Field(default="", description="Top-K sentences fed to NLI — for audit")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score (BP2 overwrites)")
    retained: bool = Field(default=True, description="False if stripped from the mitigated/filtered answer")
    filter_reason: str = Field(default="", description="Why this claim was stripped or flagged, e.g. 'contradiction', 'low_confidence', 'flagged:neutral'")


class RAGResponse(BaseModel):
    """Response from the RAG pipeline — answer text plus structured claims."""
    answer: str = Field(..., description="The full generated answer text")
    claims: list[Claim] = Field(default_factory=list, description="Individual claims with attributions")


class DocumentInfo(BaseModel):
    """Metadata about an ingested document.

    `publication_name`/`edition_date` are domain-agnostic despite the naming —
    they're a free-text "collection name" / "version label" pair, not tied to
    any specific document domain.
    """
    id: int = Field(..., description="Database row ID")
    filename: str
    publication_name: str
    edition_date: str
    chunk_count: int = Field(default=0)
    ingested_at: str = Field(default="")
    structured: bool = Field(default=True, description="False if section detection fell back to page-level citation")


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
    """RAGAS-style scorecard of pure-math trust/quality metrics (no LLM-as-judge)."""
    context_relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean cosine(query, retrieved chunk)")
    faithfulness: float = Field(default=0.0, ge=0.0, le=1.0, description="Alias for faithfulness_post on AuditReport")
    citation_precision: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of claims with a real source citation")
    edition_conflict_risk: bool = Field(default=False)
    paraphrase_stability: float = Field(default=1.0, ge=0.0, le=1.0, description="NLI claim-agreement with a second sample; 1.0 (skipped) when the primary answer is already Safe")
    answer_relevancy: float = Field(default=0.0, ge=0.0, le=1.0, description="cosine(query, answer) — simplified proxy, not RAGAS's LLM-based metric")
    context_utilization: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of retrieved chunks actually cited by >=1 claim")
    context_diversity: float = Field(default=0.0, ge=0.0, le=1.0, description="1 - mean pairwise cosine of retrieved chunk embeddings")


# ──────────────────────────────────────────────
#  BP3: Compliance & Audit Models
# ──────────────────────────────────────────────

class BRDRequirement(BaseModel):
    """A single requirement extracted from a BRD document."""
    id: str = Field(..., description="Requirement ID, e.g. REQ-001")
    text: str = Field(..., description="Requirement text")
    category: str = Field(default="", description="e.g. 'payment processing', 'KYC'")
    regulatory_relevance: str = Field(default="", description="Relevant domain/category of the requirement")
    mapped_sections: list[str] = Field(default_factory=list)
    alignment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    gaps: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)
    remediation: str = Field(default="")


class RetrievalMatrix(BaseModel):
    chunk_ids: list[str] = Field(description="section IDs, length n")
    similarity_scores: list[float] = Field(description="S vector, length n")

class EntailmentMatrix(BaseModel):
    claim_texts:   list[str] = Field(description="length m")
    passage_ids:   list[str] = Field(description="section IDs only, e.g. COLLECTION-§I.2.1")
    passage_texts: list[str] = Field(default_factory=list, description="raw passage text for audit display")
    scores:        list[list[list[float]]] = Field(description="shape (m, n, 3) — E[i][j] = [contradiction, entailment, neutral] probs")
    labels:        list[str] = Field(default=["contradiction", "entailment", "neutral"])

class AttributionMatrix(BaseModel):
    sentence_texts:       list[str] = Field(description="length m")
    chunk_ids:            list[str] = Field(description="length n")
    scores:               list[list[float]] = Field(description="shape (m, n) — A matrix")
    primary_attributions: list[dict] = Field(
        default_factory=list,
        description="Pre-computed top attribution per sentence with runner-up and confidence gap"
    )

class ConflictMatrix(BaseModel):
    old_section_ids: list[str] = Field(description="length p")
    new_section_ids: list[str] = Field(description="length q")
    scores: list[list[float]] = Field(description="shape (p, q) — C matrix, contradiction probs")
    threshold: float = Field(description="what was used to flag conflicts")

class ShapleyContributions(BaseModel):
    claim_texts: list[str]
    shapley_values: list[float] = Field(description="φᵢ per claim")
    penalty_reasons: list[list[str]] = Field(description="reasons per claim")
    overall_score: float

class XAIArtifacts(BaseModel):
    retrieval: RetrievalMatrix
    entailment: EntailmentMatrix
    attribution: AttributionMatrix
    conflict: Optional[ConflictMatrix] = None
    shapley: ShapleyContributions


class RelatedQuery(BaseModel):
    """A past query semantically similar to the current query."""
    id: int
    timestamp: str
    query: str
    cosine_similarity: float = Field(description="cos(q_current, q_past) ∈ [0,1]")
    trust_status: str = Field(default="")


class AuditReport(BaseModel):
    """Full audit trail for a governance interaction."""
    id: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    query: str = Field(default="")
    response: str = Field(default="", description="Raw, unfiltered Gemini answer — preserved for traceability")
    claims: list[Claim] = Field(default_factory=list)
    verifications: list[VerificationResult] = Field(default_factory=list)
    trust_gate: Optional[TrustGate] = None
    edition_conflicts: list[EditionConflict] = Field(default_factory=list)
    # XAI Mathematical Artifacts
    xai_artifacts: Optional[XAIArtifacts] = None
    related_queries: List[RelatedQuery] = Field(default_factory=list,
        description="Past queries with high cosine similarity to this query")
    # Hallucination mitigation (claim-level filtering + abstention)
    filtered_claims: list[str] = Field(default_factory=list,
        description="Retained (grounded) claim texts — the 'Verified statements' view, not a reconstructed narrative")
    faithfulness_raw: float = Field(default=0.0, ge=0.0, le=1.0, description="entailed/total over ALL claims")
    faithfulness_post: float = Field(default=0.0, ge=0.0, le=1.0, description="entailed/total over RETAINED claims only")
    retained_trust_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Trust Gate score recomputed over retained claims only")
    abstained: bool = Field(default=False)
    abstention_reason: str = Field(default="")
    scorecard: Optional[TrustScorecard] = None
    # Observability (latency & Gemini call cost)
    latency_ms: Dict[str, float] = Field(default_factory=dict, description="Per-stage wall-clock time")
    gemini_call_count: int = Field(default=0)
    # Tamper-evident audit chain (Feature 1) — SHA-256 link to the prior record
    prev_hash: Optional[str] = Field(default=None, description="record_hash of the preceding audit in the chain")
    record_hash: Optional[str] = Field(default=None, description="SHA-256 of this record chained on prev_hash")


# ──────────────────────────────────────────────
#  API Request Models
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the /api/query endpoint."""
    query: str = Field(..., min_length=1, description="The user's question")
    filters: Optional[dict] = Field(default=None, description="Optional filters: publication_name, edition_date")


# ──────────────────────────────────────────────
#  Governance: Human-in-the-loop Review
# ──────────────────────────────────────────────

class ReviewAction(BaseModel):
    """A single human resolution appended to the tamper-evident review chain."""
    id: Optional[int] = None
    audit_log_id: int
    reviewer: str = Field(..., description="Reviewer identity (supplied in the request — no auth in this system)")
    action: str = Field(..., description="approve | override | reject")
    note: str = Field(default="")
    timestamp: str = Field(default="")
    prev_hash: Optional[str] = None
    record_hash: Optional[str] = None


class ResolveReviewRequest(BaseModel):
    """Request body for POST /api/review/{audit_id}/resolve."""
    reviewer: str = Field(..., min_length=1)
    action: str = Field(..., description="approve | override | reject")
    note: str = Field(default="")
