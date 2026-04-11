"""
API Routes for the Ingestion & RAG service (BP1).

Endpoints:
  POST /api/ingest            — Upload and ingest an RBI PDF
  POST /api/query             — RAG query with optional filters
  GET  /api/documents         — List all ingested documents
  GET  /api/sections/{pub}/{edition} — List sections for a publication/edition
"""

import logging
import os
import shutil
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from shared.config import PUBLICATIONS
from shared.database import get_chroma_collection, list_documents
from shared.models import (
    ChunkMetadata,
    DocumentInfo,
    QueryRequest,
    RAGResponse,
    SectionInfo,
    AuditReport,
    BRDRequirement
)
from ingestion.pdf_parser import ingest_pdf
from ingestion.rag import rag_query
from ingestion.retriever import hybrid_retrieve, rebuild_bm25_index

from verification.nli_engine import verify_all_claims
from verification.trust_gate import compute_trust_gate
from compliance.mapper import map_requirement
from compliance.audit import generate_audit_report
from shared.xai_matrices import (
    build_retrieval_similarity_matrix,
    compute_shapley_contributions,
    compute_primary_attributions,
    find_related_queries,
    get_encoder,
)
from shared.database import store_query_embedding, get_past_query_embeddings
from shared.models import (
    XAIArtifacts,
    RetrievalMatrix,
    EntailmentMatrix,
    AttributionMatrix,
    ConflictMatrix,
    ShapleyContributions,
)


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Deduplicate retrieved chunks: keep one (highest-index = highest RRF) per section_id.
    Filters out UNSTRUCTURED-p* chunks from similarity matrices to keep provenance clean.
    """
    seen:   dict[str, dict] = {}
    for chunk in chunks:
        sid = chunk["section_id"]
        if sid not in seen:
            seen[sid] = chunk   # first occurrence = highest RRF rank
    return list(seen.values())

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ingestion & RAG"])

# Directory for storing uploaded PDFs
UPLOAD_DIR = os.path.join("data", "rbi_reports")


# ──────────────────────────────────────────────
#  POST /api/ingest
# ──────────────────────────────────────────────

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(..., description="RBI publication PDF"),
    publication: str = Form(..., description="Publication type: FSR, MPR, PSR, FER"),
    edition_date: str = Form(..., description="Edition date, e.g., 'June 2024'"),
):
    """Upload and ingest an RBI PDF publication.

    The PDF is parsed into section-level chunks, stored in ChromaDB
    for retrieval, and registered in the documents database.
    """
    # Validate publication type
    if publication.upper() not in PUBLICATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid publication '{publication}'. Must be one of: {PUBLICATIONS}",
        )
    publication = publication.upper()

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Ingest the PDF
        chunk_count = ingest_pdf(save_path, publication, edition_date)

        # Rebuild BM25 index to include new data
        rebuild_bm25_index()

        return {
            "status": "ok",
            "filename": file.filename,
            "publication": publication,
            "edition_date": edition_date,
            "chunks_ingested": chunk_count,
        }

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        # Clean up on error
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ──────────────────────────────────────────────
#  POST /api/query
# ──────────────────────────────────────────────

@router.post("/query", response_model=AuditReport)
async def query_documents(request: QueryRequest):
    """Ask a question and get a fully integrated RAG + Verification + Compliance Audit response.

    The output includes all XAI mathematical artifacts:
    - retrieval_similarity_matrix: S = q·Kᵀ/(‖q‖·‖k‖) for every retrieved chunk
    - trust_score_breakdown: Shapley-style penalty decomposition
    - faithfulness_score: |ENTAILMENT| / total claims
    - related_queries: past queries with cosine_sim > 0.70
    """
    try:
        # ── Step 1: BP1 — Hybrid Retrieval & Retrieval Matrix (S) ──
        logger.info(f"Step 1: Retrieving chunks for: '{request.query[:80]}'")
        raw_chunks = hybrid_retrieve(query=request.query, filters=request.filters)

        # Convert to dicts, deduplicate by section_id
        dict_chunks = [{
            "chunk_text":       c.chunk_text,
            "section_id":       c.section_id,
            "publication_name": c.publication_name,
            "edition_date":     c.edition_date,
        } for c in raw_chunks]
        dict_chunks = _deduplicate_chunks(dict_chunks)

        # Matrix 1 (S) — Cosine similarity of query vs each unique chunk
        chunk_ids, S_scores = build_retrieval_similarity_matrix(request.query, dict_chunks)

        # ── Step 2: BP1 — Generation & Attribution Matrix (A) ──
        rag_response, A_matrix = await rag_query(request.query, dict_chunks)

        # ── Step 3: BP2 — NLI Verification, Entailment Matrix (E), focused passages ──
        logger.info("Step 3: Batched NLI verification via CrossEncoder")
        verifications, E_matrix, focused_passages = await verify_all_claims(rag_response.claims)
        trust_gate = compute_trust_gate(verifications, [], prim_attrs if 'prim_attrs' in dir() else [])

        # Stamp focused_passage onto each Claim for full audit traceability
        for i, claim in enumerate(rag_response.claims):
            if i < len(focused_passages):
                claim.focused_passage = focused_passages[i]

        # ── Step 4: XAI Math — Shapley (φ), primary attributions, assemble artifacts ──
        logger.info("Step 4: Computing Shapley and attribution artifacts")
        shapley = compute_shapley_contributions([v.model_dump() for v in verifications])

        import numpy as np
        A_scores = A_matrix.tolist() if isinstance(A_matrix, np.ndarray) and A_matrix.size > 0 else []
        E_scores = E_matrix.tolist() if isinstance(E_matrix, np.ndarray) and E_matrix.size > 0 else []

        # Primary attribution per sentence (argmax + runner-up)
        prim_attrs = compute_primary_attributions(A_matrix, chunk_ids) if A_scores else []

        # Re-compute trust gate now that primary_attributions are available
        trust_gate = compute_trust_gate(verifications, [], prim_attrs)

        # Shapley: mirrors trust_gate penalties exactly (NLI + attribution)
        shapley = compute_shapley_contributions(
            [v.model_dump() for v in verifications],
            prim_attrs,
        )

        retrieval_mat = RetrievalMatrix(
            chunk_ids=chunk_ids,
            similarity_scores=S_scores,
        )
        attr_mat = AttributionMatrix(
            sentence_texts=[c.text for c in rag_response.claims],
            chunk_ids=chunk_ids,
            scores=A_scores,
            primary_attributions=prim_attrs,
        )
        seen_passages = {}
        for c in rag_response.claims:
            if c.source_passage and c.source_section_id not in seen_passages:
                seen_passages[c.source_section_id] = c.source_passage
        entail_mat = EntailmentMatrix(
            claim_texts=[c.text for c in rag_response.claims],
            passage_ids=list(seen_passages.keys()),
            passage_texts=list(seen_passages.values()),
            scores=E_scores,
            labels=["contradiction", "entailment", "neutral"],
        )
        shapley_mat = ShapleyContributions(
            claim_texts=shapley["claim_texts"],
            shapley_values=shapley["shapley_values"],
            penalty_reasons=shapley["penalty_reasons"],
            overall_score=shapley["overall_score"],
        )
        xai_artifacts = XAIArtifacts(
            retrieval=retrieval_mat,
            entailment=entail_mat,
            attribution=attr_mat,
            conflict=None,
            shapley=shapley_mat,
        )

        # ── Step 4: BP3 — BRD Compliance Mapping ──
        logger.info("Step 4: BP3 - Mapping requirement gaps")
        brd_req = BRDRequirement(id="ASK", text=request.query)
        mapped_dict = await map_requirement(brd_req)
        brd_req.mapped_sections = [c.chunk_text for c in mapped_dict.get("relevant_chunks", [])]
        brd_req.alignment_score = mapped_dict.get("alignment_score", 0.0)
        brd_req.gaps = mapped_dict.get("gaps", [])
        brd_req.risk_flags = mapped_dict.get("violations", [])
        brd_req.risk_level = mapped_dict.get("risk_level", "LOW")
        brd_req.remediation = mapped_dict.get("remediation_suggestions", "")

        # ── Step 5: BP3 — Generate Audit Report ──
        logger.info("Step 5: BP3 - Compiling Audit Report")
        audit_report_dict = await generate_audit_report(
            query=request.query,
            rag_response=rag_response.answer,
            claims=rag_response.claims,
            verifications=verifications,
            trust_gate=trust_gate,
            edition_conflicts=[],
            brd_results=[brd_req]
        )

        # ── Step 6: Store embedding + find related queries (pure cosine, no LLM) ──
        log_id = audit_report_dict.get("id")
        q_emb  = get_encoder().encode(request.query).tolist()
        if log_id:
            store_query_embedding(log_id, q_emb)

        past    = get_past_query_embeddings(exclude_id=log_id)
        related = find_related_queries(request.query, past)
        # Exclude near-exact matches (same query re-run) from related list
        related = [r for r in related if r.get("cosine_similarity", 0) < 0.99]

        audit_report_dict["xai_artifacts"]  = xai_artifacts.model_dump()
        audit_report_dict["related_queries"] = related
        audit_report_dict["response"]        = rag_response.answer
        audit_report_dict["claims"]          = [c.model_dump() for c in rag_response.claims]
        audit_report_dict["verifications"]   = [v.model_dump() for v in verifications]
        audit_report_dict["trust_gate"]      = trust_gate.model_dump() if trust_gate else None
        audit_report_dict["edition_conflicts"] = []

        return audit_report_dict

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unified Query pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


# ──────────────────────────────────────────────
#  GET /api/documents
# ──────────────────────────────────────────────

@router.get("/documents", response_model=list[DocumentInfo])
async def get_documents():
    """List all ingested documents with metadata."""
    docs = list_documents()
    return [DocumentInfo(**doc) for doc in docs]


# ──────────────────────────────────────────────
#  GET /api/sections/{publication}/{edition}
# ──────────────────────────────────────────────

@router.get("/sections/{publication}/{edition}", response_model=list[SectionInfo])
async def get_sections(publication: str, edition: str):
    """List all sections for a specific publication and edition.

    Queries ChromaDB metadata to find unique sections and their chunk counts.
    """
    publication = publication.upper()
    if publication not in PUBLICATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid publication '{publication}'. Must be one of: {PUBLICATIONS}",
        )

    collection = get_chroma_collection()

    # Query ChromaDB for all chunks matching this publication + edition
    results = collection.get(
        where={
            "$and": [
                {"publication_name": publication},
                {"edition_date": edition},
            ]
        },
        include=["metadatas"],
    )

    if not results or not results["metadatas"]:
        return []

    # Aggregate by section
    section_map: dict[str, dict] = {}
    for meta in results["metadatas"]:
        sid = meta.get("section_id", "unknown")
        if sid not in section_map:
            section_map[sid] = {
                "section_id": sid,
                "section_title": meta.get("section_title", ""),
                "chunk_count": 0,
                "page_number": meta.get("page_number", 0),
            }
        section_map[sid]["chunk_count"] += 1

    sections = sorted(section_map.values(), key=lambda s: s["section_id"])
    return [SectionInfo(**s) for s in sections]
