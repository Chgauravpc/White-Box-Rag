"""
API Routes for the Ingestion & RAG service (BP1).

Endpoints:
  POST /api/ingest            — Upload and ingest a PDF into a collection
  POST /api/query             — RAG query with optional filters
  GET  /api/documents         — List all ingested documents
  GET  /api/sections/{collection}/{edition} — List sections for a collection/edition
"""

import logging
import os
import re
import shutil
import tempfile
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from shared.config import MAX_COLLECTION_LABEL_LENGTH
from shared.database import get_chroma_collection, list_documents
from shared.models import (
    DocumentInfo,
    QueryRequest,
    SectionInfo,
    AuditReport,
)
from ingestion.pdf_parser import ingest_pdf
from ingestion.retriever import rebuild_bm25_index
from ingestion.pipeline import run_query_pipeline


_COLLECTION_LABEL_RE = re.compile(r"^[\w\s\-.,&/()]+$", re.UNICODE)


def _validate_collection_label(label: str) -> str:
    """Validate a free-text collection/publication label.

    Domain-agnostic: any non-empty, reasonably-sized label of ordinary
    characters is accepted — there is no fixed enum of allowed values.
    """
    label = (label or "").strip()
    if not label:
        raise HTTPException(status_code=400, detail="Collection/publication label cannot be empty.")
    if len(label) > MAX_COLLECTION_LABEL_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Collection/publication label too long (max {MAX_COLLECTION_LABEL_LENGTH} characters).",
        )
    if not _COLLECTION_LABEL_RE.match(label):
        raise HTTPException(status_code=400, detail="Collection/publication label contains invalid characters.")
    return label.upper()


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ingestion & RAG"])

# Directory for storing uploaded PDFs
UPLOAD_DIR = os.path.join("data", "documents")


# ──────────────────────────────────────────────
#  POST /api/ingest
# ──────────────────────────────────────────────

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(..., description="Source PDF document"),
    publication: str = Form(..., description="Collection label, e.g. 'CONTRACTS', 'FSR', 'ENG_SPECS' — any free-text name"),
    edition_date: str = Form(default="", description="Version/date label, e.g. 'June 2024'. Optional — defaults to the ingestion date if omitted."),
):
    """Upload and ingest a PDF document into a named collection.

    The PDF is parsed into section-level chunks, stored in ChromaDB
    for retrieval, and registered in the documents database. The collection
    label is free-text — this system is not restricted to any single domain.
    """
    publication = _validate_collection_label(publication)

    # "Edition" only makes sense for genuinely versioned corpora — default to
    # the ingestion date rather than forcing every document into that model.
    edition_date = (edition_date or "").strip() or datetime.now().strftime("%Y-%m-%d")

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

    Thin wrapper around ingestion/pipeline.py::run_query_pipeline — the same
    function the offline evaluation harness calls directly (no HTTP round-trip).
    """
    try:
        return await run_query_pipeline(request.query, request.filters)
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
    """List all sections for a specific collection and edition.

    Queries ChromaDB metadata to find unique sections and their chunk counts.
    """
    publication = _validate_collection_label(publication)

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
