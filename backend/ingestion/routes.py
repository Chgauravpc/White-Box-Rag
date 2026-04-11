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
)
from ingestion.pdf_parser import ingest_pdf
from ingestion.rag import rag_query
from ingestion.retriever import rebuild_bm25_index

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

@router.post("/query", response_model=RAGResponse)
async def query_documents(request: QueryRequest):
    """Ask a question and get a RAG-generated answer with claim-level citations.

    Optionally filter by publication_name and/or edition_date.
    """
    try:
        response = await rag_query(
            query=request.query,
            filters=request.filters,
        )
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
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
