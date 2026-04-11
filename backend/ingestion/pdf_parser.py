"""
PDF Parser — Extracts section-level chunks from RBI publication PDFs.

Pipeline:  PDF → page extraction → section detection → overlapping chunking → ChromaDB storage
"""

import logging
import os
import re

import fitz  # PyMuPDF

from shared.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
from shared.database import get_chroma_collection, insert_document
from shared.models import ChunkMetadata

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Regex patterns for RBI section headers
# ──────────────────────────────────────────────

# Matches:  1.1 Overview,  2.3.1 Credit Risk,  3.12 Capital Adequacy
NUMBERED_SECTION = re.compile(
    r"^(\d+\.\d+(?:\.\d+)?)\s+([A-Z][\w\s\-:,&/()]+)", re.MULTILINE
)

# Matches:  Chapter I,  Chapter II,  Section III,  Section IV, Part-II
ROMAN_SECTION = re.compile(
    r"^(Chapter|Section|Part)-?\s*(I{1,3}|IV|V|VI{0,3}|IX|X)[:\.\s]+(.+)",
    re.MULTILINE | re.IGNORECASE,
)

# Matches: I.8.4 Investment in bonds
ROMAN_NUMBERED_SECTION = re.compile(
    r"^((?:I{1,3}|IV|V|VI{0,3}|IX|X)(?:\.\d+)+)(?:\.)?\s+([A-Z][\w\s\-:,&/()]+)",
    re.MULTILINE | re.IGNORECASE,
)

# Matches:  Box 1.1,  Box 2.3
BOX_SECTION = re.compile(
    r"^(Box\s+\d+\.\d+)[:\s]*(.*)", re.MULTILINE
)


# ──────────────────────────────────────────────
#  Step 1: Extract raw text from PDF pages
# ──────────────────────────────────────────────

def extract_pages(filepath: str) -> list[dict]:
    """Extract text from each page of a PDF.

    Returns:
        List of {"page_number": int (1-indexed), "text": str}
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")

    doc = fitz.open(filepath)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_number": i + 1,
            "text": text.strip(),
        })
    doc.close()

    logger.info(f"Extracted {len(pages)} pages from {os.path.basename(filepath)}")
    return pages


# ──────────────────────────────────────────────
#  Step 2: Detect section boundaries
# ──────────────────────────────────────────────

def detect_sections(pages: list[dict]) -> list[dict]:
    """Identify section boundaries across all pages.

    Returns:
        List of {
            "section_id": str,
            "section_title": str,
            "text": str,          # accumulated text for this section
            "start_page": int,
        }
    """
    sections = []
    current_section = None

    for page in pages:
        text = page["text"]
        page_num = page["page_number"]

        if not text:
            # Skip empty / scanned-image pages
            logger.warning(f"Page {page_num}: no extractable text (possibly scanned)")
            continue

        # Try to find section headers in this page
        lines = text.split("\n")
        buffer = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                buffer.append("")
                continue

            matched = False

            # Try numbered: 1.1 Overview
            m = NUMBERED_SECTION.match(line_stripped)
            if m:
                # Flush buffer to current section
                if current_section is not None:
                    current_section["text"] += "\n".join(buffer) + "\n"
                    sections.append(current_section)
                buffer = []

                current_section = {
                    "section_id": m.group(1),
                    "section_title": m.group(2).strip(),
                    "text": "",
                    "start_page": page_num,
                }
                matched = True

            # Try roman: Chapter II: Banking Sector
            if not matched:
                m = ROMAN_SECTION.match(line_stripped)
                if m:
                    if current_section is not None:
                        current_section["text"] += "\n".join(buffer) + "\n"
                        sections.append(current_section)
                    buffer = []

                    section_type = m.group(1)
                    roman = m.group(2)
                    title = m.group(3).strip()
                    current_section = {
                        "section_id": f"{section_type}_{roman}",
                        "section_title": title,
                        "text": "",
                        "start_page": page_num,
                    }
                    matched = True

            # Try roman numbered: I.8.4 Investment
            if not matched:
                m = ROMAN_NUMBERED_SECTION.match(line_stripped)
                if m:
                    if current_section is not None:
                        current_section["text"] += "\n".join(buffer) + "\n"
                        sections.append(current_section)
                    buffer = []

                    current_section = {
                        "section_id": m.group(1),
                        "section_title": m.group(2).strip(),
                        "text": "",
                        "start_page": page_num,
                    }
                    matched = True

            # Try box: Box 1.1
            if not matched:
                m = BOX_SECTION.match(line_stripped)
                if m:
                    if current_section is not None:
                        current_section["text"] += "\n".join(buffer) + "\n"
                        sections.append(current_section)
                    buffer = []

                    current_section = {
                        "section_id": m.group(1).replace(" ", "_"),
                        "section_title": m.group(2).strip() if m.group(2) else m.group(1),
                        "text": "",
                        "start_page": page_num,
                    }
                    matched = True

            if not matched:
                buffer.append(line_stripped)

        # End of page — flush remaining buffer
        if buffer and current_section is not None:
            current_section["text"] += "\n".join(buffer) + "\n"

    # Flush final section
    if current_section is not None:
        sections.append(current_section)

    # Fallback: if NO sections detected, treat entire document as one section
    if not sections:
        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        sections.append({
            "section_id": "0.0",
            "section_title": "Full Document",
            "text": full_text,
            "start_page": 1,
        })
        logger.warning("No section headers detected — treating entire PDF as a single section")

    logger.info(f"Detected {len(sections)} sections")
    return sections


# ──────────────────────────────────────────────
#  Step 3: Chunk sections with overlap
# ──────────────────────────────────────────────

def chunk_section(
    section: dict,
    publication_name: str,
    edition_date: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[ChunkMetadata]:
    """Split a section into overlapping chunks.

    Uses whitespace tokenisation (simple, fast). Each chunk inherits
    the section's metadata.
    """
    text = section["text"].strip()
    if not text:
        return []

    words = text.split()

    # If section is short enough, return as single chunk
    if len(words) <= max_tokens:
        return [
            ChunkMetadata(
                publication_name=publication_name,
                edition_date=edition_date,
                section_id=section["section_id"],
                section_title=section["section_title"],
                page_number=section["start_page"],
                chunk_text=text,
            )
        ]

    # Sliding window with overlap
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(
            ChunkMetadata(
                publication_name=publication_name,
                edition_date=edition_date,
                section_id=section["section_id"],
                section_title=section["section_title"],
                page_number=section["start_page"],
                chunk_text=chunk_text,
            )
        )

        # Advance by (max_tokens - overlap)
        start += max_tokens - overlap_tokens

        # Avoid tiny trailing chunks
        if start + overlap_tokens >= len(words):
            break

    return chunks


# ──────────────────────────────────────────────
#  Step 4: End-to-end ingestion
# ──────────────────────────────────────────────

def ingest_pdf(filepath: str, publication: str, edition_date: str) -> int:
    """Full pipeline: parse PDF → detect sections → chunk → store in ChromaDB + SQLite.

    Args:
        filepath: Path to the PDF file.
        publication: Publication code (FSR, MPR, PSR, FER).
        edition_date: Edition identifier, e.g. "June 2024".

    Returns:
        Number of chunks ingested.
    """
    # 1. Extract pages
    pages = extract_pages(filepath)

    # 2. Detect sections
    sections = detect_sections(pages)

    # 3. Chunk all sections
    all_chunks: list[ChunkMetadata] = []
    for section in sections:
        chunks = chunk_section(section, publication, edition_date)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning(f"No chunks generated from {filepath}")
        return 0

    # 4. Store in ChromaDB
    collection = get_chroma_collection()

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(all_chunks):
        # Deterministic ID to prevent duplicates on re-ingestion
        chunk_id = f"{publication}_{edition_date}_{chunk.section_id}_chunk{i}"
        ids.append(chunk_id)
        documents.append(chunk.chunk_text)
        metadatas.append({
            "publication_name": chunk.publication_name,
            "edition_date": chunk.edition_date,
            "section_id": chunk.section_id,
            "section_title": chunk.section_title,
            "page_number": chunk.page_number,
        })

    # Upsert to handle re-ingestion gracefully
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # 5. Record in SQLite
    filename = os.path.basename(filepath)
    insert_document(filename, publication, edition_date, len(all_chunks))

    logger.info(
        f"Ingested {len(all_chunks)} chunks from {filename} "
        f"({publication} · {edition_date}, {len(sections)} sections)"
    )
    return len(all_chunks)
