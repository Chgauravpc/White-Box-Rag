"""
PDF Parser — Extracts section-level chunks from PDF documents.

Pipeline:  PDF → page extraction → section detection → overlapping chunking → ChromaDB storage
"""

import logging
import os
import re

import fitz  # PyMuPDF

from shared.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
from shared.database import get_chroma_collection, upsert_document
from shared.models import ChunkMetadata
from shared.chunk_key import build_chunk_key
from shared.hashing import hash_file

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Parser identity
# ──────────────────────────────────────────────
# Bump by hand whenever section detection or chunking changes in a way that
# alters chunk boundaries or chunk_key assignment. A git commit hash is too
# coarse (every commit would invalidate a frozen corpus) and no version at all
# is too loose — a labeled dataset pins this value via the corpus manifest
# (shared/corpus_manifest.py) so a silent re-chunk can't invalidate labels
# without being detected.
PARSER_VERSION = "1.0"

# ──────────────────────────────────────────────
#  Regex patterns for structured (financial-report-style) section headers
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
#  Generic (domain-agnostic) fallback patterns
#
#  Only tried when NONE of the structured patterns above find a single
#  section anywhere in the document — so they never interfere with
#  financial-report-style PDFs that already parse correctly. They exist so
#  that other reasonably-structured document types (contracts, specs,
#  single-level-numbered documents) still get section-level attribution
#  instead of silently degrading to page-level citation.
# ──────────────────────────────────────────────

# Matches:  1. Definitions,  2 Term and Termination
GENERIC_NUMBERED_SECTION = re.compile(
    r"^(\d+)\.?\s+([A-Z][\w\s\-:,&/()]{2,70})$"
)


def _looks_like_generic_heading(line: str) -> bool:
    """Heuristic: a short, standalone Title-Case or ALL-CAPS line that reads
    like a heading rather than a sentence of body prose. Intentionally
    conservative — this is a best-effort fallback, not a layout analyzer.
    """
    if not line or len(line) > 80:
        return False
    if line[-1] in ".,;:":
        return False
    words = line.split()
    if not (1 <= len(words) <= 10):
        return False
    if line.isupper():
        return True
    return all(w[0].isupper() for w in words if w[:1].isalpha())


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

def detect_sections(pages: list[dict]) -> tuple[list[dict], bool]:
    """Identify section boundaries across all pages.

    Tries the structured (financial-report-style) patterns first. If those
    find nothing at all, falls back to a generic domain-agnostic pass before
    finally falling back to page-level pseudo-sections.

    Returns:
        (sections, structured) where structured is False only when neither
        pattern tier found anything and every page became its own section.
        sections: List of {
            "section_id": str,
            "section_title": str,
            "text": str,          # accumulated text for this section
            "start_page": int,
        }
    """
    sections = _detect_structured_sections(pages)
    if sections:
        return sections, True

    sections = _detect_generic_sections(pages)
    if sections:
        logger.info(f"No financial-report-style headers found — used generic fallback, {len(sections)} sections")
        return sections, True

    # Final fallback: if NO sections detected by either tier, treat each page as its own section
    logger.warning("No section headers detected — treating each page as its own section")
    for page in pages:
        if page["text"]:
            sections.append({
                "section_id":    f"UNSTRUCTURED-p{page['page_number']}",
                "section_title": f"Page {page['page_number']} (unstructured)",
                "text":          page["text"],
                "start_page":    page["page_number"],
            })
    return sections, False


def _detect_generic_sections(pages: list[dict]) -> list[dict]:
    """Domain-agnostic fallback: single-level numbered headers and short
    Title-Case/ALL-CAPS heading lines. Best-effort — see `_looks_like_generic_heading`.
    """
    sections = []
    current_section = None

    for page in pages:
        text = page["text"]
        page_num = page["page_number"]
        if not text:
            continue

        for line in text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            m = GENERIC_NUMBERED_SECTION.match(line_stripped)
            is_heading = bool(m) or _looks_like_generic_heading(line_stripped)

            if is_heading:
                if current_section is not None:
                    sections.append(current_section)
                section_id = m.group(1) if m else str(len(sections) + 1)
                title = m.group(2).strip() if m else line_stripped
                current_section = {
                    "section_id":    section_id,
                    "section_title": title,
                    "text":          "",
                    "start_page":    page_num,
                }
            elif current_section is not None:
                current_section["text"] += line_stripped + "\n"

    if current_section is not None:
        sections.append(current_section)
    return sections


def _detect_structured_sections(pages: list[dict]) -> list[dict]:
    """Financial-report-style section detection (numbered, roman, box headers)."""
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

    logger.info(f"Structured-pattern pass detected {len(sections)} sections")
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

def _delete_existing_chunks(collection, publication: str, edition_date: str) -> int:
    """Remove every chunk previously ingested for this (publication, edition).

    Returns the number of chunks deleted. Best-effort: a backend that doesn't
    support `where`-deletes shouldn't break ingestion, so failures are logged
    and swallowed — the subsequent upsert still refreshes every chunk this
    ingest produced.
    """
    where = {"$and": [
        {"publication_name": {"$eq": publication}},
        {"edition_date": {"$eq": edition_date}},
    ]}
    try:
        existing = collection.get(where=where, include=[])
        stale_ids = existing.get("ids", []) or []
        if stale_ids:
            collection.delete(ids=stale_ids)
        return len(stale_ids)
    except Exception as exc:  # pragma: no cover - backend-specific
        logger.warning(f"Could not clear prior chunks for {publication}/{edition_date}: {exc}")
        return 0


def ingest_pdf(filepath: str, publication: str, edition_date: str) -> int:
    """Full pipeline: parse PDF → detect sections → chunk → store in ChromaDB + SQLite.

    Args:
        filepath: Path to the PDF file.
        publication: Collection label (free-text, e.g. "FSR", "CONTRACTS").
        edition_date: Version/date identifier, e.g. "June 2024".

    Returns:
        Number of chunks ingested.
    """
    # 1. Extract pages
    pages = extract_pages(filepath)

    # 2. Detect sections
    sections, structured = detect_sections(pages)

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

    # `ordinal` must be the index of the chunk WITHIN ITS SECTION, not a
    # document-global counter. With a global counter, one extra chunk in an
    # early section renumbers every chunk after it, silently re-pointing every
    # labeled relevant_chunk_key in an eval dataset at different text.
    section_ordinals: dict[str, int] = {}
    for chunk in all_chunks:
        ordinal = section_ordinals.get(chunk.section_id, 0)
        section_ordinals[chunk.section_id] = ordinal + 1
        # Canonical, globally-unique chunk identity (see shared/chunk_key.py) —
        # doubles as the deterministic ChromaDB id, so re-ingestion upserts
        # rather than duplicating.
        chunk_key = build_chunk_key(publication, edition_date, chunk.section_id, ordinal)
        chunk.chunk_key = chunk_key
        ids.append(chunk_key)
        documents.append(chunk.chunk_text)
        metadatas.append({
            "publication_name": chunk.publication_name,
            "edition_date": chunk.edition_date,
            "section_id": chunk.section_id,
            "section_title": chunk.section_title,
            "page_number": chunk.page_number,
            "chunk_key": chunk_key,
        })

    # Re-ingestion must not leave orphans. `upsert` alone only overwrites the
    # ids it is given: if a re-parse yields fewer chunks (or different keys
    # after a chunking-config change), the previous ingest's surplus chunks
    # stay in the collection and remain retrievable forever. Delete this
    # (publication, edition) slice first so the corpus is exactly what this
    # ingest produced.
    _delete_existing_chunks(collection, publication, edition_date)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # 5. Record in SQLite
    filename = os.path.basename(filepath)
    upsert_document(
        filename,
        publication,
        edition_date,
        len(all_chunks),
        structured=structured,
        source_sha256=hash_file(filepath),
        source_path=os.path.abspath(filepath),
        parser_version=PARSER_VERSION,
    )

    logger.info(
        f"Ingested {len(all_chunks)} chunks from {filename} "
        f"({publication} · {edition_date}, {len(sections)} sections, structured={structured})"
    )
    return len(all_chunks)
