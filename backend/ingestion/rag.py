"""
RAG Pipeline — Retrieval-Augmented Generation with claim-level attribution.

Retrieves relevant chunks, builds a Gemini prompt, then parses the response
into individual claims with source citations.
"""

import logging
import re

import spacy
import numpy as np
from shared.gemini import call_gemini
from shared.models import ChunkMetadata, Claim, RAGResponse
from shared.xai_matrices import build_attribution_matrix, attribute_sentence

logger = logging.getLogger(__name__)

# Load spacy for sentence splitting
nlp = spacy.load("en_core_web_sm")

# ──────────────────────────────────────────────
#  Prompt Templates
# ──────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are an RBI regulatory expert. Answer ONLY using the provided sources.
Do NOT hallucinate or add information not present in the sources.
Write a clear, professional narrative. Do NOT include arbitrary inline citations like brackets or keys.
Structure your answer clearly with proper paragraphs."""

RAG_USER_TEMPLATE = """Sources:
{formatted_sources}

Question: {query}"""


# ──────────────────────────────────────────────
#  Source Formatting
# ──────────────────────────────────────────────

def format_sources(chunks: list[dict]) -> str:
    """Format chunks for inclusion in the RAG prompt."""
    formatted = []
    for chunk in chunks:
        key = f"{chunk['publication_name']}·{chunk['edition_date']}·{chunk['section_id']}"
        formatted.append(f"[{key}] {chunk['chunk_text']}")
    return "\n\n".join(formatted)


# ──────────────────────────────────────────────
#  Claim Extraction (Matrix Engine)
# ──────────────────────────────────────────────

def parse_claims(answer: str, chunks: list[dict]) -> tuple[list[Claim], np.ndarray]:
    """Parse claims deterministically using Matrix 3 (Attribution Matrix).
    
    1. Splits response into sentences via spacy.
    2. Builds Attribution Matrix A using sentence-transformers.
    3. Maps each sentence mathematically to the best chunk (if score >= threshold).
    """
    doc = nlp(answer)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if not sentences or not chunks:
        return [], np.array([])
        
    # Build Matrix 3 (A)
    A = build_attribution_matrix(sentences, chunks)
    
    claims = []
    for i, sentence in enumerate(sentences):
        attr = attribute_sentence(A[i], chunks)
        
        if attr:
            claims.append(Claim(
                text=sentence,
                source_publication=attr["publication_name"],
                source_edition=attr["edition_date"],
                source_section_id=attr["section_id"],
                source_passage=attr["source_passage"],
                confidence=attr["attribution_score"]  # Initial proxy confidence
            ))
        else:
            # Unattributable
            claims.append(Claim(
                text=sentence,
                source_publication="",
                source_edition="",
                source_section_id="",
                source_passage="",
                confidence=0.0
            ))
            
    return claims, A


# ──────────────────────────────────────────────
#  Main RAG Function
# ──────────────────────────────────────────────

async def rag_query(
    query: str, chunks: list[dict]
) -> tuple[RAGResponse, np.ndarray]:
    """Execute the full RAG pipeline using pure Math extraction.

    Args:
        query: The user's natural language question.
        chunks: Pre-retrieved list of dictionaries representing chunks.

    Returns:
        (RAGResponse, Matrix A)
    """
    if not chunks:
        return RAGResponse(
            answer="No relevant documents found. Please ingest RBI publications first.",
            claims=[],
        )

    # 2. Build prompt
    sources_text = format_sources(chunks)
    user_prompt = RAG_USER_TEMPLATE.format(
        formatted_sources=sources_text,
        query=query,
    )

    # 3. Call Gemini
    logger.info(f"RAG gen for query '{query[:80]}...' with {len(chunks)} sources")

    answer = await call_gemini(
        prompt=user_prompt,
        system_instruction=RAG_SYSTEM_PROMPT,
        temperature=0.2,
    )

    # 4. Math Attribution (Matrix 3)
    claims, A_matrix = parse_claims(answer, chunks)

    logger.info(f"RAG Math Attribution: {len(claims)} sentences analyzed")

    return RAGResponse(answer=answer, claims=claims), A_matrix
