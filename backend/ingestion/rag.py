"""
RAG Pipeline — Retrieval-Augmented Generation with claim-level attribution.

Retrieves relevant chunks, builds a Gemini prompt, then parses the response
into individual claims with source citations.
"""

import logging
import re

from shared.gemini import call_gemini
from shared.models import ChunkMetadata, Claim, RAGResponse
from ingestion.retriever import hybrid_retrieve

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Prompt Templates
# ──────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are an RBI regulatory expert. Answer ONLY using the provided source sections.
For EACH claim or sentence in your answer, cite the source using the exact citation key in square brackets, like [FSR·Jun2024·1.1].
If the sources don't support a claim, explicitly say "insufficient evidence".
Do NOT hallucinate or add information not present in the sources.
Structure your answer clearly with proper paragraphs."""

RAG_USER_TEMPLATE = """Sources:
{formatted_sources}

Question: {query}"""


# ──────────────────────────────────────────────
#  Source Formatting
# ──────────────────────────────────────────────

def _make_citation_key(chunk: ChunkMetadata) -> str:
    """Create a citation key like FSR·Jun2024·1.1"""
    edition_short = chunk.edition_date.replace(" ", "")
    return f"{chunk.publication_name}·{edition_short}·{chunk.section_id}"


def format_sources(chunks: list[ChunkMetadata]) -> str:
    """Format chunks for inclusion in the RAG prompt.

    Each chunk is prefixed with its citation key so Gemini
    can reference it in the answer.
    """
    formatted = []
    for chunk in chunks:
        key = _make_citation_key(chunk)
        formatted.append(f"[{key}] {chunk.chunk_text}")
    return "\n\n".join(formatted)


# ──────────────────────────────────────────────
#  Claim Extraction
# ──────────────────────────────────────────────

# Regex to find citation references: [FSR·Jun2024·1.1]
CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")


def _find_chunk_by_key(
    citation_key: str, chunks: list[ChunkMetadata]
) -> ChunkMetadata | None:
    """Find the chunk matching a citation key."""
    for chunk in chunks:
        if _make_citation_key(chunk) == citation_key:
            return chunk
    return None


def parse_claims(answer: str, chunks: list[ChunkMetadata]) -> list[Claim]:
    """Parse the Gemini response to extract individual claims with citations.

    Strategy:
      1. Split the answer into sentences.
      2. For each sentence, extract citation keys via regex.
      3. Match citation keys back to source chunks.
      4. Create a Claim object per sentence.
    """
    # Split into sentences (rough split on ., !, ? followed by space/newline)
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())

    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Find all citation keys in this sentence
        citations = CITATION_PATTERN.findall(sentence)

        if citations:
            # Use the first citation as the primary source
            primary_key = citations[0]
            source_chunk = _find_chunk_by_key(primary_key, chunks)

            claims.append(Claim(
                text=sentence,
                source_publication=source_chunk.publication_name if source_chunk else "",
                source_edition=source_chunk.edition_date if source_chunk else "",
                source_section_id=source_chunk.section_id if source_chunk else "",
                source_passage=source_chunk.chunk_text[:500] if source_chunk else "",
                confidence=1.0,  # BP2's NLI engine will overwrite this
            ))
        else:
            # Sentence with no citation
            claims.append(Claim(
                text=sentence,
                source_publication="",
                source_edition="",
                source_section_id="",
                source_passage="",
                confidence=0.5,  # Lower confidence for uncited claims
            ))

    return claims


# ──────────────────────────────────────────────
#  Main RAG Function
# ──────────────────────────────────────────────

async def rag_query(
    query: str, filters: dict | None = None
) -> RAGResponse:
    """Execute the full RAG pipeline.

    1. Retrieve top-10 chunks via hybrid retrieval.
    2. Format sources and build the Gemini prompt.
    3. Call Gemini Pro to generate an attributed answer.
    4. Parse claims from the response.
    5. Return a structured RAGResponse.

    Args:
        query: The user's natural language question.
        filters: Optional metadata filters (publication_name, edition_date).

    Returns:
        RAGResponse with answer text and structured claims list.
    """
    # 1. Retrieve
    chunks = hybrid_retrieve(query, top_k=10, filters=filters)

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
    logger.info(f"RAG query: '{query[:80]}...' with {len(chunks)} source chunks")

    answer = await call_gemini(
        prompt=user_prompt,
        system_instruction=RAG_SYSTEM_PROMPT,
        temperature=0.2,
    )

    # 4. Parse claims
    claims = parse_claims(answer, chunks)

    logger.info(f"RAG response: {len(claims)} claims extracted")

    return RAGResponse(answer=answer, claims=claims)
