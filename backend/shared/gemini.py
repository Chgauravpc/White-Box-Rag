"""
Gemini API client wrapper.
Provides a singleton client and a high-level call function with retry logic.
"""

import asyncio
import logging
import time

from google import genai
from google.genai import types

from shared.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

_client = None


def get_gemini_client() -> genai.Client:
    """Return a configured Gemini client (singleton)."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add it to your .env file."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


async def call_gemini(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> str:
    """Send a prompt to Gemini and return the text response.

    Args:
        prompt: The user prompt / question.
        system_instruction: Optional system-level instruction.
        temperature: Sampling temperature (low = more deterministic).
        max_retries: Number of retries on rate-limit / transient errors.

    Returns:
        The model's text response.

    Raises:
        Exception: After all retries are exhausted.
    """
    client = get_gemini_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
    )
    if system_instruction:
        config.system_instruction = system_instruction

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            return response.text

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Retry on rate-limit or transient server errors
            if "resource exhausted" in error_str or "429" in error_str or "500" in error_str:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            else:
                # Non-retriable error — raise immediately
                raise

    raise last_error
