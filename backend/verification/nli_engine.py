import json
from typing import List
from shared.models import Claim, VerificationResult, NLIVerdict
from shared.gemini import call_gemini

async def verify_claim(claim: Claim) -> VerificationResult:
    """Verifies a single claim against its source passage using NLI."""
    prompt = f"""
    You are a Natural Language Inference expert. Determine the relationship between:

    PREMISE (source passage): {claim.source_passage}
    HYPOTHESIS (generated claim): {claim.text}

    Classify as exactly one of:
    - ENTAILMENT: The premise fully supports the hypothesis
    - CONTRADICTION: The premise contradicts the hypothesis
    - NEUTRAL: The premise neither supports nor contradicts

    Respond with JSON exactly in this format without any additional text:
    {{
      "verdict": "ENTAILMENT" | "CONTRADICTION" | "NEUTRAL",
      "confidence": <float between 0.0 and 1.0>,
      "explanation": "<detailed reasoning>"
    }}
    """
    
    # Prompt Gemini directly.
    try:
        response_text = await call_gemini(prompt, temperature=0.1)
        response_text = response_text.strip()
        if response_text.startswith("```json"): response_text = response_text[7:]
        if response_text.startswith("```"): response_text = response_text[3:]
        if response_text.endswith("```"): response_text = response_text[:-3]
        response_data = json.loads(response_text.strip())
    except Exception as e:
         return VerificationResult(
            claim_text=claim.text,
            verdict=NLIVerdict.NEUTRAL,
            entailment_score=0.0,
            explanation=f"Error evaluating claim: {e}"
         )
         
    return VerificationResult(
        claim_text=claim.text,
        verdict=response_data.get("verdict", NLIVerdict.NEUTRAL),
        entailment_score=float(response_data.get("confidence", 0.0)),
        explanation=response_data.get("explanation", "No reasoning provided.")
    )

import asyncio
async def verify_all_claims(claims: List[Claim]) -> List[VerificationResult]:
    """Verifies a list of claims using NLI."""
    tasks = [verify_claim(claim) for claim in claims]
    return await asyncio.gather(*tasks)

