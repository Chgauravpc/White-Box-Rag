import json
import logging
import asyncio
from typing import List, Dict, Any
from shared.models import BRDRequirement
from shared.gemini import call_gemini

logger = logging.getLogger(__name__)

# Integration Point: Connect with BP1 Hybrid Retriever
try:
    from ingestion.retriever import hybrid_retrieve
except ImportError:
    logger.warning("BP1's hybrid_retrieve not found. Using fallback mock.")
    def hybrid_retrieve(query: str, top_k: int = 5, filters=None):
        class MockChunk:
            publication_name = "Mock_FSR"
            edition_date = "2024-06"
            section_id = "1.1"
            chunk_text = "Data localization and KYC guidelines must be strictly adhered to as per RBI mandate."
        return [MockChunk()]

async def map_requirement(req: BRDRequirement) -> Dict[str, Any]:
    """
    Evaluates a single BRDRequirement against relevant RBI sections 
    with timeout and retry logic.
    """
    logger.info(f"Mapping requirement: {req.text}")
    
    # 1. Retrieve top sections
    try:
        retrieved_chunks = hybrid_retrieve(req.text, top_k=5)
    except Exception as e:
        logger.error(f"Hybrid retrieve failed: {e}")
        retrieved_chunks = []
    
    formatted_sections = ""
    for chunk in retrieved_chunks:
        try:
            formatted_sections += f"[{chunk.publication_name} | {chunk.edition_date} | Section {chunk.section_id}]\n{chunk.chunk_text}\n\n"
        except AttributeError:
            formatted_sections += f"{chunk}\n\n"
        
    if not formatted_sections.strip():
        formatted_sections = "No relevant RBI sections found for this requirement."

    # 2. Build Prompt
    prompt = f"""
You are a senior RBI regulatory compliance expert.
Your task is to evaluate how well a Business Requirement aligns with RBI guidelines based ONLY on the provided RBI sections.

---
### INPUT:
BRD Requirement: {req.text}
Relevant RBI Sections: {formatted_sections}
---
### OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "alignment_score": 75,
  "gaps": ["Specific gaps"],
  "violations": ["Specific violations"],
  "risk_level": "High",
  "overall_compliance_score": 65,
  "remediation_suggestions": ["Actionable steps"]
}}
"""

    max_retries = 2
    response_text = ""
    success = False
    
    for attempt in range(max_retries + 1):
        try:
            response_text = await call_gemini(prompt, temperature=0.1)
            success = True
            break
        except Exception as e:
            logger.error(f"Gemini API attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(2)
            else:
                logger.error("All Gemini API retries failed. Returning fallback.")
                return {
                    "requirement": req.text,
                    "alignment_score": 50,
                    "gaps": ["Unable to fully analyze requirement"],
                    "violations": [],
                    "risk_level": "Medium",
                    "overall_compliance_score": 50,
                    "remediation_suggestions": ["Retry analysis"]
                }
                
    if not success:
        return {
            "requirement": req.text,
            "alignment_score": 50,
            "gaps": ["Unable to fully analyze requirement"],
            "violations": [],
            "risk_level": "Medium",
            "overall_compliance_score": 50,
            "remediation_suggestions": ["Retry analysis"]
        }

    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    try:
        results = json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Validation output: {e}\nRaw={response_text}")
        return {
            "requirement": req.text,
            "alignment_score": 50,
            "gaps": ["JSON parsing failed"],
            "violations": [],
            "risk_level": "Medium",
            "overall_compliance_score": 50,
            "remediation_suggestions": ["Retry analysis"]
        }

    return {
        "requirement": req.text,
        "alignment_score": results.get("alignment_score", 0),
        "gaps": results.get("gaps", []),
        "violations": results.get("violations", []),
        "risk_level": results.get("risk_level", "Medium").capitalize(),
        "overall_compliance_score": results.get("overall_compliance_score", 0),
        "remediation_suggestions": results.get("remediation_suggestions", [])
    }
