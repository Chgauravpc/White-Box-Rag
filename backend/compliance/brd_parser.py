import os
import json
import asyncio
import fitz  # PyMuPDF
from docx import Document
from shared import config
from shared.models import BRDRequirement
from shared.llm import call_llm
from compliance.domain_profiles import get_domain_profile

def extract_text_from_file(filepath: str) -> str:
    """Extract text from a PDF, DOCX, or TXT file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    ext = os.path.splitext(filepath)[1].lower()
    text = ""
    
    if ext == ".pdf":
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
    elif ext == ".docx":
        doc = Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    return text.strip()

async def parse_brd(filepath: str) -> list[BRDRequirement]:
    """
    Extracts text from a BRD document and uses the LLM 
    to output structured requirements.
    """
    # 1. Get raw text from document
    text = extract_text_from_file(filepath)
    if not text:
        raise ValueError("No text could be extracted from the document.")

    # 2. Prepare the prompt — persona/categories/relevance come from the active
    # domain profile (shared.config.DOMAIN_PROFILE), not a hardcoded domain.
    profile = get_domain_profile()
    categories_json = json.dumps(profile["categories"])
    if profile["relevance_fixed_enum"]:
        relevance_instruction = (
            "Choose one or more of:\n    " + json.dumps(profile["relevance_fixed_enum"])
        )
        example_2_relevance = json.dumps(profile["relevance_fixed_enum"][:2])
    else:
        hint = ", ".join(profile["relevance_hint_examples"])
        relevance_instruction = (
            f"A free-text label for the policy/regulation/standard this requirement relates to "
            f"(examples: {hint}). Use whatever labels actually appear in or are implied by the "
            f"source document — do not invent a fixed code if none is evident."
        )
        example_2_relevance = json.dumps(profile["relevance_hint_examples"][:2])
    example_category = profile["categories"][0]
    example_category_2 = profile["categories"][min(4, len(profile["categories"]) - 1)]

    prompt = f"""
{profile["persona"]}

Your task is to extract structured business requirements from a Business Requirements Document (BRD).

---

### INSTRUCTIONS:

1. Identify ALL explicit and implicit business requirements from the document.
2. Each requirement must be:
   - Clear
   - Atomic (one requirement per item)
   - Not merged with others

3. Assign the following fields for each requirement:

- requirement_id: Unique ID in format REQ-001, REQ-002, ...
- requirement_text: Exact or slightly cleaned version of the requirement
- category: One of:
    {categories_json}

- regulatory_relevance: {relevance_instruction}

4. If a requirement is not clearly regulatory, still include it but classify appropriately.

5. DO NOT hallucinate. Only extract from given text.

---

### OUTPUT FORMAT (STRICT JSON ONLY):

Return ONLY a JSON array. No explanation.

Example:

[
  {{
    "requirement_id": "REQ-001",
    "requirement_text": "System must verify user identity before granting access",
    "category": "{example_category}",
    "regulatory_relevance": {example_2_relevance}
  }},
  {{
    "requirement_id": "REQ-002",
    "requirement_text": "All actions must be logged for audit purposes",
    "category": "{example_category_2}",
    "regulatory_relevance": {example_2_relevance}
  }}
]

---

### DOCUMENT TEXT:
{text}

---

### IMPORTANT:
- Output MUST be valid JSON
- Do NOT include markdown (no ```json)
- Do NOT include explanations
- Ensure IDs are sequential
"""

    # 3. Call the LLM (BP3 uses it to structure the extracted requirements)
    response_text = await call_llm(prompt, temperature=config.COMPLIANCE_TEMPERATURE)

    # Clean up markdown if the model includes it despite our instruction
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    # 4. Parse the strict JSON
    try:
        raw_reqs = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output as JSON: {e}\nRaw Response:\n{response_text}")

    # 5. Convert JSON into Pydantic models required for the shared contract
    requirements = []
    for req in raw_reqs:
        # Handling the list mapping since our Pydantic model takes a string for regulatory_relevance
        reg_rel = req.get("regulatory_relevance")
        if isinstance(reg_rel, list):
            reg_rel = ", ".join(reg_rel)

        brd_req = BRDRequirement(
            id=req.get("requirement_id", "REQ-UNKNOWN"),
            text=req.get("requirement_text", ""),
            category=req.get("category"),
            regulatory_relevance=reg_rel
        )
        requirements.append(brd_req)

    return requirements
