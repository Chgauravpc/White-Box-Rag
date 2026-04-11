import json
from typing import List
from shared.models import EditionConflict
from shared.gemini import call_gemini

async def detect_conflicts(publication: str, topic: str, older_date: str, older_text: str, newer_date: str, newer_text: str, section_id: str) -> EditionConflict:
    """Detects conflicts between old and new editions of a publication's section."""
    
    prompt = f"""
    Compare these two passages from different editions of {publication} concerning the topic '{topic}':

    OLDER EDITION ({older_date}, Section {section_id}):
    {older_text}

    NEWER EDITION ({newer_date}, Section {section_id}):
    {newer_text}

    Are there any contradictions or significant changes in position? 
    Which edition's position should be considered current?
    
    Respond as JSON exactly in this format without any additional text:
    {{
      "has_conflict": <bool true or false>,
      "conflict_description": "<description of conflict, or empty string if none>",
      "superseding_edition": "<usually the newer date if conflict, or null>",
      "details": "<any extra details or reasoning>"
    }}
    """
    
    try:
        response_text = await call_gemini(prompt, temperature=0.1)
        response_text = response_text.strip()
        if response_text.startswith("```json"): response_text = response_text[7:]
        if response_text.startswith("```"): response_text = response_text[3:]
        if response_text.endswith("```"): response_text = response_text[:-3]
        response_data = json.loads(response_text.strip())
    except Exception as e:
        return EditionConflict(
            publication=publication,
            section_id=section_id,
            older_edition=older_date,
            newer_edition=newer_date,
            has_conflict=False,
            conflict_description=f"Error detecting conflicts: {e}",
            superseding_edition="",
            details="Error occurred during Gemini execution."
        )

    return EditionConflict(
        publication=publication,
        section_id=section_id,
        older_edition=older_date,
        newer_edition=newer_date,
        has_conflict=bool(response_data.get("has_conflict", False)),
        conflict_description=response_data.get("conflict_description", ""),
        superseding_edition=response_data.get("superseding_edition", ""),
        details=response_data.get("details", "")
    )
