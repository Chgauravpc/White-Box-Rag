from typing import List
from shared.models import EditionConflict
from shared.xai_matrices import build_conflict_matrix

async def detect_conflicts(publication: str, topic: str, older_date: str, older_text: str, newer_date: str, newer_text: str, section_id: str) -> EditionConflict:
    """Detects conflicts between old and new editions using CrossEncoder probabilities."""
    old_chunk = {"chunk_text": older_text, "section_id": section_id, "edition_date": older_date}
    new_chunk = {"chunk_text": newer_text, "section_id": section_id, "edition_date": newer_date}
    
    C_matrix = build_conflict_matrix([old_chunk], [new_chunk])
    if C_matrix.size == 0:
        return EditionConflict(
            publication=publication, section_id=section_id, older_edition=older_date, newer_edition=newer_date,
            has_conflict=False, conflict_description="Empty input.", superseding_edition="", details=""
        )
        
    contradiction_prob = float(C_matrix[0][0])
    has_conflict = contradiction_prob > 0.70
    
    return EditionConflict(
        publication=publication,
        section_id=section_id,
        older_edition=older_date,
        newer_edition=newer_date,
        has_conflict=has_conflict,
        conflict_description=f"Contradiction Probability: {contradiction_prob:.2%}" if has_conflict else "",
        superseding_edition=newer_date if has_conflict else "",
        details=f"CrossEncoder probability of newer text contradicting older text: {contradiction_prob:.4f}"
    )
