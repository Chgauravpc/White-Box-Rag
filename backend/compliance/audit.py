import json
from datetime import datetime
from shared.models import Claim, VerificationResult, TrustGate, EditionConflict, BRDRequirement
from shared.gemini import call_gemini
from shared.database import get_sqlite_conn

async def generate_audit_report(
    query: str,
    rag_response: str,
    claims: list[Claim],
    verifications: list[VerificationResult],
    trust_gate: TrustGate,
    edition_conflicts: list[EditionConflict],
    brd_results: list[BRDRequirement]
) -> dict:
    """
    Generates a structured audit report using Gemini Pro, 
    and commits the log to the SQLite database.
    """
    
    # 1. Format inputs for the prompt
    claims_json = json.dumps([c.model_dump() for c in claims], indent=2)
    verifications_json = json.dumps([v.model_dump() for v in verifications], indent=2)
    trust_gate_json = trust_gate.model_dump_json(indent=2)
    conflicts_json = json.dumps([c.model_dump() for c in edition_conflicts], indent=2)
    brd_results_json = json.dumps([b.model_dump() for b in brd_results], indent=2)

    # 2. Build the exact prompt requested
    prompt = f"""
You are an RBI compliance auditor and Explainable AI (XAI) specialist.

Your task is to generate a structured audit report explaining how a system arrived at its response and whether it is compliant.

---

### INPUT DATA:

User Query:
{query}

RAG Response:
{rag_response}

Claims:
{claims_json}

Verification Results (NLI):
{verifications_json}

Trust Gate Decision:
{trust_gate_json}

Edition Conflicts:
{conflicts_json}

BRD Compliance Results:
{brd_results_json}

---

### INSTRUCTIONS:

Generate a structured audit report with full traceability and explainability.

---

### INCLUDE THE FOLLOWING SECTIONS:

1. **timestamp**
- Current system timestamp

---

2. **query**
- Original user query

---

3. **rag_response**
- Exact copy of the RAG Response provided above

---

4. **decision_log**
For EACH claim:
- claim_text
- source (publication, edition, section)
- verification_verdict (ENTAILMENT / CONTRADICTION / NEUTRAL)
- confidence_score
- explanation (why accepted/rejected)

---

4. **section_reference_registry**
List ALL RBI sections used:
- publication_name
- edition_date
- section_id
- section_title (if available)
- short_summary of section

---

5. **edition_traceability**
- Which editions were consulted
- Any detected conflicts
- Which edition is considered authoritative
- reasoning

---

6. **compliance_evidence**
From BRD validation:
- requirement_id
- alignment_score
- gaps
- violations
- risk_level
- supporting_sections

---

7. **final_audit_summary**
- overall_trust_status (Safe / Needs_Human_Review / Non_Compliant)
- key_risks
- compliance_score_summary
- reasoning (clear explanation in 2–4 lines)

---

### OUTPUT FORMAT (STRICT JSON ONLY):

Return ONLY valid JSON. No explanation. No markdown.

Example:

  "timestamp": "2026-04-11T12:00:00",
  "query": "...",
  "rag_response": "...",
  "decision_log": [
    {{
      "claim_text": "...",
      "source": "FSR·Dec2024·3.1",
      "verification_verdict": "ENTAILMENT",
      "confidence_score": 0.92,
      "explanation": "The source clearly supports the claim"
    }}
  ],
  "section_reference_registry": [
    {{
      "publication_name": "FSR",
      "edition_date": "Dec 2024",
      "section_id": "3.1",
      "section_title": "Financial Stability Risks",
      "summary": "Discusses major risks to financial stability"
    }}
  ],
  "edition_traceability": {{
    "editions_compared": ["June 2024", "Dec 2024"],
    "conflicts_detected": true,
    "authoritative_edition": "Dec 2024",
    "reasoning": "Latest edition overrides previous guidance"
  }},
  "compliance_evidence": [
    {{
      "requirement_id": "REQ-001",
      "alignment_score": 0.8,
      "gaps": ["Missing fraud detection"],
      "violations": [],
      "risk_level": "MEDIUM",
      "supporting_sections": ["FSR·3.1"]
    }}
  ],
  "final_audit_summary": {{
    "overall_trust_status": "Needs_Human_Review",
    "key_risks": ["Incomplete compliance coverage"],
    "compliance_score_summary": "Average score: 75",
    "reasoning": "Some claims lack full support and minor compliance gaps exist"
  }}
}}

---

### IMPORTANT RULES:

- DO NOT hallucinate
- Use ONLY provided data
- Be precise and audit-focused
- Maintain consistency with inputs
- Output MUST be valid JSON
"""

    # 3. Call Gemini
    import asyncio
    try:
        response_text = await call_gemini(prompt, temperature=0.1)
    except Exception as e:
        raise RuntimeError(f"Error calling Gemini to generate Audit Report: {e}")

    # 4. Clean JSON response from markdown wrappers
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    # 5. Parse Output
    try:
        audit_json = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse audit JSON: {e}\nRaw Response:\n{response_text}")

    # Update with true local runtime timestamp and raw strings
    # We guarantee ALL fields of AuditReport are populated natively!
    audit_json["timestamp"] = datetime.utcnow().isoformat()
    audit_json["query"] = query
    audit_json["response"] = rag_response
    audit_json["claims"] = [c.model_dump() for c in claims]
    audit_json["verifications"] = [v.model_dump() for v in verifications]
    audit_json["trust_gate"] = trust_gate.model_dump() if trust_gate else None
    audit_json["edition_conflicts"] = [c.model_dump() for c in edition_conflicts]

    # 6. Save the Audit Log into SQLite Database (including trust gate status)
    trust_status_str = trust_gate.status if trust_gate else ""
    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO audit_logs (timestamp, query, trust_gate_status, audit_data_json)
        VALUES (?, ?, ?, ?)
    ''', (audit_json["timestamp"], query, trust_status_str, json.dumps(audit_json)))
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Inject the database ID so the frontend can reference it
    audit_json["id"] = log_id
    return audit_json

def get_all_logs() -> list:
    """Retrieves list of all audit logs with extracted key metrics."""
    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, query, audit_data_json FROM audit_logs ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    logs = []
    for row in rows:
        try:
            data = json.loads(row["audit_data_json"])
            risk_level = "Unknown"
            score = "N/A"
            if "final_audit_summary" in data:
                risk_level = data["final_audit_summary"].get("overall_trust_status", "Unknown")
                score = data["final_audit_summary"].get("compliance_score_summary", "N/A")
                
            logs.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "query": row["query"],
                "risk_level": risk_level,
                "compliance_score": score
            })
        except Exception:
            logs.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "query": row["query"],
                "risk_level": "Error",
                "compliance_score": "Error"
            })
    return logs

def get_audit_by_id(log_id: int) -> dict:
    """Retrieves a single full audit log by its ID."""
    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT audit_data_json FROM audit_logs WHERE id = ?", (log_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return json.loads(row["audit_data_json"])
    return None
