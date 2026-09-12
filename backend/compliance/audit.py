import json
from datetime import datetime, timezone
from shared import config
from shared.models import Claim, VerificationResult, TrustGate, EditionConflict, BRDRequirement
from shared.llm import call_llm
from shared.database import get_sqlite_conn
from compliance.domain_profiles import get_domain_profile

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
    Generates a structured audit report using the configured LLM, 
    and commits the log to the SQLite database.
    """
    
    # 1. Format inputs for the prompt
    claims_json = json.dumps([c.model_dump() for c in claims], indent=2)
    verifications_json = json.dumps([v.model_dump() for v in verifications], indent=2)
    trust_gate_json = trust_gate.model_dump_json(indent=2)
    conflicts_json = json.dumps([c.model_dump() for c in edition_conflicts], indent=2)
    brd_results_json = json.dumps([b.model_dump() for b in brd_results], indent=2)

    # 2. Build the prompt — few-shot example values come from the active
    # domain profile (shared.config.DOMAIN_PROFILE), not a hardcoded domain.
    profile = get_domain_profile()
    fs_pub = profile["few_shot_publication"]
    fs_ed = profile["few_shot_edition"]
    fs_sec = profile["few_shot_section_id"]
    fs_key = f"{fs_pub}·{fs_ed}·{fs_sec}"
    fs_title = profile["few_shot_section_title"]
    fs_summary = profile["few_shot_summary"]

    prompt = f"""
You are a compliance auditor and Explainable AI (XAI) specialist.

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
List ALL knowledge base sections used:
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
      "source": "{fs_key}",
      "verification_verdict": "ENTAILMENT",
      "confidence_score": 0.92,
      "explanation": "The source clearly supports the claim"
    }}
  ],
  "section_reference_registry": [
    {{
      "publication_name": "{fs_pub}",
      "edition_date": "{fs_ed}",
      "section_id": "{fs_sec}",
      "section_title": "{fs_title}",
      "summary": "{fs_summary}"
    }}
  ],
  "edition_traceability": {{
    "editions_compared": ["{fs_ed}"],
    "conflicts_detected": false,
    "authoritative_edition": "{fs_ed}",
    "reasoning": "Only one edition was consulted for this query"
  }},
  "compliance_evidence": [
    {{
      "requirement_id": "REQ-001",
      "alignment_score": 0.8,
      "gaps": ["Example gap"],
      "violations": [],
      "risk_level": "MEDIUM",
      "supporting_sections": ["{fs_pub}·{fs_sec}"]
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

    # 3. Call the LLM — degrade, don't raise (finding #17). A failed/malformed
    # LLM narrative must not fail the whole query for a reason unrelated to
    # what's actually being measured: retrieval, verification, and trust
    # gating are all already computed deterministically by the time this
    # function runs, and every one of those fields is still populated below
    # regardless of whether the narrative call succeeded.
    audit_report_error = ""
    response_text = None
    try:
        response_text = await call_llm(prompt, temperature=config.COMPLIANCE_TEMPERATURE)
    except Exception as e:
        audit_report_error = f"LLM call failed: {e}"

    audit_json = None
    if response_text is not None:
        # 4. Clean JSON response from markdown wrappers
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # 5. Parse Output
        try:
            audit_json = json.loads(cleaned)
        except json.JSONDecodeError as e:
            audit_report_error = f"Failed to parse audit JSON: {e}"

    if audit_json is None:
        # Minimal fallback: only the LLM's narrative fields are missing —
        # decision_log prose, section_reference_registry, and
        # final_audit_summary's free-text reasoning. Everything mathematical
        # is still stamped in below.
        audit_json = {"final_audit_summary": {}}

    # Update with a true UTC runtime timestamp and raw strings
    # We guarantee ALL mathematically-derived fields of AuditReport are populated natively!
    audit_json["timestamp"] = datetime.now(timezone.utc).isoformat()
    audit_json["query"] = query
    audit_json["response"] = rag_response
    audit_json["claims"] = [c.model_dump() for c in claims]
    audit_json["verifications"] = [v.model_dump() for v in verifications]
    audit_json["trust_gate"] = trust_gate.model_dump() if trust_gate else None
    audit_json["edition_conflicts"] = [c.model_dump() for c in edition_conflicts]
    audit_json["audit_report_error"] = audit_report_error

    # The LLM's narrative summary must never be allowed to contradict the
    # deterministic Trust Gate — overwrite post-hoc rather than trust the LLM's
    # restatement of a value we already computed mathematically.
    if trust_gate and isinstance(audit_json.get("final_audit_summary"), dict):
        audit_json["final_audit_summary"]["overall_trust_status"] = trust_gate.status

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
    cursor.execute("SELECT id, timestamp, query, trust_gate_status, audit_data_json FROM audit_logs ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    logs = []
    for row in rows:
        try:
            data = json.loads(row["audit_data_json"])
            # risk_level comes from the deterministic trust_gate_status SQLite
            # column (mathematically computed by Trust Gate), not the LLM's
            # narrative final_audit_summary — the two could otherwise disagree.
            risk_level = row["trust_gate_status"] or "Unknown"
            score = "N/A"
            if "final_audit_summary" in data:
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
