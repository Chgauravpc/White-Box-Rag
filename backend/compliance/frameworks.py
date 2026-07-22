"""
frameworks.py — static mapping of this system's capabilities to real AI-governance
control frameworks (EU AI Act, NIST AI RMF).

Reframes the project from "a RAG tool" to "an AI-governance-aligned layer" by
stating, per control, which feature satisfies it and where the evidence lives.
Deliberately static (no LLM, no per-query cost) and honest: status is one of
satisfied / partial / planned, and `partial` is used wherever the system
addresses part of a control but not the full regulatory obligation.

`page` deep-links a control to the Streamlit view that demonstrates it.
"""

_STATUS_WEIGHT = {"satisfied": 1.0, "partial": 0.5, "planned": 0.0}


FRAMEWORKS = [
    {
        "framework": "EU AI Act",
        "reference": "Regulation (EU) 2024/1689 — high-risk system obligations (Title III, Ch. 2)",
        "controls": [
            {
                "control_id": "Art. 9",
                "title": "Risk management system",
                "requirement": "Establish and maintain a risk management process across the AI lifecycle.",
                "satisfied_by": ["Trust Gating", "Offline evaluation harness"],
                "evidence": ["Shapley-style penalty gating (trust_gate.py)", "Eval trend tracking with pass/fail thresholds"],
                "status": "partial",
                "page": "views/evaluation.py",
            },
            {
                "control_id": "Art. 10",
                "title": "Data and data governance",
                "requirement": "Use documented, traceable data with quality controls.",
                "satisfied_by": ["Ingestion provenance", "structured flag"],
                "evidence": ["Per-chunk publication/edition/section provenance", "structured flag marks page-level fallback"],
                "status": "partial",
                "page": "views/ingest.py",
            },
            {
                "control_id": "Art. 12",
                "title": "Record-keeping (logging)",
                "requirement": "Automatically record events over the system's lifetime to ensure traceability.",
                "satisfied_by": ["Tamper-evident audit log"],
                "evidence": ["SHA-256 hash-chained audit records", "GET /api/audit/verify-integrity"],
                "status": "satisfied",
                "page": "views/audit_trail.py",
            },
            {
                "control_id": "Art. 13",
                "title": "Transparency & provision of information",
                "requirement": "Enable users to interpret output and use the system appropriately.",
                "satisfied_by": ["Trust Scorecard", "XAI artifacts", "Counterfactual explanations", "Citations"],
                "evidence": ["8-metric RAGAS-style scorecard", "Per-claim attribution & Shapley", "'What would change the verdict' panel"],
                "status": "satisfied",
                "page": "views/query.py",
            },
            {
                "control_id": "Art. 14",
                "title": "Human oversight",
                "requirement": "Enable effective oversight by natural persons, including override.",
                "satisfied_by": ["Human-in-the-loop review queue"],
                "evidence": ["NEEDS_HUMAN_REVIEW routes to a review queue", "Chained approve/override/reject resolutions"],
                "status": "satisfied",
                "page": "views/review_queue.py",
            },
            {
                "control_id": "Art. 15",
                "title": "Accuracy, robustness & cybersecurity",
                "requirement": "Achieve appropriate accuracy and robustness, and declare metrics.",
                "satisfied_by": ["Conformal abstention", "Claim filtering", "Eval harness"],
                "evidence": ["Statistically-calibrated abstention (coverage guarantee)", "Contradicted/low-entailment claims stripped"],
                "status": "partial",
                "page": "views/evaluation.py",
            },
        ],
    },
    {
        "framework": "NIST AI RMF 1.0",
        "reference": "NIST AI Risk Management Framework — Govern / Map / Measure / Manage",
        "controls": [
            {
                "control_id": "GOVERN",
                "title": "Governance & accountability",
                "requirement": "Policies, roles, and accountability for AI risk are in place.",
                "satisfied_by": ["Policy thresholds", "Tamper-evident trail"],
                "evidence": ["Centralised, tunable trust/mitigation thresholds", "Non-repudiable audit chain"],
                "status": "partial",
                "page": "views/audit_trail.py",
            },
            {
                "control_id": "MAP",
                "title": "Context & provenance",
                "requirement": "Establish context and map sources/limitations of the system.",
                "satisfied_by": ["Hybrid retrieval provenance", "Domain-agnostic ingestion"],
                "evidence": ["Per-claim source citations", "Retrieval similarity matrix", "structured flag"],
                "status": "satisfied",
                "page": "views/ingest.py",
            },
            {
                "control_id": "MEASURE",
                "title": "Measurement & evaluation",
                "requirement": "Quantitatively and qualitatively analyze and track risk.",
                "satisfied_by": ["Trust Scorecard", "Eval harness", "Latency/cost tracking", "Conformal coverage"],
                "evidence": ["faithfulness_raw vs post", "p50/p95 latency + Gemini call counts", "Calibrated coverage target"],
                "status": "satisfied",
                "page": "views/evaluation.py",
            },
            {
                "control_id": "MANAGE",
                "title": "Risk treatment & response",
                "requirement": "Prioritize and act on risks; escalate to humans as needed.",
                "satisfied_by": ["Claim filtering", "Abstention", "HITL", "Edition-conflict gating"],
                "evidence": ["Ungrounded claims stripped from answers", "Abstain when evidence is too weak", "Review queue for flagged audits"],
                "status": "satisfied",
                "page": "views/review_queue.py",
            },
        ],
    },
]


def get_frameworks() -> list[dict]:
    """Return the catalog with per-framework coverage and status counts computed."""
    out = []
    for fw in FRAMEWORKS:
        controls = fw["controls"]
        n = len(controls)
        coverage = round(sum(_STATUS_WEIGHT.get(c["status"], 0.0) for c in controls) / n, 4) if n else 0.0
        status_counts = {s: sum(1 for c in controls if c["status"] == s) for s in _STATUS_WEIGHT}
        out.append({
            "framework": fw["framework"],
            "reference": fw["reference"],
            "num_controls": n,
            "coverage": coverage,
            "status_counts": status_counts,
            "controls": controls,
        })
    return out
