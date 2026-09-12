"""
domain_profiles.py — swappable personas/vocabularies for the compliance prompts.

`brd_parser.py` and `audit.py` used to hardcode an RBI/Indian-financial-
regulation persona, a fixed category/regulatory_relevance enum, and RBI-
specific few-shot examples on every prompt — regardless of what was actually
ingested. That contradicts the project's domain-agnostic design elsewhere
(ingestion accepts any free-text collection label). Selected via
`shared.config.DOMAIN_PROFILE`; "generic" is the default.
"""

from shared.config import DOMAIN_PROFILE

GENERIC = {
    "persona": "You are a senior requirements and compliance analyst.",
    "categories": [
        "Functional", "Non-Functional", "Security", "Data Handling",
        "Reporting", "Operational", "Legal/Regulatory", "Other",
    ],
    # None => regulatory_relevance is free text, not a fixed enum.
    "relevance_fixed_enum": None,
    "relevance_hint_examples": ["internal policy", "contractual SLA", "applicable regulation"],
    "few_shot_publication": "DOC-A",
    "few_shot_edition": "2024-06",
    "few_shot_section_id": "3.1",
    "few_shot_section_title": "Key Definitions and Scope",
    "few_shot_summary": "Defines key terms and the scope of the document",
}

FINANCIAL_REPORTS = {
    "persona": "You are a financial regulatory analyst specializing in RBI (Reserve Bank of India) compliance.",
    "categories": [
        "KYC", "Payments", "Lending", "Risk Management", "Reporting",
        "Compliance", "Fraud Detection", "Customer Onboarding", "Data Security", "Other",
    ],
    "relevance_fixed_enum": ["FSR", "MPR", "PSR", "FER"],
    "relevance_hint_examples": ["FSR", "MPR", "PSR", "FER"],
    "few_shot_publication": "FSR",
    "few_shot_edition": "Dec 2024",
    "few_shot_section_id": "3.1",
    "few_shot_section_title": "Financial Stability Risks",
    "few_shot_summary": "Discusses major risks to financial stability",
}

_PROFILES = {"generic": GENERIC, "financial_reports": FINANCIAL_REPORTS}


def get_domain_profile(name: str | None = None) -> dict:
    """Return the named profile, or the active DOMAIN_PROFILE, or GENERIC as fallback."""
    return _PROFILES.get(name or DOMAIN_PROFILE, GENERIC)
