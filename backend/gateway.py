"""
API Gateway — Main FastAPI application.

Mounts all service routers (ingestion, verification, compliance) under /api.
Run with:  uvicorn gateway:app --reload --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# BP1 — Ingestion & RAG
from ingestion.routes import router as ingestion_router

# BP3 — Compliance & Audit
from compliance.routes import router as compliance_router

# BP2 — Verification & Trust
from verification.routes import router as verification_router

# Offline Evaluation Harness
from eval.routes import router as eval_router

# ──────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ──────────────────────────────────────────────
#  Application
# ──────────────────────────────────────────────

app = FastAPI(
    title="White Box RAG — Explainable Governance Framework",
    description="Domain-agnostic, compliance-grade XAI governance layer for any document corpus.",
    version="0.3.0",
)

# CORS — allow all origins for local development / hackathon
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
#  Mount Routers
# ──────────────────────────────────────────────

app.include_router(ingestion_router, prefix="/api")
app.include_router(compliance_router, prefix="/api")

app.include_router(verification_router, prefix="/api")
app.include_router(eval_router, prefix="/api")


# ──────────────────────────────────────────────
#  Health Check
# ──────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "white-box-rag", "version": "0.3.0"}
