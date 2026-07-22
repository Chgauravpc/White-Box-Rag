"""
app.py — Streamlit entrypoint. Registers pages via st.navigation (not the
classic numbered `pages/` folder, which would auto-generate a conflicting
second sidebar nav) so the sidebar can be grouped into sections matching the
original app's IA: Overview / Verification / System / Evaluation.

Run alongside the backend:
    uvicorn gateway:app --reload --port 8000      (from backend/)
    streamlit run streamlit_app/app.py            (from repo root)
"""

import streamlit as st

from lib.ui import render_global_css

st.set_page_config(
    page_title="White Box RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
render_global_css()

dashboard = st.Page("views/dashboard.py", title="Dashboard", icon="📊", default=True)
query_page = st.Page("views/query.py", title="RAG Query", icon="💬")
verify_page = st.Page("views/verify.py", title="Trust Gating", icon="🛡️")
brd_page = st.Page("views/brd_validator.py", title="BRD Validator", icon="📋")
ingest_page = st.Page("views/ingest.py", title="Ingest Documents", icon="📥")
audit_page = st.Page("views/audit_trail.py", title="Audit Trail", icon="🗂️")
review_page = st.Page("views/review_queue.py", title="Review Queue", icon="🧑‍⚖️")
mapping_page = st.Page("views/regulatory_mapping.py", title="Regulatory Mapping", icon="⚖️")
eval_page = st.Page("views/evaluation.py", title="Evaluation", icon="📈")

nav = st.navigation({
    "Overview": [dashboard, query_page],
    "Verification": [verify_page, brd_page],
    "System": [ingest_page, audit_page],
    "Governance": [review_page, mapping_page],
    "Evaluation": [eval_page],
})

nav.run()
