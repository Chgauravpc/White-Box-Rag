"""Ingest Documents — upload a PDF into a named collection."""

import pandas as pd
import streamlit as st

from lib import api_client
from lib.api_client import ApiError

st.title("📥 Ingest Documents")
st.caption("Upload a PDF into any named collection — this system is domain-agnostic, not tied to any fixed set of publications.")

try:
    documents = api_client.list_documents()
except ApiError as e:
    documents = []
    st.error(f"Could not load documents: {e}")

existing_collections = sorted({d["publication_name"] for d in documents}) if documents else []

col_upload, col_kb = st.columns([1, 1])

with col_upload:
    st.subheader("Upload a PDF")
    # Wrapped in st.form so the uploader doesn't re-trigger ingestion on every
    # rerun (e.g. the st.rerun() after a successful upload) — Streamlit keeps
    # the uploaded file in the widget state across reruns otherwise.
    with st.form("ingest_form", clear_on_submit=True):
        uploaded = st.file_uploader("Source PDF", type=["pdf"])

        collection_choice = st.selectbox(
            "Collection",
            options=existing_collections + ["+ New collection..."],
            index=len(existing_collections) if not existing_collections else 0,
        )
        new_collection = ""
        if collection_choice == "+ New collection..." or not existing_collections:
            new_collection = st.text_input("New collection name", placeholder="e.g. CONTRACTS, FSR, ENG_SPECS")

        edition_date = st.text_input(
            "Version / date label (optional)",
            placeholder="e.g. 'June 2024' — defaults to today's date if left blank",
        )

        submitted = st.form_submit_button("Ingest", type="primary")

    if submitted:
        collection = new_collection.strip() if (collection_choice == "+ New collection..." or not existing_collections) else collection_choice
        if not uploaded:
            st.error("Please select a PDF file.")
        elif not collection:
            st.error("Please provide a collection name.")
        else:
            with st.spinner("Parsing and ingesting — this can take a moment for large PDFs..."):
                try:
                    result = api_client.ingest_document(uploaded.name, uploaded.getvalue(), collection, edition_date)
                    st.success(f"Ingested {result.get('chunks_ingested', 0)} chunks from {result.get('filename')} into '{result.get('publication')}'.")
                    st.rerun()
                except ApiError as e:
                    st.error(f"Ingestion failed: {e}")

with col_kb:
    st.subheader(f"Active Knowledge Base ({len(documents)})")
    if documents:
        df = pd.DataFrame(documents)
        df = df[["publication_name", "edition_date", "chunk_count", "structured", "filename"]]
        df.columns = ["Collection", "Version", "Chunks", "Structured", "Filename"]
        st.dataframe(df, width="stretch", hide_index=True)
        if (~pd.DataFrame(documents)["structured"]).any():
            st.caption("⚠️ Documents marked 'Structured=False' had no detectable section headers — they're cited at page granularity, and cross-edition conflict checking is disabled for them.")
    else:
        st.info("No documents ingested yet.")
