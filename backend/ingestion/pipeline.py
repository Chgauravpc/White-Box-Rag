"""
Query Pipeline — the full RAG + Verification + Compliance Audit orchestration.

Extracted from ingestion/routes.py so it has one canonical call path usable
both from the HTTP route and directly (no HTTP round-trip) by the offline
evaluation harness (backend/eval/harness.py).
"""

import logging
import time

import numpy as np

from shared.models import (
    AuditReport,
    BRDRequirement,
    XAIArtifacts,
    RetrievalMatrix,
    EntailmentMatrix,
    AttributionMatrix,
    ShapleyContributions,
)
from ingestion.rag import rag_query
from ingestion.retriever import hybrid_retrieve

from verification.nli_engine import verify_all_claims
from verification.trust_gate import compute_trust_gate
from verification.mitigation import filter_claims, should_abstain, ABSTENTION_MESSAGE_TEMPLATE
from verification.scorecard import generate_scorecard
from verification.stability import compute_paraphrase_stability
from verification.edition_conflict import discover_and_check_conflicts
from compliance.mapper import map_requirement
from compliance.audit import generate_audit_report
from shared.xai_matrices import (
    build_retrieval_similarity_matrix,
    compute_shapley_contributions,
    compute_primary_attributions,
    find_related_queries,
    get_encoder,
    compute_answer_relevancy,
    compute_context_utilization,
    compute_context_diversity,
)
from shared.database import store_query_embedding, get_past_query_embeddings
from shared.models import NLIVerdict

logger = logging.getLogger(__name__)


def _faithfulness(verifications: list) -> float:
    """entailed/total — 1.0 (vacuously faithful) when there's nothing to verify."""
    if not verifications:
        return 1.0
    entailed = sum(1 for v in verifications if v.verdict in (NLIVerdict.ENTAILMENT, NLIVerdict.SUPPORTED))
    return round(entailed / len(verifications), 6)


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Deduplicate retrieved chunks: keep one (highest-index = highest RRF) per section_id.
    Filters out UNSTRUCTURED-p* chunks from similarity matrices to keep provenance clean.
    """
    seen: dict[str, dict] = {}
    for chunk in chunks:
        sid = chunk["section_id"]
        if sid not in seen:
            seen[sid] = chunk  # first occurrence = highest RRF rank
    return list(seen.values())


async def run_query_pipeline(query: str, filters: dict | None = None) -> AuditReport:
    """Ask a question and get a fully integrated RAG + Verification + Compliance Audit response.

    The output includes all XAI mathematical artifacts:
    - retrieval_similarity_matrix: S = q·Kᵀ/(‖q‖·‖k‖) for every retrieved chunk
    - shapley: Shapley-style penalty decomposition
    - faithfulness: |ENTAILMENT| / total claims
    - related_queries: past queries with cosine_sim > 0.70
    """
    latency_ms: dict[str, float] = {}
    gemini_call_count = 0

    # ── Step 1: Hybrid Retrieval & Retrieval Matrix (S) ──
    logger.info(f"Step 1: Retrieving chunks for: '{query[:80]}'")
    _t0 = time.monotonic()
    raw_chunks = hybrid_retrieve(query=query, filters=filters)

    # Convert to dicts, deduplicate by section_id
    dict_chunks = [{
        "chunk_text":       c.chunk_text,
        "section_id":       c.section_id,
        "publication_name": c.publication_name,
        "edition_date":     c.edition_date,
    } for c in raw_chunks]
    dict_chunks = _deduplicate_chunks(dict_chunks)

    # Matrix 1 (S) — Cosine similarity of query vs each unique chunk
    chunk_ids, S_scores = build_retrieval_similarity_matrix(query, dict_chunks)
    latency_ms["retrieval_ms"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── Step 2: Generation & Attribution Matrix (A) ──
    _t0 = time.monotonic()
    rag_response, A_matrix = await rag_query(query, dict_chunks)
    gemini_call_count += 1
    latency_ms["generation_ms"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── Step 3: NLI Verification, Entailment Matrix (E), focused passages ──
    logger.info("Step 3: Batched NLI verification via CrossEncoder")
    _t0 = time.monotonic()
    verifications, E_matrix, focused_passages = await verify_all_claims(rag_response.claims)
    latency_ms["verification_ms"] = round((time.monotonic() - _t0) * 1000, 2)

    # Stamp focused_passage onto each Claim for full audit traceability
    for i, claim in enumerate(rag_response.claims):
        if i < len(focused_passages):
            claim.focused_passage = focused_passages[i]

    # ── Step 4: XAI Math — Shapley (φ), primary attributions, assemble artifacts ──
    logger.info("Step 4: Computing Shapley and attribution artifacts")

    A_scores = A_matrix.tolist() if isinstance(A_matrix, np.ndarray) and A_matrix.size > 0 else []
    E_scores = E_matrix.tolist() if isinstance(E_matrix, np.ndarray) and E_matrix.size > 0 else []

    # Primary attribution per sentence (argmax + runner-up)
    prim_attrs = compute_primary_attributions(A_matrix, chunk_ids) if A_scores else []

    # Edition conflicts — real cross-edition contradiction discovery. Guarded
    # internally against UNSTRUCTURED (page-level) sections and only fires for
    # sections with genuine multi-edition versioning in the knowledge base.
    _t0 = time.monotonic()
    conflicts = await discover_and_check_conflicts(rag_response.claims)
    latency_ms["conflict_detection_ms"] = round((time.monotonic() - _t0) * 1000, 2)

    # Trust gate — computed once prim_attrs are available (NLI + attribution quality)
    trust_gate = compute_trust_gate(verifications, conflicts, prim_attrs)

    # Shapley: mirrors trust_gate penalties exactly (NLI + attribution)
    shapley = compute_shapley_contributions(
        [v.model_dump() for v in verifications],
        prim_attrs,
    )

    retrieval_mat = RetrievalMatrix(
        chunk_ids=chunk_ids,
        similarity_scores=S_scores,
    )
    attr_mat = AttributionMatrix(
        sentence_texts=[c.text for c in rag_response.claims],
        chunk_ids=chunk_ids,
        scores=A_scores,
        primary_attributions=prim_attrs,
    )
    seen_passages = {}
    for c in rag_response.claims:
        if c.source_passage and c.source_section_id not in seen_passages:
            seen_passages[c.source_section_id] = c.source_passage
    entail_mat = EntailmentMatrix(
        claim_texts=[c.text for c in rag_response.claims],
        passage_ids=list(seen_passages.keys()),
        passage_texts=list(seen_passages.values()),
        scores=E_scores,
        labels=["contradiction", "entailment", "neutral"],
    )
    shapley_mat = ShapleyContributions(
        claim_texts=shapley["claim_texts"],
        shapley_values=shapley["shapley_values"],
        penalty_reasons=shapley["penalty_reasons"],
        overall_score=shapley["overall_score"],
    )
    xai_artifacts = XAIArtifacts(
        retrieval=retrieval_mat,
        entailment=entail_mat,
        attribution=attr_mat,
        conflict=None,
        shapley=shapley_mat,
    )

    # ── Step 4b: Hallucination mitigation — claim-level filtering + abstention ──
    logger.info("Step 4b: Applying claim-level filtering and abstention checks")
    retained_texts, retained_flags, filter_reasons = filter_claims(rag_response.claims, verifications, prim_attrs)
    for i, claim in enumerate(rag_response.claims):
        if i < len(retained_flags):
            claim.retained = retained_flags[i]
            claim.filter_reason = filter_reasons[i]

    abstained, raw_abstain_reason, retained_trust_score = should_abstain(
        rag_response.claims, verifications, prim_attrs, retained_flags, conflicts
    )
    abstention_reason = ABSTENTION_MESSAGE_TEMPLATE.format(reason=raw_abstain_reason) if abstained else ""

    faithfulness_raw = _faithfulness(verifications)
    retained_verifications = [v for v, keep in zip(verifications, retained_flags) if keep]
    faithfulness_post = _faithfulness(retained_verifications)

    # ── Step 4c: Trust Scorecard (real metrics, pure math + conditional self-consistency) ──
    logger.info("Step 4c: Computing trust scorecard")
    _t0 = time.monotonic()
    paraphrase_stability, stability_note = await compute_paraphrase_stability(
        query, dict_chunks, rag_response.claims, trust_gate.status
    )
    latency_ms["stability_check_ms"] = round((time.monotonic() - _t0) * 1000, 2)
    if not stability_note.startswith("Skipped"):
        gemini_call_count += 1
    answer_relevancy = compute_answer_relevancy(query, rag_response.answer)
    context_utilization = compute_context_utilization(prim_attrs, chunk_ids)
    context_diversity = compute_context_diversity(dict_chunks)

    scorecard = generate_scorecard(
        query=query,
        response=rag_response.answer,
        verifications=retained_verifications,
        conflicts=conflicts,
        claims=rag_response.claims,
        retrieval_scores=S_scores,
        paraphrase_stability=paraphrase_stability,
        answer_relevancy=answer_relevancy,
        context_utilization=context_utilization,
        context_diversity=context_diversity,
    )

    # ── Step 5: Requirement/Compliance Mapping (BP3) ──
    logger.info("Step 5: Mapping the query itself as a requirement-alignment gap check")
    _t0 = time.monotonic()
    brd_req = BRDRequirement(id="ASK", text=query)
    mapped_dict = await map_requirement(brd_req)
    gemini_call_count += 1
    brd_req.mapped_sections = [c.chunk_text for c in mapped_dict.get("relevant_chunks", [])]
    brd_req.alignment_score = mapped_dict.get("alignment_score", 0.0)
    brd_req.gaps = mapped_dict.get("gaps", [])
    brd_req.risk_flags = mapped_dict.get("violations", [])
    brd_req.risk_level = mapped_dict.get("risk_level", "LOW")
    brd_req.remediation = mapped_dict.get("remediation_suggestions", "")
    latency_ms["compliance_mapping_ms"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── Step 6: Generate Audit Report ──
    logger.info("Step 6: Compiling Audit Report")
    _t0 = time.monotonic()
    audit_report_dict = await generate_audit_report(
        query=query,
        rag_response=rag_response.answer,
        claims=rag_response.claims,
        verifications=verifications,
        trust_gate=trust_gate,
        edition_conflicts=conflicts,
        brd_results=[brd_req],
    )
    gemini_call_count += 1
    latency_ms["audit_report_ms"] = round((time.monotonic() - _t0) * 1000, 2)

    # ── Step 7: Store embedding + find related queries (pure cosine, no LLM) ──
    log_id = audit_report_dict.get("id")
    q_emb = get_encoder().encode(query).tolist()
    if log_id:
        store_query_embedding(log_id, q_emb)

    past = get_past_query_embeddings(exclude_id=log_id)
    related = find_related_queries(query, past)
    # Exclude near-exact matches (same query re-run) from related list
    related = [r for r in related if r.get("cosine_similarity", 0) < 0.99]

    audit_report_dict["xai_artifacts"] = xai_artifacts.model_dump()
    audit_report_dict["related_queries"] = related
    audit_report_dict["response"] = rag_response.answer
    audit_report_dict["claims"] = [c.model_dump() for c in rag_response.claims]
    audit_report_dict["verifications"] = [v.model_dump() for v in verifications]
    audit_report_dict["trust_gate"] = trust_gate.model_dump() if trust_gate else None
    audit_report_dict["edition_conflicts"] = [c.model_dump() if hasattr(c, "model_dump") else c for c in conflicts]
    audit_report_dict["filtered_claims"] = retained_texts
    audit_report_dict["faithfulness_raw"] = faithfulness_raw
    audit_report_dict["faithfulness_post"] = faithfulness_post
    audit_report_dict["retained_trust_score"] = retained_trust_score
    audit_report_dict["abstained"] = abstained
    audit_report_dict["abstention_reason"] = abstention_reason
    audit_report_dict["scorecard"] = scorecard.model_dump()
    audit_report_dict["latency_ms"] = latency_ms
    audit_report_dict["gemini_call_count"] = gemini_call_count

    return AuditReport(**audit_report_dict)
