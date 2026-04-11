import axios from "axios";
import type { 
  DocumentInfo, 
  AuditReport,
  AuditLogItem,
  Claim
} from "./types";

const api = axios.create({ baseURL: "/api" });

/**
 * Ingests an RBI PDF document.
 */
export const ingestDocument = (file: File, publication: string, edition: string) => {
  const form = new FormData();
  form.append("file", file); 
  form.append("publication", publication); 
  form.append("edition_date", edition);
  return api.post("/ingest", form);
};

/**
 * Lists all ingested publications.
 */
export const listDocuments = () => api.get<DocumentInfo[]>("/documents");

/**
 * Unified RAG Query pipeline.
 * Returns the full AuditReport (Answer + Verification + Compliance Monitoring).
 */
export const queryRAG = (query: string, filters?: Record<string, any>) => {
  return api.post<AuditReport>("/query", { query, filters });
};

/**
 * Lists summaries of previous audit logs.
 */
export const listAuditLogs = () => api.get<AuditLogItem[]>("/audit/logs");

/**
 * Retrieves a full detailed audit report by database ID.
 */
export const getAuditReport = (id: number) => api.get<AuditReport>(`/audit/${id}`);

/**
 * Downloads a PDF/JSON version of the audit report.
 */
export const downloadAudit = (id: number) => api.get(`/audit/${id}/download`, { responseType: "blob" });

/**
 * Uploads a BRD document for parsing.
 */
export const uploadBRD = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/brd/upload", form);
};

/**
 * Validates a list of requirement strings against the knowledge base.
 */
export const validateBRD = (requirements: string[]) => {
  return api.post("/brd/validate", { requirements });
};

/**
 * Verifies an AI response's claims under Trust Gating.
 */
export const verifyClaims = (answer: string, claims: Claim[]) => {
  return api.post("/verify/", { answer, claims });
};

/**
 * Checks for edition conflicts between texts.
 */
export const checkConflicts = (payload: any) => {
  return api.post("/verify/check-conflicts", payload);
};
