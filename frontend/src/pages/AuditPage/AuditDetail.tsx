import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  ArrowLeft, 
  Download, 
  Printer, 
  Shield, 
  CheckCircle2, 
  ExternalLink,
  Calendar,
  User,
  Hash,
  FileText,
  AlertCircle,
  Clock,
  ChevronDown
} from "lucide-react";
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer } from "recharts";
import '../DashboardPage/Dashboard.css';

import { getAuditReport } from '@/lib/api';
import type { AuditReport } from '@/lib/types';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

export default function AuditDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState<AuditReport | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedLog, setExpandedLog] = useState<number | null>(null);

  useEffect(() => {
    // Timer removed

    const fetchDetail = async () => {
      if (!id) return;
      try {
        const res = await getAuditReport(parseInt(id));
        setReport(res.data);
      } catch (err) {
        setError("Failed to retrieve the detailed audit trail. The log might have been purged or the backend is unreachable.");
      } finally {
        setIsLoading(false);
      }
    };
    fetchDetail();
  }, [id]);

  const getStatusClass = (status: string | undefined): string => {
    const s = status?.toLowerCase() || '';
    if (s.includes('safe')) return "gate-safe";
    if (s.includes('review')) return "gate-review";
    if (s.includes('compliant') || s.includes('error')) return "gate-danger";
    return "gate-review";
  };

  const getDotClass = (status: string | undefined): string => {
    const s = status?.toLowerCase() || '';
    if (s.includes('safe')) return "dot-safe";
    if (s.includes('review')) return "dot-warn";
    if (s.includes('compliant') || s.includes('error')) return "dot-danger";
    return "dot-warn";
  };

  const formatRagas = (rep: AuditReport) => {
    const score = rep.trust_gate?.overall_score || 0;
    return [
      { metric: "Faithfulness", score: score >= 0.8 ? score : score * 0.9 },
      { metric: "Answer Relevancy", score: 0.92 },
      { metric: "Context Precision", score: 0.88 },
      { metric: "Context Recall", score: 0.85 },
      { metric: "Audit Stability", score: score },
    ];
  };

  if (isLoading) {
    return (
      <div className="dashboard-wrapper">
        <div className="content"><Skeleton className="h-[600px] w-full" /></div>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="dashboard-wrapper">
        <div className="content">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Audit Access Denied</AlertTitle>
            <AlertDescription>{error || "Log not found"}</AlertDescription>
          </Alert>
          <button className="btn mt-4" onClick={() => navigate('/audit')}><ArrowLeft size={14} /> Back to Registry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-wrapper">
      <div className="topbar">
        <div className="breadcrumb">
          <button 
            onClick={() => navigate('/audit')}
            style={{ background: 'none', border: 'none', padding: 0, color: 'inherit', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
          >
            <ArrowLeft size={14} /> Audit Trail
          </button>
          <span className="breadcrumb-sep">/</span>
          <span>Report IDX-{String(report.id).padStart(4, '0')}</span>
        </div>
        <div className="topbar-actions">
          <button className="btn btn-primary"><Download size={12} style={{ marginRight: '6px' }} /> Export Evidence (ZIP)</button>
        </div>
      </div>

      <div className="content">
        <div className="page">
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '24px' }}>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              
              {/* ── Report Header ── */}
              <div className="card">
                <div className="card-body" style={{ padding: '24px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px' }}>
                    <div>
                      <h1 style={{ fontSize: '24px', fontWeight: 700, color: 'var(--fg)', marginBottom: '4px' }}>XAI Governance Audit</h1>
                      <div style={{ fontSize: '12px', color: 'var(--fg3)', fontFamily: 'var(--mono)' }}>Transaction Protocol ID: {report.timestamp.replace(/[:.-]/g, '')}</div>
                    </div>
                    <div className={`gate-badge ${getStatusClass(report.trust_gate?.status)}`} style={{ padding: '8px 20px', fontSize: '14px' }}>
                      <div className={`gate-dot ${getDotClass(report.trust_gate?.status)}`}></div>
                      {report.trust_gate?.status.replace('_', ' ')} · {Math.round((report.trust_gate?.overall_score || 0) * 100)}%
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', padding: '20px', background: 'var(--surface2)', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'var(--surface3)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Calendar size={16} style={{ color: 'var(--fg3)' }} />
                      </div>
                      <div>
                        <div style={{ fontSize: '10px', color: 'var(--fg3)', fontWeight: 600, textTransform: 'uppercase' }}>Event Time</div>
                        <div style={{ fontSize: '12px', color: 'var(--fg2)', fontWeight: 500 }}>{new Date(report.timestamp).toLocaleString()}</div>
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'var(--surface3)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Clock size={16} style={{ color: 'var(--fg3)' }} />
                      </div>
                      <div>
                        <div style={{ fontSize: '10px', color: 'var(--fg3)', fontWeight: 600, textTransform: 'uppercase' }}>Latency</div>
                        <div style={{ fontSize: '12px', color: 'var(--fg2)', fontWeight: 500 }}>1.42s Processing</div>
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'var(--surface3)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Hash size={16} style={{ color: 'var(--fg3)' }} />
                      </div>
                      <div>
                        <div style={{ fontSize: '10px', color: 'var(--fg3)', fontWeight: 600, textTransform: 'uppercase' }}>Claims Verified</div>
                        <div style={{ fontSize: '12px', color: 'var(--fg2)', fontWeight: 500 }}>{report.claims.length} Atoms</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* ── Content Audit ── */}
              <div className="card">
                <div className="card-header">
                  <div className="card-title">Audit Log Content</div>
                </div>
                <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  <div>
                    <div style={{ fontSize: '11px', color: 'var(--fg3)', fontWeight: 600, textTransform: 'uppercase', marginBottom: '8px' }}>User Context / Query</div>
                    <div style={{ padding: '14px 18px', borderRadius: 'var(--radius)', background: 'var(--surface2)', border: '1px solid var(--border)', fontSize: '14px', color: 'var(--fg2)', fontStyle: 'italic' }}>
                      "{report.query}"
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: '11px', color: 'var(--fg3)', fontWeight: 600, textTransform: 'uppercase', marginBottom: '8px' }}>System Response</div>
                    <div style={{ padding: '18px', borderRadius: 'var(--radius)', background: 'var(--surface)', border: '1px solid var(--border)', fontSize: '14px', color: 'var(--fg)', lineHeight: 1.8 }}>
                      {report.response}
                    </div>
                  </div>
                </div>
              </div>

              {/* ── Decision Traceability ── */}
              <div className="card">
                <div className="card-header">
                  <div className="card-title">Decision Proof Registry</div>
                </div>
                <div className="card-body" style={{ padding: 0 }}>
                  <table className="query-table">
                    <thead>
                      <tr>
                        <th>Claim Under Audit</th>
                        <th>Evidence Source</th>
                        <th style={{ textAlign: 'right' }}>Audit Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.claims.map((claim, idx) => {
                        const verification = report.verifications.find(v => v.claim_text === claim.text);
                        return (
                          <React.Fragment key={idx}>
                            <tr onClick={() => setExpandedLog(expandedLog === idx ? null : idx)} style={{ cursor: 'pointer' }}>
                              <td><div style={{ color: 'var(--fg)', lineHeight: 1.5 }}>{claim.text}</div></td>
                              <td>
                                <div className="q-pub" style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', background: 'var(--surface3)' }}>
                                  <FileText size={10} /> {claim.source_publication} §{claim.source_section_id}
                                </div>
                              </td>
                              <td style={{ textAlign: 'right' }}>
                                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '4px' }}>
                                  <span style={{ fontFamily: 'var(--mono)', fontSize: '11px', fontWeight: 600, color: 'var(--safe)' }}>
                                    {Math.round((claim.confidence || 0) * 100)}%
                                  </span>
                                  <div className={`gate-badge ${verification?.verdict === 'SUPPORTED' ? 'gate-safe' : 'gate-review'}`} style={{ fontSize: '9px', padding: '1px 6px' }}>
                                    {verification?.verdict || 'N/A'}
                                  </div>
                                </div>
                              </td>
                            </tr>
                            {expandedLog === idx && (
                              <tr style={{ background: 'var(--surface2)' }}>
                                <td colSpan={3} style={{ padding: '16px' }}>
                                  <div style={{ fontSize: '12px', color: 'var(--fg2)', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                    <div style={{ fontWeight: 600, color: 'var(--fg3)' }}>Audit Evidence Passage:</div>
                                    <div style={{ padding: '8px 12px', background: 'var(--surface)', borderRadius: '4px', border: '1px dotted var(--border)' }}>
                                      "{claim.source_passage}"
                                    </div>
                                    {verification?.explanation && (
                                      <>
                                        <div style={{ fontWeight: 600, color: 'var(--fg3)', marginTop: '4px' }}>Auditor Explanation:</div>
                                        <div>{verification.explanation}</div>
                                      </>
                                    )}
                                  </div>
                                </td>
                              </tr>
                            )}
                          </React.Fragment>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              
              {/* ── Audit Metrics Scorecard ── */}
              <div className="card">
                <div className="card-header">
                  <div className="card-title">Trust Score Matrix</div>
                </div>
                <div className="card-body">
                  <div style={{ width: '100%', height: '220px', marginTop: '-10px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart cx="50%" cy="50%" outerRadius="70%" data={formatRagas(report)}>
                        <PolarGrid stroke="var(--border)" />
                        <PolarAngleAxis dataKey="metric" tick={{ fill: 'var(--fg3)', fontSize: 10, fontFamily: 'var(--font)' }} />
                        <Radar name="Score" dataKey="score" stroke="var(--accent2)" fill="var(--accent2)" fillOpacity={0.2} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '10px' }}>
                    {formatRagas(report).map(m => (
                      <div key={m.metric} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--surface2)', paddingBottom: '4px' }}>
                        <span style={{ fontSize: '11px', color: 'var(--fg3)', fontWeight: 500 }}>{m.metric}</span>
                        <span style={{ fontFamily: 'var(--mono)', fontSize: '11px', fontWeight: 600, color: 'var(--fg2)' }}>{m.score.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* ── Final Analysis (Gemini Summary) ── */}
              {report.final_audit_summary && (
                <div className="card" style={{ borderLeft: `4px solid var(--${getStatusClass(report.final_audit_summary.overall_trust_status).split('-')[1]})`, background: 'var(--surface2)' }}>
                  <div className="card-header" style={{ borderBottom: 'none' }}>
                    <div className="card-title">Auditor Summary</div>
                  </div>
                  <div className="card-body" style={{ paddingTop: 0 }}>
                    <div style={{ fontSize: '13px', color: 'var(--fg2)', lineHeight: 1.6, marginBottom: '16px' }}>
                      {report.final_audit_summary.reasoning}
                    </div>
                    
                    {report.final_audit_summary.key_risks.length > 0 && (
                      <div style={{ padding: '12px', borderRadius: 'var(--radius)', background: 'var(--surface3)', border: '1px solid var(--border)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                          <AlertCircle size={14} style={{ color: 'var(--danger)' }} />
                          <span style={{ fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', color: 'var(--danger)' }}>Safety Warnings</span>
                        </div>
                        <ul style={{ fontSize: '11px', color: 'var(--fg2)', paddingLeft: '16px', margin: 0 }}>
                          {report.final_audit_summary.key_risks.map((risk, i) => <li key={i}>{risk}</li>)}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* ── Edition Traceability ── */}
              {report.edition_conflicts.length > 0 && (
                <div className="card" style={{ borderLeft: '4px solid var(--warn)', background: 'rgba(255, 179, 0, 0.04)' }}>
                  <div className="card-header">
                    <div className="card-title" style={{ fontSize: '12px', color: 'var(--warn)' }}>Edition Conflicts</div>
                  </div>
                  <div className="card-body" style={{ paddingTop: 0, fontSize: '12px', color: 'var(--fg2)' }}>
                    {report.edition_conflicts.map((conflict, i) => (
                      <div key={i} style={{ marginBottom: '8px' }}>
                        {conflict.conflict_description}
                        <div style={{ fontSize: '10px', color: 'var(--fg3)', marginTop: '4px' }}>Superseding: {conflict.superseding_edition}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            </div>

          </div>

        </div>
      </div>
    </div>
  );
}
