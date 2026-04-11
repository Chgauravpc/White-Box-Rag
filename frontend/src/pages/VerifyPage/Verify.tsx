import React, { useState } from 'react';
import { Shield, GitCompare, AlertTriangle, CheckCircle2, ChevronRight, Scale } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { verifyClaims, checkConflicts } from '@/lib/api';
import type { TrustGate, VerificationResult, Claim, EditionConflict } from '@/lib/types';
import './Verify.css';
import '../DashboardPage/Dashboard.css';

export default function Verify() {
  const [activeTab, setActiveTab] = useState<'claims' | 'conflicts'>('claims');

  // Logic 1: Verify Claims
  const [demoAnswer, setDemoAnswer] = useState("Banks are required to maintain a capital adequacy ratio of 9%. This was mandated in the June 2024 FSR regulations.");
  const [claims, setClaims] = useState<Claim[]>([
    {
      text: "Banks must maintain 9% capital adequacy.",
      source_publication: "FSR",
      source_edition: "June 2024",
      source_section_id: "2.1",
      source_passage: "The capital adequacy framework requires banks to maintain a minimum ratio of 9%.",
      confidence: 0.95
    },
    {
      text: "This was mandated in June 2024.",
      source_publication: "FSR",
      source_edition: "June 2024",
      source_section_id: "Summary",
      source_passage: "As of this latest June 2024 release, the requirement holds.",
      confidence: 0.88
    }
  ]);
  const [isVerifying, setIsVerifying] = useState(false);
  const [trustGateResult, setTrustGateResult] = useState<TrustGate | null>(null);
  const [verificationsResult, setVerificationsResult] = useState<VerificationResult[] | null>(null);
  const [verifyError, setVerifyError] = useState<string | null>(null);

  const runVerification = async () => {
    setIsVerifying(true);
    setTrustGateResult(null);
    setVerificationsResult(null);
    setVerifyError(null);
    
    try {
      const response = await verifyClaims(demoAnswer, claims);
      setTrustGateResult(response.data.trust_gate);
      setVerificationsResult(response.data.verifications);
    } catch (err: any) {
      setVerifyError(err.message || "Failed to reach verification api");
    } finally {
      setIsVerifying(false);
    }
  };

  // Logic 2: Cross-Edition Conflicts
  const [isChecking, setIsChecking] = useState(false);
  const [conflictResult, setConflictResult] = useState<EditionConflict | null>(null);
  const [conflictError, setConflictError] = useState<string | null>(null);

  const [conflictForm, setConflictForm] = useState({
    publication: "FSR",
    topic: "Capital Requirements",
    older_date: "Dec 2023",
    older_text: "The capital requirement is 8%.",
    newer_date: "June 2024",
    newer_text: "The capital requirement has been updated to 9%.",
    section_id: "2.1"
  });

  const runConflictCheck = async () => {
    setIsChecking(true);
    setConflictResult(null);
    setConflictError(null);
    try {
      const response = await checkConflicts(conflictForm);
      setConflictResult(response.data);
    } catch (err: any) {
      setConflictError(err.message || "Failed to check conflicts");
    } finally {
      setIsChecking(false);
    }
  };

  const getStatusClass = (status: string | undefined): string => {
    if (status === "Safe") return "gate-safe";
    if (status === "Needs_Human_Review") return "gate-review";
    if (status === "Non_Compliant") return "gate-danger";
    return "gate-review";
  };

  return (
    <div className="dashboard-wrapper">
      <div className="topbar">
        <div className="breadcrumb">
          <span>Overview</span>
          <span className="breadcrumb-sep">/</span>
          <span>Trust Gating</span>
        </div>
      </div>
      
      <div className="content page">
        {/* Intro */}
        <div style={{ marginBottom: '32px' }}>
          <h2 style={{ fontSize: '24px', fontWeight: 600, color: 'var(--fg)', marginBottom: '8px' }}>Independent Trust Gate</h2>
          <p style={{ color: 'var(--fg3)', fontSize: '14px', maxWidth: '700px', lineHeight: 1.6 }}>
            Run the verification engine manually on specific claims and cross-edition texts. 
            The Trust Gate logic prevents unverified or contradictory information from reaching consumers.
          </p>
        </div>

        {/* Tab Selection */}
        <div className="flex gap-4 mb-8" style={{ borderBottom: '1px solid var(--border)', paddingBottom: '12px' }}>
          <button 
            className={`verify-tab ${activeTab === 'claims' ? 'active' : ''}`}
            onClick={() => setActiveTab('claims')}
          >
            <Shield size={18} />
            Evaluate Claims
          </button>
          <button 
            className={`verify-tab ${activeTab === 'conflicts' ? 'active' : ''}`}
            onClick={() => setActiveTab('conflicts')}
          >
            <GitCompare size={18} />
            Cross-Edition Check
          </button>
        </div>

        {/* CLAIMS PANEL */}
        {activeTab === 'claims' && (
          <div className="grid-2 animate-fade-in">
            {/* Input Panel */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">Test Content</div>
                <div className="card-subtitle">Provide AI-generated response text to verify</div>
              </div>
              <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <textarea 
                  value={demoAnswer}
                  onChange={(e) => setDemoAnswer(e.target.value)}
                  className="w-full text-[13px] bg-slate-950/50 text-white border border-[#2d333b] rounded-md p-4 min-h-[140px] focus:outline-none focus:border-[#444c56]"
                  placeholder="Paste response here..."
                />
                
                <div className="flex justify-between items-center mb-1 mt-2">
                  <div className="text-[12px] font-semibold text-slate-400 uppercase">Extracted Claims to Verify</div>
                  <button 
                    className="text-[11px] text-blue-400 hover:text-blue-300 flex items-center gap-1"
                    onClick={() => {
                      setClaims([...claims, {
                        text: "", source_publication: "Manual", source_edition: "Current", 
                        source_section_id: "N/A", source_passage: "", confidence: 1.0
                      }]);
                    }}
                  >
                    + Add Claim
                  </button>
                </div>

                <div className="flex flex-col gap-3">
                  {claims.map((claim, idx) => (
                    <div key={idx} className="p-3 bg-[#0d1117] rounded-md border border-[#2d333b] flex flex-col gap-2 relative">
                      <button 
                        className="absolute top-2 right-2 text-slate-500 hover:text-red-400"
                        onClick={() => {
                          const newClaims = [...claims];
                          newClaims.splice(idx, 1);
                          setClaims(newClaims);
                        }}
                      >
                        ×
                      </button>
                      <div className="text-[10px] bg-blue-900/30 text-blue-400 px-2 py-0.5 rounded border border-blue-800/50 w-max uppercase font-bold">Claim {idx + 1}</div>
                      
                      <div className="flex flex-col gap-1">
                        <label className="text-[10px] text-slate-500 font-semibold">CLAIM TEXT</label>
                        <input 
                          value={claim.text}
                          onChange={e => {
                            const newClaims = [...claims];
                            newClaims[idx].text = e.target.value;
                            setClaims(newClaims);
                          }}
                          className="form-input text-[12px] py-1.5"
                          placeholder="e.g. Banks must maintain 9%..."
                        />
                      </div>

                      <div className="flex flex-col gap-1">
                        <label className="text-[10px] text-slate-500 font-semibold">SOURCE PASSAGE (For NLI Entailment)</label>
                        <textarea 
                          value={claim.source_passage}
                          onChange={e => {
                            const newClaims = [...claims];
                            newClaims[idx].source_passage = e.target.value;
                            setClaims(newClaims);
                          }}
                          className="form-input text-[12px] min-h-[60px] py-1.5"
                          placeholder="Paste the source text from publication here..."
                        />
                      </div>
                    </div>
                  ))}
                  {claims.length === 0 && (
                    <div className="text-[12px] text-slate-500 italic p-4 text-center border border-dashed border-[#2d333b] rounded">
                      No claims added. Trust Gate requires claims to verify.
                    </div>
                  )}
                </div>

                <button 
                  className="btn btn-primary mt-4 w-max"
                  onClick={runVerification}
                  disabled={isVerifying || claims.length === 0}
                >
                  {isVerifying ? 'Running NLI Engine...' : 'Run Trust Gate Validation'}
                </button>
              </div>
            </div>

            {/* Results Panel */}
            <div className="card" style={{ minHeight: '300px' }}>
              <div className="card-header">
                <div className="card-title">Verification Dashboard</div>
              </div>
              <div className="card-body">
                {verifyError && (
                  <Alert variant="destructive">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{verifyError}</AlertDescription>
                  </Alert>
                )}
                
                {!isVerifying && !trustGateResult && !verifyError && (
                  <div className="flex flex-col items-center justify-center pt-12 text-slate-500">
                    <Scale size={32} className="mb-4 opacity-40" />
                    <p className="text-[13px]">Run validation to see trust score and verdicts</p>
                  </div>
                )}

                {isVerifying && (
                  <div className="flex justify-center py-20">
                    <div className="pulse" style={{ width: '24px', height: '24px' }}></div>
                  </div>
                )}

                {trustGateResult && verificationsResult && (
                  <div className="animate-fade-in flex flex-col gap-6">
                    {/* Gate Decision Banner */}
                    <div className="p-6 rounded-lg border flex flex-col gap-3" 
                      style={{ 
                        background: trustGateResult.status === 'Safe' ? 'rgba(35, 134, 54, 0.05)' : 'rgba(218, 54, 51, 0.05)',
                        borderColor: trustGateResult.status === 'Safe' ? 'rgba(35, 134, 54, 0.2)' : 'rgba(218, 54, 51, 0.2)'
                      }}
                    >
                      <div className="flex justify-between items-center">
                        <span className="text-[12px] uppercase font-bold text-slate-400">Trust Gate Status</span>
                        <span className={`gate-badge ${getStatusClass(trustGateResult.status)} text-[14px] px-3 py-1 font-semibold`}>
                          {trustGateResult.status.replace('_', ' ')}
                        </span>
                      </div>
                      
                      <div>
                        <div className="text-[11px] text-slate-500 mb-1 uppercase tracking-wider">Engine Reasoning</div>
                        <div className="text-[13px] text-slate-300 leading-relaxed font-medium">
                          {trustGateResult.reasoning}
                        </div>
                      </div>
                    </div>

                    {/* Verifications List */}
                    <div>
                      <h3 className="text-[14px] font-semibold text-slate-200 mb-3 border-b border-[#2d333b] pb-2">Individual Claim Verdicts</h3>
                      <div className="flex flex-col gap-3">
                        {verificationsResult.map((v, i) => (
                          <div key={i} className="p-4 bg-[#0d1117] rounded border border-[#2d333b]">
                            <div className="flex justify-between items-start mb-2">
                              <span className="text-[13px] text-slate-200 font-medium">{v.claim_text}</span>
                              <span className={`text-[10px] px-2 py-0.5 rounded font-bold uppercase ${v.verdict === 'SUPPORTED' || v.verdict === 'ENTAILMENT' ? 'text-[#3fb950] bg-[#3fb950]/10' : 'text-[#f85149] bg-[#f85149]/10'}`}>
                                {v.verdict}
                              </span>
                            </div>
                            <div className="text-[12px] text-slate-400 italic">
                              "{v.explanation || 'Matches established source semantics.'}"
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* CONFLICTS PANEL */}
        {activeTab === 'conflicts' && (
          <div className="grid-2 animate-fade-in">
             <div className="card">
              <div className="card-header">
                <div className="card-title">Edition Inputs</div>
                <div className="card-subtitle">Check logic displacement across publication versions</div>
              </div>
              <div className="card-body flex flex-col gap-4">
                
                <div className="flex gap-4">
                  <div className="flex flex-col gap-1 flex-1">
                    <label className="text-[11px] font-semibold text-slate-400 uppercase">Publication</label>
                    <input type="text" value={conflictForm.publication} onChange={e => setConflictForm({...conflictForm, publication: e.target.value})} className="form-input" />
                  </div>
                  <div className="flex flex-col gap-1 flex-1">
                    <label className="text-[11px] font-semibold text-slate-400 uppercase">Topic</label>
                    <input type="text" value={conflictForm.topic} onChange={e => setConflictForm({...conflictForm, topic: e.target.value})} className="form-input" />
                  </div>
                </div>

                <div className="p-4 bg-slate-900/40 rounded border border-[#2d333b] flex flex-col gap-3">
                  <div className="text-[12px] font-bold text-slate-300">Older Edition</div>
                  <input type="text" value={conflictForm.older_date} onChange={e => setConflictForm({...conflictForm, older_date: e.target.value})} className="form-input w-1/2" placeholder="e.g. Dec 2023" />
                  <textarea value={conflictForm.older_text} onChange={e => setConflictForm({...conflictForm, older_text: e.target.value})} className="form-input min-h-[80px]" placeholder="Older passage text..." />
                </div>

                <div className="p-4 bg-slate-900/40 rounded border border-[#2d333b] flex flex-col gap-3">
                  <div className="text-[12px] font-bold text-slate-300">Newer Edition</div>
                  <input type="text" value={conflictForm.newer_date} onChange={e => setConflictForm({...conflictForm, newer_date: e.target.value})} className="form-input w-1/2" placeholder="e.g. June 2024" />
                  <textarea value={conflictForm.newer_text} onChange={e => setConflictForm({...conflictForm, newer_text: e.target.value})} className="form-input min-h-[80px]" placeholder="Newer passage text..." />
                </div>

                <button className="btn btn-primary" onClick={runConflictCheck} disabled={isChecking}>
                  {isChecking ? 'Running Diff Checks...' : 'Detect Contradictions'}
                </button>
              </div>
             </div>

             <div className="card">
               <div className="card-header">
                 <div className="card-title">Conflict Analysis</div>
               </div>
               <div className="card-body">
                 {!isChecking && !conflictResult && !conflictError && (
                    <div className="flex flex-col items-center justify-center pt-12 text-slate-500">
                      <GitCompare size={32} className="mb-4 opacity-40" />
                      <p className="text-[13px]">Run detection to verify timeline integrity</p>
                    </div>
                  )}

                  {isChecking && (
                    <div className="flex justify-center py-20">
                      <div className="pulse" style={{ width: '24px', height: '24px' }}></div>
                    </div>
                  )}

                  {conflictError && (
                    <Alert variant="destructive">
                      <AlertTitle>Error</AlertTitle>
                      <AlertDescription>{conflictError}</AlertDescription>
                    </Alert>
                  )}

                  {conflictResult && (
                    <div className="animate-fade-in">
                      {conflictResult.has_conflict ? (
                        <div className="p-6 rounded-lg border bg-amber-950/20 border-amber-900/50 mb-6 flex items-start gap-4">
                           <AlertTriangle className="text-amber-500 mt-1" size={24} />
                           <div>
                             <h4 className="text-amber-500 font-semibold mb-2">Superseding Conflict Detected</h4>
                             <p className="text-[13px] text-amber-200/80 leading-relaxed mb-3">
                               {conflictResult.conflict_description}
                             </p>
                             <div className="text-[12px] text-amber-500/70">
                               System will favor <span className="font-bold">{conflictResult.superseding_edition}</span> in future RAG queries via timeline-aware sorting.
                             </div>
                           </div>
                        </div>
                      ) : (
                        <div className="p-6 rounded-lg border bg-green-950/20 border-green-900/50 mb-6 flex items-start gap-4">
                           <CheckCircle2 className="text-emerald-500 mt-1" size={24} />
                           <div>
                             <h4 className="text-emerald-500 font-semibold mb-1">Consistency Maintained</h4>
                             <p className="text-[13px] text-emerald-200/70">No logical displacement detected between the two editions.</p>
                           </div>
                        </div>
                      )}
                      
                      <div className="px-4 py-3 bg-[#0d1117] rounded border border-[#2d333b] text-[12px] font-mono text-slate-400">
                         {JSON.stringify(conflictResult, null, 2)}
                      </div>
                    </div>
                  )}
               </div>
             </div>
          </div>
        )}

      </div>
    </div>
  );
}
