import React, { useState, useEffect } from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer } from "recharts";

import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircle, AlertTriangle, CheckCircle2, ChevronDown, ExternalLink, Search, Sparkles, Shield, FileText, Zap } from "lucide-react";
import '../DashboardPage/Dashboard.css';

const examplePrompts = [
  "What are the capital adequacy requirements for NBFCs?",
  "Explain the regulatory framework for digital lending",
  "What are the KYC norms for customer onboarding?",
];

const mockResponse = {
  answer: "According to the Financial Stability Report (December 2025), capital adequacy requirements for Non-Banking Financial Companies (NBFCs) mandate a minimum Capital to Risk-Weighted Assets Ratio (CRAR) of 15% for systemically important NBFCs. The Tier I capital must be at least 10% of the risk-weighted assets. These requirements ensure NBFCs maintain adequate capital buffers to absorb potential losses and maintain financial stability.",
  claims: [
    {
      id: 1,
      text: "Minimum CRAR of 15% for systemically important NBFCs",
      source: "FSR·December 2025·Section 4.2.3",
      nli: "Entailment",
      confidence: 0.98
    },
    {
      id: 2,
      text: "Tier I capital must be at least 10% of risk-weighted assets",
      source: "FSR·December 2025·Section 4.2.4",
      nli: "Entailment",
      confidence: 0.95
    },
  ],
  trustGate: {
    status: "safe",
    score: 96,
    reasoning: "All claims are strongly supported by official RBI publications with high confidence scores. No conflicts detected across editions."
  },
  ragas: [
    { metric: "Faithfulness", score: 0.96 },
    { metric: "Answer Relevancy", score: 0.94 },
    { metric: "Context Precision", score: 0.92 },
    { metric: "Context Recall", score: 0.89 },
    { metric: "Answer Correctness", score: 0.93 },
  ]
};

/* ── Pipeline Steps ── */
const PIPELINE_STEPS = [
  { label: "Searching Knowledge Base", icon: Search, duration: 600 },
  { label: "Retrieving Context Chunks", icon: FileText, duration: 500 },
  { label: "Generating RAG Response", icon: Sparkles, duration: 700 },
  { label: "Verifying Claims (NLI)", icon: Shield, duration: 500 },
  { label: "Trust Gate Decision", icon: Zap, duration: 400 },
];

export default function Query() {
  const [time, setTime] = useState('00:00:00');
  const [queryText, setQueryText] = useState('');
  const [hasQueried, setHasQueried] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pipelineStep, setPipelineStep] = useState(-1);
  const [expandedClaim, setExpandedClaim] = useState<number | null>(null);
  const [showConflict, setShowConflict] = useState(false);

  useEffect(() => {
    const tick = () => {
      const n = new Date();
      const pad = (x: number) => String(x).padStart(2, '0');
      setTime(`${pad(n.getHours())}:${pad(n.getMinutes())}:${pad(n.getSeconds())}`);
    };
    const timer = setInterval(tick, 1000);
    tick();
    return () => clearInterval(timer);
  }, []);

  const runQuery = () => {
    if (!queryText.trim()) return;
    setIsLoading(true);
    setHasQueried(false);
    setPipelineStep(0);
    setExpandedClaim(null);
    setError(null);
    setShowConflict(false);

    // Animate through pipeline steps sequentially
    let step = 0;
    const advancePipeline = () => {
      if (step < PIPELINE_STEPS.length - 1) {
        step++;
        setPipelineStep(step);
        setTimeout(advancePipeline, PIPELINE_STEPS[step].duration);
      } else {
        // All steps complete
        setTimeout(() => {
          // Simulation: If query is about specific term, show conflict
          if (queryText.toLowerCase().includes('nbofc') || queryText.toLowerCase().includes('nbfc')) {
            setShowConflict(true);
          }
          setIsLoading(false);
          setHasQueried(true);
        }, 600);
      }
    };
    
    // Simulate API fail
    if (queryText === "FAIL") {
      setTimeout(() => {
        setError("RBI Connection Timeout: The regulatory knowledge base is currently undergoing synchronization. Please try again in 5 minutes.");
        setIsLoading(false);
      }, 1000);
      return;
    }

    setTimeout(advancePipeline, PIPELINE_STEPS[0].duration);
  };

  return (
    <div className="dashboard-wrapper">
      <div className="topbar">
        <div className="breadcrumb">
          <span>Overview</span>
          <span className="breadcrumb-sep">/</span>
          <span>RAG Query & Verify</span>
        </div>
        <div className="topbar-actions">
          <div className="time-chip" id="clock">{time}</div>
        </div>
      </div>

      <div className="content">
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Response Generation Failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="page" id="page-query">
          
          {/* ── Query Input Card ── */}
          <div className="card" style={{ marginBottom: '20px' }}>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <input 
                  type="text" 
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && runQuery()}
                  placeholder="Ask about any RBI publication... e.g. 'What are the capital adequacy requirements for NBFCs?'" 
                  style={{ flex: 1, fontFamily: 'var(--font)', fontSize: '13px', padding: '10px 14px', border: '1px solid var(--border)', borderRadius: 'var(--radius)', background: 'var(--surface)', color: 'var(--fg)', outline: 'none' }} 
                />
                <select style={{ fontFamily: 'var(--font)', fontSize: '12px', padding: '10px 12px', border: '1px solid var(--border)', borderRadius: 'var(--radius)', background: 'var(--surface)', color: 'var(--fg)', cursor: 'pointer' }}>
                  <option>All Publications</option>
                  <option>FSR</option>
                  <option>MPR</option>
                  <option>PSR</option>
                  <option>FER</option>
                </select>
                <button className="btn btn-primary" onClick={runQuery} disabled={isLoading || !queryText.trim()}>
                  {isLoading ? 'Analyzing...' : 'Run Query ↗'}
                </button>
              </div>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
                <span style={{ fontSize: '11px', color: 'var(--fg3)', marginRight: '4px' }}>Try these:</span>
                {examplePrompts.map((prompt, idx) => (
                  <button 
                    key={idx} 
                    className="btn" 
                    style={{ padding: '4px 10px', fontSize: '11px', borderRadius: '12px' }}
                    onClick={() => setQueryText(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* ── Pipeline Animation (shows during loading) ── */}
          {isLoading && (
            <div className="card" style={{ marginBottom: '20px' }}>
              <div className="card-header">
                <div className="card-title">Verification Pipeline</div>
                <div className="card-subtitle">Processing your query through the XAI pipeline...</div>
              </div>
              <div className="pipeline" style={{ padding: '16px 20px 20px' }}>
                {PIPELINE_STEPS.map((step, idx) => {
                  const StepIcon = step.icon;
                  const isActive = idx === pipelineStep;
                  const isDone = idx < pipelineStep;
                  const isPending = idx > pipelineStep;
                  return (
                    <div className="pipe-step" key={idx}>
                      <div 
                        className={`pipe-node ${isDone ? 'done' : isActive ? 'active' : ''}`}
                        style={isActive ? { 
                          borderColor: 'var(--accent2)', 
                          boxShadow: '0 0 12px rgba(26,108,200,0.4)',
                          animation: 'pulse-glow 1.2s ease-in-out infinite'
                        } : {}}
                      >
                        {isDone ? (
                          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="var(--safe)" strokeWidth="2">
                            <path d="M4 10l4 4 8-8"/>
                          </svg>
                        ) : isActive ? (
                          <StepIcon size={16} style={{ color: 'var(--accent2)' }} />
                        ) : (
                          <span style={{ color: 'var(--fg3)', fontSize: '10px' }}>{idx + 1}</span>
                        )}
                      </div>
                      <div className="pipe-label" style={{ 
                        color: isDone ? 'var(--safe)' : isActive ? 'var(--fg)' : 'var(--fg3)',
                        fontWeight: isActive ? 600 : 400
                      }}>
                        {step.label}
                        {isActive && <span className="loading-dots" style={{ marginLeft: '4px' }}>...</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── Empty State (before any query) ── */}
          {!hasQueried && !isLoading && (
            <div className="card" style={{ marginBottom: '20px' }}>
              <div className="card-body" style={{ 
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px',
                padding: '48px 32px', textAlign: 'center'
              }}>
                <div style={{ 
                  width: '64px', height: '64px', borderRadius: '50%',
                  background: 'linear-gradient(135deg, rgba(26,200,122,0.1), rgba(26,108,200,0.1))',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  border: '1px solid var(--border)'
                }}>
                  <Search size={28} style={{ color: 'var(--accent2)' }} />
                </div>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--fg)', marginBottom: '8px' }}>
                    Ask the Regulatory Knowledge Base
                  </h3>
                  <p style={{ fontSize: '13px', color: 'var(--fg3)', maxWidth: '500px', lineHeight: 1.6 }}>
                    Type your question above. We'll search across RBI publications (FSR, MPR, PSR, FER), 
                    generate a cited answer, verify every claim with NLI, and give you a Trust Gate score.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '24px', marginTop: '8px' }}>
                  {[
                    { icon: FileText, step: "1", text: "Search & Retrieve" },
                    { icon: Sparkles, step: "2", text: "Generate Answer" },
                    { icon: Shield, step: "3", text: "Verify & Score" },
                  ].map((item) => {
                    const Icon = item.icon;
                    return (
                      <div key={item.step} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                        <div style={{ 
                          width: '40px', height: '40px', borderRadius: '10px',
                          background: 'var(--surface2)', border: '1px solid var(--border)',
                          display: 'flex', alignItems: 'center', justifyContent: 'center'
                        }}>
                          <Icon size={18} style={{ color: 'var(--fg2)' }} />
                        </div>
                        <span style={{ fontSize: '11px', color: 'var(--fg3)', fontWeight: 500 }}>
                          {item.text}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
          
          {/* ── Conflict Alert ── */}
          {hasQueried && showConflict && (
            <Alert variant="default" className="mb-6 bg-amber-50 border-amber-200 text-amber-900 shadow-sm animate-fade-in">
              <AlertTriangle className="h-4 w-4 text-amber-600" />
              <AlertTitle className="font-bold">Regulatory Edition Conflict Detected</AlertTitle>
              <AlertDescription className="text-[13px] leading-relaxed">
                Found conflicting guidance between <strong>FSR December 2024</strong> and <strong>FSR December 2025</strong> regarding NBFC capital requirements. 
                The RAG engine prioritizes the latest edition, but human verification is recommended for legacy project compliance.
              </AlertDescription>
            </Alert>
          )}

          {/* ── Results Grid ── */}
          {hasQueried && (
            <div className="grid-3">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                {/* Response Card */}
                <div className="card" id="query-result-card">
                  <div className="card-header">
                    <div>
                      <div className="card-title">RAG Response</div>
                      <div className="card-subtitle">Results for your query</div>
                    </div>
                    <span className="gate-badge gate-safe">
                      <div className="gate-dot dot-safe"></div>Safe · {mockResponse.trustGate.score}
                    </span>
                  </div>
                  <div className="card-body" style={{ fontSize: '13px', color: 'var(--fg2)', lineHeight: 1.8 }}>
                    <div>
                      According to the Financial Stability Report (December 2025), capital adequacy requirements for Non-Banking Financial Companies (NBFCs) mandate a minimum Capital to Risk-Weighted Assets Ratio (CRAR) of 15% for systemically important NBFCs 
                      <span className="q-pub" style={{ margin: '0 6px', color: 'var(--safe)', borderColor: 'var(--safe)' }}>FSR·Dec 2025·§4.2.3</span>. 
                      The Tier I capital must be at least 10% of the risk-weighted assets 
                      <span className="q-pub" style={{ margin: '0 6px', color: 'var(--safe)', borderColor: 'var(--safe)' }}>FSR·Dec 2025·§4.2.4</span>. 
                      These requirements ensure NBFCs maintain adequate capital buffers to absorb potential losses and maintain financial stability.
                    </div>
                  </div>
                </div>

                {/* Claims Table */}
                <div className="card">
                  <div className="card-header">
                    <div>
                      <div className="card-title">Claim Attribution</div>
                      <div className="card-subtitle">Source verification & confidence mapping</div>
                    </div>
                  </div>
                  <div className="card-body" style={{ padding: 0 }}>
                    <table className="query-table">
                      <thead>
                        <tr>
                          <th>Claim</th>
                          <th>Source</th>
                          <th>Verification</th>
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {mockResponse.claims.map((claim) => (
                          <React.Fragment key={claim.id}>
                            <tr onClick={() => setExpandedClaim(expandedClaim === claim.id ? null : claim.id)} style={{ cursor: 'pointer' }}>
                              <td><div style={{ color: 'var(--fg)', lineHeight: 1.4, paddingRight: '12px' }}>{claim.text}</div></td>
                              <td><div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--accent2)', fontFamily: 'var(--font)', cursor: 'pointer' }}>{claim.source} <ExternalLink size={10} /></div></td>
                              <td>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                  <span className="gate-badge gate-safe" style={{ fontSize: '10px', padding: '2px 8px' }}>{claim.nli}</span>
                                  <ChevronDown size={14} style={{ color: 'var(--fg3)', transform: expandedClaim === claim.id ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
                                </div>
                              </td>
                              <td><span style={{ color: 'var(--safe)', fontWeight: 600 }}>{(claim.confidence * 100).toFixed(0)}%</span></td>
                            </tr>
                            {expandedClaim === claim.id && (
                              <tr style={{ background: 'var(--surface2)', cursor: 'default' }}>
                                <td colSpan={4} style={{ padding: '16px', borderBottom: '1px solid var(--border)' }}>
                                  <div style={{ fontSize: '12px', color: 'var(--fg3)', display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                                    <CheckCircle2 size={14} style={{ color: 'var(--safe)', marginTop: '2px', flexShrink: 0 }} />
                                    <span>NLI analysis confirms this claim is directly entailed by the source document with no contradictions detected. The source chunk perfectly matches the semantic intent.</span>
                                  </div>
                                </td>
                              </tr>
                            )}
                          </React.Fragment>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* Right Column */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                {/* RAGAS Scorecard */}
                <div className="card">
                  <div className="card-header">
                    <div className="card-title">RAGAS Live Scorecard</div>
                  </div>
                  <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ width: '100%', height: '240px', marginTop: '-20px', marginLeft: '-24px' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={mockResponse.ragas}>
                          <PolarGrid stroke="var(--border)" />
                          <PolarAngleAxis dataKey="metric" tick={{ fill: 'var(--fg3)', fontSize: 10, fontFamily: 'var(--font)' }} />
                          <Radar name="Score" dataKey="score" stroke="var(--accent2)" fill="var(--accent2)" fillOpacity={0.2} />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                    
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      {mockResponse.ragas.map((metric) => (
                        <div className="sparkline-row" key={metric.metric}>
                          <div className="spark-label">{metric.metric}</div>
                          <div className="spark-track"><div className={`spark-fill ${metric.score >= 0.9 ? 's' : 'w'}`} style={{ width: `${metric.score * 100}%` }}></div></div>
                          <div className="spark-val" style={{ color: metric.score >= 0.9 ? 'var(--safe)' : 'var(--warn)' }}>{metric.score.toFixed(2)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Trust Gate Card */}
                <div className="card" style={{ background: 'var(--surface2)', borderColor: 'var(--safe)' }}>
                  <div className="card-body" style={{ padding: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                      <CheckCircle2 size={24} style={{ color: 'var(--safe)', flexShrink: 0 }} />
                      <div>
                        <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--fg)', marginBottom: '4px' }}>
                          Trust Gate: Safe (Score: {mockResponse.trustGate.score})
                        </div>
                        <div style={{ fontSize: '12px', color: 'var(--fg3)', lineHeight: 1.5 }}>
                          {mockResponse.trustGate.reasoning}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
