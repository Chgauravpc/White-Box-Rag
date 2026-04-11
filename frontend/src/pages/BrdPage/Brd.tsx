import React, { useEffect, useState } from 'react';
import '../DashboardPage/Dashboard.css';

import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircle, FileCheck, History, RefreshCcw } from 'lucide-react';

import BrdUpload from './components/BrdUpload';
import ComplianceScore from './components/ComplianceScore';
import RequirementsTable, { type Requirement } from './components/RequirementsTable';
import RemediationPanel from './components/RemediationPanel';
import RAGQuery from './components/RAGQuery';

import { uploadBRD, validateBRD } from '@/lib/api';

export default function Brd() {
  // ── History Tracking ──
  interface BrdHistoryItem {
    id: string;
    filename: string;
    timestamp: string;
    requirements: Requirement[];
    score: number;
  }
  const [history, setHistory] = useState<BrdHistoryItem[]>(() => {
    const saved = localStorage.getItem('brdHistory');
    return saved ? JSON.parse(saved) : [];
  });

  useEffect(() => {
    localStorage.setItem('brdHistory', JSON.stringify(history));
  }, [history]);
  const [currentFilename, setCurrentFilename] = useState("");

  const [time, setTime] = useState('00:00:00');
  const [uploaded, setUploaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requirements, setRequirements] = useState<Requirement[]>([]);
  const [overallScore, setOverallScore] = useState(0);

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

  const handleBrdUpload = async (file: File) => {
    setIsLoading(true);
    setUploaded(false);
    setError(null);

    try {
      // 1. Upload & Parse
      const uploadRes = await uploadBRD(file);
      const parsedData = uploadRes.data.data; // List of {id, text}

      const reqTexts = parsedData.map((r: any) => r.text);

      // 2. Validate against Knowledge Base
      const validateRes = await validateBRD(reqTexts);
      const validationResults = validateRes.data.data;

      // 3. Map to UI Model
      const mappedReqs = validationResults.map((v: any, idx: number) => {
        const rLevel = (v.risk_level || "").toLowerCase();
        const status = (rLevel.includes('low') || rLevel.includes('safe')) ? 'compliant' 
                     : (rLevel.includes('medium') || rLevel.includes('review')) ? 'review' 
                     : 'violation';

        return {
          id: idx + 1,
          text: v.requirement,
          status: status,
          alignmentScore: v.alignment_score || v.overall_compliance_score || 0,
          mappedSections: [], // Backend doesn't return precise citations for BRD yet
          gaps: v.gaps || [],
          violations: v.violations || []
        };
      });

      setRequirements(mappedReqs);

      const totalScore = mappedReqs.reduce((acc: number, r: any) => acc + r.alignmentScore, 0);
      const calculatedScore = Math.round(totalScore / mappedReqs.length);
      setOverallScore(calculatedScore);

      setCurrentFilename(file.name);
      setHistory(prev => [{
        id: Date.now().toString(),
        filename: file.name,
        timestamp: new Date().toLocaleTimeString(),
        requirements: mappedReqs,
        score: calculatedScore
      }, ...prev]);

      setUploaded(true);
    } catch (err: any) {
      console.error("BRD Validation Error:", err);
      setError(err?.response?.data?.message || "The platform failed to parse or validate the BRD. Ensure the PDF is not encrypted.");
    } finally {
      setIsLoading(false);
    }
  };

  const compliantCount = requirements.filter(r => r.status === "compliant").length;
  const reviewCount = requirements.filter(r => r.status === "review").length;
  const violationCount = requirements.filter(r => r.status === "violation").length;

  return (
    <div className="dashboard-wrapper">
      <div className="topbar">
        <div className="breadcrumb">
          <span>Overview</span>
          <span className="breadcrumb-sep">/</span>
          <span>BRD Validator</span>
          {uploaded && currentFilename && (
            <>
              <span className="breadcrumb-sep">/</span>
              <span style={{ color: 'var(--accent2)' }}>{currentFilename}</span>
            </>
          )}
        </div>
        <div className="topbar-actions">
          <div className="time-chip" id="clock">{time}</div>
          {uploaded && <button className="btn" onClick={() => window.print()}>Export Evidence</button>}
          {uploaded && <button className="btn btn-primary" onClick={() => { setUploaded(false); setIsLoading(false); }}><RefreshCcw size={14} style={{ marginRight: '6px' }} /> New Scan</button>}
        </div>
      </div>

      <div className="content">
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Validation Pipeline Interrupted</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        <div className="page" id="page-brd">

          {isLoading ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <div style={{ padding: '48px', textAlign: 'center', background: 'var(--surface2)', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border)' }}>
                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '16px' }}>
                  <div className="loading-spinner" style={{ width: '40px', height: '40px', border: '3px solid var(--border)', borderTopColor: 'var(--accent2)', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
                </div>
                <h3 style={{ fontSize: '18px', fontWeight: 600 }}>Deep Mapping Requirements...</h3>
                <p style={{ color: 'var(--fg3)', fontSize: '14px', marginTop: '8px' }}>Parsing sections, extracting claims, and cross-referencing RBI vector store.</p>
              </div>
              <div className="grid-2">
                <Skeleton className="h-[200px] w-full" />
                <Skeleton className="h-[200px] w-full" />
              </div>
            </div>
          ) : !uploaded ? (
            <div className="grid-2">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <BrdUpload onUploadFile={handleBrdUpload} />
              </div>

              <div className="card">
                <div className="card-header">
                  <div>
                    <div className="card-title">Validation History</div>
                    <div className="card-subtitle">Previous compliance snapshots</div>
                  </div>
                </div>
                <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '0', padding: 0 }}>
                  {history.length === 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'center', justifyContent: 'center', padding: '60px 20px', color: 'var(--fg3)' }}>
                      <History size={32} style={{ opacity: 0.2, marginBottom: '12px' }} />
                      <p style={{ fontSize: '13px' }}>No previous validations synced in this session.</p>
                    </div>
                  ) : (
                    <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                      {history.map((item) => (
                        <div 
                          key={item.id} 
                          className="audit-entry" 
                          style={{ cursor: 'pointer', padding: '16px 20px', borderBottom: '1px solid var(--surface2)' }}
                          onClick={() => {
                            setRequirements(item.requirements);
                            setOverallScore(item.score);
                            setCurrentFilename(item.filename);
                            setUploaded(true);
                          }}
                        >
                          <div style={{ position: 'relative', width: '14px' }}>
                            <div className="audit-dot safe"></div>
                          </div>
                          <div style={{ flex: 1 }}>
                            <div className="audit-action">{item.filename}</div>
                            <div className="audit-meta">
                              Score: {item.score}% · {item.requirements.length} Requirements Verified
                            </div>
                          </div>
                          <div className="audit-time">{item.timestamp}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

              <div className="stats-row">
                <div className="stat-card c-blue text-left">
                  <div className="stat-label">Total Requirements</div>
                  <div className="stat-value">{requirements.length}</div>
                </div>
                <div className="stat-card c-safe text-left">
                  <div className="stat-label">Full Compliance</div>
                  <div className="stat-value text-safe">{compliantCount}</div>
                </div>
                <div className="stat-card c-warn text-left">
                  <div className="stat-label">Risk Review</div>
                  <div className="stat-value text-review">{reviewCount}</div>
                </div>
                <div className="stat-card c-danger text-left">
                  <div className="stat-label">Direct Violations</div>
                  <div className="stat-value text-danger">{violationCount}</div>
                </div>
              </div>

              <div className="grid-3" style={{ gridTemplateColumns: '1fr 1.8fr' }}>
                <ComplianceScore scoreTarget={overallScore} />
                <RequirementsTable requirements={requirements} />
              </div>

              <RemediationPanel requirements={requirements} />
              <RAGQuery />

            </div>
          )}

        </div>
      </div>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
