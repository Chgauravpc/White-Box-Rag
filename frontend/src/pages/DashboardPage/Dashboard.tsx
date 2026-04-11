import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Dashboard.css';

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircle, FileText, Activity, ShieldCheck, AlertTriangle } from 'lucide-react';

import { listAuditLogs, listDocuments } from '@/lib/api';
import type { AuditLogItem, DocumentInfo } from '@/lib/types';

export default function Dashboard() {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [logs, setLogs] = useState<AuditLogItem[]>([]);

  const donutRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [docsRes, logsRes] = await Promise.all([
          listDocuments(),
          listAuditLogs()
        ]);
        setDocs(docsRes.data);
        setLogs(logsRes.data);
      } catch (err) {
        setError("Synchronization Error: Failed to fetch live governance metrics.");
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();

    // Timer removed
  }, []);

  useEffect(() => {
    // Draw Donut for Trust Distribution
    if (donutRef.current && !isLoading && logs.length > 0) {
      const c = donutRef.current;
      const ctx = c.getContext('2d');
      if (ctx) {
        const cx = 80, cy = 80, r = 60, inner = 38;
        const total = logs.length;
        const counts = logs.reduce((acc, l) => {
          const risk = l.risk_level?.toLowerCase() || '';
          if (risk.includes('safe')) acc.safe++;
          else if (risk.includes('review')) acc.review++;
          else acc.danger++;
          return acc;
        }, { safe: 0, review: 0, danger: 0 });

        const data = [
          { v: counts.safe / total, color: '#1ac87a' }, 
          { v: counts.review / total, color: '#c8921a' }, 
          { v: counts.danger / total, color: '#c8371a' }
        ];

        let start = -Math.PI / 2;
        ctx.clearRect(0, 0, 160, 160);
        data.forEach(d => {
          if (d.v === 0) return;
          const end = start + d.v * 2 * Math.PI;
          ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, r, start, end); ctx.closePath();
          ctx.fillStyle = d.color; ctx.fill();
          start = end;
        });

        const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        ctx.beginPath(); ctx.arc(cx, cy, inner, 0, 2 * Math.PI);
        ctx.fillStyle = isDark ? '#131820' : '#ffffff'; ctx.fill();
        ctx.fillStyle = isDark ? '#e8e4da' : '#1a1a2e';
        ctx.font = '700 20px "Space Mono", monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(total.toString(), cx, cy - 6);
        ctx.font = '400 9px "Space Grotesk", sans-serif'; ctx.fillStyle = isDark ? '#7a7468' : '#6b7280';
        ctx.fillText('total logs', cx, cy + 10);
      }
    }
  }, [isLoading, logs]);

  const getStats = () => {
    const totalChunks = docs.reduce((acc, d) => acc + d.chunk_count, 0);
    const safeCount = logs.filter(l => l.risk_level?.toLowerCase().includes('safe')).length;
    const reviewCount = logs.filter(l => l.risk_level?.toLowerCase().includes('review')).length;
    const compliancePct = logs.length > 0 ? (safeCount / logs.length) * 100 : 0;

    return {
      docs: docs.length,
      chunks: totalChunks,
      queries: logs.length,
      review: reviewCount,
      compliance: Math.round(compliancePct)
    };
  };

  const stats = getStats();

  return (
    <div className="dashboard-wrapper">
      <div className="topbar">
        <div className="breadcrumb">
          <span>Overview</span>
          <span className="breadcrumb-sep">/</span>
          <span>Dashboard</span>
        </div>
        <div className="topbar-actions">
          <button className="btn btn-primary" onClick={() => navigate('/query')}>+ New Analysis</button>
        </div>
      </div>

      <div className="content">
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>System Partial Outage</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="stats-row">
          {[
            { label: 'Publications Ingested', val: stats.docs, delta: `${stats.chunks} total chunks`, up: 'Active Database', color: 'c-blue', path: '/ingest' },
            { label: 'Queries Logged', val: stats.queries, delta: 'Total trace history', up: 'Immutable Logs', color: 'c-safe', path: '/audit' },
            { label: 'Risk Flags', val: stats.review, delta: 'Awaiting human eye', up: 'Human-in-the-loop', color: 'c-warn', path: '/audit' },
            { label: 'Safety Threshold', val: stats.compliance, suffix: '%', delta: 'NLI Entailment Rate', up: 'Verified Logic', color: 'c-safe', path: '/audit' }
          ].map((s, i) => (
            <div key={i} className={`stat-card ${s.color}`} onClick={() => navigate(s.path)} style={{ cursor: 'pointer' }}>
              <div className="stat-label">{s.label}</div>
              {isLoading ? (
                <Skeleton className="h-9 w-24 mt-2" />
              ) : (
                <div className="stat-value">{s.val}{s.suffix}</div>
              )}
              <div className="stat-delta">{s.delta} <span className="up">{s.up}</span></div>
            </div>
          ))}
        </div>

        <div className="grid-3">
          {/* Recent Audits */}
          <div className="card">
            <div className="card-header">
              <div><div className="card-title">Recent Audits</div><div className="card-subtitle">Real-time system pulse</div></div>
              <button className="card-action" onClick={() => navigate('/audit')}>Full Registry →</button>
            </div>
            <div className="card-body" style={{ padding: 0 }}>
              <table className="query-table">
                <thead><tr>
                  <th>Query excerpt</th>
                  <th>Trust Status</th>
                  <th>Audit ID</th>
                </tr></thead>
                <tbody>
                  {isLoading ? (
                    [...Array(5)].map((_, i) => (
                      <tr key={i}>
                        <td colSpan={3} className="p-4"><Skeleton className="h-4 w-full" /></td>
                      </tr>
                    ))
                  ) : logs.length === 0 ? (
                    <tr><td colSpan={3} style={{ textAlign: 'center', padding: '32px', color: 'var(--fg3)' }}>No queries processed yet.</td></tr>
                  ) : (
                    logs.slice(0, 5).map(log => (
                      <tr key={log.id} onClick={() => navigate(`/audit/${log.id}`)} style={{ cursor: 'pointer' }}>
                        <td><div className="q-text" style={{ maxWidth: '200px' }}>{log.query}</div></td>
                        <td>
                          <span className={`gate-badge gate-${(log.risk_level || '').toLowerCase().includes('safe') ? 'safe' : (log.risk_level || '').toLowerCase().includes('review') ? 'review' : 'danger'}`}>
                            <div className={`gate-dot dot-${(log.risk_level || '').toLowerCase().includes('safe') ? 'safe' : (log.risk_level || '').toLowerCase().includes('review') ? 'warn' : 'danger'}`}></div>
                            {log.risk_level?.split('_')[0] || 'Unknown'}
                          </span>
                        </td>
                        <td style={{ fontFamily: 'var(--mono)', fontSize: '10px', color: 'var(--accent2)' }}>IDX-{log.id}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Trust Analytics */}
          <div className="card">
            <div className="card-header">
              <div><div className="card-title">Global Trust Distribution</div><div className="card-subtitle">Aggregated from all logs</div></div>
            </div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
              <canvas ref={donutRef} width="160" height="160"></canvas>
              <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--safe)' }}></div> Safe
                  </div>
                  <div style={{ fontWeight: 600 }}>{logs.filter(l => (l.risk_level || '').toLowerCase().includes('safe')).length}</div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--warn)' }}></div> Needs Review
                  </div>
                  <div style={{ fontWeight: 600 }}>{logs.filter(l => (l.risk_level || '').toLowerCase().includes('review')).length}</div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--danger)' }}></div> Non-Compliant
                  </div>
                  <div style={{ fontWeight: 600 }}>{logs.filter(l => (l.risk_level || '').toLowerCase().includes('non_compliant')).length}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
