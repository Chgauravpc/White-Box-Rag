import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Download, Eye, FileText, ChevronRight, Loader2 } from "lucide-react";
import '../DashboardPage/Dashboard.css';
import { listAuditLogs } from '@/lib/api';
import type { AuditLogItem } from '@/lib/types';

export default function Audit() {
  const navigate = useNavigate();
  const [logs, setLogs] = useState<AuditLogItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const res = await listAuditLogs();
        setLogs(res.data);
      } catch (err) {
        console.error("Failed to fetch logs", err);
      } finally {
        setIsLoading(false);
      }
    };
    fetchLogs();
  }, []);

  const filteredLogs = logs.filter((log) => {
    return log.query.toLowerCase().includes(searchTerm.toLowerCase()) || String(log.id).includes(searchTerm.toLowerCase());
  });

  const getStatusColor = (riskLevel: string) => {
    const risk = riskLevel.toLowerCase();
    if (risk.includes("safe")) return "safe";
    if (risk.includes("review")) return "warn";
    if (risk.includes("non")) return "danger";
    return "safe";
  };

  return (
    <div className="dashboard-wrapper">
      <div className="topbar">
        <div className="breadcrumb">
          <span>Overview</span>
          <span className="breadcrumb-sep">/</span>
          <span>Governance</span>
          <span className="breadcrumb-sep">/</span>
          <span>Audit Trail</span>
        </div>
        <div className="topbar-actions">
          <button className="btn btn-primary" onClick={() => navigate('/query')}>+ New Governance Query</button>
        </div>
      </div>

      <div className="content">
        <div className="page" id="page-audit">
          
          <div className="stats-row">
            <div className="stat-card c-blue">
              <div className="stat-label">Total Logs</div>
              <div className="stat-value">{logs.length}</div>
            </div>
            <div className="stat-card c-safe">
              <div className="stat-label">Safe Decisions</div>
              <div className="stat-value text-safe">{logs.filter(l => l.risk_level.toLowerCase().includes("safe")).length}</div>
            </div>
            <div className="stat-card c-warn">
              <div className="stat-label">Under Review</div>
              <div className="stat-value text-review">{logs.filter(l => l.risk_level.toLowerCase().includes("review")).length}</div>
            </div>
            <div className="stat-card c-danger">
              <div className="stat-label">Non-Compliant</div>
              <div className="stat-value text-danger">{logs.filter(l => l.risk_level.toLowerCase().includes("non")).length}</div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">Governance Registry</div>
                  <div className="card-subtitle">Searchable log of all system decisions</div>
                </div>
              </div>
              <div className="card-body" style={{ padding: '0 20px 20px' }}>
                <div className="audit-search" style={{ marginTop: '20px' }}>
                  <div style={{ flex: 1, position: 'relative' }}>
                    <Search size={14} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--fg3)' }} />
                    <input 
                      type="text" 
                      placeholder="Search by ID or query text..." 
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      style={{ paddingLeft: '34px', width: '100%', fontFamily: 'var(--font)' }}
                    />
                  </div>
                  <button className="btn">
                    <Download size={12} style={{ marginRight: '6px', display: 'inline-block', verticalAlign: 'middle' }} /> Export
                  </button>
                </div>

                <div style={{ overflowX: 'auto' }}>
                  <table className="query-table" style={{ whiteSpace: 'nowrap' }}>
                    <thead>
                      <tr>
                        <th>Audit ID</th>
                        <th>Timestamp (UTC)</th>
                        <th>Query</th>
                        <th>Status</th>
                        <th>Score</th>
                        <th style={{ width: '40px' }}></th>
                      </tr>
                    </thead>
                    <tbody>
                      {isLoading ? (
                        <tr><td colSpan={6} style={{ textAlign: 'center', padding: '40px', color: 'var(--fg3)' }}><Loader2 className="animate-spin" style={{ display: 'inline' }} /> Loading Immutable Logs...</td></tr>
                      ) : filteredLogs.map((log) => {
                        const sColor = getStatusColor(log.risk_level);
                        return (
                          <tr key={log.id} onClick={(e) => { e.stopPropagation(); navigate(`/audit/${log.id}`); }} style={{ cursor: 'pointer' }}>
                            <td><span style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--accent2)', fontWeight: 600 }}>IDX-{log.id}</span></td>
                            <td><span style={{ fontSize: '12px', color: 'var(--fg)' }}>{new Date(log.timestamp).toLocaleTimeString()}</span></td>
                            <td><span className="q-pub" style={{ fontSize: '12px' }}>{log.query.length > 50 ? log.query.substring(0, 50) + "..." : log.query}</span></td>
                            <td>
                              <span className={`gate-badge gate-${sColor}`} style={{ fontSize: '10px', padding: '3px 8px' }}>
                                <div className={`gate-dot dot-${sColor === 'warn' ? 'warn' : sColor}`}></div>
                                {log.risk_level.toUpperCase().replace(/_/g, ' ')}
                              </span>
                            </td>
                            <td><span style={{ fontFamily: 'var(--mono)', fontWeight: 600, color: `var(--${sColor})` }}>{log.compliance_score}</span></td>
                            <td>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent2)' }}>
                                <Eye size={14} /> 
                                <span style={{ fontSize: '11px', fontWeight: 600 }}>JSON</span>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                      {!isLoading && filteredLogs.length === 0 && (
                        <tr><td colSpan={6} style={{ textAlign: 'center', padding: '40px', color: 'var(--fg3)' }}>No logs matched your criteria.</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
        </div>
        </div>
      </div>
    </div>
  );
}
