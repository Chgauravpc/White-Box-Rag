import React, { useState } from 'react';
import { FileText, AlertTriangle, XCircle, ChevronDown } from 'lucide-react';

export interface Requirement {
  id: number;
  text: string;
  status: 'compliant' | 'review' | 'violation';
  alignmentScore: number;
  mappedSections: string[];
  gaps: string[];
  violations: string[];
}

interface RequirementsTableProps {
  requirements: Requirement[];
}

export default function RequirementsTable({ requirements }: RequirementsTableProps) {
  const [expandedReq, setExpandedReq] = useState<number | null>(null);

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">Requirement Analysis</div>
          <div className="card-subtitle">{requirements.length} requirements extracted</div>
        </div>
        <button className="btn">Download Report</button>
      </div>
      <div className="card-body" style={{ padding: 0 }}>
        {requirements.map((req) => (
          <div key={req.id} style={{ borderBottom: '1px solid var(--border)' }}>
            <div 
              className="req-row" 
              onClick={() => setExpandedReq(expandedReq === req.id ? null : req.id)} 
              style={{ borderBottom: 'none', paddingBottom: expandedReq === req.id ? '10px' : '14px', alignItems: 'center' }}
            >
              <div style={{ width: '20px', color: 'var(--fg3)' }}><ChevronDown size={14} style={{ transform: expandedReq === req.id ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} /></div>
              <span className="req-id">REQ-00{req.id}</span>
              <div className="req-text">{req.text}</div>
              <span className={`gate-badge gate-${req.status === 'compliant' ? 'safe' : req.status === 'review' ? 'review' : 'danger'}`} style={{ fontSize: '10px', padding: '3px 8px' }}>
                <div className={`gate-dot dot-${req.status === 'compliant' ? 'safe' : req.status === 'review' ? 'warn' : 'danger'}`}></div>
                {req.status === 'compliant' ? 'COMPLIANT' : req.status === 'review' ? 'REVIEW' : 'VIOLATION'}
              </span>
              <div className={`req-score ${req.status === 'compliant' ? 'high' : req.status === 'review' ? 'mid' : 'low'}`} style={{ marginLeft: '10px' }}>{req.alignmentScore}</div>
            </div>

            {expandedReq === req.id && (
              <div style={{ padding: '0 20px 20px 62px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  
                  <div>
                    <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--fg3)', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: '2px' }}>Mapped RBI Sections</div>
                    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '6px' }}>
                      {req.mappedSections.map(sec => (
                        <span key={sec} className="q-pub" style={{ padding: '4px 8px', fontSize: '10px' }}><FileText size={10} style={{ display: 'inline', marginRight: '4px' }} />{sec}</span>
                      ))}
                    </div>
                  </div>

                  {req.gaps.length > 0 && (
                    <div>
                      <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--fg3)', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: '2px' }}>Identified Gaps</div>
                      <div style={{ marginTop: '6px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {req.gaps.map((gap, i) => (
                          <div key={i} style={{ background: 'rgba(200,146,26,.1)', color: '#a0720e', padding: '8px 12px', borderRadius: '4px', fontSize: '12px', display: 'flex', gap: '8px', alignItems: 'center' }}>
                            <AlertTriangle size={14} /> {gap}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {req.violations.length > 0 && (
                    <div>
                      <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--fg3)', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: '2px' }}>Compliance Violations</div>
                      <div style={{ marginTop: '6px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {req.violations.map((vio, i) => (
                          <div key={i} style={{ background: 'rgba(200,55,26,.1)', color: '#a02414', padding: '8px 12px', borderRadius: '4px', fontSize: '12px', display: 'flex', gap: '8px', alignItems: 'center' }}>
                            <XCircle size={14} /> {vio}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
