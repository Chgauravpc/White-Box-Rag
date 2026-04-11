import React from 'react';
import type { Requirement } from './RequirementsTable';
import { AlertTriangle, Lightbulb } from 'lucide-react';

interface RemediationPanelProps {
  requirements: Requirement[];
}

export default function RemediationPanel({ requirements }: RemediationPanelProps) {
  const issues = requirements.filter(r => r.status !== 'compliant');

  if (issues.length === 0) {
    return (
      <div className="card" style={{ marginTop: '20px', background: 'var(--surface2)', borderColor: 'var(--safe)' }}>
        <div className="card-body" style={{ textAlign: 'center', padding: '32px', color: 'var(--safe)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
            <Lightbulb size={18} />
            <span style={{ fontWeight: 600 }}>All requirements are compliant. No remediation needed.</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card" style={{ marginTop: '20px' }}>
      <div className="card-header">
        <div>
          <div className="card-title">Remediation Guide</div>
          <div className="card-subtitle">AI-generated fixes to align with RBI directives</div>
        </div>
      </div>
      <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {issues.map((issue, idx) => (
          <div 
            key={idx} 
            style={{ 
              padding: '16px', 
              background: 'var(--surface2)', 
              borderRadius: 'var(--radius)', 
              border: '1px solid var(--border)', 
              borderLeft: `4px solid var(--${issue.status === 'violation' ? 'danger' : 'warn'})` 
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--fg)' }}>
                {issue.status === 'violation' ? 'Regulatory Violation' : 'Compliance Gap'} Identified
              </div>
              <span className={`gate-badge gate-${issue.status}`} style={{ fontSize: '10px' }}>
                Priority: {issue.status === 'violation' ? 'High' : 'Medium'}
              </span>
            </div>
            
            <div style={{ fontSize: '13px', color: 'var(--fg2)', marginBottom: '12px', lineHeight: 1.5 }}>
              <strong>Requirement:</strong> {issue.text}
            </div>

            <div style={{ background: 'rgba(0,0,0,0.03)', padding: '12px', borderRadius: '4px', fontSize: '12px', color: 'var(--fg2)' }}>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                <Lightbulb size={14} style={{ color: 'var(--accent2)', marginTop: '2px' }} />
                <div>
                  <strong>Recommended Fix:</strong> {issue.status === 'violation' ? issue.violations[0] : issue.gaps[0]}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
