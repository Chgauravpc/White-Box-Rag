import React, { useEffect, useRef, useState } from 'react';

interface ComplianceScoreProps {
  scoreTarget: number;
}

export default function ComplianceScore({ scoreTarget }: ComplianceScoreProps) {
  const [score, setScore] = useState(0);
  const gaugeRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!gaugeRef.current) return;
    const c = gaugeRef.current;
    const ctx = c?.getContext('2d');
    if (!ctx) return;

    let current = 0;
    
    const anim = setInterval(() => {
      current += (scoreTarget - current) * 0.1;
      if (Math.abs(scoreTarget - current) < 0.5) current = scoreTarget;
      
      ctx.clearRect(0, 0, 180, 100);
      ctx.lineWidth = 14;
      ctx.lineCap = 'round';
      
      const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      ctx.beginPath();
      ctx.arc(90, 85, 70, Math.PI, 2 * Math.PI);
      ctx.strokeStyle = isDark ? '#1c2333' : '#e3e1d9';
      ctx.stroke();
      
      ctx.beginPath();
      const endAngle = Math.PI + (Math.PI * (current / 100));
      ctx.arc(90, 85, 70, Math.PI, endAngle);
      ctx.strokeStyle = current >= 70 ? '#1ac87a' : current >= 50 ? '#c8921a' : '#c8371a';
      ctx.stroke();
      
      setScore(Math.round(current));
      
      if (current === scoreTarget) clearInterval(anim);
    }, 16);

    return () => clearInterval(anim);
  }, [scoreTarget]);

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">Compliance Score</div>
          <div className="card-subtitle">BRD: v2.3</div>
        </div>
      </div>
      <div className="card-body">
        <div className="gauge-wrap">
          <canvas ref={gaugeRef} width="180" height="100"></canvas>
          <div className="gauge-score">{score}%</div>
          <div className="gauge-label">Overall BRD Compliance</div>
        </div>
        <div style={{ marginTop: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--fg3)', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: '4px' }}>Alignment Summary</div>
          <div className="sparkline-row">
            <div className="spark-label">FSR Alignment</div>
            <div className="spark-track"><div className="spark-fill s" style={{ width: '88%' }}></div></div>
          </div>
          <div className="sparkline-row">
            <div className="spark-label">MPR Alignment</div>
            <div className="spark-track"><div className="spark-fill s" style={{ width: '92%' }}></div></div>
          </div>
          <div className="sparkline-row">
            <div className="spark-label">PSR Alignment</div>
            <div className="spark-track"><div className="spark-fill w" style={{ width: '64%' }}></div></div>
          </div>
        </div>
      </div>
    </div>
  );
}
