import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import "../pages/DashboardPage/Dashboard.css"; // Ensure Dashboard.css is imported globally for the sidebar

export default function Layout({ children }: { children: ReactNode }) {
  const location = useLocation();

  const getLinkClass = (path: string) => {
    const isActive = path === "/" ? location.pathname === "/" : location.pathname.startsWith(path);
    return `nav-item ${isActive ? 'active' : ''}`;
  };

  return (
    <div className="dashboard-wrapper shell">
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-mark">
            <svg viewBox="0 0 18 18" fill="none">
              <path d="M9 2L15.5 13H2.5L9 2Z" fill="white" stroke="white" strokeWidth=".5" />
              <rect x="5" y="10" width="8" height="1.5" rx=".75" fill="#0d1117" />
              <rect x="7" y="12.5" width="4" height="1.5" rx=".75" fill="#0d1117" />
            </svg>
          </div>
          <div>
            <div className="logo-text">XAI Govern</div>
            <div className="logo-sub">RBI Trust Framework</div>
          </div>
        </div>

        <div className="nav-section">Overview</div>
        <Link to="/" className={getLinkClass("/")}>
          <svg className="nav-icon" viewBox="0 0 16 16" fill="currentColor">
            <rect x="1" y="1" width="6" height="6" rx="1" />
            <rect x="9" y="1" width="6" height="6" rx="1" />
            <rect x="1" y="9" width="6" height="6" rx="1" />
            <rect x="9" y="9" width="6" height="6" rx="1" />
          </svg>
          Dashboard
        </Link>
        <Link to="/query" className={getLinkClass("/query")}>
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="7" cy="7" r="4" />
            <path d="M11 11l3 3" />
          </svg>
          RAG Query
          <span className="nav-badge">3</span>
        </Link>

        <div className="nav-section">Verification</div>
        <Link to="/verify" className={getLinkClass("/verify")}>
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M14 4L6 12 2 8" />
          </svg>
          Trust Gating
          <span className="nav-badge warn">1</span>
        </Link>
        <Link to="/brd" className={getLinkClass("/brd")}>
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="2" y="1" width="12" height="14" rx="1" />
            <path d="M5 5h6M5 8h6M5 11h4" />
          </svg>
          BRD Validator
        </Link>

        <div className="nav-section">System</div>
        <Link to="/ingest" className={getLinkClass("/ingest")}>
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M8 2v8M5 7l3 3 3-3" />
            <path d="M2 12v1a1 1 0 001 1h10a1 1 0 001-1v-1" />
          </svg>
          Ingest Documents
        </Link>
        <Link to="/audit" className={getLinkClass("/audit")}>
          <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M2 4h12M2 8h8M2 12h5" />
            <circle cx="12" cy="11" r="3" />
          </svg>
          Audit Trail
          <span className="nav-badge safe">12</span>
        </Link>

        <div className="sidebar-footer">
          <div className="status-row">
            <div className="pulse"></div>
            Backend services online
          </div>
          <div className="status-row" style={{ marginBottom: 0 }}>
            <svg width="6" height="6" viewBox="0 0 6 6">
              <circle cx="3" cy="3" r="3" fill="var(--accent2)" opacity=".6" />
            </svg>
            Gemini Pro 3.0 connected
          </div>
        </div>
      </aside>

      <main className="main">
        {children}
      </main>
    </div>
  );
}
