import { BrowserRouter, Route, Routes, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import AuditDetail from "./pages/AuditPage/AuditDetail";
import Audit from "./pages/AuditPage/Audit";
import Brd from "./pages/BrdPage/Brd";
import Query from "./pages/QueryPage/Query";
import Ingest from "./pages/IngestPage/Ingest";
import Dashboard from "./pages/DashboardPage/Dashboard";
import Verify from "./pages/VerifyPage/Verify";

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/ingest" element={<Ingest />} />
          <Route path="/query" element={<Query />} />
          <Route path="/verify" element={<Verify />} />
          <Route path="/brd" element={<Brd />} />
          <Route path="/audit" element={<Audit />} />
          <Route path="/audit/:id" element={<AuditDetail />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
