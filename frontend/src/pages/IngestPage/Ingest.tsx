import React, { useState, useEffect, useRef } from 'react';
import '../DashboardPage/Dashboard.css';

import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircle, FileUp, Database, CheckCircle, Clock } from 'lucide-react';

import { listDocuments, ingestDocument } from '@/lib/api';
import type { DocumentInfo } from '@/lib/types';

export default function Ingest() {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  
  const [selectedPub, setSelectedPub] = useState('FSR');
  const [editionDate, setEditionDate] = useState('');
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchDocs();
  }, []);

  const fetchDocs = async () => {
    try {
      const res = await listDocuments();
      setDocuments(res.data);
      setIsLoading(false);
    } catch (err) {
      setError("Failed to fetch ingested documents. Please ensure the backend is running.");
      setIsLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
    }
  };

  const startUpload = async () => {
    if (!selectedFile) {
      setError("Please select a file first.");
      return;
    }
    if (!editionDate) {
      setError("Please select an edition date before uploading.");
      return;
    }
    
    setIsUploading(true);
    setError(null);

    try {
      await ingestDocument(selectedFile, selectedPub, editionDate);
      await fetchDocs();
      setIsUploading(false);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Ingestion failed. Ensure the file is a PDF and valid metadata is provided.");
      setIsUploading(false);
    }
  };

  const triggerFileSelect = () => {
    if (isUploading) return;
    fileInputRef.current?.click();
  };

  return (
    <div className="dashboard-wrapper">
      <div className="topbar">
        <div className="breadcrumb">
          <span>Overview</span>
          <span className="breadcrumb-sep">/</span>
          <span>Knowledge Base</span>
          <span className="breadcrumb-sep">/</span>
          <span>Ingest Data</span>
        </div>
        <div className="topbar-actions">
        </div>
      </div>

      <div className="content">
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Operation Failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        <div className="page" id="page-ingest">
          <div className="grid-2">
            
            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">Ingest Publication</div>
                  <div className="card-subtitle">Add new RBI reports to the XAI vector store</div>
                </div>
              </div>
              <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div 
                  className={`drop-zone ${isUploading ? 'active' : ''}`} 
                  onClick={triggerFileSelect}
                  style={{ cursor: isUploading ? 'wait' : 'pointer' }}
                >
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    onChange={handleFileChange} 
                    accept=".pdf" 
                    style={{ display: 'none' }} 
                  />
                  <div className="drop-icon">
                    {isUploading ? <Clock className="animate-spin" size={32} /> : <FileUp size={32} />}
                  </div>
                  <div className="drop-text">
                    {isUploading ? 'Processing & Vectorizing...' : (selectedFile ? selectedFile.name : 'Click to Select PDF')}
                  </div>
                  <div className="drop-sub">
                    {selectedFile ? 'File selected and ready to ingest' : 'The system will automatically parse sections and claims'}
                  </div>
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                  <div>
                    <div style={{ fontSize: '11px', color: 'var(--fg3)', marginBottom: '8px', fontWeight: 600, textTransform: 'uppercase' }}>Publication Code</div>
                    <select 
                      value={selectedPub}
                      onChange={(e) => setSelectedPub(e.target.value)}
                      style={{ width: '100%', fontFamily: 'var(--font)', fontSize: '13px', padding: '10px 12px', border: '1px solid var(--border)', borderRadius: 'var(--radius)', background: 'var(--surface)', color: 'var(--fg)', outline: 'none' }}
                    >
                      <option value="FSR">FSR - Financial Stability Report</option>
                      <option value="MPR">MPR - Monetary Policy Report</option>
                      <option value="PSR">PSR - Payment Systems Report</option>
                      <option value="FER">FER - Financial Evaluation Report</option>
                    </select>
                  </div>
                  <div>
                    <div style={{ fontSize: '11px', color: 'var(--fg3)', marginBottom: '8px', fontWeight: 600, textTransform: 'uppercase' }}>Edition Date</div>
                    <input 
                      type="text" 
                      placeholder="e.g. December 2024"
                      value={editionDate}
                      onChange={(e) => setEditionDate(e.target.value)}
                      style={{ width: '100%', fontFamily: 'var(--font)', fontSize: '13px', padding: '10px 12px', border: '1px solid var(--border)', borderRadius: 'var(--radius)', background: 'var(--surface)', color: 'var(--fg)', outline: 'none' }} 
                    />
                  </div>
                </div>
                
                <button 
                  className="btn btn-primary" 
                  style={{ width: '100%', padding: '12px', fontWeight: 600 }} 
                  onClick={startUpload}
                  disabled={isUploading || !editionDate || !selectedFile}
                >
                  {isUploading ? 'Ingestion in Progress...' : 'Start Ingestion'}
                </button>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">Active Knowledge Base</div>
                  <div className="card-subtitle">Verified publications currently in vector store</div>
                </div>
                {!isLoading && (
                  <div className="card-subtitle" style={{ float: 'right', display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <div style={{ padding: '4px 8px', background: 'var(--surface2)', borderRadius: '6px', fontSize: '11px' }}>
                      <span style={{ fontWeight: 600, color: 'var(--accent2)' }}>{documents.length}</span> Documents
                    </div>
                  </div>
                )}
              </div>
              <div className="card-body" style={{ padding: 0, overflowY: 'auto', maxHeight: '500px' }}>
                <table className="query-table">
                  <thead>
                    <tr>
                      <th>Source Type</th>
                      <th>Edition</th>
                      <th>Granularity</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {isLoading ? (
                      [...Array(5)].map((_, i) => (
                        <tr key={i}>
                          <td colSpan={4} style={{ padding: '16px' }}><Skeleton className="h-4 w-full" /></td>
                        </tr>
                      ))
                    ) : documents.length === 0 ? (
                      <tr>
                        <td colSpan={4} style={{ textAlign: 'center', padding: '48px', color: 'var(--fg3)', fontSize: '13px' }}>
                          <Database size={24} style={{ margin: '0 auto 12px', opacity: 0.5 }} />
                          No publications ingested yet.
                        </td>
                      </tr>
                    ) : (
                      documents.map((doc) => (
                        <tr key={doc.id}>
                          <td><span className="q-pub">{doc.publication_name}</span></td>
                          <td style={{ fontSize: '13px', color: 'var(--fg)' }}>{doc.edition_date}</td>
                          <td style={{ fontSize: '12px', color: 'var(--fg3)' }}>{doc.chunk_count} chunks</td>
                          <td>
                            <span className="gate-badge gate-safe" style={{ fontSize: '10px', padding: '3px 8px' }}>
                              <div className="gate-dot dot-safe"></div>Ready
                            </span>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
