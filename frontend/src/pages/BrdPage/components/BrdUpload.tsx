import React, { useState, useRef } from 'react';
import { Upload, FileUp } from 'lucide-react';

interface BrdUploadProps {
  onUploadFile: (file: File) => void;
}

export default function BrdUpload({ onUploadFile }: BrdUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUploadFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onUploadFile(e.target.files[0]);
    }
  };

  const triggerUpload = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">Upload Business Requirements Document</div>
          <div className="card-subtitle">AI will parse & map requirements to RBI norms</div>
        </div>
      </div>
      <div className="card-body">
        <div 
          className={`drop-zone ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={triggerUpload}
          style={{ padding: '60px 20px', cursor: 'pointer' }}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept=".pdf,.docx,.doc,.txt" 
            style={{ display: 'none' }} 
          />
          <div className="drop-icon" style={{ background: dragActive ? 'var(--accent2)' : 'var(--fg)', transition: 'background 0.2s' }}>
            <FileUp size={20} />
          </div>
          <div className="drop-text">Drop your BRD document here, or click to browse</div>
          <div className="drop-sub">The system will identify compliance gaps automatically</div>
        </div>
      </div>
    </div>
  );
}
