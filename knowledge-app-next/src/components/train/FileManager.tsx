import React, { useState } from 'react';
import { FileUpload } from './FileUpload';

interface UploadedFile {
  name: string;
  size: number;
  status: 'pending' | 'uploading' | 'completed' | 'failed';
  progress: number;
}

export const FileManager: React.FC = () => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);

  const handleFileUpload = (files: File[]) => {
    const newFiles: UploadedFile[] = files.map(file => ({
      name: file.name,
      size: file.size,
      status: 'pending',
      progress: 0,
    }));
    setUploadedFiles(prev => [...prev, ...newFiles]);

    // Simulate upload process
    newFiles.forEach(file => {
      // In a real application, you would call your backend upload API here
      // For example: uploadFile(file).then(...) or use a more sophisticated upload library
      console.log(`Simulating upload for ${file.name}`);
      let progress = 0;
      const interval = setInterval(() => {
        progress += 10;
        if (progress <= 100) {
          setUploadedFiles(prev =>
            prev.map(f =>
              f.name === file.name ? { ...f, status: 'uploading', progress } : f
            )
          );
        } else {
          clearInterval(interval);
          setUploadedFiles(prev =>
            prev.map(f =>
              f.name === file.name ? { ...f, status: 'completed', progress: 100 } : f
            )
          );
        }
      }, 200);
    });
  };

  return (
    <div className="file-manager-container">
      <h2>File Manager</h2>
      <FileUpload
        onFileUpload={handleFileUpload}
        acceptedTypes={['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']}
        maxSize={10 * 1024 * 1024} // 10 MB
        multiple={true}
      />

      <div className="uploaded-files-list">
        <h3>Uploaded Files</h3>
        {uploadedFiles.length === 0 ? (
          <p>No files uploaded yet.</p>
        ) : (
          <ul>
            {uploadedFiles.map((file, index) => (
              <li key={index} className={`file-item file-item-${file.status}`}>
                <span>{file.name} ({ (file.size / (1024 * 1024)).toFixed(2) } MB)</span>
                <span>Status: {file.status} {file.status === 'uploading' && `(${file.progress}%)`}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};
