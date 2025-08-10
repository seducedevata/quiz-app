import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

interface FileUploadProps {
  onFileUpload: (files: File[]) => void;
  acceptedTypes: string[];
  maxSize: number;
  multiple?: boolean;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onFileUpload,
  acceptedTypes,
  maxSize,
  multiple = true
}) => {
  const [rejectedFiles, setRejectedFiles] = useState<any[]>([]);

  const onDrop = useCallback((acceptedFiles: File[], fileRejections: any[]) => {
    if (acceptedFiles.length > 0) {
      onFileUpload(acceptedFiles);
    }
    if (fileRejections.length > 0) {
      setRejectedFiles(fileRejections);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedTypes.join(',') as any, // react-dropzone expects a string of comma-separated MIME types
    maxSize: maxSize,
    multiple: multiple,
  });

  return (
    <div className="file-upload-container">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'drag-active' : ''}`}
      >
        <input {...getInputProps()} />
        {
          isDragActive ?
            <p>Drop the files here ...</p> :
            <p>Drag 'n' drop some files here, or click to select files</p>
        }
        {acceptedTypes.length > 0 && (
          <p className="accepted-types">Accepted file types: {acceptedTypes.join(', ')}</p>
        )}
        {maxSize > 0 && (
          <p className="max-size">Max file size: {maxSize / (1024 * 1024)} MB</p>
        )}
      </div>
      {rejectedFiles.length > 0 && (
        <div className="rejected-files">
          <h4>Rejected Files:</h4>
          <ul>
            {rejectedFiles.map((rejection, index) => (
              <li key={index}>
                {rejection.file.name} - {rejection.errors.map((e: any) => e.message).join(', ')}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
