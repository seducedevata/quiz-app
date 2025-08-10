import React from 'react';

interface FilePreviewProps {
  file: File;
  onRemove: (file: File) => void;
}

const FilePreview: React.FC<FilePreviewProps> = ({ file, onRemove }) => {
  return (
    <div className="file-preview">
      <p>{file.name}</p>
      <button onClick={() => onRemove(file)}>Remove</button>
    </div>
  );
};

export default FilePreview;