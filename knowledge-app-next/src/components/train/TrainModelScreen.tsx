'use client';

import React, { useState } from 'react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { supabase } from '@/utils/supabaseClient';
import { Icon } from '@/components/common/Icon';
import { ProgressBar } from '@/components/common/ProgressBar';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';

export const TrainModelScreen: React.FC = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const handleUpload = async () => {
    setUploading(true);
    setUploadProgress(0);
    try {
      for (const file of files) {
        const { error } = await supabase.storage
          .from('training-files')
          .upload(file.name, file, {
            // You might need to implement a custom progress callback for detailed progress
            // Supabase client-side upload doesn't directly expose progress events easily
          });

        if (error) {
          throw error;
        }
        setUploadProgress((prev) => prev + (100 / files.length));
      }
      alert('Files uploaded successfully!');
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Error uploading files');
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-h1 font-h1 text-textPrimary mb-xl">Train Model</h1>

      <Card>
        <div className="flex flex-col items-center justify-center p-xl border-2 border-dashed border-borderColor rounded-lg">
          <p className="text-h4 font-h4 text-textPrimary mb-md">Drag and drop files here or click to browse</p>
          <p className="text-body text-textSecondary mb-md">Supported formats: PDF, TXT, DOCX</p>
          <input type="file" multiple onChange={handleFileChange} className="hidden" id="file-input" />
          <label htmlFor="file-input" className="bg-primaryColor text-white px-lg py-md rounded-md cursor-pointer hover:bg-primaryHover flex items-center">
            <Icon name="FaFolderOpen" className="mr-sm" />
            Browse Files
          </label>
        </div>

        {files.length > 0 && (
          <div className="mt-lg">
            <h3 className="text-h3 font-h3 text-textPrimary mb-md">Selected Files:</h3>
            <ul>
              {files.map((file, i) => (
                <li key={i} className="text-body text-textSecondary">{file.name}</li>
              ))}
            </ul>
            <Button onClick={handleUpload} className="mt-lg" disabled={uploading}>
              {uploading ? 'Uploading...' : 'Upload and Train'}
            </Button>
            {uploading && (
              <div className="mt-md">
                <ProgressBar progress={uploadProgress} />
                <LoadingSpinner />
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
};