'use client';

import { useEffect, useState, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import styles from './train.module.css';

interface UploadedFile {
  name: string;
  size: number;
}

interface TrainingProgress {
  progress: number;
  status: string;
  stage: string;
}

export default function TrainPage() {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState('');
  const [trainingStage, setTrainingStage] = useState('');
  const [socket, setSocket] = useState<Socket | null>(null);

  useEffect(() => {
    const newSocket = io('http://localhost:8000');
    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('Connected to WebSocket server');
    });

    newSocket.on('training-progress', (data: TrainingProgress) => {
      setTrainingProgress(data.progress);
      setTrainingStatus(data.status);
      setTrainingStage(data.stage);

      if (data.progress >= 100) {
        setIsTraining(false);
      }
    });

    return () => {
      newSocket.disconnect();
    };
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setUploadedFile(e.target.files[0]);
    }
  };

  const startTraining = async () => {
    if (!uploadedFile) {
      alert('Please upload a document first.');
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingStatus('Starting training...');
    setTrainingStage('initialization');

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await fetch('http://localhost:8000/api/train-model', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to start training.');
      }
    } catch (error) {
      console.error('Failed to start training:', error);
      setIsTraining(false);
    }
  };

  return (
    <div className={styles.trainContainer}>
      <div className={styles.trainHeader}>
        <div className={styles.headerContent}>
          <h2>📚 Train AI Model</h2>
          <p>Upload documents to generate training data and fine-tune AI models for custom knowledge domains</p>
        </div>
      </div>

      <div className={styles.uploadArea}>
        <div className={styles.uploadIcon}>📄</div>
        <p>Drag & drop a file here or click to browse</p>
        <p className={styles.fileTypes}>Supported: PDF, TXT, DOCX, DOC, RTF</p>
        <input
          type="file"
          accept=".pdf,.txt,.docx,.doc,.rtf"
          onChange={handleFileChange}
          style={{ display: 'none' }}
          id="file-input"
        />
        <label htmlFor="file-input" className={styles.browseButton}>
          Choose File
        </label>
        {uploadedFile && (
          <div className={styles.fileItem}>
            <span className={styles.fileIcon}>📄</span>
            <div className={styles.fileInfo}>
              <div className={styles.fileName}>{uploadedFile.name}</div>
            </div>
          </div>
        )}
      </div>

      <div className={styles.trainingControls}>
        <div className={styles.controlButtons}>
          <button
            className={styles.startTrainingButton}
            onClick={startTraining}
            disabled={isTraining || !uploadedFile}
          >
            {isTraining ? 'Processing Documents...' : 'Generate Training Data & Train Model'}
          </button>
        </div>
      </div>

      {isTraining && (
        <div className={styles.trainingProgress}>
          <div className={styles.progressHeader}>
            <h3>🔄 Training Progress</h3>
          </div>

          <div className="progress-container">
            <div className={styles.progressBarContainer}>
              <div className={styles.progressBar}>
                <div
                  className={styles.progressFill}
                  style={{ width: `${trainingProgress}%` }}
                ></div>
              </div>
              <div className={styles.progressPercentage}>{trainingProgress}%</div>
            </div>

            <div className={styles.progressDetails}>
              <div className="progress-stage">Stage: {trainingStage}</div>
              <div className="progress-status">{trainingStatus}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}