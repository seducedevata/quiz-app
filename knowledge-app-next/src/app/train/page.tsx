'use client';

import React, { useState, useEffect } from 'react';
import { callPythonMethod } from '../../lib/pythonBridge';
import { AppLogger } from '../../lib/logger';

interface UploadedFile {
  name: string;
  size: number;
}

export default function TrainModelPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [trainingConfig, setTrainingConfig] = useState({
    modelType: 'text-generation',
    epochs: 3,
    batchSize: 8,
    learningRate: 0.001,
  });
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState('idle'); // idle, uploading, training, complete, error
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [trainedModels, setTrainedModels] = useState<string[]>([]);
  const [error, setError] = useState('');

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setSelectedFile(file);
      AppLogger.info('FILES', `Selected file: ${file.name}`);

      // Upload file to backend immediately
      const formData = new FormData();
      formData.append('file', file);

      try {
        AppLogger.action('FILES', 'Uploading file to backend...', { fileName: file.name });
        const result = await callPythonMethod<string>('uploadFile', file.name, Array.from(new Uint8Array(await file.arrayBuffer())));
        const data = JSON.parse(result);
        if (data.success) {
          AppLogger.success('FILES', `File uploaded successfully: ${file.name}`);
          loadExistingFiles(); // Refresh the list of uploaded files
        } else {
          AppLogger.error('FILES', `Failed to upload file: ${file.name}`, { error: data.error });
          setError(data.error || 'Failed to upload file.');
        }
      } catch (e: any) {
        AppLogger.error('FILES', `Error uploading file: ${file.name}`, { error: e.message });
        setError(e.message || 'Error uploading file.');
      }
    } else {
      setSelectedFile(null);
      AppLogger.info('FILES', 'No file selected.');
    }
  };

  const loadExistingFiles = async () => {
    AppLogger.info('FILES', 'Loading existing uploaded files');
    try {
      const filesJson = await callPythonMethod<string>('getUploadedFiles');
      const files = JSON.parse(filesJson);
      setUploadedFiles(files);
      AppLogger.success('FILES', 'Uploaded files loaded', { fileCount: files.length });
    } catch (e: any) {
      AppLogger.error('FILES', 'Failed to load uploaded files', { error: e.message });
      setError(e.message || 'Failed to load uploaded files.');
    }
  };

  const deleteFile = async (filename: string) => {
    AppLogger.user('FILES', `File deletion requested: ${filename}`);
    try {
      const result = await callPythonMethod<string>('deleteUploadedFile', filename);
      if (result === 'success') {
        AppLogger.success('FILES', `File deleted successfully: ${filename}`);
        loadExistingFiles(); // Refresh the list
      } else {
        AppLogger.error('FILES', `Failed to delete file: ${filename}`);
        setError(`Failed to delete file: ${filename}`);
      }
    } catch (e: any) {
      AppLogger.error('FILES', `Error deleting file: ${filename}`, { error: e.message });
      setError(e.message || `Error deleting file: ${filename}`);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleConfigChange = (event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = event.target;
    setTrainingConfig((prevConfig) => ({
      ...prevConfig,
      [name]: name === 'epochs' || name === 'batchSize' ? parseInt(value) : parseFloat(value),
    }));
  };

  const startTraining = async () => {
    if (!selectedFile) {
      alert('Please select a file to train.');
      return;
    }

    setTrainingStatus('training');
    setTrainingProgress(0);
    setTrainingLogs([]);
    setError('');

    try {
      AppLogger.action('TRAINING', 'Starting training process...', trainingConfig);
      const result = await callPythonMethod<string>('startTraining', JSON.stringify({
        fileName: selectedFile.name,
        ...trainingConfig,
      }));
      const data = JSON.parse(result);

      if (data.success) {
        AppLogger.success('TRAINING', 'Training initiated successfully.', data);
        setTrainingLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] Training started: ${data.message}`]);
      } else {
        AppLogger.error('TRAINING', 'Failed to initiate training.', data);
        setError(data.error || 'Failed to initiate training.');
        setTrainingStatus('error');
      }
    } catch (e: any) {
      AppLogger.error('TRAINING', 'Error calling Python bridge for training.', { error: e.message });
      setError(e.message || 'Error starting training.');
      setTrainingStatus('error');
    }
  };

  const updateTrainingProgressUI = (progressData: any) => {
    setTrainingProgress(progressData.progress);
    setTrainingLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] Epoch ${progressData.epoch}, Step ${progressData.step} - Loss: ${progressData.loss.toFixed(4)}`]);
  };

  const updateTrainingMetricsUI = (metricsData: any) => {
    AppLogger.performance('TRAINING', 'Training metrics updated', metricsData);
    // Update metrics display elements (e.g., accuracy, loss, learning rate)
    // For now, we'll just log them, but in a real UI, you'd update specific elements
    setTrainingLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] Metrics: Accuracy: ${(metricsData.accuracy * 100).toFixed(2)}%, Loss: ${metricsData.loss.toFixed(4)}`]);
  };

  const loadTrainingConfiguration = async () => {
    AppLogger.info('TRAINING', 'Loading training configuration');
    try {
      const configJson = await callPythonMethod<string>('getTrainingConfiguration');
      const config = JSON.parse(configJson);
      applyTrainingConfigToUI(config);
      AppLogger.success('TRAINING', 'Training configuration loaded', { configKeys: Object.keys(config).length });
    } catch (e: any) {
      AppLogger.error('TRAINING', 'Failed to load training configuration', { error: e.message });
    }
  };

  const applyTrainingConfigToUI = (config: any) => {
    AppLogger.debug('TRAINING', 'Applying training configuration to UI', config);
    setTrainingConfig({
      modelType: config.model_name || 'text-generation',
      epochs: config.epochs || 3,
      batchSize: config.batch_size || 8,
      learningRate: config.learning_rate || 0.001,
    });
  };

  useEffect(() => {
    loadExistingFiles();
    loadTrainingConfiguration();

    onPythonEvent('trainingProgressStructured', updateTrainingProgressUI);
    onPythonEvent('trainingStatusChanged', (data: any) => {
      setTrainingStatus(data.status);
      setTrainingLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] Status: ${data.status} - ${data.phase}`]);
    });
    onPythonEvent('trainingMetricsUpdate', updateTrainingMetricsUI);
    onPythonEvent('trainingConfigSaved', (data: any) => {
      AppLogger.success('TRAINING', 'Training configuration saved', data);
    });

    return () => {
      offPythonEvent('trainingProgressStructured');
      offPythonEvent('trainingStatusChanged');
      offPythonEvent('trainingMetricsUpdate');
      offPythonEvent('trainingConfigSaved');
    };
  }, []);

  return (
    <div className="train-page p-8">
      <h1 className="text-3xl font-bold text-text-primary mb-6">Train Your Custom Model</h1>

      <div className="training-section bg-bg-secondary p-6 rounded-lg shadow-md mb-6">
        <h2 className="text-2xl font-bold text-text-primary mb-4">1. Upload Dataset</h2>
        <div className="form-group mb-4">
          <label htmlFor="datasetFile" className="block text-text-secondary text-sm font-bold mb-2">Select Dataset File (e.g., .txt, .jsonl)</label>
          <input
            type="file"
            id="datasetFile"
            className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
            onChange={handleFileChange}
            accept=".txt,.jsonl,.csv"
          />
        </div>
        <div id="uploaded-files-list" className="uploaded-files-list border border-border-color rounded-md mt-4">
          {uploadedFiles.length === 0 ? (
            <p className="no-files text-text-secondary text-center py-4">No files uploaded yet</p>
          ) : (
            uploadedFiles.map((file) => (
              <div key={file.name} className="file-item">
                <span className="file-icon">📄</span>
                <span className="file-name">{file.name}</span>
                <span className="file-size">{formatFileSize(file.size)}</span>
                <button className="delete-file-btn" onClick={() => deleteFile(file.name)}>🗑️</button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="training-section bg-bg-secondary p-6 rounded-lg shadow-md mb-6">
        <h2 className="text-2xl font-bold text-text-primary mb-4">2. Training Configuration</h2>
        <div className="form-group mb-4">
          <label htmlFor="modelType" className="block text-text-secondary text-sm font-bold mb-2">Model Type</label>
          <select
            id="modelType"
            name="modelType"
            className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
            value={trainingConfig.modelType}
            onChange={handleConfigChange}
          >
            <option value="text-generation">Text Generation</option>
            <option value="question-answering">Question Answering</option>
            <option value="summarization">Summarization</option>
          </select>
        </div>

        <div className="form-group mb-4">
          <label htmlFor="epochs" className="block text-text-secondary text-sm font-bold mb-2">Epochs</label>
          <input
            type="number"
            id="epochs"
            name="epochs"
            className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
            value={trainingConfig.epochs}
            onChange={handleConfigChange}
            min="1"
          />
        </div>

        <div className="form-group mb-4">
          <label htmlFor="batchSize" className="block text-text-secondary text-sm font-bold mb-2">Batch Size</label>
          <input
            type="number"
            id="batchSize"
            name="batchSize"
            className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
            value={trainingConfig.batchSize}
            onChange={handleConfigChange}
            min="1"
          />
        </div>

        <div className="form-group mb-4">
          <label htmlFor="learningRate" className="block text-text-secondary text-sm font-bold mb-2">Learning Rate</label>
          <input
            type="number"
            id="learningRate"
            name="learningRate"
            className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
            value={trainingConfig.learningRate}
            onChange={handleConfigChange}
            step="0.0001"
            min="0.00001"
            max="0.1"
          />
        </div>

        <button
          className="btn-primary w-full py-3 px-4 rounded-md"
          onClick={startTraining}
          disabled={!selectedFile || trainingStatus === 'training'}
        >
          {trainingStatus === 'training' ? 'Training...' : 'Start Training'}
        </button>

        {error && <p className="error-message text-error-color mt-4"><span>Error:</span> {error}</p>}
      </div>

      {(trainingStatus === 'training' || trainingStatus === 'complete' || trainingStatus === 'error') && (
        <div className="training-section bg-bg-secondary p-6 rounded-lg shadow-md">
          <h2 className="text-2xl font-bold text-text-primary mb-4">Training Status</h2>
          <div className="training-status text-lg font-semibold mb-4">
            Status: {trainingStatus}
          </div>
          <div className="progress-bar-container bg-gray-200 rounded-full h-4 mb-4">
            <div className="progress-bar bg-primary-color h-4 rounded-full" style={{ width: `${trainingProgress}%` }}></div>
          </div>
          <p className="text-text-secondary mb-4">{trainingProgress}% Complete</p>

          <div className="metrics-grid grid grid-cols-3 gap-4 mb-4">
            <div className="metric bg-bg-primary p-4 rounded-md shadow-sm">
              <label className="block text-text-secondary text-sm">Accuracy:</label>
              <span className="text-text-primary text-xl font-bold">N/A</span>
            </div>
            <div className="metric bg-bg-primary p-4 rounded-md shadow-sm">
              <label className="block text-text-secondary text-sm">Loss:</label>
              <span className="text-text-primary text-xl font-bold">N/A</span>
            </div>
            <div className="metric bg-bg-primary p-4 rounded-md shadow-sm">
              <label className="block text-text-secondary text-sm">Learning Rate:</label>
              <span className="text-text-primary text-xl font-bold">{trainingConfig.learningRate}</span>
            </div>
          </div>

          <div className="training-logs-container bg-bg-primary p-4 rounded-md h-48 overflow-y-auto text-sm text-text-secondary">
            <h4 className="font-semibold mb-2">Logs:</h4>
            {trainingLogs.map((log, index) => (
              <p key={index} className="font-mono text-xs">{log}</p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}