import React, { useState, useEffect } from 'react';
import { callPythonMethod, onPythonEvent, offPythonEvent } from '../../lib/pythonBridge';

interface TrainingMetrics {
  accuracy?: number;
  loss?: number;
  learningRate?: number;
  epoch?: number;
  step?: number;
}

interface TrainingProgressCardProps {
  onStop: () => void;
}

export const TrainingProgressCard: React.FC<TrainingProgressCardProps> = ({ onStop }) => {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Idle');
  const [stage, setStage] = useState('N/A');
  const [metrics, setMetrics] = useState<TrainingMetrics>({});

  useEffect(() => {
    const handleTrainingUpdate = (data: any) => {
      setProgress(data.progress || 0);
      setStatus(data.status || 'Updating...');
      setStage(data.stage || 'N/A');
      setMetrics(data.metrics || {});
    };

    onPythonEvent('training_update', handleTrainingUpdate);

    return () => {
      offPythonEvent('training_update', handleTrainingUpdate);
    };
  }, []);

  return (
    <div className="training-progress">
      <div className="progress-header">
        <h3>🔄 Training Progress</h3>
        <div className="progress-stats">
          {metrics.epoch && (
            <span className="metric">Epoch: {metrics.epoch}</span>
          )}
          {metrics.step && (
            <span className="metric">Step: {metrics.step}</span>
          )}
          {metrics.loss && (
            <span className="metric">Loss: {metrics.loss.toFixed(4)}</span>
          )}
          {metrics.accuracy && (
            <span className="metric">Accuracy: {(metrics.accuracy * 100).toFixed(1)}%</span>
          )}
        </div>
      </div>
      
      <div className="progress-container">
        <div className="progress-bar-container">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <div className="progress-percentage">{progress}%</div>
        </div>
        
        <div className="progress-details">
          <div className="progress-stage">Stage: {stage}</div>
          <div className="progress-status">{status}</div>
          <button 
            className="stop-training-button"
            onClick={onStop}
            title="Stop training"
          >
            🛑 Stop
          </button>
        </div>
      </div>
    </div>
  );
};