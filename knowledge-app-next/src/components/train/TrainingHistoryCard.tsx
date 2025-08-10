import React, { useState, useEffect } from 'react';
import { callPythonMethod } from '../../lib/pythonBridge';
import TrainingTimeline from './TrainingTimeline';

interface TrainingHistory {
  runId: string;
  adapterName: string;
  baseModel: string;
  status: string;
  startTime: string;
  endTime?: string;
  improvementScore?: number;
  evaluationScore?: number;
}

interface TrainingHistoryCardProps {
  isVisible: boolean;
}

export const TrainingHistoryCard: React.FC<TrainingHistoryCardProps> = ({ isVisible }) => {
  const [history, setHistory] = useState<TrainingHistory[]>([]);

  const fetchHistory = async () => {
    const history = await callPythonMethod('get_training_history');
    setHistory(history);
  };

  useEffect(() => {
    if (isVisible) {
      fetchHistory();
    }
  }, [isVisible]);

  if (!isVisible) return null;

  return (
    <div className="training-history">
      <div className="history-header">
        <h3>📊 Training History</h3>
        <button 
          className="btn-secondary"
          onClick={fetchHistory}
        >
          🔄 Refresh
        </button>
      </div>
      
      {history.length === 0 ? (
        <div className="no-history">
          <p>No training history available</p>
          <p className="text-sm">Start your first training session to see results here</p>
        </div>
      ) : (
        <TrainingTimeline history={history} />
      )}
    </div>
  );
};