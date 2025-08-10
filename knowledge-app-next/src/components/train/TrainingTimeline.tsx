import React from 'react';

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

interface TrainingTimelineProps {
  history: TrainingHistory[];
}

const TrainingTimeline: React.FC<TrainingTimelineProps> = ({ history }) => {
  return (
    <div className="training-timeline">
      <div className="space-y-4">
        {history.map((item) => (
          <div key={item.runId} className="training-timeline-item p-4 border rounded-lg">
            <div className="flex justify-between items-center">
              <div>
                <h4 className="font-semibold">{item.adapterName}</h4>
                <p className="text-sm text-gray-600">{item.baseModel}</p>
              </div>
              <div className="text-right">
                <span className={`px-2 py-1 rounded text-sm ${
                  item.status === 'completed' ? 'bg-green-100 text-green-800' :
                  item.status === 'running' ? 'bg-blue-100 text-blue-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {item.status}
                </span>
                <p className="text-xs text-gray-500 mt-1">
                  {new Date(item.startTime).toLocaleDateString()}
                </p>
              </div>
            </div>
            {item.improvementScore && (
              <div className="mt-2 text-sm">
                <span className="text-gray-600">Improvement: </span>
                <span className="font-medium">{item.improvementScore}%</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default TrainingTimeline;