'use client';

import React, { useState, useEffect } from 'react';
import { AppLogger } from '../../lib/logger';

interface PipelineStage {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'error' | 'skipped';
  progress: number;
  startTime?: number;
  endTime?: number;
  duration?: number;
  details?: string;
}

interface PipelineVisualizationProps {
  isActive: boolean;
  onStageUpdate?: (stage: PipelineStage) => void;
}

export const PipelineVisualization: React.FC<PipelineVisualizationProps> = ({
  isActive,
  onStageUpdate
}) => {
  const [stages, setStages] = useState<PipelineStage[]>([
    {
      id: 'document_analysis',
      name: 'Document Analysis',
      description: 'Analyzing uploaded documents and extracting key concepts',
      status: 'pending',
      progress: 0
    },
    {
      id: 'deepseek_thinking',
      name: 'DeepSeek R1 Thinking',
      description: 'Advanced reasoning and concept understanding',
      status: 'pending',
      progress: 0
    },
    {
      id: 'question_generation',
      name: 'Question Generation',
      description: 'Creating PhD-level questions based on analysis',
      status: 'pending',
      progress: 0
    }
  ]);

  const [overallProgress, setOverallProgress] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  if (!isActive) {
    return (
      <div className="pipeline-visualization-disabled text-center py-8 text-gray-500">
        <div className="text-4xl mb-2">🔧</div>
        <p>Pipeline visualization is disabled</p>
        <p className="text-sm">Enable in Expert Mode settings to see real-time progress</p>
      </div>
    );
  }

  return (
    <div className="pipeline-visualization bg-gray-900/50 rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="text-2xl">🔬</div>
          <div>
            <h3 className="text-xl font-bold text-white">DeepSeek Pipeline</h3>
            <p className="text-sm text-gray-400">Two-Model Expert Question Generation</p>
          </div>
        </div>
      </div>

      <div className="text-center py-8 text-gray-400">
        <div className="text-4xl mb-2">🚧</div>
        <p>Pipeline visualization in progress</p>
        <p className="text-sm">Real-time pipeline monitoring will be available soon</p>
      </div>
    </div>
  );
};

export default PipelineVisualization;