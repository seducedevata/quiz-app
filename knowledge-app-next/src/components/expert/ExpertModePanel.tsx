'use client';

import React, { useState, useEffect } from 'react';
import { AppLogger } from '../../lib/logger';
import { callPythonMethod } from '../../lib/pythonBridge';

interface ExpertModeConfig {
  thinkingModel: string;
  jsonModel: string;
  timeout: number;
  complexity: number;
  researchDepth: number;
  enablePipelineVisualization: boolean;
  enableAdvancedMetrics: boolean;
  customPromptTemplate?: string;
}

interface ExpertModePanelProps {
  isEnabled: boolean;
  onConfigChange: (config: ExpertModeConfig) => void;
  onToggle: (enabled: boolean) => void;
}

export const ExpertModePanel: React.FC<ExpertModePanelProps> = ({
  isEnabled,
  onConfigChange,
  onToggle
}) => {
  // Configuration state
  const [config, setConfig] = useState<ExpertModeConfig>({
    thinkingModel: 'deepseek-r1',
    jsonModel: 'llama3.1',
    timeout: 180,
    complexity: 8,
    researchDepth: 3,
    enablePipelineVisualization: true,
    enableAdvancedMetrics: true
  });

  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // Handle configuration changes
  const handleConfigChange = (key: keyof ExpertModeConfig, value: any) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
    onConfigChange(newConfig);
    
    AppLogger.info('EXPERT_MODE', 'Configuration updated', {
      key,
      value,
      newConfig
    });
  };

  return (
    <div className="expert-mode-panel bg-gradient-to-br from-purple-900/20 to-indigo-900/20 border-2 border-purple-500/30 rounded-xl p-6 my-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-3">
          <div className="text-3xl">🧠</div>
          <div>
            <h2 className="text-2xl font-bold text-purple-300">Expert Mode</h2>
            <p className="text-sm text-purple-400">PhD-Level AI Question Generation</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-300">Enable Expert Mode</span>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={isEnabled}
                onChange={(e) => onToggle(e.target.checked)}
              />
              <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
            </label>
          </div>
        </div>
      </div>

      {isEnabled && (
        <div className="text-center py-8 text-gray-400">
          <div className="text-4xl mb-2">🚧</div>
          <p>Expert mode configuration panel in progress</p>
          <p className="text-sm">Advanced settings will be available soon</p>
        </div>
      )}
    </div>
  );
};

export default ExpertModePanel;