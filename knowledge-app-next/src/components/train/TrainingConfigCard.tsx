import React from 'react';

interface TrainingConfig {
  modelType: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
  adapterName?: string;
  baseModel?: string;
  trainingPreset?: string;
  selectedFiles?: string[];
}

interface TrainingConfigCardProps {
  config: TrainingConfig;
  availableModels: string[];
  showAdvanced: boolean;
  isTraining: boolean;
  onConfigChange: (config: TrainingConfig) => void;
  onToggleAdvanced: () => void;
}

export const TrainingConfigCard: React.FC<TrainingConfigCardProps> = ({
  config,
  availableModels,
  showAdvanced,
  isTraining,
  onConfigChange,
  onToggleAdvanced
}) => {
  const updateConfig = (updates: Partial<TrainingConfig>) => {
    onConfigChange({ ...config, ...updates });
  };

  return (
    <div className="training-config">
      <div className="config-header">
        <h3>⚙️ Training Configuration</h3>
        <button 
          className="btn-secondary"
          onClick={onToggleAdvanced}
          disabled={isTraining}
        >
          {showAdvanced ? '📋 Basic' : '🔧 Advanced'}
        </button>
      </div>
      
      <div className="config-grid">
        <div className="config-item">
          <label>Base Model:</label>
          <select 
            value={config.baseModel}
            onChange={(e) => updateConfig({ baseModel: e.target.value })}
            disabled={isTraining}
          >
            {availableModels.map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>
        
        <div className="config-item">
          <label>Adapter Name:</label>
          <input 
            type="text" 
            value={config.adapterName}
            onChange={(e) => updateConfig({ adapterName: e.target.value })}
            placeholder="custom_adapter"
            disabled={isTraining}
          />
        </div>
        
        <div className="config-item">
          <label>Training Preset:</label>
          <select 
            value={config.trainingPreset}
            onChange={(e) => updateConfig({ trainingPreset: e.target.value })}
            disabled={isTraining}
          >
            <option value="standard">Standard</option>
            <option value="fast">Fast</option>
            <option value="quality">High Quality</option>
            <option value="expert">Expert</option>
          </select>
        </div>
        
        <div className="config-item">
          <label>Epochs:</label>
          <input 
            type="number" 
            value={config.epochs}
            onChange={(e) => updateConfig({ epochs: parseInt(e.target.value) })}
            min="1" 
            max="10"
            disabled={isTraining}
          />
        </div>
      </div>
      
      {showAdvanced && (
        <div className="advanced-config">
          <h4>🔧 Advanced Settings</h4>
          <div className="config-grid">
            <div className="config-item">
              <label>Batch Size:</label>
              <input 
                type="number" 
                value={config.batchSize}
                onChange={(e) => updateConfig({ batchSize: parseInt(e.target.value) })}
                min="1" 
                max="32"
                disabled={isTraining}
              />
            </div>
            
            <div className="config-item">
              <label>Learning Rate:</label>
              <input 
                type="number" 
                step="0.0001"
                value={config.learningRate}
                onChange={(e) => updateConfig({ learningRate: parseFloat(e.target.value) })}
                min="0.0001" 
                max="0.01"
                disabled={isTraining}
              />
            </div>
            
            <div className="config-item">
              <label>Model Type:</label>
              <select 
                value={config.modelType}
                onChange={(e) => updateConfig({ modelType: e.target.value })}
                disabled={isTraining}
              >
                <option value="text-generation">Text Generation</option>
                <option value="question-answering">Question Answering</option>
                <option value="classification">Classification</option>
              </select>
            </div>
          </div>
          
          <div className="advanced-info">
            <p className="info-text">
              💡 <strong>Tip:</strong> Lower learning rates (0.0001-0.001) are generally safer for fine-tuning. 
              Higher batch sizes require more GPU memory but may train faster.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};