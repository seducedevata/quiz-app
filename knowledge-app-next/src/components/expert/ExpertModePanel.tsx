'use client';

import React, { useState } from 'react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { FormGroup } from '@/components/common/FormGroup';
import { Dropdown } from '@/components/common/Dropdown';

const modelOptions = [
  { label: 'DeepSeek R1 14B', value: 'deepseek-r1:14b' },
  { label: 'DeepSeek R1 32B', value: 'deepseek-r1:32b' },
  { label: 'Llama 3.1 70B', value: 'llama3.1:70b' },
];

export const ExpertModePanel: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState('deepseek-r1:14b');
  const [enablePipeline, setEnablePipeline] = useState(false);
  const [customPrompt, setCustomPrompt] = useState('');

  return (
    <Card className="mt-lg">
      <h3 className="text-h3 font-h3 text-textPrimary mb-lg">Expert Mode Configuration</h3>
      
      <FormGroup label="AI Model">
        <Dropdown
          options={modelOptions}
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
        />
      </FormGroup>

      <FormGroup label="Two-Model Pipeline">
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={enablePipeline}
            onChange={(e) => setEnablePipeline(e.target.checked)}
            className="mr-sm"
          />
          <span className="text-textPrimary">Enable advanced two-model pipeline</span>
        </label>
      </FormGroup>

      <FormGroup label="Custom Prompt Template">
        <textarea
          value={customPrompt}
          onChange={(e) => setCustomPrompt(e.target.value)}
          placeholder="Enter custom prompt template..."
          className="w-full p-md border border-borderColor rounded-md bg-bgSecondary text-textPrimary"
          rows={4}
        />
      </FormGroup>

      <Button
        onClick={() => {/* Apply settings */}}
        className="mt-lg"
      >
        Apply Expert Settings
      </Button>
    </Card>
  );
};
