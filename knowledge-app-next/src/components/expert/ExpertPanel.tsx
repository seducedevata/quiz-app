import React from 'react';
import { Card } from '@/components/common/Card';
import { ModelSelector } from './ModelSelector';
import { DeepSeekSection } from './DeepSeekSection';

export const ExpertPanel: React.FC = () => {
  const models = [
    { label: 'GPT-4', value: 'gpt-4' },
    { label: 'Claude 3.5 Sonnet', value: 'claude-3-5-sonnet' },
    { label: 'Gemini Pro', value: 'gemini-pro' },
  ];

  return (
    <div className="max-w-5xl mx-auto">
      <h1 className="text-h1 font-h1 text-textPrimary mb-xl">Expert Mode</h1>

      <Card className="mb-lg">
        <h2 className="text-h2 font-h2 text-textPrimary mb-lg">Model Selection</h2>
        <ModelSelector selectedModel="gpt-4" onSelectModel={() => {}} models={models} />
      </Card>

      <DeepSeekSection />
    </div>
  );
};