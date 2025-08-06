'use client';

import React from 'react';
import { FormGroup } from '@/components/common/FormGroup';
import { Dropdown } from '@/components/common/Dropdown';

interface ModelSelectorProps {
  selectedModel: string;
  onSelectModel: (model: string) => void;
  models: { label: string; value: string; }[];
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onSelectModel,
  models,
}) => {
  return (
    <FormGroup label="AI Model">
      <Dropdown
        options={models}
        value={selectedModel}
        onChange={(e) => onSelectModel(e.target.value)}
      />
    </FormGroup>
  );
};