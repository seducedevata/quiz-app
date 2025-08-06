'use client';

import React from 'react';
import { Card } from '@/components/common/Card';
import { FormGroup } from '@/components/common/FormGroup';
import { Input } from '@/components/common/Input';

export const DeepSeekIntegration: React.FC = () => {
  return (
    <Card className="mt-lg">
      <h3 className="text-h3 font-h3 text-textPrimary mb-lg">DeepSeek Integration</h3>
      
      <FormGroup label="DeepSeek API Key">
        <Input type="password" placeholder="Enter DeepSeek API Key" />
      </FormGroup>

      <FormGroup label="Custom Endpoint">
        <Input type="text" placeholder="Enter custom DeepSeek endpoint (optional)" />
      </FormGroup>
    </Card>
  );
};