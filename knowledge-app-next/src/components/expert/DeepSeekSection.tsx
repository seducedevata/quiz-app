
import React from 'react';
import { Card } from '@/components/common/Card';
import { Icon } from '@/components/common/Icon';

export const DeepSeekSection: React.FC = () => {
  return (
    <Card>
      <div className="flex items-center mb-md">
        <Icon name="FaBrain" className="mr-sm" />
        <h3 className="text-h3 font-h3 text-textPrimary">DeepSeek AI Pipeline</h3>
      </div>
      <p className="text-body text-textSecondary mb-sm">🔬 Two-Model Pipeline: DeepSeek R1 thinking + Llama JSON formatting</p>
      <p className="text-body text-textSecondary">🎯 Optimized for expert-level, PhD-quality questions</p>
    </Card>
  );
};
