'use client';

import React from 'react';
import { Card } from '@/components/common/Card';

export const PipelineVisualization: React.FC = () => {
  return (
    <Card className="mt-lg">
      <h3 className="text-h3 font-h3 text-textPrimary mb-lg">Two-Model Pipeline Visualization</h3>
      <div className="flex items-center justify-center space-x-4">
        <div className="p-4 border border-borderColor rounded-md text-center">
          <p className="font-bold">Model A</p>
          <p className="text-sm text-textSecondary">Initial Generation</p>
        </div>
        <span>➡️</span>
        <div className="p-4 border border-borderColor rounded-md text-center">
          <p className="font-bold">Model B</p>
          <p className="text-sm text-textSecondary">Refinement/Correction</p>
        </div>
      </div>
      <p className="text-sm text-textSecondary mt-md text-center">
        This visualizes the two-model pipeline where one model generates content and another refines it.
      </p>
    </Card>
  );
};