'use client';

import React from 'react';

interface ProgressBarProps {
  progress: number; // 0-100
  label?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ progress, label }) => {
  return (
    <div className="progress-bar-container">
      {label && <span className="progress-bar-label">{label}</span>}
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${progress}%` }}
        ></div>
      </div>
      <span className="progress-percentage">{progress.toFixed(0)}%</span>
    </div>
  );
};
