'use client';

import React from 'react';

interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  color?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'medium',
  color = 'var(--primary-color)',
}) => {
  const spinnerSize = {
    small: '20px',
    medium: '40px',
    large: '60px',
  }[size];

  const borderWidth = {
    small: '2px',
    medium: '4px',
    large: '6px',
  }[size];

  return (
    <div
      className="loading-spinner"
      style={{
        width: spinnerSize,
        height: spinnerSize,
        borderTopColor: color,
        borderRightColor: color,
        borderBottomColor: color,
        borderWidth: borderWidth,
      }}
    ></div>
  );
};
