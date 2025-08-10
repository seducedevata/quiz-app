import React from 'react';

interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  color?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'medium',
  color = '#007bff'
}) => {
  const spinnerSize = {
    small: '20px',
    medium: '40px',
    large: '60px',
  };

  const borderWidth = {
    small: '2px',
    medium: '4px',
    large: '6px',
  };

  return (
    <div
      className="loading-spinner"
      style={{
        width: spinnerSize[size],
        height: spinnerSize[size],
        border: `${borderWidth[size]} solid #f3f3f3`,
        borderTop: `${borderWidth[size]} solid ${color}`,
        borderRadius: '50%',
        animation: 'spin 1s linear infinite',
      }}
    ></div>
  );
};
