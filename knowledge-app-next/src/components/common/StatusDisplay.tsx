'use client';

import React from 'react';

interface StatusDisplayProps {
  status: 'success' | 'error' | 'info' | 'warning' | 'loading';
  message: string;
}

export const StatusDisplay: React.FC<StatusDisplayProps> = ({
  status,
  message,
}) => {
  let icon = '';
  let className = 'status-display';

  switch (status) {
    case 'success':
      icon = '✅';
      className += ' status-success';
      break;
    case 'error':
      icon = '❌';
      className += ' status-error';
      break;
    case 'info':
      icon = 'ℹ️';
      className += ' status-info';
      break;
    case 'warning':
      icon = '⚠️';
      className += ' status-warning';
      break;
    case 'loading':
      icon = '⏳';
      className += ' status-loading';
      break;
    default:
      break;
  }

  return (
    <div className={className}>
      <span className="status-icon">{icon}</span>
      <span className="status-message">{message}</span>
    </div>
  );
};
