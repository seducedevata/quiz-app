'use client';

import React from 'react';

interface StatusMessageProps {
  message: string;
  type?: 'success' | 'error' | 'info' | 'warning' | 'turbo' | 'gpu' | 'api';
  icon?: string;
  spin?: boolean;
}

export const StatusMessage: React.FC<StatusMessageProps> = ({
  message,
  type = 'info',
}) => {
  let icon = '';
  let className = 'status-message-container';

  switch (type) {
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
    default:
      break;
  }

  return (
    <div className={className}>
      <span className="status-icon">{icon}</span>
      <span className="message-text">{message}</span>
    </div>
  );
};