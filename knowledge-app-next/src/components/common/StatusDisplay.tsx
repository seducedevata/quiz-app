import React, { useState, useEffect } from 'react';

interface StatusDisplayProps {
  message: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'debug';
  details?: string;
  onClose?: () => void;
  duration?: number; // in milliseconds, 0 for persistent
}

export const StatusDisplay: React.FC<StatusDisplayProps> = ({ message, type, details, onClose, duration = 5000 }) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(() => {
        setIsVisible(false);
        if (onClose) onClose();
      }, duration);
      return () => clearTimeout(timer);
    }
  }, [duration, onClose]);

  const getStatusIcon = () => {
    const icons: { [key: string]: string } = {
      info: 'ℹ️',
      success: '✅',
      warning: '⚠️',
      error: '❌',
      debug: '🐛',
    };
    return icons[type] || 'ℹ️';
  };

  const getBackgroundColor = () => {
    switch (type) {
      case 'info': return 'bg-blue-500';
      case 'success': return 'bg-green-500';
      case 'warning': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
      case 'debug': return 'bg-purple-500';
      default: return 'bg-gray-500';
    }
  };

  if (!isVisible) return null;

  return (
    <div className={`fixed bottom-4 right-4 p-4 rounded-lg shadow-lg text-white flex items-center space-x-3 z-50 ${getBackgroundColor()}`}>
      <span className="text-2xl">{getStatusIcon()}</span>
      <div className="flex-1">
        <p className="font-semibold text-lg">{message}</p>
        {details && <p className="text-sm opacity-90">{details}</p>}
      </div>
      {onClose && (
        <button
          onClick={() => {
            setIsVisible(false);
            onClose();
          }}
          className="ml-4 text-white hover:text-gray-200 focus:outline-none"
        >
          &times;
        </button>
      )}
    </div>
  );
};