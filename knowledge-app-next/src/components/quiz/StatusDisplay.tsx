
'use client';

import React, { useState, useEffect } from 'react';
import { StatusMessage } from '@/components/common/StatusMessage';

interface StatusItem {
  id: number;
  type: 'turbo' | 'gpu' | 'success' | 'warning' | 'error' | 'info' | 'api';
  message: string;
  icon?: string;
  spin?: boolean;
}

export const StatusDisplay: React.FC = () => {
  const [statuses, setStatuses] = useState<StatusItem[]>([
    { id: 1, type: 'info', message: 'App initialized.' },
  ]);

  // Example of how to add a new status message
  const addStatus = (newStatus: Omit<StatusItem, 'id'>) => {
    setStatuses((prevStatuses) => [
      ...prevStatuses,
      { id: prevStatuses.length + 1, ...newStatus },
    ]);
  };

  useEffect(() => {
    // Example: Add a new status after 3 seconds
    const timer = setTimeout(() => {
      addStatus({ type: 'success', message: 'Data loaded successfully!' });
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="fixed bottom-lg right-lg z-50 w-80">
      {statuses.map((status) => (
        <StatusMessage
          key={status.id}
          type={status.type}
          message={status.message}
          icon={status.icon}
          spin={status.spin}
        />
      ))}
    </div>
  );
};
