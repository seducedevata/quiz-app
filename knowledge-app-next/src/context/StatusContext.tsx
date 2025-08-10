import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { AppLogger } from '@/lib/logger';

interface StatusMessage {
  id: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'debug';
  details?: string;
  timestamp: Date;
  duration?: number; // in milliseconds, 0 for persistent
}

interface StatusContextType {
  showStatus: (message: string, type?: StatusMessage['type'], details?: string, duration?: number) => void;
  clearStatus: (id: string) => void;
  statusHistory: StatusMessage[];
}

const StatusContext = createContext<StatusContextType | undefined>(undefined);

interface StatusProviderProps {
  children: ReactNode;
}

export const StatusProvider: React.FC<StatusProviderProps> = ({ children }) => {
  const [statusHistory, setStatusHistory] = useState<StatusMessage[]>([]);

  const showStatus = useCallback((message: string, type: StatusMessage['type'] = 'info', details?: string, duration?: number) => {
    const id = `status_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const newMessage: StatusMessage = {
      id,
      message,
      type,
      details,
      timestamp: new Date(),
      duration,
    };
    setStatusHistory(prev => {
      const updatedHistory = [...prev, newMessage];
      // Keep only the last 20 status messages in history
      return updatedHistory.slice(Math.max(updatedHistory.length - 20, 0));
    });
    AppLogger.info('STATUS_CONTEXT', `New status: [${type}] ${message}`, { id, details });
  }, []);

  const clearStatus = useCallback((id: string) => {
    setStatusHistory(prev => prev.filter(msg => msg.id !== id));
    AppLogger.info('STATUS_CONTEXT', `Status message cleared: ${id}`);
  }, []);

  return (
    <StatusContext.Provider value={{ showStatus, clearStatus, statusHistory }}>
      {children}
    </StatusContext.Provider>
  );
};

export const useStatus = () => {
  const context = useContext(StatusContext);
  if (context === undefined) {
    throw new Error('useStatus must be used within a StatusProvider');
  }
  return context;
};
