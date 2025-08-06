'use client';

import React, { createContext, useContext } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';

const WebSocketContext = createContext<any>(null);

export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const webSocket = useWebSocket();

  return (
    <WebSocketContext.Provider value={webSocket}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocketContext = () => {
  return useContext(WebSocketContext);
};