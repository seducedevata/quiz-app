'use client';

import React, { useEffect, useState, useRef } from 'react';

interface TokenStreamProps {
  quizId: string; // Or any identifier for the stream
  enabled: boolean;
}

export const TokenStream: React.FC<TokenStreamProps> = ({ quizId, enabled }) => {
  const [streamContent, setStreamContent] = useState('');
  const [status, setStatus] = useState('Disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!enabled) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setStatus('Disabled');
      setStreamContent('');
      return;
    }

    // Connect to WebSocket
    const connectWebSocket = () => {
      setStatus('Connecting...');
      console.log(`🔥 ENHANCED DEBUG - Attempting WebSocket connection for quizId: ${quizId}`);
      // Replace with your actual WebSocket endpoint
      const ws = new WebSocket(`ws://localhost:8000/ws/stream/${quizId}`);

      ws.onopen = () => {
        setStatus('Connected');
        console.log('WebSocket Connected');
        console.log(`🔥 ENHANCED DEBUG - WebSocket Connected for quizId: ${quizId}`);
        setStreamContent('');
      };

      ws.onmessage = (event) => {
        const token = event.data;
        console.log(`🔥 ENHANCED DEBUG - Token Received: "${token}" (Session: ${quizId})`);
        setStreamContent((prev) => {
          const newContent = prev + token;
          if (contentRef.current) {
            contentRef.current.scrollTop = contentRef.current.scrollHeight;
          }
          return newContent;
        });
      };

      ws.onclose = () => {
        setStatus('Disconnected');
        console.log('WebSocket Disconnected');
        console.log(`🔥 ENHANCED DEBUG - WebSocket Disconnected for quizId: ${quizId}`);
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (error) => {
        setStatus('Error');
        console.error('WebSocket Error:', error);
        console.error(`🔥 ENHANCED DEBUG - WebSocket Error for quizId: ${quizId}:`, error);
        ws.close();
      };

      wsRef.current = ws;
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        console.log(`🔥 ENHANCED DEBUG - Closing WebSocket for quizId: ${quizId} on component unmount/dependency change`);
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [quizId, enabled]);

  useEffect(() => {
    console.log(`🔥 ENHANCED DEBUG - TokenStream component props changed: enabled=${enabled}, quizId=${quizId}`);
    // Equivalent of originalShouldUseTokenStreaming logic would be here, determining if streaming should be active
    // Equivalent of originalCreateTokenStreamUI logic would be handled by React's rendering based on 'enabled' prop
  }, [enabled, quizId]);

  return (
    <div className="token-stream-container">
      <div className="stream-header">
        <h3>AI Thinking Process (Token Stream)</h3>
        <span className="stream-status">Status: {status}</span>
      </div>
      <div ref={contentRef} className="stream-content">
        {streamContent ? (
          <span className="typewriter-text">{streamContent}</span>
        ) : (
          <p className="text-gray-500">{enabled ? 'Waiting for AI to start thinking...' : 'Token streaming is disabled.'}</p>
        )}
      </div>
    </div>
  );
};
