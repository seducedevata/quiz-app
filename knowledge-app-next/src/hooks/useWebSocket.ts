'use client';

import { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { useQuizStore } from '@/store/quizStore';

export const useWebSocket = () => {
  const socketRef = useRef<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const { addStreamingToken, setStreamingStatus } = useQuizStore();

  useEffect(() => {
    // Initialize socket connection
    socketRef.current = io('http://localhost:8000', {
      transports: ['websocket'],
      autoConnect: true,
    });

    const socket = socketRef.current;

    // Connection event handlers
    socket.on('connect', () => {
      console.log('✅ WebSocket connected:', socket.id);
      setIsConnected(true);
      setStreamingStatus('Connected');
    });

    socket.on('disconnect', () => {
      console.log('❌ WebSocket disconnected');
      setIsConnected(false);
      setStreamingStatus('Disconnected');
    });

    socket.on('connect_error', (error) => {
      console.error('❌ WebSocket connection error:', error);
      setIsConnected(false);
      setStreamingStatus('Connection Error');
    });

    // Quiz streaming event handlers
    socket.on('quiz-token', (token: string) => {
      console.log('📡 Received token:', token);
      addStreamingToken(token);
    });

    socket.on('quiz-complete', () => {
      console.log('✅ Quiz generation complete');
      setStreamingStatus('Complete');
    });

    socket.on('quiz-error', (error: string) => {
      console.error('❌ Quiz generation error:', error);
      setStreamingStatus(`Error: ${error}`);
    });

    socket.on('streaming-started', (data: any) => {
      console.log('🌊 Streaming started:', data);
      setStreamingStatus('Generating...');
    });

    // Cleanup on unmount
    return () => {
      if (socket) {
        socket.disconnect();
      }
    };
  }, [addStreamingToken, setStreamingStatus]);

  const startStreamingQuiz = (params: any) => {
    if (socketRef.current && isConnected) {
      console.log('🚀 Starting streaming quiz:', params);
      setStreamingStatus('Initializing...');
      socketRef.current.emit('start-streaming-quiz', params);
      return true;
    } else {
      console.error('❌ WebSocket not connected');
      setStreamingStatus('Not Connected');
      return false;
    }
  };

  const stopStreaming = () => {
    if (socketRef.current) {
      socketRef.current.emit('stop-streaming');
      setStreamingStatus('Stopped');
    }
  };

  return {
    isConnected,
    startStreamingQuiz,
    stopStreaming,
  };
};