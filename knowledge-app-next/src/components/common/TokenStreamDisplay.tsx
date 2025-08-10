import React, { useState, useEffect, useRef } from 'react';
import { onPythonEvent, offPythonEvent } from '../../lib/pythonBridge';
import { AppLogger } from '../../lib/logger';

interface TokenStreamDisplayProps {
  isVisible: boolean;
}

export const TokenStreamDisplay: React.FC<TokenStreamDisplayProps> = ({ isVisible }) => {
  const [streamedText, setStreamedText] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [streamSpeed, setStreamSpeed] = useState<number>(1); // 1x, 2x, 0.5x
  const [tokensPerSecond, setTokensPerSecond] = useState<number>(0);
  const lastTokenTimeRef = useRef<number>(Date.now());
  const tokenCountRef = useRef<number>(0);
  const streamRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isVisible) {
      setStreamedText('');
      setIsStreaming(false);
      setIsPaused(false);
      return;
    }

    const handleToken = (data: { token: string; is_end: boolean }) => {
      if (!isPaused) {
        setStreamedText(prev => prev + data.token);
        // Calculate tokens per second
        tokenCountRef.current += data.token.length; // Assuming each char is a token for simplicity
        const now = Date.now();
        const timeElapsed = (now - lastTokenTimeRef.current) / 1000;
        if (timeElapsed >= 1) { // Update every second
          setTokensPerSecond(tokenCountRef.current / timeElapsed);
          tokenCountRef.current = 0;
          lastTokenTimeRef.current = now;
        }

        if (streamRef.current) {
          streamRef.current.scrollTop = streamRef.current.scrollHeight;
        }
      }
      if (data.is_end) {
        setIsStreaming(false);
        setTokensPerSecond(0); // Reset TPS on stream end
        AppLogger.info('TOKEN_STREAM', 'Token stream ended.');
      }
    };

    const handleStreamStart = () => {
      setStreamedText('');
      setIsStreaming(true);
      setIsPaused(false);
      AppLogger.info('TOKEN_STREAM', 'Token stream started.');
    };

    const handleStreamError = (error: any) => {
      setIsStreaming(false);
      AppLogger.error('TOKEN_STREAM', 'Token stream error.', error);
      setStreamedText(prev => prev + `\n\nError: ${error.message || 'Unknown streaming error'}`);
    };

    onPythonEvent('token_stream_start', handleStreamStart);
    onPythonEvent('token_stream_token', handleToken);
    onPythonEvent('token_stream_error', handleStreamError);

    return () => {
      offPythonEvent('token_stream_start', handleStreamStart);
      offPythonEvent('token_stream_token', handleToken);
      offPythonEvent('token_stream_error', handleStreamError);
    };
  }, [isVisible, isPaused]);

  const togglePause = () => {
    setIsPaused(prev => !prev);
    AppLogger.action('TOKEN_STREAM', `Token stream ${isPaused ? 'resumed' : 'paused'}.`);
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div className="token-stream-display bg-bg-secondary rounded-lg p-4 shadow-md mt-4">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-lg font-semibold text-text-primary">🌊 Live Token Stream</h3>
        <div className="flex items-center space-x-2">
          {isStreaming && (
            <span className="text-sm text-primary-color animate-pulse">Streaming... ({tokensPerSecond.toFixed(2)} tokens/s)</span>
          )}
          <button
            onClick={togglePause}
            className="btn btn-sm btn-secondary"
            disabled={!isStreaming}
          >
            {isPaused ? '▶️ Resume' : '⏸️ Pause'}
          </button>
          <button
            onClick={() => setStreamedText('')}
            className="btn btn-sm btn-secondary"
            title="Clear Stream"
          >
            ✖️ Clear
          </button>
        </div>
      </div>
      <div ref={streamRef} className="token-stream-content bg-bg-primary p-3 rounded-md text-text-secondary text-sm overflow-y-auto h-48 font-mono whitespace-pre-wrap">
        {streamedText || (isStreaming ? 'Waiting for tokens...' : 'Stream idle.')}
      </div>
    </div>
  );
};
