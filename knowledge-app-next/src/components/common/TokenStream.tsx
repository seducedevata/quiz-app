import React, { useState, useEffect, useRef } from 'react';

interface TokenStreamProps {
  stream: string[];
}

const TokenStream: React.FC<TokenStreamProps> = ({ stream }) => {
  const [displayedText, setDisplayedText] = useState('');
  const [showCursor, setShowCursor] = useState(true);
  const [isPaused, setIsPaused] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const currentIndexRef = useRef(0);

  useEffect(() => {
    const startStream = () => {
      intervalRef.current = setInterval(() => {
        if (currentIndexRef.current < stream.length) {
          setDisplayedText(prevText => prevText + stream[currentIndexRef.current]);
          currentIndexRef.current++;
        } else {
          if (intervalRef.current) {
            clearInterval(intervalRef.current);
          }
          setShowCursor(false);
        }
      }, 100);
    };

    if (!isPaused) {
      startStream();
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [stream, isPaused]);

  const handlePause = () => {
    setIsPaused(true);
  };

  const handleResume = () => {
    setIsPaused(false);
  };

  const progress = (displayedText.length / stream.join('').length) * 100;

  return (
    <div className="token-stream-container">
      <div className="progress-bar-container">
        <div className="progress-bar" style={{ width: `${progress}%` }}></div>
      </div>
      <p>{displayedText}{showCursor && <span className="cursor">|</span>}</p>
      <div className="stream-controls">
        {isPaused ? (
          <button onClick={handleResume}>Resume</button>
        ) : (
          <button onClick={handlePause}>Pause</button>
        )}
      </div>
    </div>
  );
};

export default TokenStream;