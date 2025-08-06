'use client';

import React, { useEffect } from 'react';
import { useQuizStore } from '@/store/quizStore';

export const QuizTimer: React.FC = () => {
  const { timeRemaining, updateTimer, gameMode } = useQuizStore();

  useEffect(() => {
    if (gameMode === 'serious') {
      const timer = setInterval(() => {
        updateTimer();
      }, 1000);

      return () => clearInterval(timer);
    }
  }, [updateTimer, gameMode]);

  if (gameMode === 'casual') {
    return (
      <div className="quiz-timer casual">
        <span className="timer-icon">🎵</span>
        <span className="timer-text">Casual Mode - No Time Limit</span>
      </div>
    );
  }

  const getTimerClass = () => {
    if (timeRemaining <= 5) return 'quiz-timer critical';
    if (timeRemaining <= 10) return 'quiz-timer warning';
    return 'quiz-timer normal';
  };

  return (
    <div className={getTimerClass()}>
      <span className="timer-icon">⏱️</span>
      <span className="timer-text">
        Time: {Math.floor(timeRemaining / 60)}:{(timeRemaining % 60).toString().padStart(2, '0')}
      </span>
    </div>
  );
};
