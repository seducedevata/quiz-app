'use client';

import React from 'react';

interface QuizProgressProps {
  current: number;
  total: number;
}

export const QuizProgress: React.FC<QuizProgressProps> = ({ current, total }) => {
  const progress = (current / total) * 100;

  return (
    <div className="quiz-progress">
      <div className="progress-bar-container">
        <div className="progress-bar" style={{ width: `${progress}%` }}></div>
      </div>
      <div className="progress-text">Question {current} of {total}</div>
    </div>
  );
};
