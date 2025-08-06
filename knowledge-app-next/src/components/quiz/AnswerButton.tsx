'use client';

import React from 'react';

interface AnswerButtonProps {
  option: string;
  index: number;
  isSelected: boolean;
  onSelect: (index: number) => void;
  isCorrect?: boolean;
  showFeedback?: boolean;
}

export const AnswerButton: React.FC<AnswerButtonProps> = ({
  option,
  index,
  isSelected,
  onSelect,
  isCorrect,
  showFeedback,
}) => {
  let buttonClass = "answer-button";
  if (showFeedback) {
    if (isCorrect) {
      buttonClass += " correct";
    } else if (isSelected) {
      buttonClass += " incorrect";
    }
  } else if (isSelected) {
    buttonClass += " selected";
  }

  return (
    <button className={buttonClass} onClick={() => onSelect(index)} disabled={showFeedback}>
      {option}
    </button>
  );
};