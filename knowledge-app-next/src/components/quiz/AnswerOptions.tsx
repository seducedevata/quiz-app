'use client';

import React from 'react';
import { Button } from '@/components/common/Button';

interface AnswerOptionsProps {
  options: string[];
  onSelectAnswer: (index: number) => void;
  selectedAnswer: number;
}

export const AnswerOptions: React.FC<AnswerOptionsProps> = ({ options, onSelectAnswer, selectedAnswer }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-md">
      {options.map((option, index) => (
        <Button
          key={index}
          onClick={() => onSelectAnswer(index)}
          className={`btn-answer ${selectedAnswer === index ? 'selected' : ''}`}
        >
          {option}
        </Button>
      ))}
    </div>
  );
};
