'use client';

import React from 'react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';

interface QuizFeedbackProps {
  isCorrect: boolean;
  explanation: string;
  onNext: () => void;
}

export const QuizFeedback: React.FC<QuizFeedbackProps> = ({ isCorrect, explanation, onNext }) => {
  return (
    <Card className="mt-lg">
      <h3 className={`text-h3 font-h3 mb-md ${isCorrect ? 'text-success-color' : 'text-danger-color'}`}>
        {isCorrect ? 'Correct!' : 'Incorrect!'}
      </h3>
      <p className="text-textPrimary mb-lg">{explanation}</p>
      <Button onClick={onNext} className="btn-primary">
        Next Question
      </Button>
    </Card>
  );
};
