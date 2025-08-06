
import React from 'react';
import { MathJax } from 'better-react-mathjax';

interface QuestionDisplayProps {
  question: string;
}

export const QuestionDisplay: React.FC<QuestionDisplayProps> = ({ question }) => {
  return (
    <div className="text-h2 font-h2 text-textPrimary mb-lg">
      <MathJax>{question}</MathJax>
    </div>
  );
};
