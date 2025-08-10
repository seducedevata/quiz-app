import React from 'react';

interface QuestionCardProps {
  question: Question;
  onClick: (question: Question) => void;
}

interface Question {
  id: string;
  question: string;
  answer: string;
  topic: string;
  difficulty: string;
  // Add other relevant question properties
}

export const QuestionCard: React.FC<QuestionCardProps> = ({
  question,
  onClick
}) => {
  return (
    <div className="question-card" onClick={() => onClick(question)}>
      <h3>{question.question}</h3>
      <p>Topic: {question.topic} | Difficulty: {question.difficulty}</p>
    </div>
  );
};
