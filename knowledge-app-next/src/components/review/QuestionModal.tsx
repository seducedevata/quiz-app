import React from 'react';

interface QuestionModalProps {
  question: Question | null;
  onClose: () => void;
}

interface Question {
  id: string;
  question: string;
  answer: string;
  topic: string;
  difficulty: string;
  // Add other relevant question properties
}

export const QuestionModal: React.FC<QuestionModalProps> = ({
  question,
  onClose
}) => {
  if (!question) {
    return null;
  }

  return (
    <div className="question-modal-overlay" onClick={onClose}>
      <div className="question-modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>Question Details</h2>
        <h3>{question.question}</h3>
        <p><strong>Answer:</strong> {question.answer}</p>
        <p><strong>Topic:</strong> {question.topic}</p>
        <p><strong>Difficulty:</strong> {question.difficulty}</p>
        {/* Display other question details as needed */}
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
};
