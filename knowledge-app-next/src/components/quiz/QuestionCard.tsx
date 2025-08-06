'use client';

import React from 'react';

interface Question {
  id: string;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
  topic: string;
  difficulty: string;
}

interface QuestionCardProps {
  question: Question;
  selectedAnswer: number;
  showFeedback: boolean;
  onAnswerSelect: (index: number) => void;
}

export const QuestionCard: React.FC<QuestionCardProps> = ({
  question,
  selectedAnswer,
  showFeedback,
  onAnswerSelect,
}) => {
  const getAnswerClass = (index: number) => {
    let baseClass = 'answer-option';
    
    if (selectedAnswer === index) {
      baseClass += ' selected';
    }
    
    if (showFeedback) {
      if (index === question.correctAnswer) {
        baseClass += ' correct';
      } else if (selectedAnswer === index && index !== question.correctAnswer) {
        baseClass += ' incorrect';
      }
    }
    
    return baseClass;
  };

  return (
    <div className="question-card">
      <div className="question-header">
        <span className="question-topic">{question.topic}</span>
        <span className="question-difficulty">{question.difficulty}</span>
      </div>
      
      <div className="question-content">
        <h3 className="question-text">{question.question}</h3>
      </div>
      
      <div className="answer-options">
        {question.options.map((option, index) => (
          <button
            key={index}
            className={getAnswerClass(index)}
            onClick={() => onAnswerSelect(index)}
            disabled={showFeedback}
          >
            <span className="option-letter">
              {String.fromCharCode(65 + index)}
            </span>
            <span className="option-text">{option}</span>
            {showFeedback && index === question.correctAnswer && (
              <span className="correct-indicator">✓</span>
            )}
            {showFeedback && selectedAnswer === index && index !== question.correctAnswer && (
              <span className="incorrect-indicator">✗</span>
            )}
          </button>
        ))}
      </div>
      
      {showFeedback && question.explanation && (
        <div className="explanation">
          <h4>Explanation:</h4>
          <p>{question.explanation}</p>
        </div>
      )}
    </div>
  );
};