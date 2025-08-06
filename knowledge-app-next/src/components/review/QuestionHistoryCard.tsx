'use client';

import React, { useState } from 'react';

interface Question {
  id: string;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
  topic: string;
  difficulty: string;
  timestamp: string;
  userAnswer?: number;
  isCorrect?: boolean;
}

interface QuestionHistoryCardProps {
  question: Question;
}

export const QuestionHistoryCard: React.FC<QuestionHistoryCardProps> = ({ question }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'easy': return '#10b981';
      case 'medium': return '#f59e0b';
      case 'hard': return '#ef4444';
      case 'expert': return '#8b5cf6';
      default: return '#6b7280';
    }
  };

  const getResultIcon = () => {
    if (question.userAnswer === undefined) return '❓';
    return question.isCorrect ? '✅' : '❌';
  };

  const getResultClass = () => {
    if (question.userAnswer === undefined) return 'unanswered';
    return question.isCorrect ? 'correct' : 'incorrect';
  };

  return (
    <div className={`question-history-card ${getResultClass()}`}>
      {/* Card Header */}
      <div className="card-header">
        <div className="question-meta">
          <span className="topic-badge">{question.topic}</span>
          <span 
            className="difficulty-badge"
            style={{ backgroundColor: getDifficultyColor(question.difficulty) }}
          >
            {question.difficulty}
          </span>
        </div>
        <div className="question-result">
          <span className="result-icon">{getResultIcon()}</span>
          <span className="timestamp">{formatDate(question.timestamp)}</span>
        </div>
      </div>

      {/* Question Preview */}
      <div className="question-preview">
        <h3 className="question-text">
          {question.question.length > 100 && !isExpanded
            ? `${question.question.substring(0, 100)}...`
            : question.question
          }
        </h3>
      </div>

      {/* Expand/Collapse Button */}
      <button
        className="expand-btn"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {isExpanded ? '🔼 Show Less' : '🔽 Show More'}
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="expanded-content">
          {/* Answer Options */}
          <div className="answer-options">
            <h4>Answer Options:</h4>
            {question.options.map((option, index) => (
              <div
                key={index}
                className={`option-item ${
                  index === question.correctAnswer ? 'correct-option' : ''
                } ${
                  index === question.userAnswer && index !== question.correctAnswer ? 'user-wrong' : ''
                } ${
                  index === question.userAnswer && index === question.correctAnswer ? 'user-correct' : ''
                }`}
              >
                <span className="option-letter">
                  {String.fromCharCode(65 + index)}
                </span>
                <span className="option-text">{option}</span>
                {index === question.correctAnswer && (
                  <span className="correct-indicator">✓ Correct</span>
                )}
                {index === question.userAnswer && index !== question.correctAnswer && (
                  <span className="user-indicator">Your Answer</span>
                )}
              </div>
            ))}
          </div>

          {/* Explanation */}
          {question.explanation && (
            <div className="explanation-section">
              <h4>Explanation:</h4>
              <p className="explanation-text">{question.explanation}</p>
            </div>
          )}

          {/* Performance Summary */}
          <div className="performance-summary">
            <div className="summary-item">
              <span className="summary-label">Your Answer:</span>
              <span className="summary-value">
                {question.userAnswer !== undefined 
                  ? `${String.fromCharCode(65 + question.userAnswer)} - ${question.options[question.userAnswer]}`
                  : 'Not answered'
                }
              </span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Correct Answer:</span>
              <span className="summary-value">
                {String.fromCharCode(65 + question.correctAnswer)} - {question.options[question.correctAnswer]}
              </span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Result:</span>
              <span className={`summary-value ${getResultClass()}`}>
                {question.userAnswer === undefined 
                  ? 'Unanswered' 
                  : question.isCorrect ? 'Correct' : 'Incorrect'
                }
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};