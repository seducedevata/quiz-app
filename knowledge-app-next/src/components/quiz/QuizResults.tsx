'use client';

import React from 'react';
import { useQuizStore } from '@/store/quizStore';

interface QuizResultsProps {
  onRestart: () => void;
}

export const QuizResults: React.FC<QuizResultsProps> = ({ onRestart }) => {
  const { score, questions, topic, difficulty } = useQuizStore();
  
  const percentage = Math.round((score / questions.length) * 100);
  
  const getPerformanceMessage = () => {
    if (percentage >= 90) return { message: "Outstanding! 🏆", class: "excellent" };
    if (percentage >= 80) return { message: "Great job! 🎉", class: "good" };
    if (percentage >= 70) return { message: "Well done! 👍", class: "average" };
    if (percentage >= 60) return { message: "Not bad! 📚", class: "below-average" };
    return { message: "Keep practicing! 💪", class: "needs-improvement" };
  };

  const performance = getPerformanceMessage();

  return (
    <div className="quiz-results">
      <div className="results-header">
        <h2>Quiz Complete!</h2>
        <p className="quiz-info">
          Topic: {topic} | Difficulty: {difficulty}
        </p>
      </div>
      
      <div className="score-circle">
        <div className={`circle ${performance.class}`}>
          <span className="percentage">{percentage}%</span>
        </div>
      </div>
      
      <div className="score-details">
        <h3 className={`performance-message ${performance.class}`}>
          {performance.message}
        </h3>
        <p className="score-text">
          You scored {score} out of {questions.length} questions correctly
        </p>
      </div>
      
      <div className="results-actions">
        <button className="btn-primary" onClick={onRestart}>
          🔄 New Quiz
        </button>
        <button className="btn-secondary" onClick={() => window.location.href = '/review'}>
          📚 Review Questions
        </button>
      </div>
      
      <div className="results-breakdown">
        <h4>Question Breakdown:</h4>
        <div className="breakdown-list">
          {questions.map((question, index) => (
            <div key={question.id} className="breakdown-item">
              <span className="question-number">Q{index + 1}</span>
              <span className="question-topic">{question.topic}</span>
              <span className={`question-result ${index < score ? 'correct' : 'incorrect'}`}>
                {index < score ? '✓' : '✗'}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
