'use client';

import React, { useState, useEffect, useCallback } from 'react';

interface Option {
  id: string;
  text: string;
}

interface Question {
  id: string;
  questionText: string;
  options: Option[];
  correctAnswerId: string;
  explanation: string;
}

interface QuizInterfaceProps {
  quizData: Question[];
  onQuizComplete: (score: number) => void;
}

const mockQuizData: Question[] = [
  {
    id: '1',
    questionText: 'What is the capital of France?',
    options: [
      { id: 'a', text: 'Berlin' },
      { id: 'b', text: 'Madrid' },
      { id: 'c', text: 'Paris' },
      { id: 'd', text: 'Rome' },
    ],
    correctAnswerId: 'c',
    explanation: 'Paris is the capital and most populous city of France.',
  },
  {
    id: '2',
    questionText: 'Which planet is known as the Red Planet?',
    options: [
      { id: 'a', text: 'Earth' },
      { id: 'b', text: 'Mars' },
      { id: 'c', text: 'Jupiter' },
      { id: 'd', text: 'Venus' },
    ],
    correctAnswerId: 'b',
    explanation: 'Mars is often referred to as the Red Planet because of its reddish appearance, which is caused by iron oxide (rust) on its surface.',
  },
  {
    id: '3',
    questionText: 'What is the chemical symbol for water?',
    options: [
      { id: 'a', text: 'O2' },
      { id: 'b', text: 'H2O' },
      { id: 'c', text: 'CO2' },
      { id: 'd', text: 'NaCl' },
    ],
    correctAnswerId: 'b',
    explanation: 'The chemical symbol for water is H2O, meaning it has two hydrogen atoms and one oxygen atom.',
  },
];

export const QuizInterface: React.FC<QuizInterfaceProps> = ({ quizData = mockQuizData, onQuizComplete }) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [score, setScore] = useState(0);
  const [quizFinished, setQuizFinished] = useState(false);

  const currentQuestion = quizData[currentQuestionIndex];
  const totalQuestions = quizData.length;
  const progress = ((currentQuestionIndex + (quizFinished ? 1 : 0)) / totalQuestions) * 100;

  const handleAnswerSelect = (optionId: string) => {
    if (selectedAnswer) return; // Prevent changing answer after selection
    setSelectedAnswer(optionId);
    setShowExplanation(true);

    if (optionId === currentQuestion.correctAnswerId) {
      setScore((prevScore) => prevScore + 1);
    }
  };

  const handleNextQuestion = useCallback(() => {
    if (currentQuestionIndex < totalQuestions - 1) {
      setCurrentQuestionIndex((prevIndex) => prevIndex + 1);
      setSelectedAnswer(null);
      setShowExplanation(false);
    } else {
      setQuizFinished(true);
      onQuizComplete(score);
    }
  }, [currentQuestionIndex, totalQuestions, score, onQuizComplete]);

  useEffect(() => {
    if (quizFinished) {
      onQuizComplete(score);
    }
  }, [quizFinished, score, onQuizComplete]);

  if (quizFinished) {
    return (
      <div className="quiz-results">
        <h2>Quiz Completed!</h2>
        <p>You scored {score} out of {totalQuestions} questions.</p>
        <button className="btn-primary" onClick={() => window.location.reload()}>Start New Quiz</button>
        <button className="btn-secondary" onClick={() => alert('Implement Review Feature')}>Review Answers</button>
      </div>
    );
  }

  if (!currentQuestion) {
    return <div className="quiz-loading">Loading quiz...</div>;
  }

  return (
    <div className="quiz-interface-container">
      <div className="quiz-progress">
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }}></div>
        </div>
        <span className="question-count">Question {currentQuestionIndex + 1} / {totalQuestions}</span>
        {/* <span className="timer">00:00</span> // Placeholder for timer */}
      </div>

      <div className="question-card">
        <p className="question-text">{currentQuestion.questionText}</p>
        <div className="options-grid">
          {currentQuestion.options.map((option) => (
            <button
              key={option.id}
              className={`option-button
                ${selectedAnswer === option.id ? (option.id === currentQuestion.correctAnswerId ? 'correct' : 'incorrect') : ''}
                ${showExplanation && option.id === currentQuestion.correctAnswerId && selectedAnswer !== option.id ? 'correct' : ''}
              `}
              onClick={() => handleAnswerSelect(option.id)}
              disabled={!!selectedAnswer}
            >
              <span className="option-letter">{option.id.toUpperCase()}</span>
              <span>{option.text}</span>
            </button>
          ))}
        </div>

        {showExplanation && (
          <div className="explanation-section">
            <h3>Explanation</h3>
            <p>{currentQuestion.explanation}</p>
            <button className="btn-primary" onClick={handleNextQuestion}>
              {currentQuestionIndex < totalQuestions - 1 ? 'Next Question' : 'Finish Quiz'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
