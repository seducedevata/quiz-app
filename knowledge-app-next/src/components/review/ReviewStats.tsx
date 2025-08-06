'use client';

import React, { useEffect, useState } from 'react';

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

interface ReviewStatsProps {
  questions: Question[];
  onClose: () => void;
}

export const ReviewStats: React.FC<ReviewStatsProps> = ({ questions, onClose }) => {
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [correctAnswers, setCorrectAnswers] = useState(0);
  const [incorrectAnswers, setIncorrectAnswers] = useState(0);
  const [unansweredQuestions, setUnansweredQuestions] = useState(0);
  const [averageScore, setAverageScore] = useState(0);
  const [topicBreakdown, setTopicBreakdown] = useState<Record<string, { total: number; correct: number }>>({});
  const [difficultyBreakdown, setDifficultyBreakdown] = useState<Record<string, { total: number; correct: number }>>({});

  useEffect(() => {
    calculateStats();
  }, [questions]);

  const calculateStats = () => {
    let total = questions.length;
    let correct = 0;
    let incorrect = 0;
    let unanswered = 0;
    const topics: Record<string, { total: number; correct: number }> = {};
    const difficulties: Record<string, { total: number; correct: number }> = {};

    questions.forEach(q => {
      if (q.userAnswer !== undefined) {
        if (q.isCorrect) {
          correct++;
        } else {
          incorrect++;
        }
      } else {
        unanswered++;
      }

      // Topic breakdown
      if (!topics[q.topic]) {
        topics[q.topic] = { total: 0, correct: 0 };
      }
      topics[q.topic].total++;
      if (q.isCorrect) {
        topics[q.topic].correct++;
      }

      // Difficulty breakdown
      if (!difficulties[q.difficulty]) {
        difficulties[q.difficulty] = { total: 0, correct: 0 };
      }
      difficulties[q.difficulty].total++;
      if (q.isCorrect) {
        difficulties[q.difficulty].correct++;
      }
    });

    setTotalQuestions(total);
    setCorrectAnswers(correct);
    setIncorrectAnswers(incorrect);
    setUnansweredQuestions(unanswered);
    setAverageScore(total > 0 ? (correct / total) * 100 : 0);
    setTopicBreakdown(topics);
    setDifficultyBreakdown(difficulties);
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content review-stats-modal">
        <div className="modal-header">
          <h2>📊 Quiz Statistics</h2>
          <button className="close-button" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <div className="stats-grid">
            <div className="stat-card-small">
              <h3>{totalQuestions}</h3>
              <p>Total Questions</p>
            </div>
            <div className="stat-card-small">
              <h3>{correctAnswers}</h3>
              <p>Correct Answers</p>
            </div>
            <div className="stat-card-small">
              <h3>{incorrectAnswers}</h3>
              <p>Incorrect Answers</p>
            </div>
            <div className="stat-card-small">
              <h3>{unansweredQuestions}</h3>
              <p>Unanswered</p>
            </div>
            <div className="stat-card-small">
              <h3>{averageScore.toFixed(1)}%</h3>
              <p>Average Score</p>
            </div>
          </div>

          <div className="breakdown-section">
            <h3>Topic Breakdown</h3>
            {Object.entries(topicBreakdown).length > 0 ? (
              <ul className="breakdown-list">
                {Object.entries(topicBreakdown).map(([topic, data]) => (
                  <li key={topic}>
                    <span>{topic}:</span>
                    <span>{data.correct}/{data.total} ({((data.correct / data.total) * 100).toFixed(1)}%)</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p>No topic data available.</p>
            )}
          </div>

          <div className="breakdown-section">
            <h3>Difficulty Breakdown</h3>
            {Object.entries(difficultyBreakdown).length > 0 ? (
              <ul className="breakdown-list">
                {Object.entries(difficultyBreakdown).map(([difficulty, data]) => (
                  <li key={difficulty}>
                    <span>{difficulty}:</span>
                    <span>{data.correct}/{data.total} ({((data.correct / data.total) * 100).toFixed(1)}%)</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p>No difficulty data available.</p>
            )}
          </div>
        </div>
        <div className="modal-footer">
          <button className="btn-primary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
};