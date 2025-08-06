'use client';

import React from 'react';
import { Card } from '@/components/common/Card';

interface StatisticsPanelProps {
  totalQuizzes: number;
  averageScore: number;
  totalQuestionsAnswered: number;
}

export const StatisticsPanel: React.FC<StatisticsPanelProps> = ({
  totalQuizzes,
  averageScore,
  totalQuestionsAnswered,
}) => {
  return (
    <Card className="mt-lg">
      <h3 className="text-h3 font-h3 text-textPrimary mb-lg">Overall Statistics</h3>
      <div className="stats-grid">
        <div className="stat-card">
          <h3>{totalQuizzes}</h3>
          <p>Quizzes Taken</p>
        </div>
        <div className="stat-card">
          <h3>{averageScore.toFixed(2)}%</h3>
          <p>Average Score</p>
        </div>
        <div className="stat-card">
          <h3>{totalQuestionsAnswered}</h3>
          <p>Questions Answered</p>
        </div>
      </div>
    </Card>
  );
};