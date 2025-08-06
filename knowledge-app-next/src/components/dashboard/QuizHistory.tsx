'use client';

import React from 'react';
import { Card } from '@/components/common/Card';

interface QuizHistoryProps {
  // This would typically take an array of past quiz results
  history?: any[]; 
}

export const QuizHistory: React.FC<QuizHistoryProps> = ({ history }) => {
  return (
    <Card className="mt-lg">
      <h3 className="text-h3 font-h3 text-textPrimary mb-lg">Quiz History</h3>
      {history && history.length > 0 ? (
        <div>
          {/* Render historical quiz data here */}
          <p>Displaying {history.length} past quizzes.</p>
        </div>
      ) : (
        <p className="text-textSecondary">No quiz history available yet. Start a quiz to see your progress!</p>
      )}
    </Card>
  );
};
