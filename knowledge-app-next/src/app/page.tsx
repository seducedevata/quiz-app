'use client';

import React from 'react';

export default function Home() {
  return (
    <div className="welcome-card">
      <h2>Welcome to Knowledge App</h2>
      <p>Test your knowledge with AI-powered quizzes</p>
      
      <div className="stats-grid">
        <div className="stat-item">
          <div className="value" data-stat="quizzes">0</div>
          <div className="label">Quizzes Taken</div>
        </div>
        <div className="stat-item">
          <div className="value" data-stat="score">0%</div>
          <div className="label">Average Score</div>
        </div>
        <div className="stat-item">
          <div className="value" data-stat="questions">0</div>
          <div className="label">Questions Answered</div>
        </div>
      </div>
      
      <div className="dashboard-info">
        <p>📊 View your learning progress and statistics above</p>
        <p>🎯 Use the sidebar to start a new quiz or review questions</p>
      </div>
    </div>
  );
}
