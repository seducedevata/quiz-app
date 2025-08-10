'use client';


export default function Home() {
  return (
    <div className="screen active">
      <div className="welcome-card">
        <h2>Welcome to Knowledge App</h2>
        <p>Test your knowledge with AI-powered quizzes</p>
        <div className="stats-grid">
          <div className="stat-card">
            <h3>0</h3>
            <p>Quizzes Taken</p>
          </div>
          <div className="stat-card">
            <h3>0%</h3>
            <p>Average Score</p>
          </div>
          <div className="stat-card">
            <h3>0</h3>
            <p>Questions Answered</p>
          </div>
        </div>
        {/* Dashboard Analytics - Use sidebar to navigate to Quiz */}
        <div className="dashboard-info">
          <p>📊 View your learning progress and statistics above</p>
          <p>🎯 Use the sidebar to start a new quiz or review questions</p>
        </div>
      </div>
    </div>
  );
}
