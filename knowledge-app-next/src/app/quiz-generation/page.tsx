'use client';

import React, { useState } from 'react';
import { callPythonMethod } from '../../lib/pythonBridge';

const QuizGenerationPage: React.FC = () => {
  const [topic, setTopic] = useState<string>('');
  const [difficulty, setDifficulty] = useState<string>('medium');
  const [numQuestions, setNumQuestions] = useState<number>(5);
  const [quizId, setQuizId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setQuizId(null);

    const config = {
      topic,
      difficulty,
      numQuestions,
      // Add other parameters as needed based on your Python backend's generate_mcq_quiz expected config
      questionType: 'mixed',
      mode: 'auto',
      gameMode: 'casual',
      timer: '30s',
    };

    try {
      const generatedQuizId = await callPythonMethod<string>('generate_mcq_quiz', config);
      setQuizId(generatedQuizId);
      console.log('Generated Quiz ID:', generatedQuizId);
    } catch (err: any) {
      setError(err.message || 'Failed to generate quiz.');
      console.error('Quiz generation error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="quiz-setup-card">
      <h2>Generate New Quiz</h2>
      <p>Configure your quiz settings below.</p>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="topic">Topic</label>
          <input
            type="text"
            id="topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g., Quantum Physics, World History"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="difficulty">Difficulty</label>
          <select
            id="difficulty"
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
          >
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
            <option value="expert">Expert</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="numQuestions">Number of Questions</label>
          <input
            type="number"
            id="numQuestions"
            value={numQuestions}
            onChange={(e) => setNumQuestions(parseInt(e.target.value))}
            min="1"
            max="20"
            required
          />
        </div>

        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? 'Generating...' : 'Generate Quiz'}
        </button>
      </form>

      {quizId && (
        <div className="mt-6 p-4 bg-green-100 text-green-800 rounded-md">
          <p className="font-semibold">Quiz Generated Successfully!</p>
          <p>Quiz ID: <span className="font-mono">{quizId}</span></p>
          <p>You can now proceed to take the quiz.</p>
        </div>
      )}

      {error && (
        <div className="mt-6 p-4 bg-red-100 text-red-800 rounded-md">
          <p className="font-semibold">Error:</p>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

export default QuizGenerationPage;
