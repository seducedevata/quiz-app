'use client';

import React, { useState, useEffect } from 'react';
import { io, Socket } from 'socket.io-client';

import { AppLogger } from '../../lib/logger';

const QuizGenerationPage: React.FC = () => {
  const [topic, setTopic] = useState<string>('Quantum Physics');
  const [difficulty, setDifficulty] = useState<string>('medium');
  const [numQuestions, setNumQuestions] = useState<number>(3);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [streamingText, setStreamingText] = useState<string>('');
  const [finalQuestions, setFinalQuestions] = useState<any[]>([]);
  const [uiSuggestions, setUiSuggestions] = useState<any>(null);
  const [socket, setSocket] = useState<Socket | null>(null);

  useEffect(() => {
    const getSuggestions = async () => {
      if (topic) {
        try {
          const response = await fetch('http://localhost:8000/api/adapt-ui', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ topic }),
          });

          if (response.ok) {
            const data = await response.json();
            setUiSuggestions(data.suggestions);
          }
        } catch (error) {
          console.error('Failed to get UI suggestions:', error);
        }
      }
    };

    const debounceTimeout = setTimeout(() => {
      getSuggestions();
    }, 500);

    return () => clearTimeout(debounceTimeout);
  }, [topic]);

  useEffect(() => {
    const newSocket = io('http://localhost:8000');
    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('Connected to WebSocket server');
    });

    newSocket.on('mcq-stream', (data) => {
      if (data.type === 'token') {
        setStreamingText(prev => prev + data.data);
      } else if (data.type === 'question_end') {
        setFinalQuestions(prev => [...prev, streamingText]);
        setStreamingText('');
      } else if (data.type === 'stream_end') {
        setLoading(false);
      }
    });

    return () => {
      newSocket.disconnect();
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setStreamingText('');
    setFinalQuestions([]);

    const config = {
      topic,
      difficulty,
      numQuestions,
    };

    if (socket) {
      socket.emit('start-streaming-mcq', config);
      AppLogger.info('QUIZ_GENERATION', 'Generate quiz started', config);
    }
  };

  return (
    <div className="quiz-setup-card">
      <h2>Generate New Quiz (with Streaming)</h2>
      <p>Configure your quiz settings and see the questions generated in real-time.</p>

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

        {uiSuggestions && (
          <div className="suggestions-box">
            <h4>Suggestions for your topic:</h4>
            <h5>Prompt Enhancements:</h5>
            <ul>
              {uiSuggestions.prompt_enhancements.map((prompt: string, index: number) => (
                <li key={index}>{prompt}</li>
              ))}
            </ul>
            <h5>Related Topics:</h5>
            <ul>
              {uiSuggestions.suggested_topics.map((suggestedTopic: string, index: number) => (
                <li key={index}>{suggestedTopic}</li>
              ))}
            </ul>
          </div>
        )}

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

      {loading && (
        <div className="mt-6 p-4 bg-blue-100 text-blue-800 rounded-md">
          <p className="font-semibold">AI is thinking...</p>
          <p className="font-mono whitespace-pre-wrap">{streamingText}</p>
        </div>
      )}

      {finalQuestions.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold">Generated Questions:</h3>
          <ul className="list-disc pl-5">
            {finalQuestions.map((q, i) => (
              <li key={i} className="mt-2">{q}</li>
            ))}
          </ul>
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