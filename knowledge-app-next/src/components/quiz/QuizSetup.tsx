'use client';

import React, { useState } from 'react';
import { useQuizStore } from '@/store/quizStore';
import { StatusDisplay } from '@/components/common/StatusDisplay';
import { ExpertModePanel } from '@/components/expert/ExpertModePanel';

const modeOptions = [
  { label: 'Offline (Local AI - TURBO)', value: 'offline' },
  { label: 'Auto (Best Available)', value: 'auto' },
  { label: 'Online (Cloud APIs)', value: 'online' },
];

const gameModeOptions = [
  { label: '🎵 Casual Mode (Relaxed, Music)', value: 'casual' },
  { label: '⏱️ Serious Mode (Timed, Focused)', value: 'serious' },
];

const questionTypeOptions = [
  { label: '🔀 Mixed (Balanced)', value: 'mixed' },
  { label: '📊 Numerical (Math & Calculations)', value: 'numerical' },
  { label: '🧠 Conceptual (Theory & Understanding)', value: 'conceptual' },
];

const difficultyOptions = [
  { label: '🟢 Easy', value: 'easy' },
  { label: '🟡 Medium', value: 'medium' },
  { label: '🔴 Hard', value: 'hard' },
  { label: '🔥💀 EXPERT (PhD-Level) 💀🔥', value: 'expert' },
];

export const QuizSetup: React.FC = () => {
  const [topic, setTopic] = useState('');
  const [mode, setMode] = useState('auto');
  const [gameMode, setGameMode] = useState('casual');
  const [questionType, setQuestionType] = useState('mixed');
  const [difficulty, setDifficulty] = useState('medium');
  const [numQuestions, setNumQuestions] = useState('5');
  const [tokenStreaming, setTokenStreaming] = useState(true);

  const { setIsLoading, startQuiz } = useQuizStore();

  const handleStartQuiz = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/generate-quiz', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          topic,
          difficulty,
          num_questions: parseInt(numQuestions),
          mode,
          submode: questionType, // Using questionType as submode
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      
      // Assuming data.questions is an array of Question objects
      startQuiz(data.questions);
    } catch (error) {
      console.error('Failed to generate quiz:', error);
      // Handle error state in UI, e.g., show a toast message
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="quiz-setup screen">
      <h2>Quiz Setup</h2>
      
      <div className="form-group">
        <label>Topic</label>
        <input
          type="text"
          placeholder="Enter topic (e.g., Science, History)"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
        />
      </div>

      <div className="form-group">
        <label>Mode</label>
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          {modeOptions.map(option => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
        <small>🤖 Auto-selecting best available method</small>
      </div>

      <div className="form-group">
        <label>Game Mode</label>
        <select value={gameMode} onChange={(e) => setGameMode(e.target.value)}>
          {gameModeOptions.map(option => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Question Type</label>
        <select value={questionType} onChange={(e) => setQuestionType(e.target.value)}>
          {questionTypeOptions.map(option => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Difficulty</label>
        <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
          {difficultyOptions.map(option => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
      </div>

      {difficulty === 'expert' && <ExpertModePanel />}

      <div className="form-group">
        <label>Number of Questions</label>
        <input
          type="number"
          value={numQuestions}
          onChange={(e) => setNumQuestions(e.target.value)}
          min="1"
          max="10"
        />
      </div>

      <div className="form-group">
        <label>
          <input
            type="checkbox"
            checked={tokenStreaming}
            onChange={(e) => {
              setTokenStreaming(e.target.checked);
              console.log(`🔥 ENHANCED DEBUG - QuizSetup: Token streaming checkbox changed to: ${e.target.checked}`);
            }}
          />
          🌊 Live Token Streaming
        </label>
        <small>Watch AI thinking process in real-time</small>
      </div>

      <button className="btn-primary" onClick={handleStartQuiz}>
        ⭐ START QUIZ
      </button>
    </div>
  );
};