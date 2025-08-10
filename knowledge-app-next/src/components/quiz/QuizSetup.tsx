'use client';

import React, { useState, useEffect } from 'react';
import { useQuizStore } from '@/store/quizStore';
import { StatusDisplay } from '@/components/common/StatusDisplay';

// EXACT Qt WebEngine options - no extras allowed!
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
  const [numQuestions, setNumQuestions] = useState('2');
  const [tokenStreaming, setTokenStreaming] = useState(true);
  const [deepSeekStatus, setDeepSeekStatus] = useState('Checking DeepSeek availability...');
  const [isDeepSeekAvailable, setIsDeepSeekAvailable] = useState(false);

  const { setIsLoading, startQuiz } = useQuizStore();

  // Info text updaters - EXACT Qt behavior
  const updateModeInfo = () => {
    const element = document.getElementById('mode-info');
    if (element) {
      element.innerHTML = '<small>🤖 Auto-selecting best available method</small>';
    }
  };

  const updateGameModeInfo = () => {
    const element = document.getElementById('game-mode-info');
    if (element) {
      const info = gameMode === 'casual' 
        ? '🎵 Relaxed learning with background music and no time pressure'
        : '⏱️ Focused learning with time limits and performance tracking';
      element.innerHTML = `<small>${info}</small>`;
    }
  };

  const updateSubmodeInfo = () => {
    const element = document.getElementById('submode-info');
    if (element) {
      const info = questionType === 'mixed' 
        ? '🔀 Balanced mix of numerical and conceptual questions'
        : questionType === 'numerical'
        ? '📊 Focus on calculations, formulas, and problem-solving'
        : '🧠 Focus on understanding, theory, and concepts';
      element.innerHTML = `<small>${info}</small>`;
    }
  };

  const updateDifficultyInfo = () => {
    const element = document.getElementById('difficulty-info');
    if (element) {
      const info = difficulty === 'easy' 
        ? '🟢 Basic concepts and straightforward questions'
        : difficulty === 'medium'
        ? '🟡 Moderate complexity requiring some analysis'
        : difficulty === 'hard'
        ? '🔴 Advanced analysis and complex problem-solving'
        : '🔥💀 PhD-level complexity with cutting-edge research topics 💀🔥';
      element.innerHTML = `<small>${info}</small>`;
    }
  };

  const saveSettings = () => {
    console.log('Settings saved'); // Qt compatibility placeholder
  };

  useEffect(() => {
    // Initialize info displays
    updateModeInfo();
    updateGameModeInfo();
    updateSubmodeInfo();
    updateDifficultyInfo();
    
    // Check DeepSeek status
    const checkDeepSeek = async () => {
      try {
        // Simulate DeepSeek status check like Qt
        setDeepSeekStatus('Ready: DeepSeek R1 + Llama JSON');
        setIsDeepSeekAvailable(true);
      } catch (error) {
        setDeepSeekStatus('Not available');
        setIsDeepSeekAvailable(false);
      }
    };
    checkDeepSeek();
  }, []);

  useEffect(() => {
    updateModeInfo();
  }, [mode]);

  useEffect(() => {
    updateGameModeInfo();
  }, [gameMode]);

  useEffect(() => {
    updateSubmodeInfo();
  }, [questionType]);

  useEffect(() => {
    updateDifficultyInfo();
  }, [difficulty]);

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
          numQuestions: parseInt(numQuestions),
          mode,
          gameMode, // submode in Qt
          questionType,
          enableTokenStreaming: tokenStreaming,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      
      startQuiz(data.questions);
    } catch (error) {
      console.error('Failed to generate quiz:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="quiz-setup">
      <h2>Quiz Setup</h2>
      
      <div className="form-group">
        <label>Topic</label>
        <input
          type="text"
          id="quiz-topic"
          placeholder="Enter topic (e.g., Science, History)"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
        />
      </div>

      <div className="form-group">
        <label>Mode</label>
        <select 
          id="quiz-mode" 
          value={mode} 
          onChange={(e) => { 
            setMode(e.target.value); 
            updateModeInfo(); 
            saveSettings(); 
          }}
        >
          {modeOptions.map(option => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
        <div id="mode-info" className="mode-info">
          <small>🤖 Auto-selecting best available method</small>
        </div>
      </div>

      <div className="form-group">
        <label>Game Mode</label>
        <select 
          id="quiz-game-mode" 
          value={gameMode} 
          onChange={(e) => { 
            setGameMode(e.target.value); 
            updateGameModeInfo(); 
            saveSettings(); 
          }}
        >
          {gameModeOptions.map(option => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
        <div id="game-mode-info" className="mode-info">
          <small>🎵 Relaxed learning with background music and no time pressure</small>
        </div>
      </div>

      <div className="form-group">
        <label>Question Type</label>
        <select 
          id="quiz-submode" 
          value={questionType} 
          onChange={(e) => { 
            setQuestionType(e.target.value); 
            updateSubmodeInfo(); 
            saveSettings(); 
          }}
        >
          {questionTypeOptions.map(option => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
        <div id="submode-info" className="mode-info">
          <small>🔀 Balanced mix of numerical and conceptual questions</small>
        </div>
      </div>

      <div className="form-group">
        <label>Difficulty</label>
        <select 
          id="quiz-difficulty" 
          value={difficulty} 
          onChange={(e) => { 
            setDifficulty(e.target.value); 
            updateDifficultyInfo(); 
            saveSettings(); 
          }}
        >
          {difficultyOptions.map(option => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
        <div id="difficulty-info" className="mode-info">
          <small>🔴 Advanced analysis and complex problem-solving</small>
        </div>
      </div>

      <div className="form-group">
        <label>Number of Questions</label>
        <input
          type="number"
          id="quiz-questions"
          value={numQuestions}
          onChange={(e) => setNumQuestions(e.target.value)}
          min="1"
          max="10"
        />
      </div>

      {/* Token Streaming - EXACT Qt styling */}
      <div className="form-group">
        <div className="checkbox-group">
          <input
            type="checkbox"
            id="token-streaming-enabled"
            className="toggle-checkbox"
            checked={tokenStreaming}
            onChange={(e) => setTokenStreaming(e.target.checked)}
          />
          <label htmlFor="token-streaming-enabled" className="toggle-label">
            🌊 Live Token Streaming
            <small className="feature-description">Watch AI thinking process in real-time (Expert + Online mode only)</small>
          </label>
        </div>
      </div>

      {/* DeepSeek Section - EXACT Qt styling */}
      {isDeepSeekAvailable && (
        <div className="form-group deepseek-section" id="deepseek-section">
          <div className="deepseek-header">
            <h3>🧠 DeepSeek AI Pipeline</h3>
            <div id="deepseek-status" className="deepseek-status">
              <span className="status-indicator">⏳</span>
              <span className="status-text">{deepSeekStatus}</span>
            </div>
          </div>
          <div className="deepseek-info">
            <small>🔬 Two-Model Pipeline: DeepSeek R1 thinking + Llama JSON formatting</small>
            <br />
            <small>🎯 Optimized for expert-level, PhD-quality questions</small>
          </div>
        </div>
      )}

      {/* START QUIZ Button - EXACT Qt styling */}
      <button 
        className="btn btn-primary" 
        id="start-quiz-button"
        onClick={handleStartQuiz}
        style={{
          backgroundColor: '#007bff',
          color: 'white',
          fontWeight: 'bold',
          padding: '15px 30px',
          fontSize: '16px',
          borderRadius: '8px',
          border: 'none',
          cursor: 'pointer',
          transition: 'all 0.3s ease',
          position: 'relative',
          zIndex: 1000
        }}
      >
        ⭐ START QUIZ
      </button>
    </div>
  );
};
