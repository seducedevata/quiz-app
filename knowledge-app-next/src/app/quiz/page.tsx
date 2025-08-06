'use client';

import React, { useState, useEffect } from 'react';
import { MathJax } from 'better-react-mathjax';
import { callPythonMethod, onPythonEvent, offPythonEvent } from '../../lib/pythonBridge';
import { AppLogger, setCurrentScreenName } from '../../lib/logger';
import { useScreen } from '../../context/ScreenContext'; // Import useScreen

// Mock data for quiz (will be replaced by backend integration)
const mockQuizData = {
  id: 'mock-quiz-123',
  totalQuestions: 5,
  questions: [
    {
      id: 'q1',
      text: 'What is the formula for water?',
      options: [
        { id: 'opt1', text: 'H2O' },
        { id: 'opt2', text: 'CO2' },
        { id: 'opt3', text: 'O2' },
        { id: 'opt4', text: 'N2' },
      ],
      correctAnswerId: 'opt1',
      explanation: 'Water is a chemical compound with the chemical formula H2O. A water molecule contains one oxygen atom and two hydrogen atoms connected by covalent bonds.',
    },
    {
      id: 'q2',
      text: 'Calculate the derivative of $f(x) = x^2 + 3x - 5$.',
      options: [
        { id: 'opt5', text: '$2x + 3$' },
        { id: 'opt6', text: '$x + 3$' },
        { id: 'opt7', text: '$2x - 5$' },
        { id: 'opt8', text: '$x^2 + 3$' },
      ],
      correctAnswerId: 'opt5',
      explanation: 'The derivative of $x^n$ is $nx^{n-1}$, and the derivative of $ax$ is $a$. So, the derivative of $x^2$ is $2x$, and the derivative of $3x$ is $3$. The derivative of a constant $(-5)$ is $0$.',
    },
    {
      id: 'q3',
      text: 'Which of the following is a noble gas?',
      options: [
        { id: 'opt9', text: 'Oxygen' },
        { id: 'opt10', text: 'Helium' },
        { id: 'opt11', text: 'Nitrogen' },
        { id: 'opt12', text: 'Chlorine' },
      ],
      correctAnswerId: 'opt10',
      explanation: 'Noble gases are a group of chemical elements with similar properties: under standard conditions, they are all odorless, colorless, monatomic gases with very low chemical reactivity. The six naturally occurring noble gases are helium (He), neon (Ne), argon (Ar), krypton (Kr), xenon (Xe), and radon (Rn).',
    },
    {
      id: 'q4',
      text: 'What is the capital of France?',
      options: [
        { id: 'opt13', text: 'Berlin' },
        { id: 'opt14', text: 'Madrid' },
        { id: 'opt15', text: 'Paris' },
        { id: 'opt16', text: 'Rome' },
      ],
      correctAnswerId: 'opt15',
      explanation: 'Paris is the capital and most populous city of France.',
    },
    {
      id: 'q5',
      text: 'Solve for $x$: $2x + 5 = 15$.',
      options: [
        { id: 'opt17', text: '$x = 5$' },
        { id: 'opt18', text: '$x = 10$' },
        { id: 'opt19', text: '$x = 2.5$' },
        { id: 'opt20', text: '$x = 7.5$' },
      ],
      correctAnswerId: 'opt17',
      explanation: 'To solve $2x + 5 = 15$: Subtract 5 from both sides to get $2x = 10$. Then, divide by 2 to get $x = 5$.',
    },
  ],
};

const QuizPage: React.FC = () => {
  const { currentScreen, showScreen } = useScreen(); // Use global screen state
  const [quizInternalScreen, setQuizInternalScreen] = useState('setup'); // 'setup', 'game', 'results'
  const [topic, setTopic] = useState<string>('');
  const [mode, setMode] = useState<string>('offline');
  const [gameMode, setGameMode] = useState<string>('standard');
  const [questionType, setQuestionType] = useState<string>('mixed');
  const [difficulty, setDifficulty] = useState<string>('medium');
  const [numQuestions, setNumQuestions] = useState<number>(2);
  const [tokenStreamingEnabled, setTokenStreamingEnabled] = useState<boolean>(true);
  const [deepSeekStatus, setDeepSeekStatus] = useState<string>('⏳ Checking DeepSeek availability...');
  const [isDeepSeekAvailable, setIsDeepSeekAvailable] = useState<boolean>(false);
  const [apiStatusDisplay, setApiStatusDisplay] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<{ message: string; type: string } | null>(null);

  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedOptionId, setSelectedOptionId] = useState<string | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [quizCompleted, setQuizCompleted] = useState(false);
  const [score, setScore] = useState(0);
  const [answeredQuestions, setAnsweredQuestions] = useState<{
    questionId: string;
    selectedOptionId: string | null;
    isCorrect: boolean;
  }[]>([]);

  const currentQuestion = mockQuizData.questions[currentQuestionIndex];

  useEffect(() => {
    // This component is now only active when currentScreen is 'quiz'
    // Its internal screen state is managed by quizInternalScreen
    AppLogger.info('QUIZ_SCREEN', `Quiz component active. Internal screen: ${quizInternalScreen}`);
  }, [quizInternalScreen]);

  // Status Display functions
  const getStatusIcon = (type: string) => {
    const icons: { [key: string]: string } = {
      'info': '🔄',
      'success': '✅',
      'warning': '⚠️',
      'error': '❌',
      'turbo': '🚀',
      'gpu': '🎮',
      'api': '🌐',
      'cloud': '☁️',
      'network': '📡',
      'deepseek': '🧠',
    };
    return icons[type] || '🔄';
  };

  const showStatusDisplay = (message: string, type: string = 'info') => {
    setStatusMessage({ message, type });
    AppLogger.info('STATUS_DISPLAY', message, { type });
    setTimeout(() => setStatusMessage(null), 5000); // Auto-hide after 5 seconds
  };

  // Handlers for quiz setup form
  const updateModeInfo = (currentMode: string) => {
    const modeDescriptions: { [key: string]: string } = {
      'online': '🌐 Online mode uses cloud AI providers for question generation',
      'offline': '💻 Offline mode uses local models for question generation',
      'hybrid': '🔄 Hybrid mode combines online and offline capabilities',
    };
    const infoElement = document.getElementById('mode-info');
    if (infoElement) {
      infoElement.innerHTML = `<small>${modeDescriptions[currentMode] || 'Select a quiz mode'}</small>`;
    }
    AppLogger.info('QUIZ_SETUP', `Mode changed to: ${currentMode}`);
  };

  const updateGameModeInfo = (currentGameMode: string) => {
    const gameModeDescriptions: { [key: string]: string } = {
      'standard': '📝 Standard quiz with multiple choice questions',
      'timed': '⏱️ Timed quiz with countdown for each question',
      'practice': '🎯 Practice mode with immediate feedback',
      'exam': '📋 Exam mode with final scoring only',
    };
    const infoElement = document.getElementById('game-mode-info');
    if (infoElement) {
      infoElement.innerHTML = `<small>${gameModeDescriptions[currentGameMode] || 'Select a game mode'}</small>`;
    }
    AppLogger.info('QUIZ_SETUP', `Game Mode changed to: ${currentGameMode}`);
  };

  const updateSubmodeInfo = (currentQuestionType: string) => {
    const submodeDescriptions: { [key: string]: string } = {
      'conceptual': '💭 Conceptual questions focus on understanding and theory',
      'numerical': '🔢 Numerical questions involve calculations and problem-solving',
      'mixed': '🎲 Mixed questions combine conceptual and numerical elements',
      'application': '🛠️ Application questions test practical usage',
      'analysis': '🔍 Analysis questions require critical thinking',
    };
    const infoElement = document.getElementById('submode-info');
    if (infoElement) {
      infoElement.innerHTML = `<small>${submodeDescriptions[currentQuestionType] || 'Select a question type'}</small>`;
    }
    AppLogger.info('QUIZ_SETUP', `Question Type changed to: ${currentQuestionType}`);
  };

  const updateDifficultyInfo = (currentDifficulty: string) => {
    const difficultyDescriptions: { [key: string]: string } = {
      'beginner': '🌱 Beginner level - Basic concepts and simple problems',
      'intermediate': '🌿 Intermediate level - Moderate complexity and depth',
      'advanced': '🌳 Advanced level - Complex problems requiring expertise',
      'expert': '🎓 Expert level - PhD-level questions with deep analysis',
      'phd': '🔬 PhD level - Research-grade questions with cutting-edge concepts',
    };
    const infoElement = document.getElementById('difficulty-info');
    if (infoElement) {
      infoElement.innerHTML = `<small>${difficultyDescriptions[currentDifficulty] || 'Select a difficulty level'}</small>`;
    }
    AppLogger.info('QUIZ_SETUP', `Difficulty changed to: ${currentDifficulty}`);
  };

  const saveSettings = async () => {
    AppLogger.info('SETTINGS', 'Saving quiz settings...');
    const settingsToSave = {
      defaultQuizConfig: {
        topic,
        mode,
        gameMode,
        questionType,
        difficulty,
        numQuestions,
        tokenStreamingEnabled,
      },
    };
    try {
      await callPythonMethod('save_app_settings', settingsToSave);
      AppLogger.success('SETTINGS', 'Quiz settings saved successfully to backend.');
      localStorage.setItem('quizSettings', JSON.stringify(settingsToSave));
      AppLogger.info('SETTINGS', 'Quiz settings saved successfully to localStorage.');
    } catch (error) {
      AppLogger.error('SETTINGS', 'Failed to save quiz settings.', error);
      showStatusDisplay('Failed to save settings.', 'error');
    }
  };

  useEffect(() => {
    // Load settings from localStorage on component mount
    const savedSettings = localStorage.getItem('quizSettings');
    let initialMode = 'offline';
    let initialGameMode = 'standard';
    let initialQuestionType = 'conceptual';
    let initialDifficulty = 'expert';

    if (savedSettings) {
      try {
        const parsedSettings = JSON.parse(savedSettings).defaultQuizConfig;
        setTopic(parsedSettings.topic || '');
        setMode(parsedSettings.mode || 'auto');
        setGameMode(parsedSettings.gameMode || 'casual');
        setQuestionType(parsedSettings.questionType || 'mixed');
        setDifficulty(parsedSettings.difficulty || 'medium');
        setNumQuestions(parsedSettings.numQuestions || 2);
        setTokenStreamingEnabled(parsedSettings.tokenStreamingEnabled || true);
        AppLogger.info('SETTINGS', 'Quiz settings loaded from localStorage.', parsedSettings);
        AppLogger.debug('TOKEN_STREAM', `Token streaming enabled from localStorage: ${parsedSettings.tokenStreamingEnabled}`);

        initialMode = parsedSettings.mode || 'auto';
        initialGameMode = parsedSettings.gameMode || 'casual';
        initialQuestionType = parsedSettings.questionType || 'mixed';
        initialDifficulty = parsedSettings.difficulty || 'medium';

      } catch (e) {
        AppLogger.error('SETTINGS', 'Failed to parse quiz settings from localStorage.', e);
      }
    }

    // Initialize info texts based on loaded or default settings
    updateModeInfo(initialMode);
    updateGameModeInfo(initialGameMode);
    updateSubmodeInfo(initialQuestionType);
    updateDifficultyInfo(initialDifficulty);

    // Initialize DeepSeek status
    const initializeDeepSeekStatus = async () => {
      try {
        const status = await callPythonMethod<any>('getDeepSeekStatus');
        if (status.available && status.ready) {
          setDeepSeekStatus(`Ready: ${status.thinking_model} + ${status.json_model}`);
          setIsDeepSeekAvailable(true);
          AppLogger.info('DEEPSEEK', 'DeepSeek pipeline ready.', status);
        } else if (status.available && !status.ready) {
          setDeepSeekStatus(status.error || 'Pipeline not ready');
          setIsDeepSeekAvailable(true); // Still available, just not ready
          AppLogger.warn('DEEPSEEK', 'DeepSeek available but not ready.', status);
        } else {
          setDeepSeekStatus('Not available');
          setIsDeepSeekAvailable(false);
          AppLogger.warn('DEEPSEEK', 'DeepSeek not available.', status);
        }
      } catch (error) {
        setDeepSeekStatus('Error checking status');
        setIsDeepSeekAvailable(false);
        AppLogger.error('DEEPSEEK', 'Error checking DeepSeek status.', error);
      }
    };
    initializeDeepSeekStatus();

    // Placeholder for API status indicators (will be implemented later)
    // setApiStatusDisplay(true); // Example: show API status section

  }, []);

  const handleStartQuizClick = async (e: React.MouseEvent) => {
    e.preventDefault();
    AppLogger.action('QUIZ_START', 'Start Quiz button clicked.');

    const config = {
      topic,
      mode,
      gameMode,
      questionType,
      difficulty,
      numQuestions,
      tokenStreamingEnabled,
    };

    AppLogger.debug('TOKEN_STREAM', `Quiz start - tokenStreamingEnabled: ${tokenStreamingEnabled}`);
    // Simulate shouldUseTokenStreaming logic
    if (tokenStreamingEnabled && (difficulty === 'expert' || mode === 'online')) {
      AppLogger.info('TOKEN_STREAM', 'Token streaming is enabled and conditions met.');
      // Simulate startTokenStreamingVisualization logic
      AppLogger.info('TOKEN_STREAM', 'Simulating startTokenStreamingVisualization.', { topic, difficulty, questionType });
    } else {
      AppLogger.info('TOKEN_STREAM', 'Token streaming is disabled or conditions not met.');
    }

    try {
      showStatusDisplay(`🧠 Generating quiz about ${topic}...`, 'info');
      AppLogger.info('QUIZ_GEN', 'Calling backend for quiz generation.', config);
      const generatedQuizId = await callPythonMethod<string>('generate_mcq_quiz', config);
      AppLogger.success('QUIZ_GEN', 'Quiz generated successfully.', { quizId: generatedQuizId });
      showStatusDisplay('Quiz generated successfully!', 'success');
      // In a real app, you'd store the quiz data and transition to game screen
      setQuizInternalScreen('game');
    } catch (error) {
      AppLogger.error('QUIZ_GEN', 'Failed to generate quiz.', error);
      showStatusDisplay(`Failed to generate quiz: ${(error as Error).message}`, 'error');
    }
  };

  // Handlers for quiz game
  const handleOptionSelect = (optionId: string) => {
    if (!showExplanation) {
      setSelectedOptionId(optionId);
      AppLogger.user('QUIZ_GAME', 'Option selected.', { optionId });
    }
  };

  const handleSubmitAnswer = () => {
    if (selectedOptionId) {
      const isCorrect = selectedOptionId === currentQuestion.correctAnswerId;
      if (isCorrect) {
        setScore(prevScore => prevScore + 1);
      }
      setAnsweredQuestions(prev => [
        ...prev,
        { questionId: currentQuestion.id, selectedOptionId, isCorrect },
      ]);
      setShowExplanation(true);
      AppLogger.action('QUIZ_GAME', 'Answer submitted.', { questionId: currentQuestion.id, selectedOptionId, isCorrect });
    }
  };

  const handleNextQuestion = () => {
    if (currentQuestionIndex < mockQuizData.totalQuestions - 1) {
      setCurrentQuestionIndex(prevIndex => prevIndex + 1);
      setSelectedOptionId(null);
      setShowExplanation(false);
      AppLogger.action('QUIZ_GAME', 'Next question.', { newIndex: currentQuestionIndex + 1 });
    } else {
      setQuizCompleted(true);
      setQuizInternalScreen('results');
      AppLogger.action('QUIZ_GAME', 'Quiz completed.', { finalScore: score });
    }
  };

  const resetQuiz = () => {
    setQuizInternalScreen('setup');
    setCurrentQuestionIndex(0);
    setSelectedOptionId(null);
    setShowExplanation(false);
    setQuizCompleted(false);
    setScore(0);
    setAnsweredQuestions([]);
    AppLogger.action('QUIZ_GAME', 'Quiz reset.');
  };

  const progress = ((currentQuestionIndex + 1) / mockQuizData.totalQuestions) * 100;

  return (
    <div className="quiz-container max-w-[800px] mx-auto">
      <div id="quiz-setup" className={`quiz-setup bg-bg-secondary rounded-2xl p-8 shadow-md screen ${quizInternalScreen === 'setup' ? 'active' : ''}`}>
        <h2 className="text-2xl font-bold mb-2">Quiz Setup</h2>
        <div className="form-group mb-6">
          <label htmlFor="quiz-topic" className="block mb-2 font-medium text-text-secondary">Topic</label>
          <input
            type="text"
            id="quiz-topic"
            className="w-full p-3 border border-border-color rounded-lg bg-bg-primary text-text-primary text-base transition-colors duration-300 focus:outline-none focus:border-primary-color"
            placeholder="Enter topic (e.g., Science, History)"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
          />
        </div>

        <div className="form-group mb-6">
          <label htmlFor="quiz-mode" className="block mb-2 font-medium text-text-secondary">Mode</label>
          <select 
            id="quiz-mode" 
            className="w-full p-3 border border-border-color rounded-lg bg-bg-primary text-text-primary text-base transition-colors duration-300 focus:outline-none focus:border-primary-color"
            value={mode} 
            onChange={(e) => { setMode(e.target.value); updateModeInfo(); saveSettings(); }}>
            <option value="offline">Offline (Local AI - TURBO)</option>
            <option value="hybrid">Hybrid (Combines Online & Offline)</option>
            <option value="online">Online (Cloud APIs)</option>
          </select>
          <div id="mode-info" className="mode-info mt-1 py-1 text-text-secondary">
            <small>💻 Offline mode uses local models for question generation</small>
          </div>
        </div>

        {/* API Status - Placeholder for dynamic content */}
        {apiStatusDisplay && (
          <div id="api-status" className="api-status mt-2 p-4 bg-bg-primary rounded-lg border border-border-color">
            <div className="api-providers" id="api-providers"></div>
            <div className="api-help pt-2 border-t border-border-color text-text-secondary text-xs leading-tight">
              <small>💡 <strong>Supported APIs:</strong> OpenAI GPT-4, Anthropic Claude, Google Gemini, Groq, OpenRouter</small>
              <br />
              <small>🔑 <strong>Setup:</strong> Set environment variables OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, etc.</small>
            </div>
          </div>
        )}

        <div className="form-group mb-6">
          <label htmlFor="quiz-game-mode" className="block mb-2 font-medium text-text-secondary">Game Mode</label>
          <select 
            id="quiz-game-mode" 
            className="w-full p-3 border border-border-color rounded-lg bg-bg-primary text-text-primary text-base transition-colors duration-300 focus:outline-none focus:border-primary-color"
            value={gameMode} 
            onChange={(e) => { setGameMode(e.target.value); updateGameModeInfo(); saveSettings(); }}>
            <option value="standard">📝 Standard Quiz</option>
            <option value="timed">⏱️ Timed Quiz</option>
            <option value="practice">🎯 Practice Mode</option>
            <option value="exam">📋 Exam Mode</option>
          </select>
          <div id="game-mode-info" className="mode-info mt-1 py-1 text-text-secondary">
            <small>📝 Standard quiz with multiple choice questions</small>
          </div>
        </div>

        <div className="form-group mb-6">
          <label htmlFor="quiz-submode" className="block mb-2 font-medium text-text-secondary">Question Type</label>
          <select 
            id="quiz-submode" 
            className="w-full p-3 border border-border-color rounded-lg bg-bg-primary text-text-primary text-base transition-colors duration-300 focus:outline-none focus:border-primary-color"
            value={questionType} 
            onChange={(e) => { setQuestionType(e.target.value); updateSubmodeInfo(); saveSettings(); }}>
            <option value="conceptual">💭 Conceptual Questions</option>
            <option value="numerical">🔢 Numerical Problems</option>
            <option value="mixed">🎲 Mixed Question Types</option>
            <option value="mcq">✅ Multiple Choice</option>
            <option value="true_false">⚖️ True/False</option>
            <option value="fill_blank">✍️ Fill in the Blank</option>
          </select>
          <div id="submode-info" className="mode-info mt-1 py-1 text-text-secondary">
                        <small>💭 Conceptual questions focus on understanding and theory</small>
          </div>
        </div>

        <div className="form-group mb-6">
          <label htmlFor="quiz-difficulty" className="block mb-2 font-medium text-text-secondary">Difficulty</label>
          <select 
            id="quiz-difficulty" 
            className="w-full p-3 border border-border-color rounded-lg bg-bg-primary text-text-primary text-base transition-colors duration-300 focus:outline-none focus:border-primary-color"
            value={difficulty} 
            onChange={(e) => { setDifficulty(e.target.value); updateDifficultyInfo(); saveSettings(); }}>
            <option value="beginner">🌱 Beginner</option>
            <option value="intermediate">🌿 Intermediate</option>
            <option value="advanced">🌳 Advanced</option>
            <option value="expert">🎓 Expert</option>
            <option value="phd">🔬 PhD</option>
          </select>
          <div id="difficulty-info" className="mode-info mt-1 py-1 text-text-secondary">
            <small>🎓 Expert level - PhD-level questions with deep analysis</small>
          </div>
        </div>

        <div className="form-group mb-6">
          <label htmlFor="quiz-questions" className="block mb-2 font-medium text-text-secondary">Number of Questions</label>
          <input
            type="number"
            id="quiz-questions"
            className="w-full p-3 border border-border-color rounded-lg bg-bg-primary text-text-primary text-base transition-colors duration-300 focus:outline-none focus:border-primary-color"
            value={numQuestions}
            onChange={(e) => setNumQuestions(parseInt(e.target.value))}
            min="1"
            max="50"
          />
        </div>

        {/* Token Streaming Option */}
        <div className="form-group mb-6">
          <div className="checkbox-group flex items-center">
            <input
              type="checkbox"
              id="token-streaming-enabled"
              className="toggle-checkbox w-auto mr-3 transform scale-125"
              checked={tokenStreamingEnabled}
              onChange={(e) => {
                setTokenStreamingEnabled(e.target.checked);
                AppLogger.debug('TOKEN_STREAM', `Token streaming checkbox changed to: ${e.target.checked}`);
              }}
            />
            <label htmlFor="token-streaming-enabled" className="toggle-label mb-0 font-medium text-text-primary">
              🌊 Live Token Streaming
              <small className="feature-description block text-text-secondary text-sm">Watch AI thinking process in real-time (Expert + Online mode only)</small>
            </label>
          </div>
        </div>

        {/* DeepSeek Integration */}
        {isDeepSeekAvailable && (
          <div id="deepseek-section" className="form-group deepseek-section bg-gradient-to-br from-purple-100/10 to-purple-200/10 border-2 border-purple-300/20 rounded-xl p-4 my-4">
            <div className="deepseek-header flex justify-between items-center mb-2">
              <h3 className="m-0 text-primary-color text-lg">🧠 DeepSeek AI Pipeline</h3>
              <div id="deepseek-status" className="deepseek-status flex items-center gap-2 text-sm">
                <span className="status-indicator text-xl">⏳</span>
                <span className="status-text">{deepSeekStatus}</span>
              </div>
            </div>
            <div className="deepseek-info text-text-secondary text-sm leading-snug">
              <small>🔬 Two-Model Pipeline: DeepSeek R1 thinking + Llama JSON formatting</small>
              <br />
              <small>🎯 Optimized for expert-level, PhD-quality questions</small>
            </div>
          </div>
        )}

        <button
          className="btn btn-primary bg-primary-color text-white font-bold py-4 px-8 text-lg rounded-lg border-none cursor-pointer transition-all duration-300 relative z-10 w-full hover:bg-primary-hover hover:translate-y-[-2px] hover:shadow-lg"
          id="start-quiz-button"
          onClick={handleStartQuizClick}
        >
          ⭐ START QUIZ
        </button>
        <button
          className="btn btn-secondary bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-4 px-8 text-lg rounded-lg border-none cursor-pointer transition-all duration-300 relative z-10 w-full hover:translate-y-[-2px] hover:shadow-lg mt-4"
          id="quick-quiz-button"
          onClick={() => handleStartQuizClick(new Event('click'))}
        >
          ⚡ QUICK QUIZ
        </button>
      </div>

      <div id="quiz-game" className={`quiz-game bg-bg-secondary rounded-2xl p-8 shadow-md screen ${quizInternalScreen === 'game' ? 'active' : ''}`}>
        {/* Status Display */}
        {statusMessage && (
          <div id="status-display" className="status-display">
            <div className={`status-message ${statusMessage.type}`}>
              <div className="status-icon">{getStatusIcon(statusMessage.type)}</div>
              <div className="status-text">{statusMessage.message}</div>
              <div className="status-spinner"></div>
            </div>
          </div>
        )}

        <div className="quiz-header flex justify-between items-center mb-8 text-text-secondary">
          <span id="question-number" className="text-lg font-semibold">Question {currentQuestionIndex + 1} / {mockQuizData.totalQuestions}</span>
          <span id="timer-display" className="timer-normal text-xl font-bold">⏱️ 30s</span>
        </div>
        <div className="question-container mb-8">
          <h3 id="question-text" className="mb-6 text-2xl font-semibold text-text-primary leading-relaxed"><MathJax>{currentQuestion.text}</MathJax></h3>
          <div id="options-container" className="options-container flex flex-col gap-3">
            {currentQuestion.options.map(option => {
              const isSelected = selectedOptionId === option.id;
              const isCorrectOption = showExplanation && option.id === currentQuestion.correctAnswerId;
              const isIncorrectSelected = showExplanation && isSelected && !isCorrectOption;

              let optionClassName = "option-button";
              if (isSelected && !showExplanation) {
                optionClassName += " selected";
              } else if (isCorrectOption) {
                optionClassName += " correct-answer";
              } else if (isIncorrectSelected) {
                optionClassName += " incorrect-answer";
              }

              return (
                <button
                  key={option.id}
                  className={optionClassName}
                  onClick={() => handleOptionSelect(option.id)}
                  disabled={showExplanation}
                >
                  <span className="option-letter">
                    {String.fromCharCode(65 + currentQuestion.options.indexOf(option))}.
                  </span>
                  <span className="option-content"><MathJax>{option.text}</MathJax></span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Feedback Container */}
        <div id="feedback-container" className="feedback-container" style={{ display: showExplanation ? 'block' : 'none' }}>
          <div className={`feedback-message ${selectedOptionId === currentQuestion.correctAnswerId ? 'correct' : 'incorrect'}`}>
            <div className="feedback-icon">{selectedOptionId === currentQuestion.correctAnswerId ? '✅' : '❌'}</div>
            <div className="feedback-text">
              {selectedOptionId === currentQuestion.correctAnswerId ? 'Correct!' : 'Incorrect'}
            </div>
          </div>
        </div>

        {/* Explanation Container */}
        <div id="explanation-container" className="explanation-container" style={{ display: showExplanation ? 'block' : 'none' }}>
          <h4 className="text-lg text-primary-color mb-3 font-semibold">💡 Explanation:</h4>
          <p className="text-text-primary leading-relaxed m-0"><MathJax>{currentQuestion.explanation}</MathJax></p>
        </div>

        <div className="quiz-footer flex justify-between gap-4 mt-6">
          {!showExplanation ? (
            <button
              id="submit-btn"
              className="btn btn-primary w-full"
              onClick={handleSubmitAnswer}
              disabled={!selectedOptionId}
            >
              Submit Answer
            </button>
          ) : (
            <div id="quiz-navigation" className="quiz-navigation w-full">
              <button id="prev-question-btn" className="btn btn-secondary" onClick={() => console.log('Previous clicked')}>← Previous</button>
              <button id="next-question-btn" className="btn btn-primary" onClick={handleNextQuestion}>Next →</button>
              <button id="finish-quiz-btn" className="btn btn-secondary" onClick={() => setQuizInternalScreen('results')}>Finish Quiz</button>
            </div>
          )}
        </div>
      </div>

      <div id="quiz-results" className={`quiz-results screen ${quizInternalScreen === 'results' ? 'active' : ''}`}>
        <h2>Quiz Complete!</h2>
        <div className="score">Score: {Math.round((score / mockQuizData.totalQuestions) * 100)}%</div>
        <div className="details">
          {score}/{mockQuizData.totalQuestions} correct
        </div>
        <button className="btn btn-primary" onClick={resetQuiz}>New Quiz</button>
      </div>
    </div>
  );
};

export default QuizPage;