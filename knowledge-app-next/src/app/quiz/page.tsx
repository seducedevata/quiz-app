'use client';

import { MathJax } from 'better-react-mathjax';
import React, { useEffect, useState } from 'react';
import { useScreen } from '../../context/ScreenContext'; // Import useScreen
import { AppLogger } from '../../lib/logger';
import { callPythonMethod } from '../../lib/pythonBridge';
import { StatusDisplay } from '../../components/common/StatusDisplay';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { TokenStreamDisplay } from '../../components/common/TokenStreamDisplay';

// Define types for our quiz data
interface QuizOption {
  id: string;
  text: string;
}

interface QuizQuestion {
  id: string;
  text: string;
  options: QuizOption[];
  correctAnswerId: string;
  explanation: string;
}

interface QuizData {
  id: string;
  totalQuestions: number;
  questions: QuizQuestion[];
}

const QuizPage: React.FC = () => {
  const { currentScreen, showScreen } = useScreen(); // Use global screen state
  const [quizInternalScreen, setQuizInternalScreen] = useState('setup'); // 'setup', 'game', 'results'
  const [topic, setTopic] = useState<string>('');
  const [mode, setMode] = useState<string>('auto');
  const [gameMode, setGameMode] = useState<string>('casual');
  const [questionType, setQuestionType] = useState<string>('mixed');
  const [difficulty, setDifficulty] = useState<string>('medium');
  const [numQuestions, setNumQuestions] = useState<number>(2);
  const [tokenStreamingEnabled, setTokenStreamingEnabled] = useState<boolean>(true);
  const [deepSeekStatus, setDeepSeekStatus] = useState<string>('⏳ Checking DeepSeek availability...');
  const [isDeepSeekAvailable, setIsDeepSeekAvailable] = useState<boolean>(false);
    const [statusMessage, setStatusMessage] = useState<{ message: string; type: string } | null>(null);
  const [answeredQuestions, setAnsweredQuestions] = useState<{
    questionId: string;
    selectedOptionId: string | null;
    isCorrect: boolean;
  }[]>([]);
  const [timeRemaining, setTimeRemaining] = useState(30);
  const [isExpertMode, setIsExpertMode] = useState(false);
  const [quizData, setQuizData] = useState<QuizData | null>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedOptionId, setSelectedOptionId] = useState<string | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [quizCompleted, setQuizCompleted] = useState(false);
  const [score, setScore] = useState(0);

  const currentQuestion = quizData?.questions[currentQuestionIndex];

  // Handlers for quiz game
  const handleSubmitAnswer = () => {
    if (selectedOptionId && currentQuestion) {
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

  useEffect(() => {
    if (quizInternalScreen === 'game' && !quizCompleted) {
      if (isExpertMode) {
        setTimeRemaining(15);
      }
      const timer = setInterval(() => {
        setTimeRemaining(prevTime => {
          if (prevTime === 1) {
            handleSubmitAnswer(); // Auto-submit when timer reaches 0
            return 0;
          }
          return prevTime - 1;
        });
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [quizInternalScreen, currentQuestionIndex, quizCompleted, isExpertMode, selectedOptionId, currentQuestion]);



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
      'casual': '🎵 Relaxed learning with background music and no time pressure',
      'serious': '⏱️ Focused learning with time limits and performance tracking',
    };
    const infoElement = document.getElementById('game-mode-info');
    if (infoElement) {
      infoElement.innerHTML = `<small>${gameModeDescriptions[currentGameMode] || 'Select a game mode'}</small>`;
    }
    AppLogger.info('QUIZ_SETUP', `Game Mode changed to: ${currentGameMode}`);
  };

  const updateSubmodeInfo = (currentQuestionType: string) => {
    const submodeDescriptions: { [key: string]: string } = {
      'mixed': ' Balanced mix of numerical and conceptual questions',
      'numerical': ' Focus on calculations, formulas, and problem-solving',
      'conceptual': '🧠 Focus on understanding, theory, and concepts',
    };
    const infoElement = document.getElementById('submode-info');
    if (infoElement) {
      infoElement.innerHTML = `<small>${submodeDescriptions[currentQuestionType] || 'Select a question type'}</small>`;
    }
    AppLogger.info('QUIZ_SETUP', `Question Type changed to: ${currentQuestionType}`);
  };

  const updateDifficultyInfo = (currentDifficulty: string) => {
    const difficultyDescriptions: { [key: string]: string } = {
      'easy': '🟢 Basic concepts and straightforward questions',
      'medium': '🟡 Moderate complexity requiring some analysis',
      'hard': '🔴 Advanced analysis and complex problem-solving',
      'expert': '🔥💀 PhD-level complexity with cutting-edge research topics 💀🔥',
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
    let initialMode = 'auto';
    let initialGameMode = 'casual';
    let initialQuestionType = 'mixed';
    let initialDifficulty = 'medium';

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

    if (difficulty === 'expert') {
      setIsExpertMode(true);
    }

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
      const generatedQuiz = await callPythonMethod<QuizData>('generate_mcq_quiz', config);
      AppLogger.success('QUIZ_GEN', 'Quiz generated successfully.', { quizId: generatedQuiz.id });
      setQuizData(generatedQuiz);
      showStatusDisplay('Quiz generated successfully!', 'success');
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

  const handleNextQuestion = () => {
    if (quizData && currentQuestionIndex < quizData.totalQuestions - 1) {
      setCurrentQuestionIndex(prevIndex => prevIndex + 1);
      setSelectedOptionId(null);
      setShowExplanation(false);
      setTimeRemaining(30); // Reset timer for next question
      AppLogger.action('QUIZ_GAME', 'Next question.', { newIndex: currentQuestionIndex + 1 });
    } else {
      setQuizCompleted(true);
      setQuizInternalScreen('results');
      AppLogger.action('QUIZ_GAME', 'Quiz completed.', { finalScore: score });
    }
  };

  const resetQuiz = () => {
    setQuizInternalScreen('setup');
    setQuizData(null);
    setCurrentQuestionIndex(0);
    setSelectedOptionId(null);
    setShowExplanation(false);
    setQuizCompleted(false);
    setScore(0);
    setAnsweredQuestions([]);
    setTimeRemaining(30);
    AppLogger.action('QUIZ_GAME', 'Quiz reset.');
  };

  if (quizInternalScreen === 'game' && !quizData) {
    return (
      <div className="quiz-container max-w-[800px] mx-auto">
        <div className="quiz-game bg-bg-secondary rounded-2xl p-8 shadow-md screen active">
          <div className="text-center">
            <h2 className="text-2xl font-bold mb-4">Loading Quiz...</h2>
            <p>Please wait while we generate your quiz.</p>
          </div>
        </div>
      </div>
    );
  }

  const progress = quizData ? ((currentQuestionIndex + 1) / quizData.totalQuestions) * 100 : 0;

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
            onChange={(e) => { setMode(e.target.value); updateModeInfo(e.target.value); saveSettings(); }}>
            <option value="offline">Offline (Local AI - TURBO)</option>
            <option value="auto">Auto (Best Available)</option>
            <option value="online">Online (Cloud APIs)</option>
          </select>
          <div id="mode-info" className="mode-info mt-1 py-1 text-text-secondary">
            <small>🤖 Auto-selecting best available method</small>
          </div>
        </div>

        <div className="form-group mb-6">
          <label htmlFor="quiz-game-mode" className="block mb-2 font-medium text-text-secondary">Game Mode</label>
          <select 
            id="quiz-game-mode" 
            className="w-full p-3 border border-border-color rounded-lg bg-bg-primary text-text-primary text-base transition-colors duration-300 focus:outline-none focus:border-primary-color"
            value={gameMode} 
            onChange={(e) => { setGameMode(e.target.value); updateGameModeInfo(e.target.value); saveSettings(); }}>
            <option value="casual">🎵 Casual Mode (Relaxed, Music)</option>
            <option value="serious">⏱️ Serious Mode (Timed, Focused)</option>
          </select>
          <div id="game-mode-info" className="mode-info mt-1 py-1 text-text-secondary">
            <small>🎵 Relaxed learning with background music and no time pressure</small>
          </div>
        </div>

        <div className="form-group mb-6">
          <label htmlFor="quiz-submode" className="block mb-2 font-medium text-text-secondary">Question Type</label>
          <select 
            id="quiz-submode" 
            className="w-full p-3 border border-border-color rounded-lg bg-bg-primary text-text-primary text-base transition-colors duration-300 focus:outline-none focus:border-primary-color"
            value={questionType} 
            onChange={(e) => { setQuestionType(e.target.value); updateSubmodeInfo(e.target.value); saveSettings(); }}>
            <option value="mixed"> Mixed (Balanced)</option>
            <option value="numerical"> Numerical (Math & Calculations)</option>
            <option value="conceptual">🧠 Conceptual (Theory & Understanding)</option>
          </select>
          <div id="submode-info" className="mode-info mt-1 py-1 text-text-secondary">
            <small> Balanced mix of numerical and conceptual questions</small>
          </div>
        </div>

        <div className="form-group mb-6">
          <label htmlFor="quiz-difficulty" className="block mb-2 font-medium text-text-secondary">Difficulty</label>
          <select 
            id="quiz-difficulty" 
            className="w-full p-3 border border-border-color rounded-lg bg-bg-primary text-text-primary text-base transition-colors duration-300 focus:outline-none focus:border-primary-color"
            value={difficulty} 
            onChange={(e) => { setDifficulty(e.target.value); updateDifficultyInfo(e.target.value); saveSettings(); }}>
            <option value="easy">🟢 Easy</option>
            <option value="medium">🟡 Medium</option>
            <option value="hard">🔴 Hard</option>
            <option value="expert">🔥💀 EXPERT (PhD-Level) 💀</option>
          </select>
          <div id="difficulty-info" className="mode-info mt-1 py-1 text-text-secondary">
            <small>🔴 Advanced analysis and complex problem-solving</small>
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
                saveSettings();
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
      </div>

      {quizData && currentQuestion && (
        <div id="quiz-game" className={`quiz-game bg-bg-secondary rounded-2xl p-8 shadow-md screen ${quizInternalScreen === 'game' ? 'active' : ''}`}>
          {/* Status Display */}
          {statusMessage && (
            <StatusDisplay message={statusMessage.message} type={statusMessage.type as any} />
          )}
          {statusMessage?.type === 'info' && <LoadingSpinner size="small" />}

          <div className="quiz-header flex justify-between items-center mb-8 text-text-secondary">
            <span id="question-number" className="text-lg font-semibold">Question {currentQuestionIndex + 1} / {quizData.totalQuestions}</span>
            <span id="timer-display" className="timer-display timer-normal text-xl font-bold">⏱️ {timeRemaining}s</span>
          </div>
          
          <div className="question-container mb-8">
            <h3 id="question-text" className="question-text mb-6 text-2xl font-semibold text-text-primary leading-relaxed">
              <MathJax>{currentQuestion.text}</MathJax>
            </h3>
            <div id="options-container" className="options-container">
              {currentQuestion.options.map((option, index) => {
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
                      {String.fromCharCode(65 + index)}.
                    </span>
                    <span className="option-content">
                      <MathJax>{option.text}</MathJax>
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Feedback Container */}
          {showExplanation && (
            <div id="feedback-container" className="feedback-container mb-6">
              <div className={`feedback-message ${selectedOptionId === currentQuestion.correctAnswerId ? 'correct' : 'incorrect'}`}>
                <div className="feedback-icon">
                  {selectedOptionId === currentQuestion.correctAnswerId ? '✅' : '❌'}
                </div>
                <div className="feedback-text">
                  {selectedOptionId === currentQuestion.correctAnswerId ? 'Correct!' : 'Incorrect'}
                </div>
              </div>
            </div>
          )}

          {/* Explanation Container */}
          {showExplanation && (
            <div id="explanation-container" className="explanation-container mb-6">
              <h4 className="text-lg text-primary-color mb-3 font-semibold">💡 Explanation:</h4>
              <p className="text-text-primary leading-relaxed m-0">
                <MathJax>{currentQuestion.explanation}</MathJax>
              </p>
            </div>
          )}

          <div className="quiz-footer">
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
              <div id="quiz-navigation" className="quiz-navigation">
                <button 
                  id="prev-question-btn" 
                  className="btn btn-secondary" 
                  onClick={() => console.log('Previous clicked')}
                  disabled={currentQuestionIndex === 0}
                >
                  ← Previous
                </button>
                <button 
                  id="next-question-btn" 
                  className="btn btn-primary" 
                  onClick={handleNextQuestion}
                >
                  {quizData && currentQuestionIndex < quizData.totalQuestions - 1 ? 'Next →' : 'Finish Quiz'}
                </button>
              </div>
            )}
          </div>
          <TokenStreamDisplay isVisible={tokenStreamingEnabled && quizInternalScreen === 'game'} />
        </div>
      )}

      <div id="quiz-results" className={`quiz-results bg-bg-secondary rounded-2xl p-8 shadow-md screen ${quizInternalScreen === 'results' ? 'active' : ''}`}>
        <div className="results-container text-center">
          <h2 className="text-3xl font-bold text-text-primary mb-6">Quiz Complete!</h2>
          <div className="score-display mb-6">
                        <div className="score-circle mx-auto mb-4 w-32 h-32 rounded-full bg-primary-color text-white flex items-center justify-center">
              <span className="text-2xl font-bold">{quizData ? Math.round((score / quizData.totalQuestions) * 100) : 0}%</span>
            </div>
            <div className="score-details text-lg text-text-secondary">
              {score}/{quizData ? quizData.totalQuestions : 0} correct answers
            </div>
          </div>
          <div className="results-actions">
            <button 
              className="btn btn-primary mr-4" 
              onClick={resetQuiz}
            >
              🆕 New Quiz
            </button>
            <button 
              className="btn btn-secondary" 
              onClick={() => showScreen('review')}
            >
              📋 Review Answers
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QuizPage;
