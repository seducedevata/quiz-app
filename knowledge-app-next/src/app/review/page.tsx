'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { callPythonMethod } from '../../lib/pythonBridge';
import { AppLogger } from '../../lib/logger';
import { MathJax } from 'better-react-mathjax';

interface QuestionHistoryItem {
  id: string;
  question: string;
  options: string[];
  correct: number;
  topic: string;
  difficulty: 'Easy' | 'Medium' | 'Hard' | 'Expert';
  timestamp: string;
  explanation: string;
  user_answer?: number;
  is_correct?: boolean;
}

export default function ReviewPage() {
  const router = useRouter();
  const [questions, setQuestions] = useState<QuestionHistoryItem[]>([]);
  const [topics, setTopics] = useState<string[]>([]);
  const [filterTopic, setFilterTopic] = useState('');
  const [filterDifficulty, setFilterDifficulty] = useState('');
  const [filterStartDate, setFilterStartDate] = useState('');
  const [filterEndDate, setFilterEndDate] = useState('');
  const [filterMinScore, setFilterMinScore] = useState<string>('');
  const [filterMaxScore, setFilterMaxScore] = useState<string>('');
  const [filterModel, setFilterModel] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [modalQuestion, setModalQuestion] = useState<QuestionHistoryItem | null>(null);
  const [showStats, setShowStats] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [questionsPerPage] = useState(10); // Number of questions per page
  const [totalPages, setTotalPages] = useState(0);
  const [stats, setStats] = useState({
    total: 0,
    topics: 0,
    expert: 0
  });

  // Mock data for demo until backend is connected
  const mockQuestions: QuestionHistoryItem[] = [
    {
      id: 'q_007/08/2025_11:42:54',
      question: 'Sample question 0?',
      options: ['Option A', 'Option B', 'Option C', 'Option D'],
      correct: 0,
      topic: 'General',
      difficulty: 'Easy',
      timestamp: '2025-08-07T11:42:54',
      explanation: 'This is the explanation for the sample question.'
    },
    {
      id: 'q_106/08/2025_11:42:54',
      question: 'Sample question 1?',
      options: ['Option A', 'Option B', 'Option C', 'Option D'],
      correct: 1,
      topic: 'General',
      difficulty: 'Medium',
      timestamp: '2025-08-06T11:42:54',
      explanation: 'This is another explanation.'
    },
    {
      id: 'q_205/08/2025_11:42:54',
      question: 'Sample question 2?',
      options: ['Option A', 'Option B', 'Option C', 'Option D'],
      correct: 2,
      topic: 'General',
      difficulty: 'Hard',
      timestamp: '2025-08-05T11:42:54',
      explanation: 'A more complex explanation here.'
    }
  ];

  const loadQuestionHistory = async (offset = 0, limit = 50, topicFilter = '', difficultyFilter = '', searchTerm = '', startDate = '', endDate = '', minScore: string | number = '', maxScore: string | number = '', model = '') => {
    AppLogger.info('REVIEW', 'Loading question history', { offset, limit, topicFilter, difficultyFilter, searchTerm, startDate, endDate, minScore, maxScore, model });
    setLoading(true);
    setError('');
    try {
      const response = await callPythonMethod('get_question_history', {
        offset,
        limit,
        topic: topicFilter,
        difficulty: difficultyFilter,
        search: searchTerm,
        start_date: startDate,
        end_date: endDate,
        min_score: minScore !== '' ? Number(minScore) : undefined,
        max_score: maxScore !== '' ? Number(maxScore) : undefined,
        model: model,
      });
      const backendQuestions = response.questions || [];
      const totalCount = response.total_count || backendQuestions.length; // Assuming backend returns total_count
      setTotalPages(Math.ceil(totalCount / questionsPerPage));
      
      if (backendQuestions.length > 0) {
        setQuestions(backendQuestions);
        updateTopicFilter(backendQuestions);
        updateStats(backendQuestions);
        AppLogger.success('REVIEW', `Loaded ${backendQuestions.length} questions from backend`);
      } else {
        // Fallback to mock data if backend returns empty
        setQuestions(mockQuestions);
        updateTopicFilter(mockQuestions);
        updateStats(mockQuestions);
        AppLogger.info('REVIEW', 'Using mock data - no questions in backend history');
      }
    } catch (e: any) {
      AppLogger.error('REVIEW', 'Failed to load question history from backend, using mock data', e);
      // Fallback to mock data on error
      setQuestions(mockQuestions);
      updateTopicFilter(mockQuestions);
      updateStats(mockQuestions);
      setError(`Backend connection failed: ${e.message}. Using sample data.`);
    } finally {
      setLoading(false);
    }
  };

  const nuclearLoadQuestions = () => {
    console.log('🚀 NUCLEAR: Force loading questions...');
    loadQuestionHistory();
  };

  const nuclearTestDisplay = () => {
    console.log('🧪 NUCLEAR: Testing display...');
    setQuestions(mockQuestions);
  };

  const searchQuestions = () => {
    setCurrentPage(1);
    loadQuestionHistory(0, questionsPerPage, filterTopic, filterDifficulty, searchTerm);
  };

  const filterQuestionsByTopic = (topic: string) => {
    setFilterTopic(topic);
    setCurrentPage(1);
    loadQuestionHistory(0, questionsPerPage, topic, filterDifficulty, searchTerm);
  };

  const filterQuestionsByDifficulty = (difficulty: string) => {
    setFilterDifficulty(difficulty);
    setCurrentPage(1);
    loadQuestionHistory(0, questionsPerPage, filterTopic, difficulty, searchTerm);
  };

  const updateTopicFilter = (questions: QuestionHistoryItem[]) => {
    const uniqueTopics = [...new Set(questions.map(q => q.topic || 'General'))].sort();
    setTopics(uniqueTopics);
  };

  const updateStats = (questions: QuestionHistoryItem[]) => {
    const uniqueTopics = new Set(questions.map(q => q.topic)).size;
    const expertCount = questions.filter(q => q.difficulty === 'Expert').length;
    
    setStats({
      total: questions.length,
      topics: uniqueTopics,
      expert: expertCount
    });
  };

  const showQuestionModal = (question: QuestionHistoryItem) => {
    setModalQuestion(question);
  };

  const closeQuestionModal = () => {
    setModalQuestion(null);
  };

  const toggleStats = () => {
    setShowStats(!showStats);
  };

  const clearFilters = () => {
    setSearchTerm('');
    setFilterTopic('');
    setFilterDifficulty('');
    setFilterStartDate('');
    setFilterEndDate('');
    setFilterMinScore('');
    setFilterMaxScore('');
    setFilterModel('');
    setCurrentPage(1);
    loadQuestionHistory(0, questionsPerPage, '', '', '', '', '', '', '', '');
  };

  const retakeQuiz = (question: QuestionHistoryItem) => {
    // Navigate back to quiz setup page instead of directly to quiz
    // This allows users to configure settings before retaking
    router.push('/quiz');
  };

  const retakeTopicQuiz = (topic: string) => {
    // Navigate back to quiz setup page
    router.push('/quiz');
  };

  const startNewQuiz = () => {
    // Navigate to quiz setup page for a fresh start
    router.push('/quiz');
  };

  useEffect(() => {
    loadQuestionHistory(
      (currentPage - 1) * questionsPerPage,
      questionsPerPage,
      filterTopic,
      filterDifficulty,
      searchTerm,
      filterStartDate,
      filterEndDate,
      filterMinScore,
      filterMaxScore,
      filterModel
    );
  }, [currentPage, filterTopic, filterDifficulty, searchTerm, filterStartDate, filterEndDate, filterMinScore, filterMaxScore, filterModel]);

  const getDifficultyBadgeColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'easy': return 'difficulty-easy';
      case 'medium': return 'difficulty-medium';
      case 'hard': return 'difficulty-hard';
      case 'expert': return 'difficulty-expert';
      default: return 'difficulty-medium';
    }
  };

  const getDifficultyIcon = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'easy': return '🟢';
      case 'medium': return '🟡';
      case 'hard': return '🔴';
      case 'expert': return '🔥💀';
      default: return '🟡';
    }
  };

  return (
    <div className="review-container">
      <h2>📚 Question Review & History</h2>
      <p>Review all your previously generated questions without burning API calls or GPU resources!</p>

      {/* NUCLEAR OPTION: Direct Question Display - EXACT Qt styling */}
      <div className="nuclear-review-section" style={{ background: '#ffe6e6', padding: '10px', margin: '10px', border: '2px solid #ff0000' }}>
        <h3>🚀 NUCLEAR DEBUG MODE</h3>
        <button className="btn btn-primary" onClick={nuclearLoadQuestions} style={{ margin: '5px' }}>
          🚀 FORCE LOAD QUESTIONS
        </button>
        <button className="btn btn-secondary" onClick={nuclearTestDisplay} style={{ margin: '5px' }}>
          🧪 TEST DISPLAY
        </button>
        <div style={{ background: '#f0f0f0', padding: '10px', margin: '10px', fontFamily: 'monospace', fontSize: '12px', maxHeight: '200px', overflowY: 'auto' }}>
          {loading ? 'Loading questions...' : `Found ${questions.length} questions in history`}
        </div>
      </div>

      {/* Review Controls - EXACT Qt layout */}
      <div className="review-controls">
        <div className="filter-section">
          <h3>🔍 Filters</h3>
          <div className="filter-grid">
            <div className="filter-item">
              <label>Search Questions</label>
              <input
                type="text"
                placeholder="Search by content..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyUp={(e) => { if (e.key === 'Enter') searchQuestions(); }}
              />
            </div>
            <div className="filter-item">
              <label>Filter by Topic</label>
              <select
                value={filterTopic}
                onChange={(e) => {
                  setFilterTopic(e.target.value);
                  filterQuestionsByTopic(e.target.value);
                }}
              >
                <option value="">All Topics</option>
                {topics.map(topic => (
                  <option key={topic} value={topic}>{topic}</option>
                ))}
              </select>
            </div>
            <div className="filter-item">
              <label>Filter by Difficulty</label>
              <select
                value={filterDifficulty}
                onChange={(e) => {
                  setFilterDifficulty(e.target.value);
                  filterQuestionsByDifficulty(e.target.value);
                }}
              >
                <option value="">All Difficulties</option>
                <option value="easy">🟢 Easy</option>
                <option value="medium">🟡 Medium</option>
                <option value="hard">🔴 Hard</option>
                <option value="expert">🔥💀 Expert</option>
              </select>
            </div>
            <div className="filter-item">
              <label>Start Date</label>
              <input
                type="date"
                value={filterStartDate}
                onChange={(e) => setFilterStartDate(e.target.value)}
              />
            </div>
            <div className="filter-item">
              <label>End Date</label>
              <input
                type="date"
                value={filterEndDate}
                onChange={(e) => setFilterEndDate(e.target.value)}
              />
            </div>
            <div className="filter-item">
              <label>Min Score</label>
              <input
                type="number"
                value={filterMinScore}
                onChange={(e) => setFilterMinScore(e.target.value)}
                min="0"
                max="100"
              />
            </div>
            <div className="filter-item">
              <label>Max Score</label>
              <input
                type="number"
                value={filterMaxScore}
                onChange={(e) => setFilterMaxScore(e.target.value)}
                min="0"
                max="100"
              />
            </div>
            <div className="filter-item">
              <label>Model Used</label>
              <input
                type="text"
                value={filterModel}
                onChange={(e) => setFilterModel(e.target.value)}
                placeholder="e.g., deepseek-coder"
              />
            </div>
          </div>
          <div className="review-actions">
            <button className="btn btn-primary" onClick={() => loadQuestionHistory()}>🔄 Refresh History</button>
            <button className="btn btn-secondary" onClick={toggleStats}>📊 Statistics</button>
            <button className="btn btn-secondary" onClick={clearFilters}>🗑️ Clear Filters</button>
          </div>
        </div>
      </div>
      
      {/* Statistics Display - EXACT Qt styling */}
      {showStats && (
        <div className="question-stats">
          <h3>📊 Question Statistics</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <h3>{stats.total}</h3>
              <p>Total Questions</p>
            </div>
            <div className="stat-card">
              <h3>{stats.topics}</h3>
              <p>Topics Covered</p>
            </div>
            <div className="stat-card">
              <h3>{stats.expert}</h3>
              <p>Expert Level</p>
            </div>
          </div>
        </div>
      )}

      {/* Questions Display */}
      <div className="question-history-container">
        {loading && (
          <div className="loading-message">
            <p>⏳ Loading question history...</p>
          </div>
        )}

        {error && (
          <div className="error-message">
            <span className="error-icon">❌</span>
            <p>{error}</p>
            <button className="btn btn-primary" onClick={() => loadQuestionHistory()}>Retry</button>
          </div>
        )}

        {!loading && !error && questions.length === 0 && (
          <div className="no-questions">
            <p>📝 No questions found. Generate some questions first!</p>
            <button className="btn btn-primary" onClick={startNewQuiz}>Start Quiz Setup</button>
          </div>
        )}

        {!loading && !error && questions.length > 0 && (
          <div className="questions-list">
            {questions.map((question, index) => (
              <div key={question.id || index} className="question-card">
                <div className="question-header">
                  <span className="question-topic">{question.topic || 'General'}</span>
                  <span className={`question-difficulty ${getDifficultyBadgeColor(question.difficulty)}`}>
                    {getDifficultyIcon(question.difficulty)} {question.difficulty || 'Medium'}
                  </span>
                </div>
                
                <div className="question-text-review">
                  <MathJax>{question.question || 'No question text'}</MathJax>
                </div>
                
                <div className="question-options-review">
                  {(question.options || []).map((option, i) => (
                    <div key={i} className={`option-item ${i === question.correct ? 'correct' : ''}`}>
                      <span className="option-letter">{String.fromCharCode(65 + i)}.</span>
                      <span className="option-text">
                        <MathJax>{option}</MathJax>
                      </span>
                      {i === question.correct && <span className="correct-indicator">✅</span>}
                    </div>
                  ))}
                </div>
                
                <div className="question-actions">
                  <button 
                    className="btn btn-small btn-info" 
                    onClick={() => showQuestionModal(question)}
                  >
                    📖 View Details
                  </button>
                  <button 
                    className="btn btn-small btn-primary" 
                    onClick={() => retakeQuiz(question)}
                  >
                    🔄 New Quiz
                  </button>
                  <button 
                    className="btn btn-small btn-secondary" 
                    onClick={() => retakeTopicQuiz(question.topic)}
                  >
                    📚 Quiz Setup
                  </button>
                </div>
                
                <div className="question-meta">
                  <span>ID: {question.id || 'Unknown'}</span>
                  <span>{new Date(question.timestamp).toLocaleString()}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Pagination Controls */}
        {totalPages > 1 && (
          <div className="pagination-controls flex justify-center items-center space-x-4 mt-8">
            <button
              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
              disabled={currentPage === 1}
              className="px-4 py-2 bg-gray-700 text-white rounded-md hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <span className="text-white">Page {currentPage} of {totalPages}</span>
            <button
              onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
              disabled={currentPage === totalPages}
              className="px-4 py-2 bg-gray-700 text-white rounded-md hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        )}
      </div>

      {/* Question Detail Modal - Enhanced */}
      {modalQuestion && (
        <div className="modal" style={{ display: 'block' }}>
          <div className="modal-content">
            <div className="modal-header">
              <h3>Question Details</h3>
              <span className="modal-close" onClick={closeQuestionModal}>&times;</span>
            </div>
            <div className="modal-body">
              <div className="question-meta-header">
                <span className="badge topic-badge">📚 {modalQuestion.topic || 'General'}</span>
                <span className={`badge difficulty-badge ${getDifficultyBadgeColor(modalQuestion.difficulty)}`}>
                  {getDifficultyIcon(modalQuestion.difficulty)} {modalQuestion.difficulty || 'Medium'}
                </span>
              </div>

              <div className="question-text">
                <h4>Question:</h4>
                <MathJax>{modalQuestion.question || 'No question text'}</MathJax>
              </div>

              <div className="question-options">
                <h4>Answer Options:</h4>
                {(modalQuestion.options || []).map((option, i) => (
                  <div key={i} className={`option-item ${i === modalQuestion.correct ? 'correct-option' : ''}`}>
                    <strong>{String.fromCharCode(65 + i)}.</strong>
                    <MathJax>{option}</MathJax>
                    {i === modalQuestion.correct && <span className="correct-indicator">✅ (Correct Answer)</span>}
                  </div>
                ))}
              </div>

              {modalQuestion.explanation && (
                <div className="question-explanation">
                  <h4>💡 Explanation:</h4>
                  <div className="explanation-content">
                    <MathJax>{modalQuestion.explanation}</MathJax>
                  </div>
                </div>
              )}

              <div className="modal-actions">
                <button 
                  className="btn btn-primary" 
                  onClick={() => {
                    closeQuestionModal();
                    retakeQuiz(modalQuestion);
                  }}
                >
                  🔄 New Quiz
                </button>
                <button 
                  className="btn btn-secondary" 
                  onClick={() => {
                    closeQuestionModal();
                    retakeTopicQuiz(modalQuestion.topic);
                  }}
                >
                  📚 Quiz Setup
                </button>
                <button className="btn btn-secondary" onClick={closeQuestionModal}>
                  Close
                </button>
              </div>

              <div className="question-footer">
                <div><strong>Question ID:</strong> {modalQuestion.id || 'Unknown'}</div>
                <div><strong>Generated:</strong> {new Date(modalQuestion.timestamp).toLocaleString()}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
