'use client';

import React, { useState, useEffect } from 'react';
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
  user_answer?: number; // Optional, for review mode
  is_correct?: boolean; // Optional, for review mode
}

export default function ReviewPage() {
  const [questions, setQuestions] = useState<QuestionHistoryItem[]>([]);
  const [topics, setTopics] = useState<string[]>([]);
  const [filterTopic, setFilterTopic] = useState('');
  const [filterDifficulty, setFilterDifficulty] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [modalQuestion, setModalQuestion] = useState<QuestionHistoryItem | null>(null);

  const loadQuestionHistory = async (offset = 0, limit = 50) => {
    AppLogger.info('REVIEW', 'Loading question history', { offset, limit });
    setLoading(true);
    setError('');
    try {
      const result = await callPythonMethod<string>('getQuestionHistory', offset, limit);
      const data = JSON.parse(result);

      if (data.success) {
        setQuestions(data.questions || []);
        updateTopicFilter(data.questions || []);
        AppLogger.success('REVIEW', `Loaded ${data.questions?.length || 0} questions from history`);
      } else {
        AppLogger.error('REVIEW', 'Failed to load question history', { error: data.error });
        setError(data.error || 'Failed to load question history.');
      }
    } catch (e: any) {
      AppLogger.error('REVIEW', 'Error calling Python bridge for question history', { error: e.message });
      setError(e.message || 'Error loading question history.');
    } finally {
      setLoading(false);
    }
  };

  const updateTopicFilter = (questions: QuestionHistoryItem[]) => {
    const uniqueTopics = [...new Set(questions.map(q => q.topic || 'General'))].sort();
    setTopics(uniqueTopics);
    AppLogger.debug('REVIEW', `Updated topic filter with ${uniqueTopics.length} topics`);
  };

  const searchQuestions = async () => {
    AppLogger.debug('REVIEW', 'Searching questions', { searchTerm });
    setLoading(true);
    setError('');
    try {
      const result = await callPythonMethod<string>('searchQuestions', searchTerm);
      const data = JSON.parse(result);
      if (data.success) {
        setQuestions(data.questions || []);
        AppLogger.success('REVIEW', `Found ${data.questions?.length || 0} questions matching: ${searchTerm}`);
      } else {
        AppLogger.error('REVIEW', 'Error searching questions', { error: data.error });
        setError(data.error || 'Error searching questions.');
      }
    } catch (e: any) {
      AppLogger.error('REVIEW', 'Error searching questions', { error: e.message });
      setError(e.message || 'Error searching questions.');
    } finally {
      setLoading(false);
    }
  };

  const filterQuestionsByTopic = async (topic: string) => {
    AppLogger.debug('REVIEW', 'Filtering questions by topic', { topic });
    setLoading(true);
    setError('');
    try {
      const result = await callPythonMethod<string>('filterQuestionsByTopic', topic);
      const data = JSON.parse(result);
      if (data.success) {
        setQuestions(data.questions || []);
        AppLogger.success('REVIEW', `Filtered by topic: ${topic || 'All'}, found ${data.questions?.length || 0} questions`);
      } else {
        AppLogger.error('REVIEW', 'Error filtering questions by topic', { error: data.error });
        setError(data.error || 'Error filtering questions by topic.');
      }
    } catch (e: any) {
      AppLogger.error('REVIEW', 'Error filtering questions by topic', { error: e.message });
      setError(e.message || 'Error filtering questions by topic.');
    } finally {
      setLoading(false);
    }
  };

  const filterQuestionsByDifficulty = async (difficulty: string) => {
    AppLogger.debug('REVIEW', 'Filtering questions by difficulty', { difficulty });
    setLoading(true);
    setError('');
    try {
      const result = await callPythonMethod<string>('filterQuestionsByDifficulty', difficulty);
      const data = JSON.parse(result);
      if (data.success) {
        setQuestions(data.questions || []);
        AppLogger.success('REVIEW', `Filtered by difficulty: ${difficulty || 'All'}, found ${data.questions?.length || 0} questions`);
      } else {
        AppLogger.error('REVIEW', 'Error filtering questions by difficulty', { error: data.error });
        setError(data.error || 'Error filtering questions by difficulty.');
      }
    } catch (e: any) {
      AppLogger.error('REVIEW', 'Error filtering questions by difficulty', { error: e.message });
      setError(e.message || 'Error filtering questions by difficulty.');
    } finally {
      setLoading(false);
    }
  };

  const showQuestionModal = (question: QuestionHistoryItem) => {
    setModalQuestion(question);
    AppLogger.action('REVIEW', 'Question modal opened', { questionId: question.id });
  };

  const closeQuestionModal = () => {
    setModalQuestion(null);
    AppLogger.action('REVIEW', 'Question modal closed');
  };

  useEffect(() => {
    loadQuestionHistory();
  }, []);

  const getDifficultyBadgeColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'easy': return 'bg-success-color';
      case 'medium': return 'bg-warning-color';
      case 'hard': return 'bg-error-color';
      case 'expert': return 'bg-purple-600';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="review-container p-8">
      <h1 className="text-3xl font-bold text-text-primary mb-6">Question History & Review</h1>

      <div className="review-filters bg-bg-secondary p-6 rounded-lg shadow-md mb-6">
        <div className="filters-grid grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
          <div className="form-group">
            <label htmlFor="question-search" className="block text-text-secondary text-sm font-bold mb-2">Search</label>
            <input
              type="text"
              id="question-search"
              className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
              placeholder="Search questions..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onKeyUp={(e) => { if (e.key === 'Enter') searchQuestions(); }}
            />
          </div>

          <div className="form-group">
            <label htmlFor="topic-filter" className="block text-text-secondary text-sm font-bold mb-2">Topic</label>
            <select
              id="topic-filter"
              className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
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

          <div className="form-group">
            <label htmlFor="difficulty-filter" className="block text-text-secondary text-sm font-bold mb-2">Difficulty</label>
            <select
              id="difficulty-filter"
              className="w-full p-2 border border-border-color rounded-md bg-bg-primary text-text-primary"
              value={filterDifficulty}
              onChange={(e) => {
                setFilterDifficulty(e.target.value);
                filterQuestionsByDifficulty(e.target.value);
              }}
            >
              <option value="">All Difficulties</option>
              <option value="Easy">Easy</option>
              <option value="Medium">Medium</option>
              <option value="Hard">Hard</option>
              <option value="Expert">Expert</option>
            </select>
          </div>

          <div className="form-group flex items-end">
            <button
              className="btn-primary w-full py-2 px-4 rounded-md"
              onClick={() => {
                setSearchTerm('');
                setFilterTopic('');
                setFilterDifficulty('');
                loadQuestionHistory();
              }}
            >
              Clear Filters
            </button>
          </div>
        </div>
      </div>

      {loading && (
        <div id="loading-history" className="text-center text-text-secondary py-8">
          <div className="loading-spinner mx-auto mb-4"></div>
          <p>Loading question history...</p>
        </div>
      )}

      {error && (
        <div className="error-message text-center py-8">
          <span className="error-icon">❌</span>
          <p>{error}</p>
          <button className="btn-primary mt-4" onClick={() => loadQuestionHistory()}>Retry</button>
        </div>
      )}

      {!loading && !error && questions.length === 0 && (
        <div id="no-questions" className="empty-state text-center py-8">
          <span className="empty-icon text-5xl mb-4">😔</span>
          <h3 className="text-xl font-semibold text-text-primary mb-2">No Questions Found</h3>
          <p className="text-text-secondary">Adjust your filters or take some quizzes to see questions here.</p>
        </div>
      )}

      {!loading && !error && questions.length > 0 && (
        <div id="questions-list" className="questions-grid grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {questions.map((question, index) => (
            <QuestionHistoryCard
              key={question.id || index}
              question={question}
              getDifficultyBadgeColor={getDifficultyBadgeColor}
              onCardClick={showQuestionModal}
            />
          ))}
        </div>
      )}

      {modalQuestion && (
        <QuestionDetailModal
          question={modalQuestion}
          onClose={closeQuestionModal}
          getDifficultyBadgeColor={getDifficultyBadgeColor}
        />
      )}
    </div>
  );
}

interface QuestionHistoryCardProps {
  question: QuestionHistoryItem;
  getDifficultyBadgeColor: (difficulty: string) => string;
  onCardClick: (question: QuestionHistoryItem) => void;
}

const QuestionHistoryCard: React.FC<QuestionHistoryCardProps> = ({ question, getDifficultyBadgeColor, onCardClick }) => {
  return (
    <div className="question-card" onClick={() => onCardClick(question)}>
      <div className="question-header">
        <span className="question-topic">{question.topic || 'General'}</span>
        <span className={`question-difficulty ${getDifficultyBadgeColor(question.difficulty)}`}>
          {question.difficulty || 'Medium'}
        </span>
      </div>
      <div className="question-text-review">
        <MathJax>{question.question || 'No question text'}</MathJax>
      </div>
      <div className="question-options-review">
        {(question.options || []).map((option, i) => (
          <div key={i} className={`option-item ${i === question.correct ? 'correct' : ''}`}>
            {String.fromCharCode(65 + i)}. <MathJax>{option}</MathJax>
          </div>
        ))}
      </div>
      <div className="question-meta">
        <span>ID: {question.id || 'Unknown'}</span>
        <span>{new Date(question.timestamp).toLocaleString()}</span>
      </div>
    </div>
  );
};

interface QuestionDetailModalProps {
  question: QuestionHistoryItem;
  onClose: () => void;
  getDifficultyBadgeColor: (difficulty: string) => string;
}

const QuestionDetailModal: React.FC<QuestionDetailModalProps> = ({ question, onClose, getDifficultyBadgeColor }) => {
  return (
    <div id="question-detail-modal" className="modal" style={{ display: 'block' }}>
      <div className="modal-content">
        <button className="modal-close" onClick={onClose}>×</button>
        <div id="modal-question-content" className="question-detail">
          <div className="question-meta-header">
            <span className="badge">📚 {question.topic || 'General'}</span>
            <span className={`badge ${getDifficultyBadgeColor(question.difficulty)}`}>
              ⭐ {question.difficulty || 'Medium'}
            </span>
          </div>

          <div className="question-text">
            <MathJax>{question.question || 'No question text'}</MathJax>
          </div>

          <div className="question-options">
            <h4>Answer Options:</h4>
            {(question.options || []).map((option, i) => (
              <div
                key={i}
                className="option-item"
              >
                <strong>{String.fromCharCode(65 + i)}.</strong> <MathJax>{option}</MathJax>
                {i === question.correct ? ' ✅ (Correct Answer)' : ''}
              </div>
            ))}
          </div>

          {question.explanation && (
            <div className="question-explanation">
              <h4>💡 Explanation:</h4>
              <p><MathJax>{question.explanation}</MathJax></p>
            </div>
          )}

          <div className="question-footer">
            <div>Question ID: {question.id || 'Unknown'}</div>
            <div>Generated: {new Date(question.timestamp).toLocaleString()}</div>
          </div>
        </div>
      </div>
    </div>
  );
};