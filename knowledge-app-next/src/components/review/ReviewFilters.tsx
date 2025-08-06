'use client';

import React from 'react';

interface ReviewFiltersProps {
  searchTerm: string;
  onSearchChange: (term: string) => void;
  selectedTopic: string;
  onTopicChange: (topic: string) => void;
  topics: string[];
  selectedDifficulty: string;
  onDifficultyChange: (difficulty: string) => void;
  difficulties: string[];
  totalQuestions: number;
}

export const ReviewFilters: React.FC<ReviewFiltersProps> = ({
  searchTerm,
  onSearchChange,
  selectedTopic,
  onTopicChange,
  topics,
  selectedDifficulty,
  onDifficultyChange,
  difficulties,
  totalQuestions,
}) => {
  const clearAllFilters = () => {
    onSearchChange('');
    onTopicChange('');
    onDifficultyChange('');
  };

  const hasActiveFilters = searchTerm || selectedTopic || selectedDifficulty;

  return (
    <div className="review-filters">
      <div className="filters-header">
        <h3>🔍 Filter Questions</h3>
        <div className="results-count">
          <span>{totalQuestions} questions found</span>
          {hasActiveFilters && (
            <button className="clear-filters-btn" onClick={clearAllFilters}>
              🗑️ Clear Filters
            </button>
          )}
        </div>
      </div>

      <div className="filters-grid">
        {/* Search Filter */}
        <div className="filter-group">
          <label htmlFor="search">Search Questions</label>
          <div className="search-input-wrapper">
            <input
              id="search"
              type="text"
              placeholder="Search by question content or topic..."
              value={searchTerm}
              onChange={(e) => onSearchChange(e.target.value)}
              className="search-input"
            />
            {searchTerm && (
              <button
                className="clear-search-btn"
                onClick={() => onSearchChange('')}
              >
                ✕
              </button>
            )}
          </div>
        </div>

        {/* Topic Filter */}
        <div className="filter-group">
          <label htmlFor="topic">Filter by Topic</label>
          <select
            id="topic"
            value={selectedTopic}
            onChange={(e) => onTopicChange(e.target.value)}
            className="filter-select"
          >
            {topics.map(topic => (
              <option key={topic} value={topic === 'All Topics' ? '' : topic}>
                {topic}
              </option>
            ))}
          </select>
        </div>

        {/* Difficulty Filter */}
        <div className="filter-group">
          <label htmlFor="difficulty">Filter by Difficulty</label>
          <select
            id="difficulty"
            value={selectedDifficulty}
            onChange={(e) => onDifficultyChange(e.target.value)}
            className="filter-select"
          >
            {difficulties.map(difficulty => (
              <option key={difficulty} value={difficulty === 'All Difficulties' ? '' : difficulty}>
                {difficulty}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Active Filters Display */}
      {hasActiveFilters && (
        <div className="active-filters">
          <span className="active-filters-label">Active filters:</span>
          <div className="filter-tags">
            {searchTerm && (
              <span className="filter-tag">
                Search: "{searchTerm}"
                <button onClick={() => onSearchChange('')}>✕</button>
              </span>
            )}
            {selectedTopic && (
              <span className="filter-tag">
                Topic: {selectedTopic}
                <button onClick={() => onTopicChange('')}>✕</button>
              </span>
            )}
            {selectedDifficulty && (
              <span className="filter-tag">
                Difficulty: {selectedDifficulty}
                <button onClick={() => onDifficultyChange('')}>✕</button>
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};