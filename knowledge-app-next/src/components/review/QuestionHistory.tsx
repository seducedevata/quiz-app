import React from 'react';

interface Question {
  id: string;
  question: string;
  answer: string;
  topic: string;
  difficulty: string;
  // Add other relevant question properties
}

interface QuestionFilters {
  topic: string;
  difficulty: string;
  searchTerm: string;
}

interface QuestionHistoryProps {
  questions: Question[];
  onQuestionSelect: (question: Question) => void;
  filters: QuestionFilters;
  onFiltersChange: (filters: QuestionFilters) => void;
}

export const QuestionHistory: React.FC<QuestionHistoryProps> = ({
  questions,
  onQuestionSelect,
  filters,
  onFiltersChange
}) => {
  // Basic filtering logic (can be enhanced)
  const filteredQuestions = questions.filter(q => {
    const matchesTopic = filters.topic ? q.topic === filters.topic : true;
    const matchesDifficulty = filters.difficulty ? q.difficulty === filters.difficulty : true;
    const matchesSearch = filters.searchTerm ? 
      q.question.toLowerCase().includes(filters.searchTerm.toLowerCase()) ||
      q.answer.toLowerCase().includes(filters.searchTerm.toLowerCase()) : true;
    return matchesTopic && matchesDifficulty && matchesSearch;
  });

  return (
    <div className="question-history-container">
      <h2>Question History</h2>
      <div className="filters">
        <input
          type="text"
          placeholder="Search questions..."
          value={filters.searchTerm}
          onChange={(e) => onFiltersChange({ ...filters, searchTerm: e.target.value })}
        />
        {/* Add more filter inputs for topic and difficulty */}
      </div>
      <div className="question-statistics mt-4 p-4 bg-gray-100 rounded-md">
        <h3 className="text-lg font-semibold mb-2">Statistics</h3>
        <p>Total Questions: {questions.length}</p>
        <p>Easy Questions: {questions.filter(q => q.difficulty === 'easy').length}</p>
        <p>Hard Questions: {questions.filter(q => q.difficulty === 'hard').length}</p>
      </div>
      <div className="question-list">
        {filteredQuestions.length > 0 ? (
          filteredQuestions.map(question => (
            <div key={question.id} className="question-card" onClick={() => onQuestionSelect(question)}>
              <h3>{question.question}</h3>
              <p>Topic: {question.topic} | Difficulty: {question.difficulty}</p>
            </div>
          ))
        ) : (
          <p>No questions found matching your criteria.</p>
        )}
      </div>
    </div>
  );
};
