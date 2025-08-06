'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { supabase } from '@/utils/supabaseClient';
import { Icon } from '@/components/common/Icon';

interface Question {
  id: number;
  question: string;
  options: string[];
  correct: number;
  topic: string;
  difficulty: string;
  explanation: string;
}

export const ReviewScreen: React.FC = () => {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchQuestions = async () => {
      const { data, error } = await supabase
        .from('quizzes')
        .select('questions')
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Error fetching questions:', error);
      } else {
        const allQuestions = data.flatMap((quiz: any) => quiz.questions);
        setQuestions(allQuestions);
      }
      setLoading(false);
    };

    fetchQuestions();
  }, []);

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div className="max-w-5xl mx-auto">
      <h1 className="text-h1 font-h1 text-textPrimary mb-xl">Question Review</h1>

      {questions.length === 0 ? (
        <Card>
          <p className="text-body text-textPrimary text-center">No questions found. Generate some questions first!</p>
          <Button onClick={() => {}} className="mt-lg">
            Start Quiz
          </Button>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-lg">
          {questions.map((question) => (
            <Card key={question.id}>
              <p className="text-h4 font-h4 text-textPrimary mb-md">{question.question}</p>
              <p className="text-body text-textSecondary mb-md">{question.topic} - {question.difficulty}</p>
              <Button onClick={() => {}} variant="secondary">
                <Icon name="FaEye" className="mr-sm" />
                View Details
              </Button>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};