'use client';

import React, { useState, useEffect, useCallback, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { TokenStream } from '../../../components/quiz/TokenStream';

interface Question {
    id: string;
    question: string;
    options: string[];
    correctAnswer: string;
    explanation: string;
}

const mockQuestions: Question[] = [
    {
        id: '1',
        question: 'What is the capital of France?',
        options: ['Berlin', 'Madrid', 'Paris', 'Rome'],
        correctAnswer: 'Paris',
        explanation: 'Paris is the capital and most populous city of France.',
    },
    {
        id: '2',
        question: 'Which planet is known as the Red Planet?',
        options: ['Earth', 'Mars', 'Jupiter', 'Venus'],
        correctAnswer: 'Mars',
        explanation: 'Mars is often referred to as the Red Planet due to its reddish appearance caused by iron oxide on its surface.',
    },
    {
        id: '3',
        question: 'What is the largest ocean on Earth?',
        options: ['Atlantic', 'Indian', 'Arctic', 'Pacific'],
        correctAnswer: 'Pacific',
        explanation: 'The Pacific Ocean is the largest and deepest of Earth\'s five oceanic divisions.',
    },
    {
        id: '4',
        question: 'Who painted the Mona Lisa?',
        options: ['Vincent van Gogh', 'Pablo Picasso', 'Leonardo da Vinci', 'Claude Monet'],
        correctAnswer: 'Leonardo da Vinci',
        explanation: 'The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci.',
    },
    {
        id: '5',
        question: 'What is the chemical symbol for water?',
        options: ['O2', 'H2O', 'CO2', 'NaCl'],
        correctAnswer: 'H2O',
        explanation: 'H2O is the chemical formula for water, meaning it has two hydrogen atoms and one oxygen atom.',
    },
];

const QuizInterfaceContent: React.FC = () => {
    const router = useRouter();
    const searchParams = useSearchParams();
    const topic = searchParams.get('topic') || 'General Knowledge';
    const difficulty = searchParams.get('difficulty') || 'medium';
    const numQuestions = parseInt(searchParams.get('numQuestions') || '5');
    const enableStreaming = searchParams.get('enableStreaming') === 'true';
    const quizMode = searchParams.get('mode') || 'standard';

    const [questions, setQuestions] = useState<Question[]>([]);
    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
    const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
    const [showExplanation, setShowExplanation] = useState(false);
    const [score, setScore] = useState(0);
    const [quizCompleted, setQuizCompleted] = useState(false);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [timer, setTimer] = useState<number | null>(null); // For timed quizzes

    const handleNextQuestion = useCallback(() => {
        setSelectedAnswer(null);
        setShowExplanation(false);
        if (currentQuestionIndex < questions.length - 1) {
            setCurrentQuestionIndex(prevIndex => prevIndex + 1);
            if (quizMode === 'serious') {
                setTimer(30); // Reset timer for next question
            }
        } else {
            setQuizCompleted(true);
        }
    }, [currentQuestionIndex, questions.length, quizMode]);

    useEffect(() => {
        if (quizMode === 'serious') {
            const countdown = setInterval(() => {
                setTimer((prevTimer) => {
                    if (prevTimer === null) return null;
                    if (prevTimer <= 1) {
                        clearInterval(countdown);
                        // Handle quiz completion or question skip due to timeout
                        if (currentQuestionIndex < questions.length - 1) {
                            handleNextQuestion();
                        } else {
                            setQuizCompleted(true);
                        }
                        return 0;
                    }
                    return prevTimer - 1;
                });
            }, 1000);
            return () => clearInterval(countdown);
        }
    }, [quizMode, currentQuestionIndex, questions.length, handleNextQuestion]);

    const handleAnswerSelect = (answer: string) => {
        if (!selectedAnswer) { // Only allow selecting once
            setSelectedAnswer(answer);
            setShowExplanation(true);
            if (answer === questions[currentQuestionIndex].correctAnswer) {
                setScore(prevScore => prevScore + 1);
            }
        }
    };

    useEffect(() => {
        // In a real app, fetch questions based on topic, difficulty, numQuestions
        // For now, use mock data and slice based on numQuestions
        setLoading(true);
        setError(null);
        try {
            const quizQuestions = mockQuestions.slice(0, numQuestions);
            if (quizQuestions.length === 0) {
                throw new Error('No questions available for this configuration.');
            }
            setQuestions(quizQuestions);
            setLoading(false);
            // Determine quiz mode from search params or default
            if (quizMode === 'serious') {
                setTimer(quizQuestions.length * 30); // 30 seconds per question for serious mode
            }
        } catch (err: any) {
            setError(err.message || 'Failed to load quiz questions.');
            setLoading(false);
        }
    }, [topic, difficulty, numQuestions, searchParams, quizMode]);

    const currentQuestion = questions[currentQuestionIndex];
    const progress = questions.length > 0 ? ((currentQuestionIndex + 1) / questions.length) * 100 : 0;

    if (loading) {
        return (
            <div className="quiz-loading">
                <div className="loading-spinner"></div>
                <p>Loading quiz...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="quiz-error">
                <p className="error-message">Error: {error}</p>
                <button onClick={() => router.push('/quiz')} className="btn-primary">Back to Setup</button>
            </div>
        );
    }

    if (quizCompleted) {
        return (
            <div className="quiz-results">
                <h2>Quiz Completed!</h2>
                <p>You scored {score} out of {questions.length} questions.</p>
                <button onClick={() => router.push('/quiz')} className="btn-primary">Start New Quiz</button>
                <button onClick={() => router.push('/review')} className="btn-secondary">Review Questions</button>
            </div>
        );
    }

    if (!currentQuestion) {
        return (
            <div className="quiz-error">
                <p className="error-message">No questions to display.</p>
                <button onClick={() => router.push('/quiz')} className="btn-primary">Back to Setup</button>
            </div>
        );
    }

    const getOptionLetter = (index: number) => String.fromCharCode(65 + index);

    return (
        <div className="quiz-interface-container screen">
            <div className="quiz-progress">
                <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                </div>
                <span className="question-count">Question {currentQuestionIndex + 1} / {questions.length}</span>
                {quizMode === 'serious' && timer !== null && (
                    <span className="timer">Time: {timer}s</span>
                )}
            </div>

            <div className="question-card">
                <p className="question-text">{currentQuestion.question}</p>
                <div className="options-grid">
                    {currentQuestion.options.map((option, index) => (
                        <button
                            key={index}
                            className={`option-button
                                ${selectedAnswer === option ? (option === currentQuestion.correctAnswer ? 'correct' : 'incorrect') : ''}
                                ${selectedAnswer && option === currentQuestion.correctAnswer && selectedAnswer !== option ? 'correct' : ''}
                            `}
                            onClick={() => handleAnswerSelect(option)}
                            disabled={!!selectedAnswer}
                        >
                            <span className="option-letter">{getOptionLetter(index)}</span>
                            <span>{option}</span>
                        </button>
                    ))}
                </div>

                {showExplanation && (
                    <div className="explanation-section">
                        <h3>Explanation:</h3>
                        <p>{currentQuestion.explanation}</p>
                        <button onClick={handleNextQuestion} className="btn-primary">
                            {currentQuestionIndex < questions.length - 1 ? 'Next Question' : 'Finish Quiz'}
                        </button>
                    </div>
                )}
            </div>

            {enableStreaming && <TokenStream quizId="quiz-stream-1" enabled={enableStreaming} />}
        </div>
    );
};

const QuizStartPage: React.FC = () => {
  return (
    <Suspense fallback={<div>Loading quiz...</div>}>
      <QuizInterfaceContent />
    </Suspense>
  );
};

export default QuizStartPage;