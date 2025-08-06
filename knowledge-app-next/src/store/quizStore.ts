import { create } from 'zustand';

interface Question {
  id: string;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
  topic: string;
  difficulty: string;
  timestamp: string;
}

interface QuizState {
  // Quiz Configuration
  topic: string;
  difficulty: string;
  numQuestions: number;
  mode: string;
  gameMode: string;
  questionType: string;
  tokenStreaming: boolean;
  
  // Quiz Data
  questions: Question[];
  currentQuestion: Question | null;
  currentIndex: number;
  selectedAnswer: number;
  score: number;
  timeRemaining: number;
  
  // Quiz Status
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  isStreaming: boolean;
  isQuizActive: boolean;
  isQuizComplete: boolean;
  showFeedback: boolean;
  
  // Streaming
  streamingTokens: string[];
  streamingStatus: string;
  
  // Actions
  setQuizConfig: (config: Partial<QuizState>) => void;
  startQuiz: (questions: Question[]) => void;
  setCurrentQuestion: (question: Question) => void;
  selectAnswer: (index: number) => void;
  submitAnswer: () => void;
  nextQuestion: () => void;
  endQuiz: () => void;
  resetQuiz: () => void;
  addStreamingToken: (token: string) => void;
  setStreamingStatus: (status: string) => void;
  updateTimer: () => void;
}

export const useQuizStore = create<QuizState>((set, get) => ({
  // Initial State
  topic: '',
  difficulty: 'medium',
  numQuestions: 5,
  mode: 'auto',
  gameMode: 'casual',
  questionType: 'mixed',
  tokenStreaming: true,
  
  questions: [],
  currentQuestion: null,
  currentIndex: 0,
  selectedAnswer: -1,
  score: 0,
  timeRemaining: 30,
  
  isLoading: false,
  isStreaming: false,
  isQuizActive: false,
  isQuizComplete: false,
  showFeedback: false,
  
  streamingTokens: [],
  streamingStatus: '',
  
  // Actions
  setQuizConfig: (config) => set((state) => ({ ...state, ...config })),
  setIsLoading: (loading: boolean) => set({ isLoading: loading }),
  
  startQuiz: (questions) => set({
    questions,
    currentQuestion: questions[0],
    currentIndex: 0,
    selectedAnswer: -1,
    score: 0,
    timeRemaining: 30,
    isQuizActive: true,
    isQuizComplete: false,
    isLoading: false,
    isStreaming: false,
  }),
  
  setCurrentQuestion: (question) => set({ currentQuestion: question }),
  
  selectAnswer: (index) => set({ selectedAnswer: index }),
  
  submitAnswer: () => {
    const { currentQuestion, selectedAnswer, score } = get();
    if (currentQuestion && selectedAnswer !== -1) {
      const isCorrect = selectedAnswer === currentQuestion.correctAnswer;
      set({
        score: isCorrect ? score + 1 : score,
        showFeedback: true,
      });
    }
  },
  
  nextQuestion: () => {
    const { currentIndex, questions } = get();
    const nextIndex = currentIndex + 1;
    
    if (nextIndex < questions.length) {
      set({
        currentIndex: nextIndex,
        currentQuestion: questions[nextIndex],
        selectedAnswer: -1,
        timeRemaining: 30,
        showFeedback: false,
      });
    } else {
      set({
        isQuizComplete: true,
        isQuizActive: false,
      });
    }
  },
  
  endQuiz: () => set({
    isQuizComplete: true,
    isQuizActive: false,
  }),
  
  resetQuiz: () => set({
    questions: [],
    currentQuestion: null,
    currentIndex: 0,
    selectedAnswer: -1,
    score: 0,
    timeRemaining: 30,
    isLoading: false,
    isStreaming: false,
    isQuizActive: false,
    isQuizComplete: false,
    showFeedback: false,
    streamingTokens: [],
    streamingStatus: '',
  }),
  
  addStreamingToken: (token) => set((state) => ({
    streamingTokens: [...state.streamingTokens, token],
  })),
  
  setStreamingStatus: (status) => set({ streamingStatus: status }),
  
  updateTimer: () => {
    const { timeRemaining } = get();
    if (timeRemaining > 0) {
      set({ timeRemaining: timeRemaining - 1 });
    } else {
      // Auto-submit when time runs out
      get().submitAnswer();
      setTimeout(() => get().nextQuestion(), 2000);
    }
  },
}));
