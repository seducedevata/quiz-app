# Design Document

## Overview

This design document outlines the architecture for migrating the PyQt5 Knowledge App to a modern React-based web application. The migration will preserve all existing functionality while improving accessibility, maintainability, and user experience through modern web technologies.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    Home     │  │    Quiz     │  │   Review    │        │
│  │  Component  │  │  Component  │  │  Component  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐                         │
│  │   Train     │  │  Settings   │                         │
│  │  Component  │  │  Component  │                         │
│  └─────────────┘  └─────────────┘                         │
├─────────────────────────────────────────────────────────────┤
│                 State Management                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Redux     │  │   Context   │  │   React     │        │
│  │   Store     │  │   API       │  │   Query     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                 Communication Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   REST API  │  │  WebSocket  │  │   HTTP      │        │
│  │   Client    │  │   Client    │  │   Client    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Python Backend                           │
├─────────────────────────────────────────────────────────────┤
│                    FastAPI Server                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Quiz API  │  │ Training API│  │ Settings API│        │
│  │  Endpoints  │  │  Endpoints  │  │  Endpoints  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                 Existing Core Logic                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ MCQ Manager │  │ Unified     │  │ Training    │        │
│  │             │  │ Inference   │  │ Manager     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐                         │
│  │ Question    │  │ Document    │                         │
│  │ Storage     │  │ Processor   │                         │
│  └─────────────┘  └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend:**
- React 18+ with functional components and hooks
- TypeScript for type safety
- Redux Toolkit for state management
- React Query for server state management
- React Router for navigation
- Material-UI or Tailwind CSS for styling
- Socket.io-client for real-time communication
- Axios for HTTP requests

**Backend:**
- FastAPI for REST API endpoints
- WebSocket support for real-time features
- Existing Python core logic (MCQ Manager, Unified Inference Manager, etc.)
- SQLite/PostgreSQL for data persistence
- Pydantic for data validation

**Development:**
- Vite or Create React App for bundling
- ESLint and Prettier for code quality
- Jest and React Testing Library for testing
- Storybook for component development

## Components and Interfaces

### Core Components

#### 1. App Component
```typescript
interface AppProps {
  theme: 'light' | 'dark';
}

const App: React.FC<AppProps> = ({ theme }) => {
  // Main application wrapper with routing and global state
}
```

#### 2. Navigation Component
```typescript
interface NavigationProps {
  currentScreen: string;
  onNavigate: (screen: string) => void;
  isMobile: boolean;
}

const Navigation: React.FC<NavigationProps> = ({ currentScreen, onNavigate, isMobile }) => {
  // Responsive navigation with sidebar/mobile menu
}
```

#### 3. Quiz Component
```typescript
interface QuizConfig {
  topic: string;
  mode: 'offline' | 'online' | 'auto';
  gameMode: 'casual' | 'serious';
  questionType: 'mixed' | 'numerical' | 'conceptual';
  difficulty: 'easy' | 'medium' | 'hard' | 'expert';
  numQuestions: number;
  tokenStreaming: boolean;
}

interface QuizProps {
  onQuizComplete: (results: QuizResults) => void;
}

const Quiz: React.FC<QuizProps> = ({ onQuizComplete }) => {
  // Quiz setup, game, and results screens
}
```

#### 4. Question Component
```typescript
interface Question {
  id: string;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation?: string;
  topic: string;
  difficulty: string;
  timestamp: string;
}

interface QuestionProps {
  question: Question;
  onAnswer: (answerIndex: number) => void;
  showFeedback: boolean;
  selectedAnswer?: number;
}

const QuestionComponent: React.FC<QuestionProps> = ({ question, onAnswer, showFeedback, selectedAnswer }) => {
  // Individual question display with options and feedback
}
```

#### 5. Review Component
```typescript
interface ReviewProps {
  questions: Question[];
  onQuestionSelect: (question: Question) => void;
}

const Review: React.FC<ReviewProps> = ({ questions, onQuestionSelect }) => {
  // Question history with filtering and search
}
```

#### 6. Training Component
```typescript
interface TrainingConfig {
  modelName: string;
  datasetPath: string;
  epochs: number;
  learningRate: number;
}

interface TrainingProps {
  onTrainingStart: (config: TrainingConfig) => void;
  onTrainingStop: () => void;
}

const Training: React.FC<TrainingProps> = ({ onTrainingStart, onTrainingStop }) => {
  // Model training interface with progress tracking
}
```

#### 7. Settings Component
```typescript
interface SettingsProps {
  settings: AppSettings;
  onSettingsChange: (settings: Partial<AppSettings>) => void;
}

const Settings: React.FC<SettingsProps> = ({ settings, onSettingsChange }) => {
  // Application settings and API key management
}
```

### State Management

#### Redux Store Structure
```typescript
interface RootState {
  app: {
    theme: 'light' | 'dark';
    currentScreen: string;
    loading: boolean;
    error: string | null;
  };
  quiz: {
    config: QuizConfig;
    currentQuestion: Question | null;
    questions: Question[];
    results: QuizResults | null;
    isActive: boolean;
  };
  questions: {
    history: Question[];
    filters: {
      topic: string;
      difficulty: string;
      search: string;
    };
    statistics: QuestionStatistics;
  };
  training: {
    isActive: boolean;
    progress: number;
    status: string;
    config: TrainingConfig | null;
  };
  settings: {
    apiKeys: Record<string, string>;
    preferences: UserPreferences;
  };
}
```

### API Interfaces

#### REST API Endpoints
```typescript
// Quiz endpoints
POST /api/quiz/generate
GET /api/quiz/status/:id
POST /api/quiz/answer
GET /api/quiz/results/:id

// Question endpoints
GET /api/questions/history
GET /api/questions/search
GET /api/questions/statistics
DELETE /api/questions/:id

// Training endpoints
POST /api/training/start
GET /api/training/status
POST /api/training/stop
GET /api/training/models

// Settings endpoints
GET /api/settings
PUT /api/settings
POST /api/settings/api-keys/validate
```

#### WebSocket Events
```typescript
// Client to Server
interface ClientEvents {
  'quiz:start': QuizConfig;
  'quiz:answer': { questionId: string; answer: number };
  'training:start': TrainingConfig;
  'training:stop': void;
}

// Server to Client
interface ServerEvents {
  'quiz:question': Question;
  'quiz:feedback': QuizFeedback;
  'quiz:complete': QuizResults;
  'token:received': { token: string; metadata: any };
  'training:progress': { progress: number; status: string };
  'training:complete': TrainingResults;
  'error': { message: string; code: string };
}
```

## Data Models

### Question Model
```typescript
interface Question {
  id: string;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation?: string;
  topic: string;
  difficulty: 'easy' | 'medium' | 'hard' | 'expert';
  questionType: 'mixed' | 'numerical' | 'conceptual';
  timestamp: string;
  metadata: {
    generationTime: number;
    model: string;
    mode: string;
  };
}
```

### Quiz Configuration Model
```typescript
interface QuizConfig {
  topic: string;
  mode: 'offline' | 'online' | 'auto';
  gameMode: 'casual' | 'serious';
  questionType: 'mixed' | 'numerical' | 'conceptual';
  difficulty: 'easy' | 'medium' | 'hard' | 'expert';
  numQuestions: number;
  tokenStreaming: boolean;
  timeLimit?: number;
}
```

### Training Configuration Model
```typescript
interface TrainingConfig {
  modelName: string;
  datasetPath: string;
  epochs: number;
  learningRate: number;
  batchSize: number;
  validationSplit: number;
  outputPath: string;
}
```

### Settings Model
```typescript
interface AppSettings {
  theme: 'light' | 'dark';
  apiKeys: {
    openai?: string;
    anthropic?: string;
    groq?: string;
    deepseek?: string;
    tavily?: string;
  };
  preferences: {
    defaultQuizConfig: Partial<QuizConfig>;
    enableNotifications: boolean;
    autoSaveProgress: boolean;
    maxHistoryItems: number;
  };
}
```

## Error Handling

### Error Types
```typescript
enum ErrorType {
  NETWORK_ERROR = 'NETWORK_ERROR',
  API_ERROR = 'API_ERROR',
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  AUTHENTICATION_ERROR = 'AUTHENTICATION_ERROR',
  GENERATION_ERROR = 'GENERATION_ERROR',
  TRAINING_ERROR = 'TRAINING_ERROR'
}

interface AppError {
  type: ErrorType;
  message: string;
  details?: any;
  timestamp: string;
  recoverable: boolean;
}
```

### Error Handling Strategy
1. **Network Errors**: Automatic retry with exponential backoff
2. **API Errors**: Display user-friendly messages with retry options
3. **Validation Errors**: Real-time form validation with clear feedback
4. **Generation Errors**: Fallback to alternative models or cached content
5. **Training Errors**: Detailed error logs with recovery suggestions

## Testing Strategy

### Unit Testing
- Component testing with React Testing Library
- Hook testing with @testing-library/react-hooks
- Utility function testing with Jest
- API client testing with MSW (Mock Service Worker)

### Integration Testing
- End-to-end user flows with Cypress
- API integration testing
- WebSocket communication testing
- State management integration testing

### Performance Testing
- Component rendering performance
- Bundle size optimization
- Memory leak detection
- API response time monitoring

## Security Considerations

### Frontend Security
- Input sanitization for all user inputs
- XSS prevention through proper escaping
- CSRF protection for API calls
- Secure storage of API keys (encrypted localStorage)

### API Security
- Rate limiting on all endpoints
- Input validation with Pydantic
- Authentication and authorization
- CORS configuration for allowed origins

### Data Protection
- Encryption of sensitive data at rest
- Secure transmission over HTTPS/WSS
- API key rotation and validation
- User data privacy compliance

## Performance Optimization

### Frontend Optimization
- Code splitting by route and feature
- Lazy loading of heavy components
- Virtual scrolling for large lists
- Memoization of expensive calculations
- Image optimization and lazy loading

### Backend Optimization
- Database query optimization
- Caching of frequently accessed data
- Connection pooling for database
- Async processing for long-running tasks
- Resource cleanup and memory management

### Network Optimization
- Request batching where possible
- Compression of API responses
- WebSocket connection management
- Offline capability with service workers
- Progressive loading of content

## Deployment Strategy

### Development Environment
- Local development server with hot reloading
- Docker containers for consistent environment
- Environment-specific configuration
- Automated testing on code changes

### Production Deployment
- Static site hosting (Vercel, Netlify, or AWS S3)
- Backend deployment on cloud platforms
- CDN for static assets
- Database hosting and backups
- Monitoring and logging setup

### CI/CD Pipeline
- Automated testing on pull requests
- Build optimization and bundling
- Deployment automation
- Environment promotion workflow
- Rollback capabilities