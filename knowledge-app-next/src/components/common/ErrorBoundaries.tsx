// Specialized error boundaries for different application areas
import React from 'react';
import ErrorBoundary from './ErrorBoundary';
import { AppLogger } from '../../lib/logger';

// Quiz-specific error boundary
export const QuizErrorBoundary: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        AppLogger.trackQuizAction('QUIZ_ERROR', {
          error: error.message,
          componentStack: errorInfo.componentStack,
        });
      }}
      fallbackComponent={({ error, retry, goHome, errorId }) => (
        <div className="quiz-error-fallback bg-red-50 dark:bg-red-900 border border-red-200 dark:border-red-700 rounded-lg p-6 m-4">
          <div className="text-center">
            <div className="text-4xl mb-4">📝</div>
            <h3 className="text-xl font-bold text-red-800 dark:text-red-200 mb-2">
              Quiz Error
            </h3>
            <p className="text-red-700 dark:text-red-300 mb-4">
              There was an issue with the quiz interface. Your progress has been saved.
            </p>
            <div className="space-x-4">
              <button
                onClick={retry}
                className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
              >
                Retry Quiz
              </button>
              <button
                onClick={goHome}
                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
              >
                Exit Quiz
              </button>
            </div>
          </div>
        </div>
      )}
    >
      {children}
    </ErrorBoundary>
  );
};

// Training-specific error boundary
export const TrainingErrorBoundary: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        AppLogger.error('TRAINING_ERROR', 'Training module error', {
          error: error.message,
          componentStack: errorInfo.componentStack,
        });
      }}
      fallbackComponent={({ error, retry, goHome, errorId }) => (
        <div className="training-error-fallback bg-orange-50 dark:bg-orange-900 border border-orange-200 dark:border-orange-700 rounded-lg p-6 m-4">
          <div className="text-center">
            <div className="text-4xl mb-4">🧠</div>
            <h3 className="text-xl font-bold text-orange-800 dark:text-orange-200 mb-2">
              Training Error
            </h3>
            <p className="text-orange-700 dark:text-orange-300 mb-4">
              There was an issue with the training module. Any ongoing training has been paused.
            </p>
            <div className="space-x-4">
              <button
                onClick={retry}
                className="px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-700"
              >
                Resume Training
              </button>
              <button
                onClick={goHome}
                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
              >
                Exit Training
              </button>
            </div>
          </div>
        </div>
      )}
    >
      {children}
    </ErrorBoundary>
  );
};

// Review-specific error boundary
export const ReviewErrorBoundary: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        AppLogger.error('REVIEW_ERROR', 'Review module error', {
          error: error.message,
          componentStack: errorInfo.componentStack,
        });
      }}
      fallbackComponent={({ error, retry, goHome, errorId }) => (
        <div className="review-error-fallback bg-blue-50 dark:bg-blue-900 border border-blue-200 dark:border-blue-700 rounded-lg p-6 m-4">
          <div className="text-center">
            <div className="text-4xl mb-4">📚</div>
            <h3 className="text-xl font-bold text-blue-800 dark:text-blue-200 mb-2">
              Review Error
            </h3>
            <p className="text-blue-700 dark:text-blue-300 mb-4">
              There was an issue loading your question history. Your data is safe.
            </p>
            <div className="space-x-4">
              <button
                onClick={retry}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Reload History
              </button>
              <button
                onClick={goHome}
                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
              >
                Go Home
              </button>
            </div>
          </div>
        </div>
      )}
    >
      {children}
    </ErrorBoundary>
  );
};

// Settings-specific error boundary
export const SettingsErrorBoundary: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        AppLogger.error('SETTINGS_ERROR', 'Settings module error', {
          error: error.message,
          componentStack: errorInfo.componentStack,
        });
      }}
      fallbackComponent={({ error, retry, goHome, errorId }) => (
        <div className="settings-error-fallback bg-purple-50 dark:bg-purple-900 border border-purple-200 dark:border-purple-700 rounded-lg p-6 m-4">
          <div className="text-center">
            <div className="text-4xl mb-4">⚙️</div>
            <h3 className="text-xl font-bold text-purple-800 dark:text-purple-200 mb-2">
              Settings Error
            </h3>
            <p className="text-purple-700 dark:text-purple-300 mb-4">
              There was an issue with the settings interface. Your settings are preserved.
            </p>
            <div className="space-x-4">
              <button
                onClick={retry}
                className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
              >
                Reload Settings
              </button>
              <button
                onClick={goHome}
                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
              >
                Go Home
              </button>
            </div>
          </div>
        </div>
      )}
    >
      {children}
    </ErrorBoundary>
  );
};

// Python Bridge error boundary for components that heavily use the bridge
export const BridgeErrorBoundary: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        AppLogger.error('BRIDGE_ERROR', 'Python bridge error', {
          error: error.message,
          componentStack: errorInfo.componentStack,
        });
      }}
      fallbackComponent={({ error, retry, goHome, errorId }) => (
        <div className="bridge-error-fallback bg-yellow-50 dark:bg-yellow-900 border border-yellow-200 dark:border-yellow-700 rounded-lg p-6 m-4">
          <div className="text-center">
            <div className="text-4xl mb-4">🔗</div>
            <h3 className="text-xl font-bold text-yellow-800 dark:text-yellow-200 mb-2">
              Connection Error
            </h3>
            <p className="text-yellow-700 dark:text-yellow-300 mb-4">
              There was an issue connecting to the Python backend. Please check if the bridge server is running.
            </p>
            <div className="space-x-4">
              <button
                onClick={retry}
                className="px-4 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700"
              >
                Reconnect
              </button>
              <button
                onClick={goHome}
                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
              >
                Go Home
              </button>
            </div>
            <p className="text-xs text-yellow-600 dark:text-yellow-400 mt-4">
              Make sure to run: <code className="bg-yellow-100 dark:bg-yellow-800 px-1 rounded">npm run dev:bridge</code>
            </p>
          </div>
        </div>
      )}
    >
      {children}
    </ErrorBoundary>
  );
};