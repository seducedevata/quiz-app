'use client';

import { useState, useEffect, useCallback } from 'react';
import { AppLogger } from '@/lib/logger';

interface QuizDefaults {
  topic: string;
  difficulty: 'easy' | 'medium' | 'hard';
  numQuestions: number;
}

interface UserPreferences {
  theme: 'light' | 'dark';
  quizDefaults: QuizDefaults;
  // Add other preferences here as needed
}

const DEFAULT_PREFERENCES: UserPreferences = {
  theme: 'dark',
  quizDefaults: {
    topic: '',
    difficulty: 'easy',
    numQuestions: 5,
  },
};

export const usePreferences = () => {
  const [preferences, setPreferences] = useState<UserPreferences>(DEFAULT_PREFERENCES);

  useEffect(() => {
    // Load preferences from localStorage on mount
    try {
      const storedPreferences = localStorage.getItem('userPreferences');
      if (storedPreferences) {
        const parsedPreferences = JSON.parse(storedPreferences);
        // Merge with defaults to handle new preference additions
        setPreferences(prev => ({
          ...DEFAULT_PREFERENCES,
          ...prev,
          ...parsedPreferences,
          quizDefaults: {
            ...DEFAULT_PREFERENCES.quizDefaults,
            ...prev.quizDefaults,
            ...parsedPreferences.quizDefaults,
          },
        }));
        AppLogger.info('PREFERENCES', 'User preferences loaded from localStorage.', parsedPreferences);
      }
    } catch (error) {
      AppLogger.error('PREFERENCES', 'Failed to load preferences from localStorage.', error);
    }
  }, []);

  // Save preferences to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem('userPreferences', JSON.stringify(preferences));
      AppLogger.info('PREFERENCES', 'User preferences saved to localStorage.', preferences);
    } catch (error) {
      AppLogger.error('PREFERENCES', 'Failed to save preferences to localStorage.', error);
    }
  }, [preferences]);

  const updatePreference = useCallback(<K extends keyof UserPreferences>(
    key: K,
    value: UserPreferences[K]
  ) => {
    setPreferences(prev => ({
      ...prev,
      [key]: value,
    }));
  }, []);

  const resetPreferences = useCallback(() => {
    setPreferences(DEFAULT_PREFERENCES);
    AppLogger.warn('PREFERENCES', 'User preferences reset to defaults.');
  }, []);

  return {
    preferences,
    updatePreference,
    resetPreferences,
  };
};
