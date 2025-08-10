'use client';

import React, { createContext, useContext, useEffect, ReactNode } from 'react';
import { usePreferences } from './usePreferences';

interface ThemeContextType {
  isDark: boolean;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: ReactNode;
}

export const ThemeProvider = ({ children }: ThemeProviderProps) => {
  const { preferences, updatePreference } = usePreferences();
  const isDark = preferences.theme === 'dark';

  useEffect(() => {
    if (isDark) {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
  }, [isDark]);

  const toggleTheme = () => {
    updatePreference('theme', isDark ? 'light' : 'dark');
  };

  const contextValue: ThemeContextType = { isDark, toggleTheme };

  return React.createElement(
    ThemeContext.Provider,
    { value: contextValue },
    children
  );
};

export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};