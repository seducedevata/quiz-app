'use client';

import React, { useEffect } from 'react';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { useTheme } from '@/hooks/useTheme';
import { useScreen } from '../../context/ScreenContext'; // Import useScreen

// Import all screen components
import HomePage from '../../app/page';
import QuizPage from '../../app/quiz/page';
import ReviewPage from '../../app/review/page';
import TrainPage from '../../app/train/page';
import SettingsPage from '../../app/settings/page';

// Debug pages (for development/testing)
import DebugButtonTestPage from '../../app/debug-button-test/page';
import DebugNavigationPage from '../../app/debug-navigation/page';
import KnowledgeAppFixPage from '../../app/fix/page';
import MathRenderingTestPage from '../../app/math-rendering-test/page';
import StandaloneLatexTestPage from '../../app/standalone-latex-test/page';
import QuizGenerationPage from '../../app/quiz-generation/page';

interface MainLayoutProps {
  children: React.ReactNode; // This will no longer be used directly for screen content
}

export const MainLayout = ({ children }: MainLayoutProps) => {
  const { isDark } = useTheme();
  const { currentScreen } = useScreen(); // Get currentScreen from context

  useEffect(() => {
    // Apply theme class to html element
    if (isDark) {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
  }, [isDark]);

  const renderScreen = () => {
    switch (currentScreen) {
      case 'home':
        return <HomePage />;
      case 'quiz':
        return <QuizPage />;
      case 'review':
        return <ReviewPage />;
      case 'train':
        return <TrainPage />;
      case 'settings':
        return <SettingsPage />;
      // Debug/Test pages
      case 'debug-button-test':
        return <DebugButtonTestPage />;
      case 'debug-navigation':
        return <DebugNavigationPage />;
      case 'fix':
        return <KnowledgeAppFixPage />;
      case 'math-rendering-test':
        return <MathRenderingTestPage />;
      case 'standalone-latex-test':
        return <StandaloneLatexTestPage />;
      case 'quiz-generation':
        return <QuizGenerationPage />;
      default:
        return <HomePage />;
    }
  };

  return (
    <div id="app" className="flex flex-col min-h-screen bg-bg-primary font-sans text-text-primary transition-colors duration-300 ease-in-out">
      <TopBar />
      <main className="flex flex-1">
        <Sidebar />
        <div className="content-area flex-1 p-8 overflow-y-auto">
          {renderScreen()}
        </div>
      </main>
    </div>
  );
};
