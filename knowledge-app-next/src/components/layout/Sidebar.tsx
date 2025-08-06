'use client';

import React from 'react';
import { useScreen } from '../../context/ScreenContext'; // Import useScreen

const navItems = [
  { id: 'home', icon: '🏠', text: 'Home' },
  { id: 'quiz', icon: '📝', text: 'Quiz' },
  { id: 'review', icon: '📚', text: 'Review' },
  { id: 'train', icon: '🧠', text: 'Train Model' },
  { id: 'settings', icon: '⚙️', text: 'Settings' },
  // Debug/Test pages (for development/testing)
  { id: 'debug-button-test', icon: '🐞', text: 'Debug Buttons' },
  { id: 'debug-navigation', icon: '🧭', text: 'Debug Nav' },
  { id: 'fix', icon: '🔧', text: 'Fix App' },
  { id: 'math-rendering-test', icon: '🧮', text: 'Math Test' },
  { id: 'standalone-latex-test', icon: '🧪', text: 'LaTeX Test' },
  { id: 'quiz-generation', icon: '✨', text: 'Quiz Gen' },
];

export const Sidebar: React.FC = () => {
  const { currentScreen, showScreen } = useScreen(); // Get currentScreen and showScreen from context

  return (
    <nav className="side-nav w-[250px] bg-bg-secondary border-r border-border-color p-4 flex flex-col gap-2">
      {navItems.map((item) => (
        <button
          key={item.id}
          onClick={() => showScreen(item.id)}
          className={`nav-item flex items-center gap-4 py-3 px-4 bg-none border-none rounded-md text-text-primary text-base cursor-pointer transition-all duration-300 text-left w-full
            ${currentScreen === item.id ? 'active bg-primary-color text-white' : 'hover:bg-border-color hover:translate-x-1'}
          `}
        >
          <span className="icon text-xl">{item.icon}</span>
          <span>{item.text}</span>
        </button>
      ))}
    </nav>
  );
};