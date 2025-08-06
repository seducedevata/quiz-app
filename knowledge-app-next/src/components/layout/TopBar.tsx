'use client';

import React from 'react';
import { useTheme } from '@/hooks/useTheme';

export const TopBar: React.FC = () => {
  const { isDark, toggleTheme } = useTheme();

  return (
    <header className="app-header bg-bg-secondary border-b border-border-color py-4 px-8 flex justify-between items-center shadow-md min-h-[80px]">
      <div className="header-content">
        <h1 className="app-title text-[1.75rem] font-bold bg-gradient-to-br from-primary-color to-primary-hover bg-clip-text text-transparent mb-1 leading-tight">
          Knowledge App
        </h1>
        <p className="app-subtitle text-text-secondary text-sm m-0 font-normal">
          Modern Learning Platform
        </p>
      </div>
      <div className="header-controls">
        <button 
          className="theme-toggle bg-none border-none text-2xl cursor-pointer p-2 rounded-md transition-colors duration-300 hover:bg-border-color"
          onClick={toggleTheme}
        >
          {isDark ? '☀️' : '🌙'}
        </button>
      </div>
    </header>
  );
};