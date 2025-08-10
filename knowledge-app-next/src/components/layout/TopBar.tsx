'use client';

import { useTheme } from '@/hooks/useTheme';
import React from 'react';

interface TopBarProps {
  pythonBridgeConnected: boolean;
}

export const TopBar: React.FC<TopBarProps> = ({ pythonBridgeConnected }) => {
  const { isDark, toggleTheme } = useTheme();

  return (
    <>
      <div className="header-content">
        <h1 
          className="app-title"
          style={{
            fontSize: '1.75rem',
            fontWeight: '700',
            background: 'linear-gradient(135deg, var(--primary-color), var(--primary-hover))',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            marginBottom: '0.25rem',
            lineHeight: 1.2,
            margin: 0,
            /* CRITICAL: Add text shadow for depth like Qt */
            filter: 'drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1))',
            /* CRITICAL: Smooth text rendering */
            textRendering: 'optimizeLegibility',
            WebkitFontSmoothing: 'antialiased'
          }}
        >
          Knowledge App
        </h1>
        <p 
          className="app-subtitle"
          style={{
            color: 'var(--text-secondary)',
            fontSize: '0.875rem',
            margin: 0,
            fontWeight: '400',
            lineHeight: 1.3
          }}
        >
          Modern Learning Platform
        </p>
      </div>
      <div className="header-controls">
        <span 
          className="python-bridge-status mr-4"
          title={pythonBridgeConnected ? 'Python Bridge Connected' : 'Python Bridge Disconnected'}
        >
          {pythonBridgeConnected ? '🟢' : '🔴'}
        </span>
        <button 
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          style={{
            background: 'none',
            border: 'none',
            fontSize: '1.5rem',
            cursor: 'pointer',
            padding: '0.5rem',
            borderRadius: 'var(--border-radius-md)',
            transition: 'all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1)',
            position: 'relative',
            overflow: 'hidden',
            /* CRITICAL: Add subtle hover effects */
            willChange: 'transform'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--border-color)';
            e.currentTarget.style.transform = 'scale(1.1)';
            e.currentTarget.style.boxShadow = '0 4px 8px var(--shadow)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.transform = 'scale(1)';
            e.currentTarget.style.boxShadow = 'none';
          }}
          onMouseDown={(e) => {
            e.currentTarget.style.transform = 'scale(0.95)';
          }}
          onMouseUp={(e) => {
            e.currentTarget.style.transform = 'scale(1.1)';
          }}
        >
          {isDark ? '☀️' : '🌙'}
        </button>
      </div>
    </>
  );
};
