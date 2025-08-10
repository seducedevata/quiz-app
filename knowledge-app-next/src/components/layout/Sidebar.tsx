'use client';

import React from 'react';
import { useScreen } from '../../context/ScreenContext';

// Main navigation items (matching Qt exactly)
const navItems = [
  { id: 'home', icon: '🏠', text: 'Home' },
  { id: 'quiz', icon: '📝', text: 'Quiz' },
  { id: 'review', icon: '📚', text: 'Review' },
  { id: 'train', icon: '🧠', text: 'Train Model' },
  { id: 'settings', icon: '⚙️', text: 'Settings' },
];

export const Sidebar: React.FC = () => {
  const { currentScreen, showScreen } = useScreen();

  return (
    <nav 
      style={{
        width: '250px',
        minWidth: '250px',
        maxWidth: '250px',
        backgroundColor: '#1a202c',
        borderRight: '1px solid #4a5568',
        padding: '1rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem',
        height: '100%',
        overflowY: 'auto',
        flexShrink: 0
      }}
      role="navigation" 
      aria-label="Main navigation"
    >
      {navItems.map((item) => (
        <button
          key={item.id}
          onClick={() => showScreen(item.id)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            padding: '0.75rem 1rem',
            borderRadius: '0.5rem',
            textAlign: 'left',
            width: '100%',
            border: 'none',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            backgroundColor: currentScreen === item.id ? '#6366f1' : 'transparent',
            color: currentScreen === item.id ? 'white' : '#f7fafc',
            fontSize: '1rem',
            fontWeight: '500'
          }}
          onMouseEnter={(e) => {
            if (currentScreen !== item.id) {
              e.currentTarget.style.backgroundColor = '#4a5568';
              e.currentTarget.style.transform = 'translateX(4px)';
            }
          }}
          onMouseLeave={(e) => {
            if (currentScreen !== item.id) {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.transform = 'translateX(0)';
            }
          }}
          aria-current={currentScreen === item.id ? 'page' : undefined}
          aria-label={item.text}
        >
          <span style={{ fontSize: '1.25rem' }}>{item.icon}</span>
          <span>{item.text}</span>
        </button>
      ))}
    </nav>
  );
};