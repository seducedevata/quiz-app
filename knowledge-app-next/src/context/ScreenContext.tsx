'use client';

import React, { createContext, useContext, useState, ReactNode } from 'react';
import { AppLogger, setCurrentScreenName } from '../lib/logger';

interface ScreenContextType {
  currentScreen: string;
  showScreen: (screenName: string, navElement?: HTMLElement | null) => void;
}

const ScreenContext = createContext<ScreenContextType | undefined>(undefined);

export const ScreenProvider = ({ children }: { children: ReactNode }) => {
  const [currentScreen, setCurrentScreenState] = useState<string>('home');

  const showScreen = (screenName: string, navElement?: HTMLElement | null) => {
    const previousScreen = currentScreen;
    setCurrentScreenState(screenName);
    setCurrentScreenName(screenName); // Update logger's current screen

    AppLogger.action('NAVIGATION', `Navigating from ${previousScreen} to ${screenName}`, {
      method: navElement ? 'nav-click' : 'programmatic',
      timestamp: Date.now(),
    });
  };

  return (
    <ScreenContext.Provider value={{ currentScreen, showScreen }}>
      {children}
    </ScreenContext.Provider>
  );
};

export const useScreen = () => {
  const context = useContext(ScreenContext);
  if (context === undefined) {
    throw new Error('useScreen must be used within a ScreenProvider');
  }
  return context;
};
