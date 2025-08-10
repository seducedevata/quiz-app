'use client';

import { useTheme } from '../../hooks/useTheme';
import React, { useEffect, Suspense, useState } from 'react';
import { onPythonEvent, offPythonEvent } from '../../lib/pythonBridge';
import { useScreen } from '../../context/ScreenContext';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { ConnectionStatus } from '../common/ConnectionStatus';
import { 
  QuizErrorBoundary, 
  TrainingErrorBoundary, 
  ReviewErrorBoundary, 
  SettingsErrorBoundary,
  BridgeErrorBoundary 
} from '../common/ErrorBoundaries';
import { errorRecoveryService } from '../../lib/errorRecovery';
import { bridgeMonitoringService } from '../../lib/bridgeMonitorSimple';
import { AppLogger } from '../../lib/logger';

// Import all screen components
const HomePage = React.lazy(() => import('../../app/page'));
const QuizPage = React.lazy(() => import('../../app/quiz/page'));
const ReviewPage = React.lazy(() => import('../../app/review/page'));
const SettingsPage = React.lazy(() => import('../../app/settings/page'));
const TrainPage = React.lazy(() => import('../../app/train/page'));

interface MainLayoutProps {
  children: React.ReactNode;
}

export const MainLayout = ({ children }: MainLayoutProps) => {
  const { isDark } = useTheme();
  const { currentScreen } = useScreen();
  const [pythonBridgeConnected, setPythonBridgeConnected] = useState(false);
  const [fallbackMode, setFallbackMode] = useState(false);

  useEffect(() => {
    // Apply theme class to html element (matches Qt exactly)
    if (isDark) {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }

    const handleConnectionStatus = (status: { connected: boolean }) => {
      setPythonBridgeConnected(status.connected);
      
      // Attempt recovery if connection is lost
      if (!status.connected) {
        AppLogger.warn('CONNECTION', 'Python bridge connection lost, attempting recovery');
        errorRecoveryService.attemptRecovery('bridge_connection', 'Python bridge disconnected');
      }
    };

    const handleFallbackMode = () => {
      setFallbackMode(true);
      AppLogger.info('FALLBACK', 'Fallback mode activated');
    };

    const handleForceRemount = () => {
      // Force re-render by updating a state that doesn't affect UI
      AppLogger.info('RECOVERY', 'Force remount triggered');
    };

    // Check for fallback mode on mount
    setFallbackMode(localStorage.getItem('fallbackMode') === 'true');

    // Start bridge monitoring
    bridgeMonitoringService.startMonitoring();

    onPythonEvent('connection_status', handleConnectionStatus);
    window.addEventListener('enableFallbackMode', handleFallbackMode);
    window.addEventListener('forceRemount', handleForceRemount);

    return () => {
      // Stop bridge monitoring
      bridgeMonitoringService.stopMonitoring();
      
      offPythonEvent('connection_status', handleConnectionStatus);
      window.removeEventListener('enableFallbackMode', handleFallbackMode);
      window.removeEventListener('forceRemount', handleForceRemount);
    };
  }, [isDark]);

  const renderScreen = () => {
    // Wrap each screen with appropriate error boundary
    switch (currentScreen) {
      case 'home':
        return <HomePage />;
      case 'quiz':
        return (
          <QuizErrorBoundary>
            <QuizPage />
          </QuizErrorBoundary>
        );
      case 'review':
        return (
          <ReviewErrorBoundary>
            <ReviewPage />
          </ReviewErrorBoundary>
        );
      case 'train':
        return (
          <TrainingErrorBoundary>
            <TrainPage />
          </TrainingErrorBoundary>
        );
      case 'settings':
        return (
          <SettingsErrorBoundary>
            <SettingsPage />
          </SettingsErrorBoundary>
        );
      default:
        return <HomePage />;
    }
  };

  return (
    <BridgeErrorBoundary>
      <div style={{ minHeight: '100vh', backgroundColor: '#2d3748', color: '#f7fafc' }}>
        {/* Fallback Mode Indicator */}
        {fallbackMode && (
          <div className="bg-yellow-500 text-black px-4 py-2 text-center text-sm font-medium">
            ⚠️ Running in fallback mode - Some features may be limited
            <button 
              onClick={() => {
                localStorage.removeItem('fallbackMode');
                window.location.reload();
              }}
              className="ml-4 underline hover:no-underline"
            >
              Try to reconnect
            </button>
          </div>
        )}

        {/* Header */}
        <header 
          style={{
            backgroundColor: '#1a202c',
            borderBottom: '1px solid #4a5568',
            padding: '1rem 2rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            height: '80px'
          }}
        >
          <TopBar pythonBridgeConnected={pythonBridgeConnected} />
          <ConnectionStatus showDetails={false} />
        </header>
        
        {/* Main Content */}
        <div style={{ display: 'flex' }}>
          <Sidebar />
          <main style={{ flex: 1, padding: '2rem', backgroundColor: '#2d3748' }}>
            <Suspense fallback={
              <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                <span className="ml-3">Loading screen...</span>
              </div>
            }>
              {renderScreen()}
            </Suspense>
          </main>
        </div>
      </div>
    </BridgeErrorBoundary>
  );
};
