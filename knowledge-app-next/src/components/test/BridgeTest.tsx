'use client';

import React, { useEffect, useState } from 'react';
import { AppLogger } from '../../lib/logger';
import { callPythonMethod, checkBridgeHealth, offPythonEvent, onPythonEvent } from '../../lib/pythonBridge';

export const BridgeTest: React.FC = () => {
  const [bridgeStatus, setBridgeStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [testResults, setTestResults] = useState<string[]>([]);
  const [isRunningTest, setIsRunningTest] = useState(false);

  useEffect(() => {
    checkConnection();
  }, []);

  const checkConnection = async () => {
    try {
      AppLogger.info('BRIDGE_TEST', 'Checking Python bridge connection...');
      const isHealthy = await checkBridgeHealth();
      setBridgeStatus(isHealthy ? 'connected' : 'disconnected');
      AppLogger.info('BRIDGE_TEST', `Bridge health check: ${isHealthy ? 'PASS' : 'FAIL'}`);
    } catch (error) {
      setBridgeStatus('disconnected');
      AppLogger.error('BRIDGE_TEST', 'Bridge health check failed', error);
    }
  };

  const runComprehensiveTest = async () => {
    if (isRunningTest) return;
    
    setIsRunningTest(true);
    setTestResults([]);
    const results: string[] = [];

    const addResult = (test: string, passed: boolean, details?: string) => {
      const result = `${passed ? '✅' : '❌'} ${test}${details ? `: ${details}` : ''}`;
      results.push(result);
      setTestResults([...results]);
      AppLogger.info('BRIDGE_TEST', result);
    };

    try {
      // Test 1: Basic connectivity
      addResult('Bridge Health Check', bridgeStatus === 'connected');

      // Test 2: Simple method call
      try {
        const result = await callPythonMethod('get_app_settings');
        addResult('Basic Method Call (get_app_settings)', true, `Got: ${JSON.stringify(result).substring(0, 50)}...`);
      } catch (error) {
        addResult('Basic Method Call (get_app_settings)', false, (error as Error).message);
      }

      // Test 3: Quiz configuration methods
      try {
        const defaultConfig = {
          topic: 'Test Topic',
          mode: 'offline',
          gameMode: 'standard',
          questionType: 'conceptual',
          difficulty: 'medium',
          numQuestions: 2,
          tokenStreamingEnabled: false
        };
        
        await callPythonMethod('save_app_settings', defaultConfig);
        addResult('Save Settings', true, 'Quiz settings saved');
        
        const savedSettings = await callPythonMethod('get_app_settings');
        addResult('Load Settings', true, `Settings loaded: ${Object.keys(savedSettings || {}).length} keys`);
      } catch (error) {
        addResult('Settings Operations', false, (error as Error).message);
      }

      // Test 4: Question History check
      try {
        const historyStats = await callPythonMethod('get_history_stats');
        addResult('History Stats', true, `Total questions: ${historyStats?.total_questions || 0}`);
      } catch (error) {
        addResult('History Stats', false, (error as Error).message);
      }

      // Test 5: Quiz generation (dry run)
      try {
        const quizConfig = {
          topic: 'Mathematics',
          mode: 'offline',
          gameMode: 'standard',
          questionType: 'conceptual',
          difficulty: 'beginner',
          numQuestions: 1,
          tokenStreamingEnabled: false
        };
        
        // This might fail if models aren't loaded, but we can test the call
        const quizId = await callPythonMethod('generate_mcq_quiz', quizConfig);
        addResult('Quiz Generation', true, `Generated quiz ID: ${quizId}`);
      } catch (error) {
        const errorMsg = (error as Error).message;
        if (errorMsg.includes('model') || errorMsg.includes('path') || errorMsg.includes('not available')) {
          addResult('Quiz Generation', false, 'Expected: Backend components not fully configured');
        } else {
          addResult('Quiz Generation', false, errorMsg);
        }
      }

      // Test 6: WebSocket events
      let eventReceived = false;
      const testEventHandler = (data: any) => {
        eventReceived = true;
        addResult('WebSocket Events', true, `Event received: ${JSON.stringify(data).substring(0, 30)}...`);
      };

      onPythonEvent('test_event', testEventHandler);
      
      // Wait a moment for potential events
      setTimeout(() => {
        if (!eventReceived) {
          addResult('WebSocket Events', false, 'No test events received (normal for initial setup)');
        }
        offPythonEvent('test_event', testEventHandler);
      }, 2000);

    } catch (error) {
      addResult('Test Suite', false, `Unexpected error: ${(error as Error).message}`);
    }

    setIsRunningTest(false);
    AppLogger.success('BRIDGE_TEST', 'Bridge test suite completed');
  };

  const getBridgeStatusColor = () => {
    switch (bridgeStatus) {
      case 'connected': return 'var(--success-color)';
      case 'disconnected': return 'var(--error-color)';
      default: return 'var(--warning-color)';
    }
  };

  const getBridgeStatusText = () => {
    switch (bridgeStatus) {
      case 'connected': return '✅ Connected';
      case 'disconnected': return '❌ Disconnected';
      default: return '⏳ Checking...';
    }
  };

  return (
    <div style={{ 
      padding: '1.5rem', 
      backgroundColor: 'var(--bg-secondary)', 
      borderRadius: 'var(--border-radius-lg)', 
      boxShadow: 'var(--shadow-md)' 
    }}>
      <h3 style={{ 
        color: 'var(--text-primary)', 
        marginBottom: '1rem',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem'
      }}>
        🔗 Python Bridge Connection Test
        <span style={{ 
          color: getBridgeStatusColor(),
          fontSize: '0.875rem',
          fontWeight: 'normal'
        }}>
          {getBridgeStatusText()}
        </span>
      </h3>

      <div style={{ marginBottom: '1rem', display: 'flex', gap: '0.5rem' }}>
        <button
          onClick={checkConnection}
          disabled={bridgeStatus === 'checking'}
          style={{
            backgroundColor: 'var(--primary-color)',
            color: 'white',
            border: 'none',
            padding: '0.5rem 1rem',
            borderRadius: 'var(--border-radius-md)',
            cursor: 'pointer',
            opacity: bridgeStatus === 'checking' ? 0.7 : 1
          }}
        >
          {bridgeStatus === 'checking' ? 'Checking...' : 'Check Connection'}
        </button>

        <button
          onClick={runComprehensiveTest}
          disabled={isRunningTest || bridgeStatus !== 'connected'}
          style={{
            backgroundColor: isRunningTest ? 'var(--warning-color)' : 'var(--success-color)',
            color: 'white',
            border: 'none',
            padding: '0.5rem 1rem',
            borderRadius: 'var(--border-radius-md)',
            cursor: bridgeStatus !== 'connected' ? 'not-allowed' : 'pointer',
            opacity: bridgeStatus !== 'connected' ? 0.5 : 1
          }}
        >
          {isRunningTest ? 'Running Tests...' : 'Run Full Test Suite'}
        </button>
      </div>

      {testResults.length > 0 && (
        <div style={{
          backgroundColor: 'var(--bg-primary)',
          padding: '1rem',
          borderRadius: 'var(--border-radius-md)',
          border: '1px solid var(--border-color)',
          maxHeight: '300px',
          overflowY: 'auto'
        }}>
          <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
            Test Results:
          </h4>
          {testResults.map((result, index) => (
            <div 
              key={index} 
              style={{ 
                fontFamily: 'monospace',
                fontSize: '0.875rem',
                color: 'var(--text-secondary)',
                marginBottom: '0.25rem'
              }}
            >
              {result}
            </div>
          ))}
        </div>
      )}

      <div style={{ 
        marginTop: '1rem', 
        padding: '0.75rem', 
        backgroundColor: 'var(--bg-primary)', 
        borderRadius: 'var(--border-radius-md)',
        border: '1px solid var(--border-color)'
      }}>
        <h5 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
          Bridge Info:
        </h5>
        <ul style={{ 
          color: 'var(--text-secondary)', 
          fontSize: '0.875rem',
          margin: 0,
          paddingLeft: '1rem'
        }}>
          <li>URL: {process.env.NEXT_PUBLIC_PYTHON_BRIDGE_URL || 'http://localhost:8000'}</li>
          <li>Transport: WebSocket + HTTP REST</li>
          <li>Status: {getBridgeStatusText()}</li>
          <li>Last Check: {new Date().toLocaleTimeString()}</li>
        </ul>
      </div>
    </div>
  );
};
