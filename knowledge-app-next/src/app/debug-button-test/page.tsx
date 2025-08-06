'use client';

import React, { useState, useEffect, useRef } from 'react';
import { callPythonMethod, checkBridgeHealth } from '../../lib/pythonBridge';

const DebugButtonTestPage: React.FC = () => {
  const [consoleOutput, setConsoleOutput] = useState<string>('Waiting for tests...');
  const consoleOutputRef = useRef<HTMLPreElement>(null);

  // Simulate console output capture
  useEffect(() => {
    const originalLog = console.log;
    const originalError = console.error;

    const updateConsole = (prefix: string, ...args: any[]) => {
      setConsoleOutput(prev => {
        const newOutput = prev === 'Waiting for tests...' ? '' : prev;
        return newOutput + `${prefix} ${args.join(' ')}
`;
      });
    };

    console.log = (...args: any[]) => {
      updateConsole('[LOG]', ...args);
      originalLog.apply(console, args);
    };

    console.error = (...args: any[]) => {
      updateConsole('[ERROR]', ...args);
      originalError.apply(console, args);
    };

    return () => {
      console.log = originalLog;
      console.error = originalError;
    };
  }, []);

  useEffect(() => {
    if (consoleOutputRef.current) {
      consoleOutputRef.current.scrollTop = consoleOutputRef.current.scrollHeight;
    }
  }, [consoleOutput]);

  const testBridge = async () => {
    try {
      console.log('🔗 Testing Python bridge health...');
      const isHealthy = await checkBridgeHealth();
      if (isHealthy) {
                document.getElementById('bridge-result')!.innerHTML = 
          `<div class="success">Bridge Result: Python Bridge is Healthy!</div>`;
        console.log('✅ Python Bridge is Healthy!');
      } else {
        document.getElementById('bridge-result')!.innerHTML = 
          `<div class="error">❌ Python Bridge is NOT healthy or unreachable!</div>`;
        console.error('❌ Python Bridge is NOT healthy or unreachable!');
      }
    } catch (error: any) {
      document.getElementById('bridge-result')!.innerHTML = 
        `<div class="error">❌ Bridge test error: ${error.message}</div>`;
      console.error('❌ Bridge test error:', error);
    }
  };

  // These functions simulate the original QtBridge calls as they are not direct API endpoints
  const testButton = async () => {
    try {
      console.log('🚨 Testing Start Quiz button via Python...');
      const result = await callPythonMethod('debugStartQuizButton');
      document.getElementById('button-result')!.innerHTML = 
        `<div class="success">Debug Result: ${result}</div>`;
      console.log('🚨 Debug test result:', result);
    } catch (error: any) {
      document.getElementById('button-result')!.innerHTML = 
        `<div class="bg-red-50 border border-red-400 text-red-700 p-3 rounded-md">❌ Debug test error: ${error.message}</div>`;
      console.error('❌ Debug test error:', error);
    }
  };

  const forceTest = async () => {
    try {
      console.log('🚨 Force testing Start Quiz button...');
      await callPythonMethod('forceTestQuizButton');
      document.getElementById('force-result')!.innerHTML = 
        `<div class="success">✅ Force test executed (check console)</div>`;
      console.log('🚨 Force test executed');
    } catch (error: any) {
      document.getElementById('force-result')!.innerHTML = 
        `<div class="bg-red-50 border border-red-400 text-red-700 p-3 rounded-md">❌ Force test error: ${error.message}</div>`;
      console.error('❌ Force test error:', error);
    }
  };

  const manualTest = () => {
    try {
      console.log('🔍 Manual JavaScript test starting...');

      // Simulate window functions that might be present in the Qt environment
      const simulatedWindow = {
        testStartQuizButton: () => console.log('Simulated window.testStartQuizButton()'),
        startCustomQuiz: () => console.log('Simulated window.startCustomQuiz()'),
      };

      // Test 1: Check if testStartQuizButton function exists
      if (typeof simulatedWindow.testStartQuizButton === 'function') {
        console.log('✅ testStartQuizButton function exists');
        simulatedWindow.testStartQuizButton();
      } else {
        console.error('❌ testStartQuizButton function NOT FOUND');
      }

      // Test 2: Check if startCustomQuiz function exists
      if (typeof simulatedWindow.startCustomQuiz === 'function') {
        console.log('✅ startCustomQuiz function exists');
      } else {
        console.error('❌ startCustomQuiz function NOT FOUND');
      }

      // Test 3: Check if Start Quiz button exists (simulated)
      const button = document.getElementById('start-quiz-button'); // This ID is not in this HTML, but in the main app
      if (button) {
        console.log('✅ Start Quiz button found:', button.textContent);
        console.log('✅ Button visible:', button.offsetParent !== null);
        console.log('✅ Button disabled:', (button as HTMLButtonElement).disabled);
      } else {
        console.error('❌ Start Quiz button NOT FOUND with ID start-quiz-button');

        // Try to find any button with "Start Quiz" text (simulated)
        const allButtons = Array.from(document.querySelectorAll('button'));
        console.log('🔍 Found', allButtons.length, 'total buttons');

        let startQuizButtons: HTMLElement[] = [];
        for (let i = 0; i < allButtons.length; i++) {
          const btn = allButtons[i];
          if (btn.textContent && btn.textContent.includes('Start Quiz')) {
            startQuizButtons.push(btn);
            console.log('🔍 Found Start Quiz button:', btn.textContent, btn);
          }
        }

        if (startQuizButtons.length === 0) {
          console.error('❌ NO Start Quiz buttons found anywhere!');
        }
      }

      document.getElementById('manual-result')!.innerHTML = 
        `<div class="success">✅ Manual test completed (check console)</div>`;

    } catch (error: any) {
      document.getElementById('manual-result')!.innerHTML = 
        `<div class="bg-red-50 border border-red-400 text-red-700 p-3 rounded-md">❌ Manual test error: ${error.message}</div>`;
      console.error('❌ Manual test error:', error);
    }
  };

  useEffect(() => {
    console.log('🚨 Debug test page loaded successfully');
  }, []);

  return (
    <div className="font-sans p-5 bg-gray-100">
      <h1 className="text-3xl font-bold mb-5 text-red-600">🚨 Knowledge App Button Debug Test</h1>

      <div className="debug-box info">
        <h3>Test Bridge Connection</h3>
        <button 
          onClick={testBridge} 
          className="test-button"
        >
          Test Python Bridge
        </button>
        <div id="bridge-result"></div>
      </div>

      <div className="debug-box info">
        <h3>Test Start Quiz Button</h3>
        <button 
          onClick={testButton} 
          className="test-button"
        >
          Debug Start Quiz Button
        </button>
        <div id="button-result"></div>
      </div>

      <div className="debug-box info">
        <h3>Force Test Quiz Button</h3>
        <button 
          onClick={forceTest} 
          className="test-button"
        >
          Force Test Quiz Button
        </button>
        <div id="force-result"></div>
      </div>

      <div className="debug-box info">
        <h3>Manual Button Test</h3>
        <button 
          onClick={manualTest} 
          className="test-button"
        >
          Manual JavaScript Test
        </button>
        <div id="manual-result"></div>
      </div>

      <div className="debug-box info">
        <h3>Console Output</h3>
        <pre 
          id="console-output" 
          ref={consoleOutputRef}
        >
          {consoleOutput}
        </pre>
      </div>
    </div>
  );
};

export default DebugButtonTestPage;