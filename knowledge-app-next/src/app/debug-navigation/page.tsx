'use client';

import React, { useEffect, useRef, useState } from 'react';
import { callPythonMethod } from '../../lib/pythonBridge';

const DebugNavigationPage: React.FC = () => {
  const [output, setOutput] = useState<string>('');
  const outputRef = useRef<HTMLDivElement>(null);

  const log = (message: string, type: 'info' | 'success' | 'error' = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setOutput(prev => {
      const newOutput = `${prev}[${timestamp}] ${message}\n`;
      return newOutput;
    });
    console.log(message);
  };

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  const clearOutput = () => {
    setOutput('');
  };

  

  const testNavButtons = async () => {
    log('🔍 Testing navigation buttons...');
    try {
      const result = await callPythonMethod('testNavigationButtons');
      log(`✅ Python Bridge testNavigationButtons result: ${result}`);
    } catch (error: any) {
      log(`❌ Python Bridge testNavigationButtons error: ${error.message}`);
    }
  };

  const testScreens = async () => {
    log('🔍 Testing screen elements...');
    try {
      const result = await callPythonMethod('testScreenElements');
      log(`✅ Python Bridge testScreenElements result: ${result}`);
    } catch (error: any) {
      log(`❌ Python Bridge testScreenElements error: ${error.message}`);
    }
  };

  const testShowScreen = async () => {
    log('🔍 Testing showScreen function...');
    try {
      const result = await callPythonMethod('testShowScreenFunction', 'quiz', null);
      log(`✅ Python Bridge testShowScreenFunction result: ${result}`);
    } catch (error: any) {
      log(`❌ Python Bridge testShowScreenFunction error: ${error.message}`);
    }
  };

  const forceNavigation = async () => {
    log('🚀 Force navigating to quiz screen...');
    try {
      const result = await callPythonMethod('forceNavigateToQuiz');
      log(`✅ Python Bridge forceNavigateToQuiz result: ${result}`);
    } catch (error: any) {
      log(`❌ Python Bridge forceNavigateToQuiz error: ${error.message}`);
    }
  };

  const navigateTo = async (screenName: string) => {
    log(`🎯 Manually navigating to ${screenName}...`);
    try {
      const result = await callPythonMethod('navigateToScreen', screenName);
      log(`✅ Navigation to ${screenName} completed. Result: ${result}`);
    } catch (error: any) {
      log(`❌ Navigation to ${screenName} failed: ${error.message}`);
    }
  };

  useEffect(() => {
    setTimeout(() => {
      log('🚀 Auto-running basic tests...');
      testNavButtons();
      testScreens();
    }, 1000);
  }, []);

  return (
    <div className="debug-body">
      <h1>🔧 Navigation Debug Tool</h1>
      <p>This tool helps debug the navigation issues in your Knowledge App.</p>

      <div className="debug-panel">
        <h3>🧪 Quick Tests</h3>
        <button onClick={testNavButtons}>Test Navigation Buttons</button>
        <button onClick={testScreens}>Test Screen Elements</button>
        <button onClick={testShowScreen}>Test showScreen Function</button>
        <button onClick={forceNavigation}>Force Navigate to Quiz</button>
        <button onClick={clearOutput}>Clear Output</button>
      </div>

      <div className="debug-panel">
        <h3>🎯 Manual Navigation</h3>
        <button onClick={() => navigateTo('home')}>Go to Home</button>
        <button onClick={() => navigateTo('quiz')}>Go to Quiz</button>
        <button onClick={() => navigateTo('review')}>Go to Review</button>
        <button onClick={() => navigateTo('train')}>Go to Train</button>
        <button onClick={() => navigateTo('settings')}>Go to Settings</button>
      </div>

      <div className="debug-panel">
        <h3>📊 Output</h3>
        <div 
          id="output" 
          ref={outputRef}
        >
          {output}
        </div>
      </div>
    </div>
  );
};

export default DebugNavigationPage;
