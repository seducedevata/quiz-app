import { AppLogger } from './logger';
import { callPythonMethod } from './pythonBridge';

declare global {
  interface Window {
    debugApp: {
      logs: () => any;
      session: () => any;
      critical: () => any[];
      clearLogs: () => void;
      testLog: () => void;
      bridge: () => Promise<any>;
      testButtons: () => void;
      fixButtons: () => void;
      testNavigation: (screenName?: string) => void;
      clickNav: (screenName: string) => void;
      bridgeReconnect: () => Promise<boolean>;
      bridgeClearQueue: () => Promise<void>;
      bridgeReport: () => Promise<any>;
      testRecovery: (errorType?: string) => Promise<boolean>;
      recoveryStats: () => Promise<any>;
      help: () => void;
    };
  }
}

window.debugApp = {
  // Show logging summary (matching Qt implementation)
  logs: () => AppLogger.displayLogSummary(),

  // Get current session info (matching Qt implementation)
  session: () => AppLogger.getSessionSummary(),

  // Get critical logs (matching Qt implementation)
  critical: () => JSON.parse(localStorage.getItem('criticalLogs') || '[]'),

  // Clear critical logs (matching Qt implementation)
  clearLogs: () => AppLogger.clearCriticalLogs(),

  // Test logging system (matching Qt implementation)
  testLog: () => {
    AppLogger.info('TEST', 'Logging system test', { testData: 'Hello World!' });
    console.log('✅ Test log sent. Check console output above.');
  },
  // Get Python bridge status (matching Qt implementation)
  bridge: async () => {
    AppLogger.info('DEBUG_TOOL', 'Checking Python bridge status...');
    try {
      // Import bridge functions dynamically
      const { getConnectionHealth, getConnectionStats, getQueueStatus, checkBridgeHealth } = await import('./pythonBridge');
      
      const health = getConnectionHealth();
      const stats = getConnectionStats();
      const queueStatus = getQueueStatus();
      const isHealthy = await checkBridgeHealth();
      
      const bridgeStatus = {
        available: !!AppLogger.pythonBridge,
        healthy: isHealthy,
        connected: health.connected,
        quality: health.quality,
        methods: AppLogger.pythonBridge ? 
          Object.getOwnPropertyNames(AppLogger.pythonBridge)
            .filter(name => typeof AppLogger.pythonBridge[name] === 'function') : [],
        properties: AppLogger.pythonBridge ? 
          Object.getOwnPropertyNames(AppLogger.pythonBridge)
            .filter(name => typeof AppLogger.pythonBridge[name] !== 'function') : [],
        health,
        stats,
        queueStatus
      };
      
      console.group('🔗 Python Bridge Status');
      console.log('Available:', bridgeStatus.available);
      console.log('Health Check:', bridgeStatus.healthy ? 'HEALTHY' : 'UNHEALTHY');
      console.log('Methods:', bridgeStatus.methods);
      console.log('Properties:', bridgeStatus.properties);
      console.log('Connection Health:', bridgeStatus.health);
      console.log('Connection Stats:', bridgeStatus.stats);
      console.log('Queue Status:', bridgeStatus.queueStatus);
      console.groupEnd();
      
      AppLogger.success('DEBUG_TOOL', 'Python bridge status retrieved successfully', bridgeStatus);
      return bridgeStatus;
    } catch (error) {
      AppLogger.error('DEBUG_TOOL', 'Failed to get Python bridge status:', error);
      return { error: (error as Error).message };
    }
  },
  // Test button functionality (matching Qt implementation)
  testButtons: () => {
    console.log('🔍 Navigation Button Status Check:');

    try {
      // Check navigation buttons (matching Qt implementation)
      const navButtons = document.querySelectorAll('[data-nav], .nav-item, nav button, nav a');
      navButtons.forEach((button, index) => {
        const rect = button.getBoundingClientRect();
        const styles = window.getComputedStyle(button);
        const text = button.textContent?.trim() || '';
        const isButton = button.tagName === 'BUTTON';
        const isDisabled = isButton ? (button as HTMLButtonElement).disabled : false;
        
        console.log(`✅ Nav Button ${index + 1} (${text}):`, {
          found: true,
          visible: rect.width > 0 && rect.height > 0,
          pointerEvents: styles.pointerEvents,
          zIndex: styles.zIndex,
          disabled: isDisabled,
          clickable: !isDisabled && styles.pointerEvents !== 'none',
          onclick: (button as any).onclick ? 'Has onclick' : 'No onclick',
          hasEventListener: (button as any)._hasEventListener || false
        });
      });

      // Check screens (matching Qt implementation)
      console.log('🔍 Screen Status Check:');
      const screens = ['home', 'quiz', 'review', 'train', 'settings'];
      screens.forEach(screenName => {
        const screen = document.querySelector(`[data-screen="${screenName}"], #${screenName}-screen, .${screenName}-screen`);
        if (screen) {
          const isActive = screen.classList.contains('active') || screen.classList.contains('visible');
          const isVisible = (screen as HTMLElement).style.display !== 'none';
          console.log(`✅ ${screenName}-screen:`, {
            found: true,
            active: isActive,
            visible: isVisible,
            display: (screen as HTMLElement).style.display
          });
        } else {
          console.log(`❌ ${screenName}-screen: Not found`);
        }
      });

      console.log('🔍 Current State:', {
        currentScreen: AppLogger.currentScreen,
        pythonBridge: !!AppLogger.pythonBridge
      });

      AppLogger.info('DEBUG_TOOL', `Button status check completed for ${navButtons.length} navigation buttons`);
    } catch (e) {
      AppLogger.error('DEBUG_TOOL', 'Failed to check button status', e);
    }
  },
  // Force fix button handlers (matching Qt implementation)
  fixButtons: () => {
    console.log('🔧 Force-fixing button handlers...');

    try {
      const selectors = 'button, a, input[type="submit"], [role="button"], [tabindex="0"]';
      const elements = Array.from(document.querySelectorAll<HTMLElement>(selectors));
      let fixed = 0;
      
      elements.forEach((el) => {
        // Ensure pointer events and cursor
        const style = window.getComputedStyle(el);
        if (style.pointerEvents === 'none') {
          (el.style as any).pointerEvents = 'auto';
          fixed++;
        }
        if (style.cursor !== 'pointer' && el.tagName !== 'INPUT') {
          el.style.cursor = 'pointer';
        }
        // Ensure element is focusable
        if (!el.hasAttribute('tabindex') && el.tagName !== 'BUTTON' && el.tagName !== 'A') {
          el.setAttribute('tabindex', '0');
        }
        // Ensure z-index for buttons
        if (el.tagName === 'BUTTON') {
          el.style.position = 'relative';
          el.style.zIndex = '9999';
        }
      });

      // Re-run button setup functions if they exist (matching Qt implementation)
      const setupFunctions = ['setupStartQuizButton', 'setupQuickQuizButton', 'setupNavigationButtons'];
      setupFunctions.forEach(funcName => {
        if (typeof (window as any)[funcName] === 'function') {
          try {
            (window as any)[funcName]();
            console.log(`✅ Re-ran ${funcName}`);
          } catch (e) {
            console.warn(`⚠️ Failed to re-run ${funcName}:`, e);
          }
        }
      });

      console.log('✅ Button handlers re-attached');
      AppLogger.success('DEBUG_TOOL', `Button fix completed. Fixed ${fixed} elements out of ${elements.length} total.`);
    } catch (e) {
      AppLogger.error('DEBUG_TOOL', 'Failed to fix button clickability', e);
    }
  },
  // Test navigation directly (matching Qt implementation)
  testNavigation: (screenName: string = 'quiz') => {
    console.log(`🧪 Testing navigation to ${screenName}...`);
    try {
      // Try Next.js router navigation first
      if (typeof window !== 'undefined' && (window as any).next?.router) {
        (window as any).next.router.push(`/${screenName}`);
        console.log(`✅ Next.js navigation to ${screenName} completed`);
      } else {
        // Fallback to history API
        window.history.pushState({}, '', `/${screenName}`);
        console.log(`✅ History API navigation to ${screenName} completed`);
      }
      
      AppLogger.trackNavigation(AppLogger.currentScreen, screenName, 'debug_test');
      AppLogger.setCurrentScreen(screenName);
    } catch (error) {
      console.error(`❌ Navigation to ${screenName} failed:`, error);
      AppLogger.error('DEBUG_TOOL', `Navigation test failed: ${(error as Error).message}`);
    }
  },

  // Force click a navigation button (matching Qt implementation)
  clickNav: (screenName: string) => {
    const navButtons = document.querySelectorAll('[data-nav], .nav-item, nav button, nav a');
    const targetButton = Array.from(navButtons).find(btn =>
      btn.textContent?.toLowerCase().includes(screenName.toLowerCase()) ||
      (btn as HTMLElement).dataset.nav === screenName ||
      btn.getAttribute('href')?.includes(screenName)
    );

    if (targetButton) {
      console.log(`🖱️ Force clicking ${screenName} button...`);
      (targetButton as HTMLElement).click();
      AppLogger.userAction('button', 'force_click', { screenName, element: targetButton.tagName });
    } else {
      console.error(`❌ Could not find navigation button for ${screenName}`);
      AppLogger.error('DEBUG_TOOL', `Navigation button not found: ${screenName}`);
    }
  },

  // Bridge recovery tools
  bridgeReconnect: async () => {
    console.log('🔄 Attempting bridge reconnection...');
    try {
      const { forceReconnect } = await import('./pythonBridge');
      const success = await forceReconnect();
      console.log(success ? '✅ Bridge reconnection successful' : '❌ Bridge reconnection failed');
      return success;
    } catch (error) {
      console.error('❌ Bridge reconnection error:', error);
      return false;
    }
  },

  // Clear bridge queue
  bridgeClearQueue: async () => {
    console.log('🧹 Clearing bridge method call queue...');
    try {
      const { clearQueue } = await import('./pythonBridge');
      clearQueue();
      console.log('✅ Bridge queue cleared');
    } catch (error) {
      console.error('❌ Failed to clear bridge queue:', error);
    }
  },

  // Get bridge monitoring report
  bridgeReport: async () => {
    try {
      const { bridgeMonitoringService } = await import('./bridgeMonitorSimple');
      const report = bridgeMonitoringService.generateReport();
      
      console.group('📊 Bridge Monitoring Report');
      console.log('Status:', report.status);
      console.log('Performance:', report.performance);
      console.log('Metrics:', report.metrics);
      console.log('Recommendations:', report.recommendations);
      console.groupEnd();
      
      return report;
    } catch (error) {
      console.error('❌ Failed to generate bridge report:', error);
      return null;
    }
  },

  // Test error recovery
  testRecovery: async (errorType = 'bridge_connection') => {
    console.log(`🧪 Testing error recovery for: ${errorType}`);
    try {
      const { errorRecoveryService } = await import('./errorRecovery');
      const success = await errorRecoveryService.attemptRecovery(errorType, 'Debug test recovery');
      console.log(success ? '✅ Recovery test successful' : '❌ Recovery test failed');
      return success;
    } catch (error) {
      console.error('❌ Recovery test error:', error);
      return false;
    }
  },

  // Get recovery stats
  recoveryStats: async () => {
    try {
      const { errorRecoveryService } = await import('./errorRecovery');
      const stats = errorRecoveryService.getRecoveryStats();
      
      console.group('📈 Error Recovery Statistics');
      console.log('Total Attempts:', stats.totalAttempts);
      console.log('Successful Attempts:', stats.successfulAttempts);
      console.log('Success Rate:', `${stats.successRate.toFixed(1)}%`);
      console.log('Strategy Stats:', stats.strategyStats);
      console.log('Active Recoveries:', stats.activeRecoveries);
      console.groupEnd();
      
      return stats;
    } catch (error) {
      console.error('❌ Failed to get recovery stats:', error);
      return null;
    }
  },

  // Quick help (matching Qt implementation)
  help: () => {
    console.log(`
🛠️ Knowledge App Debug Commands:
- debugApp.logs()           - Show comprehensive logging summary
- debugApp.session()        - Get current session information  
- debugApp.critical()       - Get critical logs array
- debugApp.clearLogs()      - Clear stored critical logs
- debugApp.testLog()        - Test logging system
- debugApp.bridge()         - Get Python bridge status
- debugApp.testButtons()    - Check button status and clickability
- debugApp.fixButtons()     - Force re-attach button handlers
- debugApp.testNavigation() - Test navigation to a screen
- debugApp.clickNav('quiz') - Force click navigation button
- debugApp.bridgeReconnect() - Force bridge reconnection
- debugApp.bridgeClearQueue() - Clear bridge method call queue
- debugApp.bridgeReport()   - Get comprehensive bridge report
- debugApp.testRecovery()   - Test error recovery system
- debugApp.recoveryStats()  - Get error recovery statistics
- debugApp.help()           - Show this help
        `);
  }
};

// 🚨 GLOBAL ERROR HANDLERS FOR COMPREHENSIVE LOGGING (matching Qt implementation)
if (typeof window !== 'undefined') {
  window.addEventListener('error', function (event) {
    // Filter out common non-critical errors
    const message = event.error?.message || event.message || '';
    const filename = event.filename || '';
    
    // Skip logging for known non-critical errors
    if (message.includes('ResizeObserver loop limit exceeded') ||
        message.includes('Non-Error promise rejection captured') ||
        filename.includes('extensions/') ||
        filename.includes('chrome-extension://')) {
      return;
    }
    
    AppLogger.trackError('GLOBAL_ERROR', 'Unhandled JavaScript error', {
      message: message,
      filename: filename,
      lineno: event.lineno,
      colno: event.colno,
      stack: event.error?.stack,
      currentScreen: AppLogger.currentScreen,
      userAgent: navigator.userAgent.slice(0, 100)
    });
    
    // Try to recover from common errors
    if (message.includes('Cannot read property') || message.includes('Cannot read properties')) {
      console.log('🔧 Attempting to recover from property access error');
      // Add any recovery logic here
    }
  });

  window.addEventListener('unhandledrejection', function (event) {
    const reason = event.reason?.message || event.reason || '';
    
    // Skip logging for known non-critical promise rejections
    if (typeof reason === 'string' && (
        reason.includes('Load failed') ||
        reason.includes('NetworkError') ||
        reason.includes('AbortError'))) {
      return;
    }
    
    AppLogger.trackError('GLOBAL_ERROR', 'Unhandled promise rejection', {
      reason: reason,
      stack: event.reason?.stack,
      currentScreen: AppLogger.currentScreen,
      promise: event.promise
    });
    
    // Prevent the error from being logged to console if it's handled
    event.preventDefault();
  });

  // Log page visibility changes for user engagement tracking
  document.addEventListener('visibilitychange', function () {
    AppLogger.user('PAGE_VISIBILITY', document.hidden ? 'Page hidden' : 'Page visible', {
      hidden: document.hidden,
      visibilityState: document.visibilityState,
      currentScreen: AppLogger.currentScreen
    });
  });

  // Track critical DOM events
  document.addEventListener('DOMContentLoaded', function () {
    AppLogger.info('SYSTEM', 'DOM content loaded');
  });

  window.addEventListener('load', function () {
    AppLogger.performanceMetric('SYSTEM', performance.now(), true, {
      loadTime: performance.now(),
      domContentLoadedTime: performance.timing ? 
        performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart : 0,
      pageLoadTime: performance.timing ? 
        performance.timing.loadEventEnd - performance.timing.navigationStart : 0
    });
  });
}

AppLogger.info('SYSTEM', '🚀 Knowledge App Logging System Initialized', AppLogger.getSessionSummary());
AppLogger.debug('SYSTEM', 'Debug tools loaded (use debugApp.help() in console)');
AppLogger.info('SYSTEM', 'Debug console tools (window.debugApp) initialized.');
