interface CriticalLog {
  level: string;
  category: string;
  message: string;
  data: any;
  timestamp: string;
  sessionId: string;
  url: string;
  userAgent: string;
}

class AppLogger {
  static LEVELS = {
    DEBUG: { name: 'DEBUG', emoji: '🔍', color: '#6c757d' },
    INFO: { name: 'INFO', emoji: '💡', color: '#17a2b8' },
    WARN: { name: 'WARN', emoji: '⚠️', color: '#ffc107' },
    ERROR: { name: 'ERROR', emoji: '❌', color: '#dc3545' },
    SUCCESS: { name: 'SUCCESS', emoji: '✅', color: '#28a745' },
    ACTION: { name: 'ACTION', emoji: '🎯', color: '#007bff' },
    PERFORMANCE: { name: 'PERF', emoji: '⏱️', color: '#6f42c1' },
    USER: { name: 'USER', emoji: '👤', color: '#fd7e14' }
  };

  // Session tracking (matching Qt implementation)
  static sessionId: string = `session_${Date.now()}`;
  static startTime: Date = new Date();
  static actionCount: number = 0;
  static currentScreen: string = 'home';

  // Python bridge integration
  static pythonBridge: any = null;

  static log(level: string, category: string, message: string, data: any = null): void {
    const timestamp = new Date().toISOString();
    const levelInfo = AppLogger.LEVELS[level as keyof typeof AppLogger.LEVELS] || { name: level, emoji: '📝', color: '#6c757d' };
    const sessionTime = ((Date.now() - AppLogger.startTime.getTime()) / 1000).toFixed(2);

    // Increment action counter for user actions
    if (level === 'ACTION' || level === 'USER') {
      AppLogger.actionCount++;
    }

    // Format the log message (matching Qt format)
    const logPrefix = `[${timestamp}] ${levelInfo.emoji} ${levelInfo.name}`;
    const logCategory = category ? `[${category}]` : '';
    const sessionInfo = `[Session:${AppLogger.sessionId.slice(-6)} | +${sessionTime}s | Action:${AppLogger.actionCount}]`;
    const fullMessage = `${logPrefix} ${logCategory} ${sessionInfo} ${message}`;

    // Log to console with appropriate method and color
    switch (level) {
      case 'ERROR':
        console.error(`%c${fullMessage}`, `color: ${levelInfo.color}`, data || '');
        break;
      case 'WARN':
        console.warn(`%c${fullMessage}`, `color: ${levelInfo.color}`, data || '');
        break;
      default:
        console.log(`%c${fullMessage}`, `color: ${levelInfo.color}`, data || '');
    }

    // Store critical logs for potential debugging (matching Qt implementation)
    if (level === 'ERROR' || level === 'ACTION' || level === 'USER') {
      AppLogger.storeCriticalLog(level, category, message, data, timestamp);
    }

    // Send to Python backend for server-side logging (matching Qt integration)
    if (AppLogger.pythonBridge && AppLogger.pythonBridge.logClientEvent) {
      try {
        AppLogger.pythonBridge.logClientEvent(JSON.stringify({
          level: level,
          category: category,
          message: message,
          data: data,
          timestamp: timestamp,
          sessionId: AppLogger.sessionId,
          sessionTime: sessionTime,
          actionCount: AppLogger.actionCount
        }));
      } catch (e) {
        console.warn('📡 Failed to send log to backend:', (e as Error).message);
      }
    }
  }

  // Store critical logs in localStorage for debugging (matching Qt implementation)
  static storeCriticalLog(level: string, category: string, message: string, data: any, timestamp: string): void {
    try {
      const criticalLogs: CriticalLog[] = JSON.parse(localStorage.getItem('criticalLogs') || '[]');
      criticalLogs.push({
        level,
        category,
        message,
        data,
        timestamp,
        sessionId: AppLogger.sessionId,
        url: typeof window !== 'undefined' ? window.location.href : '',
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent.slice(0, 100) : ''
      });

      // Keep only last 100 critical logs (matching Qt implementation)
      if (criticalLogs.length > 100) {
        criticalLogs.splice(0, criticalLogs.length - 100);
      }

      localStorage.setItem('criticalLogs', JSON.stringify(criticalLogs));
    } catch (e) {
      console.error('Failed to store critical log:', e);
    }
  }

  // Log user interactions with UI elements (matching Qt implementation)
  static userAction(elementType: string, action: string, details: any = {}): void {
    AppLogger.user('UI_INTERACTION', `${elementType.toUpperCase()}_${action.toUpperCase()}`, {
      elementType,
      action,
      ...details,
      timestamp: Date.now(),
      screenTime: ((Date.now() - AppLogger.startTime.getTime()) / 1000).toFixed(2)
    });
  }

  // Log navigation events (matching Qt implementation)
  static navigation(from: string, to: string, method: string = 'click'): void {
    AppLogger.action('NAVIGATION', `Screen changed: ${from} → ${to}`, {
      fromScreen: from,
      toScreen: to,
      method: method,
      timestamp: Date.now()
    });
  }

  // Track navigation with comprehensive details (matching Qt implementation)
  static trackNavigation(from: string, to: string, method: string = 'click'): void {
    AppLogger.action('NAVIGATION', `Screen Navigation: ${from} → ${to}`, {
      fromScreen: from,
      toScreen: to,
      method: method,
      timestamp: Date.now(),
      sessionTime: ((Date.now() - AppLogger.startTime.getTime()) / 1000).toFixed(2)
    });
  }

  // Track user actions with context (matching Qt implementation)
  static trackUserAction(action: string, context: any = {}): void {
    AppLogger.user('USER_ACTION', action, {
      ...context,
      currentScreen: AppLogger.currentScreen,
      timestamp: Date.now(),
      sessionTime: ((Date.now() - AppLogger.startTime.getTime()) / 1000).toFixed(2)
    });
  }

  // Log performance metrics (matching Qt implementation)
  static performanceMetric(operation: string, duration: number, success: boolean = true, metadata: any = {}): void {
    AppLogger.performance('METRICS', `${operation}: ${duration}ms ${success ? '✅' : '❌'}`, {
      operation,
      duration,
      success,
      ...metadata
    });
  }

  // Get session summary (matching Qt implementation)
  static getSessionSummary(): any {
    const duration = ((Date.now() - AppLogger.startTime.getTime()) / 1000).toFixed(2);
    return {
      sessionId: AppLogger.sessionId,
      startTime: AppLogger.startTime.toISOString(),
      duration: `${duration}s`,
      actionCount: AppLogger.actionCount,
      currentScreen: AppLogger.currentScreen
    };
  }

  // Display comprehensive logging summary (matching Qt implementation)
  static displayLogSummary(): any {
    const summary = AppLogger.getSessionSummary();
    const criticalLogs: CriticalLog[] = JSON.parse(localStorage.getItem('criticalLogs') || '[]');

    console.group('📊 Knowledge App Logging Summary');
    console.log('Session ID:', summary.sessionId);
    console.log('Session Start:', summary.startTime);
    console.log('Session Duration:', summary.duration);
    console.log('Total Actions:', summary.actionCount);
    console.log('Current Screen:', summary.currentScreen);
    console.log('Critical Logs Count:', criticalLogs.length);

    if (criticalLogs.length > 0) {
      console.group('Recent Critical Logs (Last 10):');
      criticalLogs.slice(-10).forEach((log, index) => {
        console.log(`${index + 1}. [${log.level}] ${log.category}: ${log.message}`, log.data);
      });
      console.groupEnd();
    }

    console.log('Python Bridge Status:', {
      available: !!AppLogger.pythonBridge,
      methods: AppLogger.pythonBridge ? 
        Object.getOwnPropertyNames(AppLogger.pythonBridge)
          .filter(name => typeof AppLogger.pythonBridge[name] === 'function')
          .slice(0, 10) : []
    });

    console.groupEnd();
    return summary;
  }

  // Clear critical logs (matching Qt implementation)
  static clearCriticalLogs(): void {
    localStorage.removeItem('criticalLogs');
    AppLogger.info('SYSTEM', 'Critical logs cleared');
  }

  // Track quiz actions with enhanced logging (matching Qt implementation)
  static trackQuizAction(action: string, details: any = {}): void {
    AppLogger.action('QUIZ_ACTION', action, {
      ...details,
      currentScreen: AppLogger.currentScreen,
      timestamp: Date.now()
    });
  }

  // Track errors with context (matching Qt implementation)
  static trackError(category: string, error: Error | string, context: any = {}): void {
    const errorMessage = typeof error === 'string' ? error : error.message;
    const errorStack = typeof error === 'object' && error.stack ? error.stack : undefined;
    
    AppLogger.error(category, `Error: ${errorMessage}`, {
      error: errorMessage,
      stack: errorStack,
      ...context,
      currentScreen: AppLogger.currentScreen,
      timestamp: Date.now()
    });
  }

  // Set current screen (for navigation tracking)
  static setCurrentScreen(screenName: string): void {
    AppLogger.currentScreen = screenName;
    AppLogger.info('SCREEN', `Current screen set to: ${screenName}`);
  }

  // Set Python bridge reference for logging integration
  static setPythonBridge(bridge: any): void {
    AppLogger.pythonBridge = bridge;
    AppLogger.info('BRIDGE', 'Python bridge reference set for logging integration');
  }

  // Convenience methods (enhanced with session tracking)
  static debug(category: string, message: string, data?: any): void {
    AppLogger.log('DEBUG', category, message, data);
  }

  static info(category: string, message: string, data?: any): void {
    AppLogger.log('INFO', category, message, data);
  }

  static warn(category: string, message: string, data?: any): void {
    AppLogger.log('WARN', category, message, data);
  }

  static warning(category: string, message: string, data?: any): void {
    AppLogger.log('WARN', category, message, data);
  }

  static error(category: string, message: string, data?: any): void {
    AppLogger.log('ERROR', category, message, data);
  }

  static success(category: string, message: string, data?: any): void {
    AppLogger.log('SUCCESS', category, message, data);
  }

  static action(category: string, message: string, data?: any): void {
    AppLogger.log('ACTION', category, message, data);
  }

  static performance(category: string, message: string, data?: any): void {
    AppLogger.log('PERFORMANCE', category, message, data);
  }

  static user(category: string, message: string, data?: any): void {
    AppLogger.log('USER', category, message, data);
  }

  static system(category: string, message: string, data?: any): void {
    AppLogger.log('INFO', category, message, data);
  }
}

export { AppLogger };

export function setCurrentScreenName(screenName: string): void {
  AppLogger.info('SCREEN', `Current screen set to: ${screenName}`);
}
