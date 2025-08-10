// Error recovery service matching Qt implementation
import { AppLogger } from './logger';
import { sessionTracker } from './sessionTracker';
import { callPythonMethod, checkBridgeHealth } from './pythonBridge';

interface RecoveryStrategy {
  name: string;
  description: string;
  execute: () => Promise<boolean>;
  canRetry: boolean;
  maxAttempts: number;
}

interface RecoveryAttempt {
  strategyName: string;
  timestamp: number;
  success: boolean;
  error?: string;
  duration: number;
}

class ErrorRecoveryService {
  private recoveryHistory: RecoveryAttempt[] = [];
  private activeRecoveries: Map<string, Promise<boolean>> = new Map();

  // Recovery strategies matching Qt implementation patterns
  private strategies: Map<string, RecoveryStrategy> = new Map([
    ['bridge_reconnect', {
      name: 'Bridge Reconnection',
      description: 'Attempt to reconnect to Python backend',
      execute: this.reconnectBridge.bind(this),
      canRetry: true,
      maxAttempts: 3,
    }],
    ['cache_clear', {
      name: 'Cache Clear',
      description: 'Clear application cache and reload',
      execute: this.clearCacheAndReload.bind(this),
      canRetry: false,
      maxAttempts: 1,
    }],
    ['session_reset', {
      name: 'Session Reset',
      description: 'Reset user session and state',
      execute: this.resetSession.bind(this),
      canRetry: true,
      maxAttempts: 2,
    }],
    ['component_remount', {
      name: 'Component Remount',
      description: 'Force remount of React components',
      execute: this.forceComponentRemount.bind(this),
      canRetry: true,
      maxAttempts: 3,
    }],
    ['fallback_mode', {
      name: 'Fallback Mode',
      description: 'Switch to offline/fallback functionality',
      execute: this.enableFallbackMode.bind(this),
      canRetry: false,
      maxAttempts: 1,
    }],
  ]);

  // Attempt automatic recovery based on error type
  async attemptRecovery(errorType: string, errorMessage: string): Promise<boolean> {
    const recoveryKey = `${errorType}_${Date.now()}`;
    
    if (this.activeRecoveries.has(recoveryKey)) {
      AppLogger.warn('ERROR_RECOVERY', 'Recovery already in progress', { errorType });
      return false;
    }

    AppLogger.info('ERROR_RECOVERY', 'Starting automatic recovery', {
      errorType,
      errorMessage: errorMessage.substring(0, 100),
    });

    sessionTracker.logAction('AUTO_RECOVERY_START', {
      errorType,
      errorMessage: errorMessage.substring(0, 100),
    });

    const recoveryPromise = this.executeRecoverySequence(errorType, errorMessage);
    this.activeRecoveries.set(recoveryKey, recoveryPromise);

    try {
      const success = await recoveryPromise;
      this.activeRecoveries.delete(recoveryKey);
      
      AppLogger.info('ERROR_RECOVERY', 'Recovery completed', {
        errorType,
        success,
        totalAttempts: this.recoveryHistory.filter(h => h.strategyName.includes(errorType)).length,
      });

      return success;
    } catch (recoveryError) {
      this.activeRecoveries.delete(recoveryKey);
      AppLogger.error('ERROR_RECOVERY', 'Recovery failed', {
        errorType,
        error: (recoveryError as Error).message,
      });
      return false;
    }
  }

  // Execute recovery sequence based on error type
  private async executeRecoverySequence(errorType: string, errorMessage: string): Promise<boolean> {
    const strategies = this.selectRecoveryStrategies(errorType, errorMessage);
    
    for (const strategyName of strategies) {
      const strategy = this.strategies.get(strategyName);
      if (!strategy) continue;

      const recentAttempts = this.recoveryHistory.filter(
        h => h.strategyName === strategyName && 
        Date.now() - h.timestamp < 300000 // Last 5 minutes
      ).length;

      if (recentAttempts >= strategy.maxAttempts) {
        AppLogger.warn('ERROR_RECOVERY', 'Strategy max attempts reached', {
          strategyName,
          recentAttempts,
          maxAttempts: strategy.maxAttempts,
        });
        continue;
      }

      const startTime = Date.now();
      let success = false;
      let error: string | undefined;

      try {
        AppLogger.info('ERROR_RECOVERY', 'Executing recovery strategy', {
          strategyName,
          description: strategy.description,
        });

        success = await strategy.execute();
        
        if (success) {
          AppLogger.success('ERROR_RECOVERY', 'Recovery strategy succeeded', {
            strategyName,
            duration: Date.now() - startTime,
          });
        }
      } catch (strategyError) {
        error = (strategyError as Error).message;
        AppLogger.error('ERROR_RECOVERY', 'Recovery strategy failed', {
          strategyName,
          error,
          duration: Date.now() - startTime,
        });
      }

      // Record recovery attempt
      this.recoveryHistory.push({
        strategyName,
        timestamp: Date.now(),
        success,
        error,
        duration: Date.now() - startTime,
      });

      // Keep only last 50 recovery attempts
      if (this.recoveryHistory.length > 50) {
        this.recoveryHistory = this.recoveryHistory.slice(-50);
      }

      if (success) {
        sessionTracker.logAction('AUTO_RECOVERY_SUCCESS', {
          strategyName,
          errorType,
        });
        return true;
      }

      // Wait before trying next strategy
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    sessionTracker.logAction('AUTO_RECOVERY_FAILED', {
      errorType,
      strategiesAttempted: strategies.length,
    });

    return false;
  }

  // Select appropriate recovery strategies based on error type
  private selectRecoveryStrategies(errorType: string, errorMessage: string): string[] {
    const message = errorMessage.toLowerCase();

    // Network/Bridge errors
    if (message.includes('bridge') || message.includes('python') || message.includes('connection')) {
      return ['bridge_reconnect', 'session_reset', 'fallback_mode'];
    }

    // Component/UI errors
    if (message.includes('component') || message.includes('render') || message.includes('react')) {
      return ['component_remount', 'session_reset'];
    }

    // Data/Cache errors
    if (message.includes('cache') || message.includes('storage') || message.includes('data')) {
      return ['cache_clear', 'session_reset'];
    }

    // Permission/Auth errors
    if (message.includes('permission') || message.includes('unauthorized') || message.includes('forbidden')) {
      return ['session_reset', 'fallback_mode'];
    }

    // Default recovery sequence
    return ['component_remount', 'session_reset', 'cache_clear'];
  }

  // Recovery strategy implementations
  private async reconnectBridge(): Promise<boolean> {
    try {
      // Wait a moment for any ongoing operations to complete
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Check bridge health
      const isHealthy = await checkBridgeHealth();
      if (isHealthy) {
        AppLogger.success('ERROR_RECOVERY', 'Bridge reconnection successful');
        return true;
      }

      // Try to call a simple method to test connectivity
      await callPythonMethod('test_connection');
      AppLogger.success('ERROR_RECOVERY', 'Bridge test call successful');
      return true;
    } catch (error) {
      AppLogger.error('ERROR_RECOVERY', 'Bridge reconnection failed', error);
      return false;
    }
  }

  private async clearCacheAndReload(): Promise<boolean> {
    try {
      // Clear various caches
      if ('caches' in window) {
        const cacheNames = await caches.keys();
        await Promise.all(cacheNames.map(name => caches.delete(name)));
      }

      // Clear localStorage (except critical data)
      const criticalKeys = ['criticalLogs', 'sessionTracker', 'quizSettings'];
      const keysToRemove = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && !criticalKeys.includes(key)) {
          keysToRemove.push(key);
        }
      }
      keysToRemove.forEach(key => localStorage.removeItem(key));

      // Clear sessionStorage
      sessionStorage.clear();

      AppLogger.info('ERROR_RECOVERY', 'Cache cleared, reloading page');
      
      // Reload page after a short delay
      setTimeout(() => {
        window.location.reload();
      }, 1000);

      return true;
    } catch (error) {
      AppLogger.error('ERROR_RECOVERY', 'Cache clear failed', error);
      return false;
    }
  }

  private async resetSession(): Promise<boolean> {
    try {
      // Reset session tracker
      sessionTracker.clearSession();

      // Reset AppLogger session
      AppLogger.sessionId = `session_${Date.now()}`;
      AppLogger.startTime = new Date();
      AppLogger.actionCount = 0;

      // Clear non-critical localStorage
      const criticalKeys = ['criticalLogs'];
      const keysToRemove = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && !criticalKeys.includes(key)) {
          keysToRemove.push(key);
        }
      }
      keysToRemove.forEach(key => localStorage.removeItem(key));

      AppLogger.info('ERROR_RECOVERY', 'Session reset completed');
      return true;
    } catch (error) {
      AppLogger.error('ERROR_RECOVERY', 'Session reset failed', error);
      return false;
    }
  }

  private async forceComponentRemount(): Promise<boolean> {
    try {
      // Trigger a React re-render by dispatching a custom event
      const remountEvent = new CustomEvent('forceRemount', {
        detail: { timestamp: Date.now() }
      });
      window.dispatchEvent(remountEvent);

      AppLogger.info('ERROR_RECOVERY', 'Component remount triggered');
      return true;
    } catch (error) {
      AppLogger.error('ERROR_RECOVERY', 'Component remount failed', error);
      return false;
    }
  }

  private async enableFallbackMode(): Promise<boolean> {
    try {
      // Set fallback mode flag
      localStorage.setItem('fallbackMode', 'true');
      
      // Dispatch fallback mode event
      const fallbackEvent = new CustomEvent('enableFallbackMode', {
        detail: { timestamp: Date.now() }
      });
      window.dispatchEvent(fallbackEvent);

      AppLogger.info('ERROR_RECOVERY', 'Fallback mode enabled');
      return true;
    } catch (error) {
      AppLogger.error('ERROR_RECOVERY', 'Fallback mode activation failed', error);
      return false;
    }
  }

  // Get recovery statistics
  getRecoveryStats(): any {
    const now = Date.now();
    const recentAttempts = this.recoveryHistory.filter(h => now - h.timestamp < 3600000); // Last hour
    const successfulAttempts = recentAttempts.filter(h => h.success);

    const strategyStats = new Map<string, { attempts: number; successes: number; avgDuration: number }>();
    
    recentAttempts.forEach(attempt => {
      const current = strategyStats.get(attempt.strategyName) || { attempts: 0, successes: 0, avgDuration: 0 };
      current.attempts++;
      if (attempt.success) current.successes++;
      current.avgDuration = (current.avgDuration * (current.attempts - 1) + attempt.duration) / current.attempts;
      strategyStats.set(attempt.strategyName, current);
    });

    return {
      totalAttempts: recentAttempts.length,
      successfulAttempts: successfulAttempts.length,
      successRate: recentAttempts.length > 0 ? (successfulAttempts.length / recentAttempts.length) * 100 : 0,
      strategyStats: Object.fromEntries(strategyStats),
      activeRecoveries: this.activeRecoveries.size,
    };
  }
}

// Create global error recovery service instance
export const errorRecoveryService = new ErrorRecoveryService();

// Make it globally available for debugging
if (typeof window !== 'undefined') {
  (window as any).errorRecoveryService = errorRecoveryService;
}