import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AppLogger } from '../../lib/logger';
import { sessionTracker } from '../../lib/sessionTracker';
import { callPythonMethod, checkBridgeHealth } from '../../lib/pythonBridge';

interface Props {
  children?: ReactNode;
  fallbackComponent?: React.ComponentType<ErrorFallbackProps>;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  isolate?: boolean; // Whether to isolate this boundary from parent boundaries
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string | null;
  retryCount: number;
  isRecovering: boolean;
  recoveryStrategy: 'retry' | 'fallback' | 'reload' | 'navigate' | null;
}

interface ErrorFallbackProps {
  error: Error;
  errorInfo: ErrorInfo;
  retry: () => void;
  goHome: () => void;
  errorId: string;
  retryCount: number;
}

// Error classification system matching Qt implementation
enum ErrorCategory {
  NETWORK = 'NETWORK',
  PYTHON_BRIDGE = 'PYTHON_BRIDGE',
  UI_COMPONENT = 'UI_COMPONENT',
  DATA_PROCESSING = 'DATA_PROCESSING',
  PERMISSION = 'PERMISSION',
  TIMEOUT = 'TIMEOUT',
  UNKNOWN = 'UNKNOWN'
}

interface ErrorClassification {
  category: ErrorCategory;
  severity: 'low' | 'medium' | 'high' | 'critical';
  recoverable: boolean;
  autoRetry: boolean;
  fallbackAvailable: boolean;
  userActionRequired: boolean;
}

class ErrorBoundary extends Component<Props, State> {
  private retryTimeouts: NodeJS.Timeout[] = [];
  private maxRetries = 3;
  private retryDelay = 1000; // Start with 1 second

  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
    errorId: null,
    retryCount: 0,
    isRecovering: false,
    recoveryStrategy: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    // Generate unique error ID for tracking
    const errorId = `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    return { 
      hasError: true, 
      error: error, 
      errorInfo: null,
      errorId,
      retryCount: 0,
      isRecovering: false,
      recoveryStrategy: null,
    };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    const classification = this.classifyError(error);
    const sessionSummary = AppLogger.getSessionSummary();
    const sessionAnalytics = sessionTracker.getSessionAnalytics();

    // Enhanced error logging with classification
    AppLogger.trackError('REACT_ERROR_BOUNDARY', error, {
      errorId: this.state.errorId,
      classification,
      componentStack: errorInfo.componentStack,
      currentScreen: sessionSummary.currentScreen,
      sessionId: sessionSummary.sessionId,
      sessionAnalytics,
      userAgent: navigator.userAgent,
      url: window.location.href,
      timestamp: new Date().toISOString(),
    });

    // Log user action for error occurrence
    sessionTracker.logAction('ERROR_OCCURRED', {
      errorId: this.state.errorId,
      errorMessage: error.message,
      classification,
      componentStack: errorInfo.componentStack.split('\n')[1] || 'Unknown component',
    });

    this.setState({ errorInfo: errorInfo });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Attempt automatic recovery for recoverable errors
    if (classification.recoverable && classification.autoRetry && this.state.retryCount < this.maxRetries) {
      this.attemptAutoRecovery(classification);
    }

    // Send error to Python backend for server-side logging
    this.reportErrorToBackend(error, errorInfo, classification);
  }

  // Error classification system matching Qt patterns
  private classifyError(error: Error): ErrorClassification {
    const message = error.message.toLowerCase();
    const stack = error.stack?.toLowerCase() || '';

    // Network errors
    if (message.includes('network') || message.includes('fetch') || message.includes('connection')) {
      return {
        category: ErrorCategory.NETWORK,
        severity: 'medium',
        recoverable: true,
        autoRetry: true,
        fallbackAvailable: true,
        userActionRequired: false,
      };
    }

    // Python bridge errors
    if (message.includes('bridge') || message.includes('python') || stack.includes('pythonbridge')) {
      return {
        category: ErrorCategory.PYTHON_BRIDGE,
        severity: 'high',
        recoverable: true,
        autoRetry: true,
        fallbackAvailable: false,
        userActionRequired: true,
      };
    }

    // Permission errors
    if (message.includes('permission') || message.includes('denied') || message.includes('unauthorized')) {
      return {
        category: ErrorCategory.PERMISSION,
        severity: 'high',
        recoverable: false,
        autoRetry: false,
        fallbackAvailable: true,
        userActionRequired: true,
      };
    }

    // Timeout errors
    if (message.includes('timeout') || message.includes('abort')) {
      return {
        category: ErrorCategory.TIMEOUT,
        severity: 'medium',
        recoverable: true,
        autoRetry: true,
        fallbackAvailable: true,
        userActionRequired: false,
      };
    }

    // Data processing errors
    if (message.includes('json') || message.includes('parse') || message.includes('invalid')) {
      return {
        category: ErrorCategory.DATA_PROCESSING,
        severity: 'medium',
        recoverable: true,
        autoRetry: false,
        fallbackAvailable: true,
        userActionRequired: false,
      };
    }

    // UI component errors
    if (stack.includes('react') || message.includes('component') || message.includes('render')) {
      return {
        category: ErrorCategory.UI_COMPONENT,
        severity: 'low',
        recoverable: true,
        autoRetry: true,
        fallbackAvailable: true,
        userActionRequired: false,
      };
    }

    // Default classification
    return {
      category: ErrorCategory.UNKNOWN,
      severity: 'medium',
      recoverable: true,
      autoRetry: false,
      fallbackAvailable: true,
      userActionRequired: true,
    };
  }

  // Automatic recovery system matching Qt implementation
  private attemptAutoRecovery(classification: ErrorClassification): void {
    this.setState({ isRecovering: true, recoveryStrategy: 'retry' });

    const delay = this.retryDelay * Math.pow(2, this.state.retryCount); // Exponential backoff
    
    AppLogger.info('ERROR_RECOVERY', `Attempting auto-recovery in ${delay}ms`, {
      errorId: this.state.errorId,
      retryCount: this.state.retryCount + 1,
      classification,
    });

    const timeout = setTimeout(() => {
      this.handleRetry();
    }, delay);

    this.retryTimeouts.push(timeout);
  }

  // Report error to Python backend for server-side logging
  private async reportErrorToBackend(error: Error, errorInfo: ErrorInfo, classification: ErrorClassification): Promise<void> {
    try {
      await callPythonMethod('logClientError', {
        errorId: this.state.errorId,
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        classification,
        sessionId: AppLogger.sessionId,
        currentScreen: AppLogger.currentScreen,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
      });
    } catch (backendError) {
      AppLogger.warn('ERROR_REPORTING', 'Failed to report error to backend', {
        originalError: error.message,
        backendError: (backendError as Error).message,
      });
    }
  }

  private handleRetry = async () => {
    const newRetryCount = this.state.retryCount + 1;
    
    AppLogger.action('ERROR_RECOVERY', `Manual retry attempt ${newRetryCount}`, {
      errorId: this.state.errorId,
      retryCount: newRetryCount,
    });

    sessionTracker.logAction('ERROR_RETRY', {
      errorId: this.state.errorId,
      retryCount: newRetryCount,
      isManual: true,
    });

    // Check Python bridge health before retry
    if (this.state.error?.message.includes('bridge') || this.state.error?.message.includes('python')) {
      this.setState({ isRecovering: true, recoveryStrategy: 'retry' });
      
      try {
        const isHealthy = await checkBridgeHealth();
        if (!isHealthy) {
          AppLogger.warn('ERROR_RECOVERY', 'Python bridge unhealthy, attempting reconnection');
          // Give bridge time to reconnect
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
      } catch (healthError) {
        AppLogger.error('ERROR_RECOVERY', 'Bridge health check failed', healthError);
      }
    }

    this.setState({ 
      hasError: false, 
      error: null, 
      errorInfo: null,
      retryCount: newRetryCount,
      isRecovering: false,
      recoveryStrategy: null,
    });
  };

  private handleGoHome = () => {
    AppLogger.action('ERROR_RECOVERY', 'User navigating to home', {
      errorId: this.state.errorId,
      retryCount: this.state.retryCount,
    });

    sessionTracker.logAction('ERROR_NAVIGATE_HOME', {
      errorId: this.state.errorId,
      fromScreen: AppLogger.currentScreen,
    });

    // Use Next.js router if available, otherwise fallback to window.location
    if (typeof window !== 'undefined') {
      window.location.href = '/';
    }
  };

  private handleReload = () => {
    AppLogger.action('ERROR_RECOVERY', 'User reloading page', {
      errorId: this.state.errorId,
      retryCount: this.state.retryCount,
    });

    sessionTracker.logAction('ERROR_RELOAD_PAGE', {
      errorId: this.state.errorId,
    });

    window.location.reload();
  };

  private handleReportIssue = () => {
    const errorReport = {
      errorId: this.state.errorId,
      message: this.state.error?.message,
      stack: this.state.error?.stack,
      componentStack: this.state.errorInfo?.componentStack,
      sessionId: AppLogger.sessionId,
      currentScreen: AppLogger.currentScreen,
      userAgent: navigator.userAgent,
      url: window.location.href,
      timestamp: new Date().toISOString(),
    };

    // Copy error report to clipboard
    navigator.clipboard.writeText(JSON.stringify(errorReport, null, 2)).then(() => {
      AppLogger.action('ERROR_RECOVERY', 'Error report copied to clipboard', {
        errorId: this.state.errorId,
      });
      alert('Error report copied to clipboard. Please paste it when reporting the issue.');
    }).catch(() => {
      // Fallback: show error report in alert
      alert(`Error Report:\n\n${JSON.stringify(errorReport, null, 2)}`);
    });
  };

  // Cleanup timeouts on unmount
  public componentWillUnmount() {
    this.retryTimeouts.forEach(timeout => clearTimeout(timeout));
  }

  private getErrorIcon(category: ErrorCategory): string {
    switch (category) {
      case ErrorCategory.NETWORK: return '🌐';
      case ErrorCategory.PYTHON_BRIDGE: return '🔗';
      case ErrorCategory.UI_COMPONENT: return '🎨';
      case ErrorCategory.DATA_PROCESSING: return '📊';
      case ErrorCategory.PERMISSION: return '🔒';
      case ErrorCategory.TIMEOUT: return '⏱️';
      default: return '❌';
    }
  }

  private getRecoveryMessage(classification: ErrorClassification): string {
    switch (classification.category) {
      case ErrorCategory.NETWORK:
        return 'This appears to be a network connectivity issue. Please check your internet connection and try again.';
      case ErrorCategory.PYTHON_BRIDGE:
        return 'There was an issue communicating with the Python backend. The system will attempt to reconnect automatically.';
      case ErrorCategory.UI_COMPONENT:
        return 'A user interface component encountered an error. Retrying should resolve this issue.';
      case ErrorCategory.DATA_PROCESSING:
        return 'There was an issue processing data. Please try again or refresh the page.';
      case ErrorCategory.PERMISSION:
        return 'Access was denied. Please check your permissions or contact support.';
      case ErrorCategory.TIMEOUT:
        return 'The operation timed out. Please try again or check your connection.';
      default:
        return 'An unexpected error occurred. Please try the suggested recovery options below.';
    }
  }

  public render() {
    if (this.state.hasError && this.state.error) {
      // Use custom fallback component if provided
      if (this.props.fallbackComponent) {
        const FallbackComponent = this.props.fallbackComponent;
        return (
          <FallbackComponent
            error={this.state.error}
            errorInfo={this.state.errorInfo!}
            retry={this.handleRetry}
            goHome={this.handleGoHome}
            errorId={this.state.errorId!}
            retryCount={this.state.retryCount}
          />
        );
      }

      const classification = this.classifyError(this.state.error);
      const errorIcon = this.getErrorIcon(classification.category);
      const recoveryMessage = this.getRecoveryMessage(classification);
      const canRetry = this.state.retryCount < this.maxRetries;

      return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100 p-4">
          <div className="bg-white dark:bg-gray-800 shadow-lg rounded-lg p-8 max-w-2xl w-full">
            {/* Error Header */}
            <div className="text-center mb-6">
              <div className="text-6xl mb-4">{errorIcon}</div>
              <h2 className="text-3xl font-bold text-red-600 dark:text-red-400 mb-2">
                Something went wrong
              </h2>
              <p className="text-lg text-gray-600 dark:text-gray-300">
                {recoveryMessage}
              </p>
            </div>

            {/* Error Classification Info */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-gray-700 dark:text-gray-300">Error Details:</span>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  classification.severity === 'critical' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                  classification.severity === 'high' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200' :
                  classification.severity === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                  'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                }`}>
                  {classification.severity.toUpperCase()}
                </span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                <strong>Category:</strong> {classification.category.replace('_', ' ')}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                <strong>Error ID:</strong> {this.state.errorId}
              </p>
              {this.state.retryCount > 0 && (
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>Retry Attempts:</strong> {this.state.retryCount}/{this.maxRetries}
                </p>
              )}
            </div>

            {/* Recovery Status */}
            {this.state.isRecovering && (
              <div className="bg-blue-50 dark:bg-blue-900 border border-blue-200 dark:border-blue-700 rounded-lg p-4 mb-6">
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-3"></div>
                  <span className="text-blue-800 dark:text-blue-200">
                    Attempting recovery using {this.state.recoveryStrategy} strategy...
                  </span>
                </div>
              </div>
            )}

            {/* Error Message (Collapsible) */}
            <details className="mb-6">
              <summary className="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100">
                Show Technical Details
              </summary>
              <div className="mt-3 bg-gray-100 dark:bg-gray-700 p-4 rounded-md text-left overflow-auto max-h-60">
                <p className="font-semibold text-red-500 mb-2">Error Message:</p>
                <p className="font-mono text-sm break-words mb-4">{this.state.error.message}</p>
                {this.state.errorInfo && (
                  <>
                    <p className="font-semibold text-red-500 mb-2">Component Stack:</p>
                    <pre className="font-mono text-xs whitespace-pre-wrap text-gray-600 dark:text-gray-400">
                      {this.state.errorInfo.componentStack}
                    </pre>
                  </>
                )}
              </div>
            </details>

            {/* Recovery Actions */}
            <div className="space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <button
                  onClick={this.handleRetry}
                  disabled={!canRetry || this.state.isRecovering}
                  className={`px-6 py-3 rounded-md font-medium transition-colors ${
                    canRetry && !this.state.isRecovering
                      ? 'bg-blue-600 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500'
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed dark:bg-gray-600 dark:text-gray-400'
                  }`}
                >
                  {this.state.isRecovering ? 'Retrying...' : `Try Again ${canRetry ? `(${this.maxRetries - this.state.retryCount} left)` : '(Max attempts reached)'}`}
                </button>

                <button
                  onClick={this.handleGoHome}
                  className="px-6 py-3 bg-gray-300 text-gray-800 rounded-md hover:bg-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-500 transition-colors dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
                >
                  🏠 Go to Home
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <button
                  onClick={this.handleReload}
                  className="px-6 py-3 bg-orange-500 text-white rounded-md hover:bg-orange-600 focus:outline-none focus:ring-2 focus:ring-orange-500 transition-colors"
                >
                  🔄 Reload Page
                </button>

                <button
                  onClick={this.handleReportIssue}
                  className="px-6 py-3 bg-purple-500 text-white rounded-md hover:bg-purple-600 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-colors"
                >
                  📋 Copy Error Report
                </button>
              </div>
            </div>

            {/* Help Text */}
            <div className="mt-6 text-center">
              <p className="text-sm text-gray-500 dark:text-gray-400">
                If the problem persists, please copy the error report and contact support.
              </p>
              {classification.userActionRequired && (
                <p className="text-sm text-orange-600 dark:text-orange-400 mt-2">
                  ⚠️ This error may require manual intervention to resolve.
                </p>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
