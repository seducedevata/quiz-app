import { callPythonMethod } from './pythonBridge';

// Global variable to hold the current screen name
let currentScreen = 'unknown';

// Function to set the current screen, called by navigation handlers
export const setCurrentScreenName = (screenName: string) => {
    currentScreen = screenName;
};

// 📊 COMPREHENSIVE LOGGING SYSTEM
export const AppLogger = {
    // Log levels with emojis for better visibility
    LEVELS: {
        DEBUG: { name: 'DEBUG', emoji: '🔍', color: '#6c757d' },
        INFO: { name: 'INFO', emoji: '💡', color: '#17a2b8' },
        WARN: { name: 'WARN', emoji: '⚠️', color: '#ffc107' },
        ERROR: { name: 'ERROR', emoji: '❌', color: '#dc3545' },
        SUCCESS: { name: 'SUCCESS', emoji: '✅', color: '#28a745' },
        ACTION: { name: 'ACTION', emoji: '🎯', color: '#007bff' },
        PERFORMANCE: { name: 'PERF', emoji: '⏱️', color: '#6f42c1' },
        USER: { name: 'USER', emoji: '👤', color: '#fd7e14' }
    } as const,

    // Session tracking
    sessionId: 'session_' + Date.now(),
    startTime: new Date(),
    actionCount: 0,

    // Core logging function
    log(level: keyof typeof AppLogger.LEVELS, category: string, message: string, data: any = null) {
        const timestamp = new Date().toISOString();
        const levelInfo = this.LEVELS[level] || this.LEVELS.INFO;
        const sessionTime = ((Date.now() - this.startTime.getTime()) / 1000).toFixed(2);

        // Increment action counter for user actions
        if (level === 'ACTION' || level === 'USER') {
            this.actionCount++;
        }

        // Format the log message
        const logPrefix = `[${timestamp}] ${levelInfo.emoji} ${levelInfo.name}`;
        const logCategory = category ? `[${category}]` : '';
        const sessionInfo = `[Session:${this.sessionId.slice(-6)} | +${sessionTime}s | Action:${this.actionCount}]`;

        const fullMessage = `${logPrefix} ${logCategory} ${sessionInfo} ${message}`;

        // Log to console with appropriate method
        switch (level) {
            case 'ERROR':
                console.error(fullMessage, data || '');
                break;
            case 'WARN':
                console.warn(fullMessage, data || '');
                break;
            default:
                console.log(fullMessage, data || '');
        }

        // Store critical logs for potential debugging
        if (level === 'ERROR' || level === 'ACTION' || level === 'USER') {
            this.storeCriticalLog(level, category, message, data, timestamp);
        }

        // Send to Python backend for server-side logging
        // Assuming pythonBridge.logClientEvent exists and is correctly exposed
        callPythonMethod('logClientEvent', {
            level: level,
            category: category,
            message: message,
            data: data,
            timestamp: timestamp,
            sessionId: this.sessionId,
            sessionTime: sessionTime,
            actionCount: this.actionCount
        }).catch(e => {
            console.warn('📡 Failed to send log to backend:', e.message);
        });
    },

    // Store critical logs in localStorage for debugging
    storeCriticalLog(level: keyof typeof AppLogger.LEVELS, category: string, message: string, data: any, timestamp: string) {
        try {
            const criticalLogs = JSON.parse(localStorage.getItem('criticalLogs') || '[]');
            criticalLogs.push({
                level, category, message, data, timestamp,
                sessionId: this.sessionId,
                url: window.location.href,
                userAgent: navigator.userAgent.slice(0, 100)
            });

            // Keep only last 100 critical logs
            if (criticalLogs.length > 100) {
                criticalLogs.splice(0, criticalLogs.length - 100);
            }

            localStorage.setItem('criticalLogs', JSON.stringify(criticalLogs));
        } catch (e) {
            console.error('Failed to store critical log:', e);
        }
    },

    // Convenience methods for different log types
    debug(category: string, message: string, data?: any) { this.log('DEBUG', category, message, data); },
    info(category: string, message: string, data?: any) { this.log('INFO', category, message, data); },
    warn(category: string, message: string, data?: any) { this.log('WARN', category, message, data); },
    warning(category: string, message: string, data?: any) { this.log('WARN', category, message, data); }, // Alias for warn
    error(category: string, message: string, data?: any) { this.log('ERROR', category, message, data); },
    success(category: string, message: string, data?: any) { this.log('SUCCESS', category, message, data); },
    action(category: string, message: string, data?: any) { this.log('ACTION', category, message, data); },
    performance(category: string, message: string, data?: any) { this.log('PERFORMANCE', category, message, data); },
    user(category: string, message: string, data?: any) { this.log('USER', category, message, data); },
    system(category: string, message: string, data?: any) { this.log('INFO', category, message, data); }, // System alias for info

    // Log user interactions with UI elements
    userAction(elementType: string, action: string, details: any = {}) {
        this.user('UI_INTERACTION', `${elementType.toUpperCase()}_${action.toUpperCase()}`, {
            elementType,
            action,
            ...details,
            timestamp: Date.now(),
            screenTime: ((Date.now() - this.startTime.getTime()) / 1000).toFixed(2)
        });
    },

    // Track navigation events
    navigation(from: string, to: string, method: string = 'click') {
        this.action('NAVIGATION', `Screen changed: ${from} → ${to}`, {
            fromScreen: from,
            toScreen: to,
            method: method,
            timestamp: Date.now()
        });
    },

    // Track navigation with comprehensive details
    trackNavigation(from: string, to: string, method: string = 'click') {
        this.action('NAVIGATION', `Screen Navigation: ${from} → ${to}`, {
            fromScreen: from,
            toScreen: to,
            method: method,
            timestamp: Date.now(),
            sessionTime: ((Date.now() - this.startTime.getTime()) / 1000).toFixed(2)
        });
    },

    // Track user actions with context
    trackUserAction(action: string, context: any = {}) {
        this.user('USER_ACTION', action, {
            ...context,
            currentScreen,
            timestamp: Date.now(),
            sessionTime: ((Date.now() - this.startTime.getTime()) / 1000).toFixed(2)
        });
    },

    // Log performance metrics
    performanceMetric(operation: string, duration: number, success: boolean = true, metadata: any = {}) {
        this.performance('METRICS', `${operation}: ${duration}ms ${success ? '✅' : '❌'}`, {
            operation,
            duration,
            success,
            ...metadata
        });
    },

    // Get session summary
    getSessionSummary() {
        const duration = ((Date.now() - this.startTime.getTime()) / 1000).toFixed(2);
        return {
            sessionId: this.sessionId,
            startTime: this.startTime.toISOString(),
            duration: `${duration}s`,
            actionCount: this.actionCount,
            currentScreen: currentScreen
        };
    },

    // Display comprehensive logging summary (callable from console)
    displayLogSummary() {
        const summary = this.getSessionSummary();
        const criticalLogs = JSON.parse(localStorage.getItem('criticalLogs') || '[]');

        console.group('📊 Knowledge App Logging Summary');
        console.log('Session ID:', summary.sessionId);
        console.log('Session Start:', summary.startTime);
        console.log('Session Duration:', summary.duration);
        console.log('Total Actions:', summary.actionCount);
        console.log('Current Screen:', summary.currentScreen);
        console.log('Critical Logs Count:', criticalLogs.length);

        if (criticalLogs.length > 0) {
            console.group('Recent Critical Logs (Last 10):');
            criticalLogs.slice(-10).forEach((log: any, index: number) => {
                console.log(`${index + 1}. [${log.level}] ${log.category}: ${log.message}`, log.data);
            });
            console.groupEnd();
        }

        // Note: Python bridge status will be handled by pythonBridge.ts's checkBridgeHealth

        console.groupEnd();

        return summary;
    },

    // Clear critical logs (useful for debugging)
    clearCriticalLogs() {
        localStorage.removeItem('criticalLogs');
        this.info('SYSTEM', 'Critical logs cleared');
    },

    // Track quiz actions with enhanced logging
    trackQuizAction(action: string, details: any = {}) {
        this.action('QUIZ_ACTION', action, {
            ...details,
            currentScreen,
            // selectedAnswer: selectedAnswer, // This will be managed by React state
            // isReviewMode: isReviewMode, // This will be managed by React state
            timestamp: Date.now()
        });
    },

    // Track errors with context
    trackError(category: string, error: any, context: any = {}) {
        this.error(category, `Error: ${error.message || error}`, {
            error: error.message || error,
            stack: error.stack,
            ...context,
            currentScreen,
            timestamp: Date.now()
        });
    }
};

// Initialize logging
AppLogger.info('SYSTEM', '🚀 Knowledge App Logging System Initialized', AppLogger.getSessionSummary());

// 🚨 GLOBAL ERROR HANDLERS FOR COMPREHENSIVE LOGGING
if (typeof window !== 'undefined') {
    window.addEventListener('error', function (event) {
        // Filter out common non-critical errors
        const message = (event.error?.message || event.message || '') as string;
        const filename = (event.filename || '') as string;
        
        // Skip logging for known non-critical errors
        if (message.includes('ResizeObserver loop limit exceeded') ||
            message.includes('Non-Error promise rejection captured') ||
            filename.includes('extensions/') ||
            filename.includes('chrome-extension://')) {
            return;
        }
        
        AppLogger.error('GLOBAL_ERROR', 'Unhandled JavaScript error', {
            message: message,
            filename: filename,
            lineno: event.lineno,
            colno: event.colno,
            stack: (event.error as Error)?.stack,
            currentScreen: currentScreen,
            userAgent: navigator.userAgent.slice(0, 100)
        });
        
        // Try to recover from common errors
        if (message.includes('Cannot read property') || message.includes('Cannot read properties')) {
            console.log('🔧 Attempting to recover from property access error');
            // Add any recovery logic here
        }
    });
    
    window.addEventListener('unhandledrejection', function (event) {
        const reason = (event.reason?.message || event.reason || '') as string;
        
        // Skip logging for known non-critical promise rejections
        if (typeof reason === 'string' && (
            reason.includes('Load failed') ||
            reason.includes('NetworkError') ||
            reason.includes('AbortError'))) {
            return;
        }
        
        AppLogger.error('GLOBAL_ERROR', 'Unhandled promise rejection', {
            reason: reason,
            stack: (event.reason as Error)?.stack,
            currentScreen: currentScreen,
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
            currentScreen: currentScreen
        });
    });
    
    // Track critical DOM events
    document.addEventListener('DOMContentLoaded', function () {
        AppLogger.info('SYSTEM', 'DOM content loaded');
    });
    
    window.addEventListener('load', function () {
        AppLogger.performance('SYSTEM', 'Page fully loaded', {
            loadTime: performance.now(),
            domContentLoadedTime: performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart,
            pageLoadTime: performance.timing.loadEventEnd - performance.timing.navigationStart
        });
    });
}

// 🛠️ DEVELOPER CONVENIENCE FUNCTIONS (accessible from browser console)
// These will be exposed via a global object if needed, or integrated into dev tools
// For now, they are part of the AppLogger export for easy access.
export const debugApp = {
    logs: () => AppLogger.displayLogSummary(),
    session: () => AppLogger.getSessionSummary(),
    critical: () => JSON.parse(localStorage.getItem('criticalLogs') || '[]'),
    clearLogs: () => AppLogger.clearCriticalLogs(),
    testLog: () => {
        AppLogger.info('TEST', 'Logging system test', { testData: 'Hello World!' });
        console.log('✅ Test log sent. Check console output above.');
    },
    // Bridge status will be handled by pythonBridge.ts
    // testButtons, fixButtons, testNavigation, clickNav will be handled by React components
    help: () => {
        console.log(`
🛠️ Knowledge App Debug Commands:
- debugApp.logs()           - Show comprehensive logging summary
- debugApp.session()        - Get current session information  
- debugApp.critical()       - Get critical logs array
- debugApp.clearLogs()      - Clear stored critical logs
- debugApp.testLog()        - Test logging system
- debugApp.help()           - Show this help
        `);
    }
};
