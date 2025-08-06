/**
 * Quiz Generation Best Practices Implementation
 * 
 * This file implements additional best practices based on the comprehensive analysis:
 * 1. Centralized State Management
 * 2. Input Validation System
 * 3. Structured Logging
 * 4. Performance Monitoring
 * 5. Accessibility Enhancements
 */

// 1. CENTRALIZED STATE MANAGEMENT SYSTEM
class QuizStateManager {
    constructor() {
        this.state = {
            currentMode: 'auto',
            generationActive: false,
            questionCache: new Map(),
            streamingActive: false,
            lastError: null,
            performance: {
                memoryUsage: 0,
                generationTime: 0,
                renderTime: 0
            }
        };
        this.listeners = new Map();
        this.history = [];
    }

    // State management with history tracking
    setState(newState, action = 'UPDATE') {
        const previousState = { ...this.state };
        this.state = { ...this.state, ...newState };
        
        // Add to history for debugging
        this.history.push({
            timestamp: Date.now(),
            action,
            previousState,
            newState: { ...this.state }
        });
        
        // Keep history manageable
        if (this.history.length > 100) {
            this.history = this.history.slice(-50);
        }
        
        // Notify listeners
        this.notifyListeners(action, previousState, this.state);
        
        Logger.info('State updated', { action, newState });
    }

    getState() {
        return { ...this.state };
    }

    // Subscribe to state changes
    subscribe(key, callback) {
        if (!this.listeners.has(key)) {
            this.listeners.set(key, []);
        }
        this.listeners.get(key).push(callback);
    }

    // Notify all listeners
    notifyListeners(action, previousState, currentState) {
        this.listeners.forEach((callbacks, key) => {
            callbacks.forEach(callback => {
                try {
                    callback(action, previousState, currentState);
                } catch (error) {
                    Logger.error('Listener error', { key, error: error.message });
                }
            });
        });
    }

    // Complete state reset with validation
    reset(reason = 'USER_REQUEST') {
        const previousState = { ...this.state };
        
        this.state = {
            currentMode: 'auto',
            generationActive: false,
            questionCache: new Map(),
            streamingActive: false,
            lastError: null,
            performance: {
                memoryUsage: 0,
                generationTime: 0,
                renderTime: 0
            }
        };
        
        // Clear caches
        this.clearAllCaches();
        
        // Force garbage collection
        this.triggerGarbageCollection();
        
        Logger.info('State reset completed', { reason, previousState });
        this.notifyListeners('RESET', previousState, this.state);
    }

    clearAllCaches() {
        // Clear all cached data
        if (window.questionGenerationCache) {
            window.questionGenerationCache.clear();
        }
        if (window.generatedQuestionHashes) {
            window.generatedQuestionHashes.clear();
        }
        if (window.latexCache) {
            window.latexCache.clear();
        }
    }

    triggerGarbageCollection() {
        if (window.gc && typeof window.gc === 'function') {
            try {
                window.gc();
                Logger.info('Garbage collection triggered');
            } catch (e) {
                Logger.debug('Garbage collection not available');
            }
        }
    }
}

// 2. INPUT VALIDATION SYSTEM
class InputValidator {
    static rules = {
        apiKey: {
            openai: /^sk-[a-zA-Z0-9]{48,}$/,
            anthropic: /^sk-ant-[a-zA-Z0-9-]{95,}$/,
            groq: /^gsk_[a-zA-Z0-9]{52}$/,
            gemini: /^[a-zA-Z0-9-_]{39}$/
        },
        topic: {
            minLength: 2,
            maxLength: 200,
            pattern: /^[a-zA-Z0-9\s\-.,()]+$/
        },
        difficulty: {
            allowed: ['easy', 'medium', 'hard', 'expert']
        },
        mode: {
            allowed: ['online', 'offline', 'auto']
        }
    };

    static validate(type, value, provider = null) {
        try {
            switch (type) {
                case 'apiKey':
                    return this.validateApiKey(value, provider);
                case 'topic':
                    return this.validateTopic(value);
                case 'difficulty':
                    return this.validateDifficulty(value);
                case 'mode':
                    return this.validateMode(value);
                default:
                    return { valid: false, error: 'Unknown validation type' };
            }
        } catch (error) {
            Logger.error('Validation error', { type, error: error.message });
            return { valid: false, error: 'Validation failed' };
        }
    }

    static validateApiKey(key, provider) {
        if (!key || typeof key !== 'string') {
            return { valid: false, error: 'API key is required' };
        }

        if (!provider || !this.rules.apiKey[provider]) {
            return { valid: false, error: 'Invalid provider' };
        }

        const pattern = this.rules.apiKey[provider];
        if (!pattern.test(key)) {
            return { 
                valid: false, 
                error: `Invalid ${provider} API key format` 
            };
        }

        return { valid: true };
    }

    static validateTopic(topic) {
        if (!topic || typeof topic !== 'string') {
            return { valid: false, error: 'Topic is required' };
        }

        const trimmed = topic.trim();
        
        if (trimmed.length < this.rules.topic.minLength) {
            return { 
                valid: false, 
                error: `Topic must be at least ${this.rules.topic.minLength} characters` 
            };
        }

        if (trimmed.length > this.rules.topic.maxLength) {
            return { 
                valid: false, 
                error: `Topic must be less than ${this.rules.topic.maxLength} characters` 
            };
        }

        if (!this.rules.topic.pattern.test(trimmed)) {
            return { 
                valid: false, 
                error: 'Topic contains invalid characters' 
            };
        }

        return { valid: true, sanitized: trimmed };
    }

    static validateDifficulty(difficulty) {
        if (!this.rules.difficulty.allowed.includes(difficulty)) {
            return { 
                valid: false, 
                error: `Difficulty must be one of: ${this.rules.difficulty.allowed.join(', ')}` 
            };
        }
        return { valid: true };
    }

    static validateMode(mode) {
        if (!this.rules.mode.allowed.includes(mode)) {
            return { 
                valid: false, 
                error: `Mode must be one of: ${this.rules.mode.allowed.join(', ')}` 
            };
        }
        return { valid: true };
    }
}

// 3. STRUCTURED LOGGING SYSTEM
class Logger {
    static levels = {
        DEBUG: 0,
        INFO: 1,
        WARN: 2,
        ERROR: 3
    };

    static currentLevel = this.levels.INFO;
    static logs = [];
    static maxLogs = 1000;

    static setLevel(level) {
        if (typeof level === 'string') {
            level = this.levels[level.toUpperCase()];
        }
        this.currentLevel = level;
    }

    static log(level, message, data = {}) {
        if (level < this.currentLevel) return;

        const logEntry = {
            timestamp: new Date().toISOString(),
            level: Object.keys(this.levels)[level],
            message,
            data,
            stack: new Error().stack
        };

        // Add to internal log storage
        this.logs.push(logEntry);
        if (this.logs.length > this.maxLogs) {
            this.logs = this.logs.slice(-this.maxLogs / 2);
        }

        // Console output with appropriate method
        const consoleMethod = level >= this.levels.ERROR ? 'error' :
                            level >= this.levels.WARN ? 'warn' :
                            level >= this.levels.INFO ? 'info' : 'debug';

        const emoji = level >= this.levels.ERROR ? '❌' :
                     level >= this.levels.WARN ? '⚠️' :
                     level >= this.levels.INFO ? 'ℹ️' : '🔍';

        console[consoleMethod](`${emoji} [${logEntry.level}] ${message}`, data);

        // Send to external logging service if configured
        this.sendToExternalLogger(logEntry);
    }

    static debug(message, data) {
        this.log(this.levels.DEBUG, message, data);
    }

    static info(message, data) {
        this.log(this.levels.INFO, message, data);
    }

    static warn(message, data) {
        this.log(this.levels.WARN, message, data);
    }

    static error(message, data) {
        this.log(this.levels.ERROR, message, data);
    }

    static getLogs(level = null) {
        if (level === null) return [...this.logs];
        
        const levelNum = typeof level === 'string' ? 
                        this.levels[level.toUpperCase()] : level;
        
        return this.logs.filter(log => 
            this.levels[log.level] >= levelNum
        );
    }

    static sendToExternalLogger(logEntry) {
        // Placeholder for external logging service integration
        // Could integrate with Sentry, LogRocket, etc.
        if (window.externalLogger && typeof window.externalLogger.send === 'function') {
            try {
                window.externalLogger.send(logEntry);
            } catch (error) {
                console.warn('Failed to send log to external service:', error);
            }
        }
    }
}

// 4. PERFORMANCE MONITORING SYSTEM
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            memory: [],
            timing: new Map(),
            errors: [],
            userActions: []
        };
        this.startTime = performance.now();
        this.setupMonitoring();
    }

    setupMonitoring() {
        // Memory monitoring
        if (window.performance && window.performance.memory) {
            setInterval(() => {
                this.recordMemoryUsage();
            }, 5000); // Every 5 seconds
        }

        // Error monitoring
        window.addEventListener('error', (event) => {
            this.recordError(event.error, 'JavaScript Error');
        });

        window.addEventListener('unhandledrejection', (event) => {
            this.recordError(event.reason, 'Unhandled Promise Rejection');
        });
    }

    recordMemoryUsage() {
        if (!window.performance.memory) return;

        const memory = {
            timestamp: Date.now(),
            used: Math.round(window.performance.memory.usedJSHeapSize / 1024 / 1024),
            total: Math.round(window.performance.memory.totalJSHeapSize / 1024 / 1024),
            limit: Math.round(window.performance.memory.jsHeapSizeLimit / 1024 / 1024)
        };

        this.metrics.memory.push(memory);
        
        // Keep only last 100 entries
        if (this.metrics.memory.length > 100) {
            this.metrics.memory = this.metrics.memory.slice(-50);
        }

        // Alert on high memory usage
        if (memory.used > memory.limit * 0.8) {
            Logger.warn('High memory usage detected', memory);
        }
    }

    startTiming(operation) {
        this.metrics.timing.set(operation, performance.now());
    }

    endTiming(operation) {
        const startTime = this.metrics.timing.get(operation);
        if (startTime) {
            const duration = performance.now() - startTime;
            this.metrics.timing.delete(operation);
            
            Logger.info(`Performance: ${operation}`, { duration: `${duration.toFixed(2)}ms` });
            
            // Alert on slow operations
            if (duration > 5000) { // 5 seconds
                Logger.warn(`Slow operation detected: ${operation}`, { duration });
            }
            
            return duration;
        }
        return null;
    }

    recordError(error, type) {
        const errorRecord = {
            timestamp: Date.now(),
            type,
            message: error.message || error,
            stack: error.stack,
            url: window.location.href
        };

        this.metrics.errors.push(errorRecord);
        
        // Keep only last 50 errors
        if (this.metrics.errors.length > 50) {
            this.metrics.errors = this.metrics.errors.slice(-25);
        }

        Logger.error(`${type}: ${errorRecord.message}`, errorRecord);
    }

    recordUserAction(action, data = {}) {
        const actionRecord = {
            timestamp: Date.now(),
            action,
            data,
            url: window.location.href
        };

        this.metrics.userActions.push(actionRecord);
        
        // Keep only last 100 actions
        if (this.metrics.userActions.length > 100) {
            this.metrics.userActions = this.metrics.userActions.slice(-50);
        }

        Logger.debug(`User action: ${action}`, data);
    }

    getMetrics() {
        return {
            ...this.metrics,
            uptime: performance.now() - this.startTime
        };
    }

    generateReport() {
        const metrics = this.getMetrics();
        const report = {
            timestamp: new Date().toISOString(),
            uptime: `${(metrics.uptime / 1000 / 60).toFixed(2)} minutes`,
            memory: {
                current: metrics.memory[metrics.memory.length - 1],
                peak: metrics.memory.reduce((max, m) => m.used > max.used ? m : max, { used: 0 }),
                average: metrics.memory.reduce((sum, m) => sum + m.used, 0) / metrics.memory.length
            },
            errors: {
                total: metrics.errors.length,
                recent: metrics.errors.slice(-5)
            },
            userActions: {
                total: metrics.userActions.length,
                recent: metrics.userActions.slice(-10)
            }
        };

        Logger.info('Performance report generated', report);
        return report;
    }
}

// 5. ACCESSIBILITY ENHANCEMENTS
class AccessibilityManager {
    static announceToScreenReader(message, priority = 'polite') {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', priority);
        announcement.setAttribute('aria-atomic', 'true');
        announcement.className = 'sr-only';
        announcement.textContent = message;
        
        document.body.appendChild(announcement);
        
        // Remove after announcement
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }

    static enhanceFormAccessibility() {
        // Add proper labels and descriptions
        const inputs = document.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            if (!input.getAttribute('aria-label') && !input.getAttribute('aria-labelledby')) {
                const label = document.querySelector(`label[for="${input.id}"]`);
                if (label) {
                    input.setAttribute('aria-labelledby', label.id || `label-${input.id}`);
                }
            }
        });
    }

    static addKeyboardNavigation() {
        // Ensure all interactive elements are keyboard accessible
        const interactiveElements = document.querySelectorAll('button, [role="button"], .clickable');
        interactiveElements.forEach(element => {
            if (!element.hasAttribute('tabindex')) {
                element.setAttribute('tabindex', '0');
            }
            
            element.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    element.click();
                }
            });
        });
    }
}

// Initialize global instances
window.QuizStateManager = new QuizStateManager();
window.PerformanceMonitor = new PerformanceMonitor();

// Export for use in main application
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        QuizStateManager,
        InputValidator,
        Logger,
        PerformanceMonitor,
        AccessibilityManager
    };
}
