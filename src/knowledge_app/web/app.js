// Global variables - declared first to avoid temporal dead zone issues
let currentScreen = 'home';
let pythonBridge = null;
let selectedAnswer = -1;
let quizTimer = null;
let timeRemaining = 30;
let uploadedFiles = [];
let currentQuestionState = null;

// 📊 COMPREHENSIVE LOGGING SYSTEM
const AppLogger = {
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
    },

    // Session tracking
    sessionId: 'session_' + Date.now(),
    startTime: new Date(),
    actionCount: 0,

    // Core logging function
    log(level, category, message, data = null) {
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
        if (pythonBridge && pythonBridge.logClientEvent) {
            try {
                pythonBridge.logClientEvent(JSON.stringify({
                    level: level,
                    category: category,
                    message: message,
                    data: data,
                    timestamp: timestamp,
                    sessionId: this.sessionId,
                    sessionTime: sessionTime,
                    actionCount: this.actionCount
                }));
            } catch (e) {
                console.warn('📡 Failed to send log to backend:', e.message);
            }
        }
    },

    // Store critical logs in localStorage for debugging
    storeCriticalLog(level, category, message, data, timestamp) {
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
    debug(category, message, data) { this.log('DEBUG', category, message, data); },
    info(category, message, data) { this.log('INFO', category, message, data); },
    warn(category, message, data) { this.log('WARN', category, message, data); },
    warning(category, message, data) { this.log('WARN', category, message, data); }, // Alias for warn
    error(category, message, data) { this.log('ERROR', category, message, data); },
    success(category, message, data) { this.log('SUCCESS', category, message, data); },
    action(category, message, data) { this.log('ACTION', category, message, data); },
    performance(category, message, data) { this.log('PERFORMANCE', category, message, data); },
    user(category, message, data) { this.log('USER', category, message, data); },
    system(category, message, data) { this.log('INFO', category, message, data); }, // System alias for info

    // Log user interactions with UI elements
    userAction(elementType, action, details = {}) {
        this.user('UI_INTERACTION', `${elementType.toUpperCase()}_${action.toUpperCase()}`, {
            elementType,
            action,
            ...details,
            timestamp: Date.now(),
            screenTime: ((Date.now() - this.startTime.getTime()) / 1000).toFixed(2)
        });
    },

    // Log navigation events
    navigation(from, to, method = 'click') {
        this.action('NAVIGATION', `Screen changed: ${from} → ${to}`, {
            fromScreen: from,
            toScreen: to,
            method: method,
            timestamp: Date.now()
        });
    },

    // Track navigation with comprehensive details
    trackNavigation(from, to, method = 'click') {
        this.action('NAVIGATION', `Screen Navigation: ${from} → ${to}`, {
            fromScreen: from,
            toScreen: to,
            method: method,
            timestamp: Date.now(),
            sessionTime: ((Date.now() - this.startTime.getTime()) / 1000).toFixed(2)
        });
    },

    // Track user actions with context
    trackUserAction(action, context = {}) {
        this.user('USER_ACTION', action, {
            ...context,
            currentScreen,
            timestamp: Date.now(),
            sessionTime: ((Date.now() - this.startTime.getTime()) / 1000).toFixed(2)
        });
    },

    // Log performance metrics
    performanceMetric(operation, duration, success = true, metadata = {}) {
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
            criticalLogs.slice(-10).forEach((log, index) => {
                console.log(`${index + 1}. [${log.level}] ${log.category}: ${log.message}`, log.data);
            });
            console.groupEnd();
        }

        console.log('Python Bridge Status:', {
            available: !!pythonBridge,
            methods: pythonBridge ? Object.getOwnPropertyNames(pythonBridge).filter(name => typeof pythonBridge[name] === 'function').slice(0, 10) : []
        });

        console.groupEnd();

        return summary;
    },

    // Clear critical logs (useful for debugging)
    clearCriticalLogs() {
        localStorage.removeItem('criticalLogs');
        this.info('SYSTEM', 'Critical logs cleared');
    },

    // Track quiz actions with enhanced logging
    trackQuizAction(action, details = {}) {
        this.action('QUIZ_ACTION', action, {
            ...details,
            currentScreen: currentScreen,
            selectedAnswer: selectedAnswer,
            isReviewMode: isReviewMode,
            timestamp: Date.now()
        });
    },

    // Track errors with context
    trackError(category, error, context = {}) {
        this.error(category, `Error: ${error.message || error}`, {
            error: error.message || error,
            stack: error.stack,
            ...context,
            currentScreen: currentScreen,
            timestamp: Date.now()
        });
    }
};

// Initialize logging
AppLogger.info('SYSTEM', '🚀 Knowledge App Logging System Initialized', AppLogger.getSessionSummary());

// 🚨 GLOBAL ERROR HANDLERS FOR COMPREHENSIVE LOGGING
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
    
    AppLogger.error('GLOBAL_ERROR', 'Unhandled JavaScript error', {
        message: message,
        filename: filename,
        lineno: event.lineno,
        colno: event.colno,
        stack: event.error?.stack,
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
    const reason = event.reason?.message || event.reason || '';
    
    // Skip logging for known non-critical promise rejections
    if (typeof reason === 'string' && (
        reason.includes('Load failed') ||
        reason.includes('NetworkError') ||
        reason.includes('AbortError'))) {
        return;
    }
    
    AppLogger.error('GLOBAL_ERROR', 'Unhandled promise rejection', {
        reason: reason,
        stack: event.reason?.stack,
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

// 🛠️ DEVELOPER CONVENIENCE FUNCTIONS (accessible from browser console)
window.debugApp = {
    // Show logging summary
    logs: () => AppLogger.displayLogSummary(),

    // Get current session info
    session: () => AppLogger.getSessionSummary(),

    // Get critical logs
    critical: () => JSON.parse(localStorage.getItem('criticalLogs') || '[]'),

    // Clear critical logs
    clearLogs: () => AppLogger.clearCriticalLogs(),

    // Test logging system
    testLog: () => {
        AppLogger.info('TEST', 'Logging system test', { testData: 'Hello World!' });
        console.log('✅ Test log sent. Check console output above.');
    },

    // Get Python bridge status
    bridge: () => ({
        available: !!pythonBridge,
        methods: pythonBridge ? Object.getOwnPropertyNames(pythonBridge).filter(name => typeof pythonBridge[name] === 'function') : [],
        properties: pythonBridge ? Object.getOwnPropertyNames(pythonBridge).filter(name => typeof pythonBridge[name] !== 'function') : []
    }),

    // Test button functionality
    testButtons: () => {
        console.log('🔍 Navigation Button Status Check:');

        // Check navigation buttons (these are the actual ones in the HTML)
        const navButtons = document.querySelectorAll('.nav-item');
        navButtons.forEach((button, index) => {
            const rect = button.getBoundingClientRect();
            const styles = window.getComputedStyle(button);
            const text = button.textContent.trim();
            console.log(`✅ Nav Button ${index + 1} (${text}):`, {
                found: true,
                visible: rect.width > 0 && rect.height > 0,
                pointerEvents: styles.pointerEvents,
                zIndex: styles.zIndex,
                disabled: button.disabled,
                clickable: !button.disabled && styles.pointerEvents !== 'none',
                onclick: button.onclick ? 'Has onclick' : 'No onclick',
                hasEventListener: button._hasEventListener || false
            });
        });

        // Check screens
        console.log('🔍 Screen Status Check:');
        const screens = ['home', 'quiz', 'review', 'train', 'settings'];
        screens.forEach(screenName => {
            const screen = document.getElementById(`${screenName}-screen`);
            if (screen) {
                const isActive = screen.classList.contains('active');
                const isVisible = screen.style.display !== 'none';
                console.log(`✅ ${screenName}-screen:`, {
                    found: true,
                    active: isActive,
                    visible: isVisible,
                    display: screen.style.display
                });
            } else {
                console.log(`❌ ${screenName}-screen: Not found`);
            }
        });

        console.log('🔍 Current State:', {
            currentScreen: currentScreen,
            pythonBridge: !!pythonBridge
        });
    },

    // Force fix button handlers
    fixButtons: () => {
        console.log('🔧 Force-fixing button handlers...');

        // Re-run button setup
        if (typeof setupStartQuizButton === 'function') setupStartQuizButton();
        if (typeof setupQuickQuizButton === 'function') setupQuickQuizButton();
        if (typeof setupNavigationButtons === 'function') setupNavigationButtons();

        console.log('✅ Button handlers re-attached');
    },

    // Test navigation directly
    testNavigation: (screenName = 'quiz') => {
        console.log(`🧪 Testing navigation to ${screenName}...`);
        try {
            showScreen(screenName, null);
            console.log(`✅ Navigation to ${screenName} completed`);
        } catch (error) {
            console.error(`❌ Navigation to ${screenName} failed:`, error);
        }
    },

    // Force click a navigation button
    clickNav: (screenName) => {
        const navButtons = document.querySelectorAll('.nav-item');
        const targetButton = Array.from(navButtons).find(btn =>
            btn.textContent.toLowerCase().includes(screenName.toLowerCase())
        );

        if (targetButton) {
            console.log(`🖱️ Force clicking ${screenName} button...`);
            targetButton.click();
        } else {
            console.error(`❌ Could not find navigation button for ${screenName}`);
        }
    },

    // Quick help
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
- debugApp.help()           - Show this help
        `);
    }
};

// Log that debug tools are available
AppLogger.debug('SYSTEM', 'Debug tools loaded (use debugApp.help() in console)');
let isReviewMode = false;
let statusUpdateInterval = null;

// 🌊 Token streaming variables
let currentStreamSession = null;
let tokenStreamContainer = null;
let tokenStreamStats = {
    tokensReceived: 0,
    startTime: null,
    lastTokenTime: null
};

// 🔧 BUTTON FIX: Global click handler for start quiz button
function handleStartQuizClick(event) {
    try {
        AppLogger.info('BUTTON_CLICK', 'Start Quiz button clicked via inline handler');
        console.log('🚨 BUTTON DEBUG: handleStartQuizClick called', event);
        
        // Prevent any event bubbling issues
        if (event) {
            event.preventDefault();
            event.stopPropagation();
        }
        
        // Test bridge connection first
        if (pythonBridge) {
            console.log('🔗 BRIDGE TEST: Testing bridge connection...');
            try {
                const testResult = pythonBridge.testBridgeConnection('Start Quiz Button Click Test');
                console.log('🔗 BRIDGE TEST Result:', testResult);
                AppLogger.info('BRIDGE_TEST', 'Bridge test successful', { result: testResult });
            } catch (bridgeError) {
                console.error('🔗 BRIDGE TEST Error:', bridgeError);
                AppLogger.error('BRIDGE_TEST', 'Bridge test failed', bridgeError);
            }
            
            // Try to call the debug method
            try {
                const debugResult = pythonBridge.debugButtonClickability();
                console.log('🚨 BUTTON DEBUG Result:', debugResult);
                AppLogger.info('BUTTON_DEBUG', 'Debug method successful', { result: debugResult });
            } catch (debugError) {
                console.error('🚨 BUTTON DEBUG Error:', debugError);
                AppLogger.error('BUTTON_DEBUG', 'Debug method failed', debugError);
            }
            
            // Try to force start quiz with default parameters
            try {
                const forceResult = pythonBridge.forceStartQuiz();
                console.log('🚨 FORCE START Result:', forceResult);
                AppLogger.info('FORCE_START', 'Force start successful', { result: forceResult });
                
                // Show success message
                showStatusDisplay('🎉 Quiz started successfully!', 'success');
                showQuizGame(); // Show the quiz interface
            } catch (forceError) {
                console.error('🚨 FORCE START Error:', forceError);
                AppLogger.error('FORCE_START', 'Force start failed', forceError);
                showStatusDisplay('❌ Failed to start quiz: ' + forceError.message, 'error');
            }
        } else {
            console.error('❌ Python bridge not available');
            AppLogger.error('BRIDGE_ERROR', 'Python bridge not available');
            showStatusDisplay('❌ Python bridge not available. Please check app initialization.', 'error');
        }
        
        // Fallback: Call the original function if it exists
        if (typeof startCustomQuiz === 'function') {
            console.log('🔄 FALLBACK: Calling startCustomQuiz function');
            startCustomQuiz();
        } else {
            console.warn('⚠️ startCustomQuiz function not found');
        }
        
    } catch (error) {
        console.error('❌ handleStartQuizClick error:', error);
        AppLogger.error('BUTTON_ERROR', 'Start quiz click handler failed', error);
        showStatusDisplay('❌ Button click error: ' + error.message, 'error');
    }
}

// Make the function globally available
window.handleStartQuizClick = handleStartQuizClick;

// Define showStatusDisplay function early to avoid reference errors
function showStatusDisplay(message, type = 'info') {
    const statusContainer = document.getElementById('status-display');
    if (statusContainer) {
        statusContainer.innerHTML = `
        <div class="status-message ${type}">
            <div class="status-icon">${getStatusIcon(type)}</div>
            <div class="status-text">${message}</div>
            <div class="status-spinner"></div>
        </div>
    `;
        statusContainer.style.display = 'block';
    } else {
        // Fallback: create status display if it doesn't exist
        const fallbackStatus = document.createElement('div');
        fallbackStatus.id = 'status-display';
        fallbackStatus.style.cssText = `
            position: fixed; top: 20px; right: 20px; z-index: 10000;
            background: #333; color: white; padding: 10px 20px;
            border-radius: 5px; font-family: Arial, sans-serif;
        `;
        fallbackStatus.textContent = message;
        document.body.appendChild(fallbackStatus);
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            if (fallbackStatus.parentNode) {
                fallbackStatus.parentNode.removeChild(fallbackStatus);
            }
        }, 3000);
    }
}

function hideStatusDisplay() {
    const statusContainer = document.getElementById('status-display');
    if (statusContainer) {
        statusContainer.style.display = 'none';
    }
}

function getStatusIcon(type) {
    const icons = {
        'info': '💡',
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'loading': '⏳',
        'gpu': '🎮',
        'api': '🌐',
        'deepseek': '🧠'
    };
    return icons[type] || '💡';
}

// Make status functions globally available
window.showStatusDisplay = showStatusDisplay;
window.hideStatusDisplay = hideStatusDisplay;

// 🧭 Navigation System - Show different screens (removed duplicate)

// Update navigation button states
function updateNavigationButtons(activeButton) {
    // Remove active class from all nav buttons
    const navButtons = document.querySelectorAll('.nav-item');
    navButtons.forEach(button => {
        button.classList.remove('active');
    });

    // Add active class to clicked button
    if (activeButton) {
        activeButton.classList.add('active');
    }
}

// Initialize screen-specific functionality
function initializeScreen(screenName) {
    AppLogger.debug('SCREEN_INIT', `Initializing screen: ${screenName}`);

    switch (screenName) {
        case 'home':
            // Update statistics and refresh home content
            updateStats();
            break;

        case 'quiz':
            // Initialize quiz interface
            resetQuizState();
            break;

        case 'review':
            // Load review history
            if (pythonBridge && pythonBridge.loadReviewHistory) {
                pythonBridge.loadReviewHistory();
            }
            break;

        case 'train':
            // Initialize training interface
            if (typeof refreshAvailableModels === 'function') {
                refreshAvailableModels();
            }
            break;

        case 'settings':
            // Load current settings
            loadSettings();
            break;

        default:
            AppLogger.warn('SCREEN_INIT', `Unknown screen: ${screenName}`);
    }
}

// Reset quiz state when navigating to quiz screen
function resetQuizState() {
    selectedAnswer = -1;
    timeRemaining = 30;
    isReviewMode = false;

    // Clear any active timers
    if (quizTimer) {
        clearInterval(quizTimer);
        quizTimer = null;
    }

    // Reset UI elements
    const answerButtons = document.querySelectorAll('.answer-btn');
    answerButtons.forEach(btn => {
        btn.classList.remove('selected', 'correct', 'incorrect');
        btn.disabled = false;
    });

    AppLogger.debug('QUIZ_STATE', 'Quiz state reset for new navigation');
}

// Initialize the app when QWebChannel is ready
new QWebChannel(qt.webChannelTransport, function (channel) {
    const initStart = performance.now();

    pythonBridge = channel.objects.pybridge;
    
    // Make bridge globally available for logging system
    window.pybridge = pythonBridge;
    window.pythonBridge = pythonBridge; // Keep both for compatibility
    
    AppLogger.system('BRIDGE', 'Python bridge initialized', {
        channelObjects: Object.keys(channel.objects),
        bridgeAvailable: !!pythonBridge,
        globalBridgeSet: !!window.pybridge
    });

    // 🔥 CRITICAL FIX: Connect Python signals to JavaScript handlers
    AppLogger.system('BRIDGE', 'Connecting Python signal handlers', {
        signalsToConnect: ['questionReceived', 'answerFeedback', 'quizCompleted', 'errorOccurred', 'updateStatus']
    });

    pythonBridge.questionReceived.connect(handleQuestionReceived);
    pythonBridge.answerFeedback.connect(handleAnswerFeedback);
    pythonBridge.quizCompleted.connect(handleQuizCompleted);
    pythonBridge.errorOccurred.connect(handleError);
    pythonBridge.updateStatus.connect(updateStatus);

    // 🧠 NEW: Connect topic analysis signal for intelligent UI adaptation
    pythonBridge.topicProfileUpdated.connect(handleTopicProfileUpdated);
    AppLogger.system('BRIDGE', 'Topic analysis signal connected');

    // 🚀 Phase 2: Connect enhanced training signals
    const trainingSignals = [];
    if (pythonBridge.trainingProgressStructured) {
        pythonBridge.trainingProgressStructured.connect(handleTrainingProgressStructured);
        trainingSignals.push('trainingProgressStructured');
    }
    if (pythonBridge.trainingStatusChanged) {
        pythonBridge.trainingStatusChanged.connect(handleTrainingStatusChanged);
        trainingSignals.push('trainingStatusChanged');
    }
    if (pythonBridge.trainingMetricsUpdate) {
        pythonBridge.trainingMetricsUpdate.connect(handleTrainingMetricsUpdate);
        trainingSignals.push('trainingMetricsUpdate');
    }
    if (pythonBridge.trainingConfigSaved) {
        pythonBridge.trainingConfigSaved.connect(handleTrainingConfigSaved);
        trainingSignals.push('trainingConfigSaved');
    }

    AppLogger.system('BRIDGE', 'Training signals connected', { connectedSignals: trainingSignals });

    // 🌊 Connect token streaming signals
    const streamingSignals = [];
    if (pythonBridge.tokenReceived) {
        pythonBridge.tokenReceived.connect(handleTokenReceived);
        streamingSignals.push('tokenReceived');
    }
    if (pythonBridge.streamingStarted) {
        pythonBridge.streamingStarted.connect(handleStreamingStarted);
        streamingSignals.push('streamingStarted');
    }
    if (pythonBridge.streamingCompleted) {
        pythonBridge.streamingCompleted.connect(handleStreamingCompleted);
        streamingSignals.push('streamingCompleted');
    }

    AppLogger.system('BRIDGE', 'Streaming signals connected', { connectedSignals: streamingSignals });

    const initDuration = performance.now() - initStart;
    AppLogger.performance('BRIDGE', 'Python bridge initialization completed', {
        duration: `${initDuration.toFixed(2)}ms`,
        totalSignals: trainingSignals.length + streamingSignals.length + 6
    });

    // Immediately test the logging bridge now that pybridge is available
    if (window.logToPython) {
        console.log('🧪 Testing logging bridge immediately after QWebChannel init...');
        window.logToPython('SUCCESS', 'BRIDGE_INIT', 'Bridge connected via QWebChannel', {
            channelObjects: Object.keys(channel.objects),
            bridgeReady: true,
            timestamp: new Date().toISOString()
        });
    }
    
    // Set a flag to indicate bridge is ready for button debugging
    window.bridgeReady = true;
    console.log('✅ Bridge ready flag set for button debugging');

    initializeApp();
});

// 🚀 CRITICAL FIX: Add missing error handling functions
function showError(message) {
    AppLogger.error('UI', 'Error displayed to user', {
        message,
        currentScreen,
        timestamp: new Date().toISOString()
    });

    // Show error in UI
    const errorContainer = document.getElementById('error-container') || createErrorContainer();
    errorContainer.innerHTML = `
        <div class="error-message">
            <span class="error-icon">❌</span>
            <span class="error-text">${message}</span>
            <button onclick="hideError()" class="error-close">×</button>
        </div>
    `;
    errorContainer.style.display = 'block';

    // Auto-hide after 5 seconds
    setTimeout(hideError, 5000);
}

function hideError() {
    AppLogger.user('UI', 'Error message dismissed', { method: 'auto-hide or user click' });
    const errorContainer = document.getElementById('error-container');
    if (errorContainer) {
        errorContainer.style.display = 'none';
    }
}

function createErrorContainer() {
    const container = document.createElement('div');
    container.id = 'error-container';
    container.className = 'error-container';
    container.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        max-width: 400px;
        display: none;
    `;
    document.body.appendChild(container);
    return container;
}

function showQuizLoading(message) {
    AppLogger.info('UI', 'Quiz loading state displayed', { message });

    const loadingContainer = document.getElementById('loading-container') || createLoadingContainer();
    loadingContainer.innerHTML = `
        <div class="loading-message">
            <div class="loading-spinner"></div>
            <span class="loading-text">${message}</span>
        </div>
    `;
    loadingContainer.style.display = 'block';
}

function hideQuizLoading() {
    AppLogger.info('UI', 'Quiz loading state hidden');
    const loadingContainer = document.getElementById('loading-container');
    if (loadingContainer) {
        loadingContainer.style.display = 'none';
    }
}

// Missing utility functions
function shouldUseTokenStreaming(difficulty, mode) {
    // Enable streaming for expert mode or when explicitly enabled
    return difficulty === 'expert' || mode === 'streaming';
}

function getQuestionTypeLabel(gameMode, submode) {
    const labels = {
        'conceptual': 'Conceptual Questions',
        'numerical': 'Numerical Problems',
        'mixed': 'Mixed Question Types',
        'mcq': 'Multiple Choice',
        'true_false': 'True/False',
        'fill_blank': 'Fill in the Blank'
    };
    return labels[submode] || labels[gameMode] || 'Standard Questions';
}

function isTokenStreamingEnabled() {
    // Check if token streaming is enabled in settings
    const settings = JSON.parse(localStorage.getItem('userSettings') || '{}');
    return settings.enableTokenStreaming || false;
}

function createTokenStreamUI(topic, difficulty, submode) {
    AppLogger.debug('TOKEN_STREAM', 'Creating token stream UI', { topic, difficulty, submode });
    // Placeholder for token streaming UI creation
}

function startTokenStreamingSimulation(topic, difficulty, submode) {
    AppLogger.debug('TOKEN_STREAM', 'Starting token streaming simulation', { topic, difficulty, submode });
    // Placeholder for token streaming simulation
}

function hideStatusDisplay() {
    const statusContainer = document.getElementById('status-display');
    if (statusContainer) {
        statusContainer.style.display = 'none';
    }
}

// Missing utility functions for screen initialization
function updateStats() {
    AppLogger.debug('STATS', 'Updating home screen statistics');
    // Update statistics on home screen
    try {
        // Mock stats for now - in real app this would come from backend
        const stats = {
            quizzesTaken: 0,
            averageScore: 0,
            questionsAnswered: 0
        };

        // Update DOM elements if they exist
        const quizzesElement = document.querySelector('[data-stat="quizzes"]');
        const scoreElement = document.querySelector('[data-stat="score"]');
        const questionsElement = document.querySelector('[data-stat="questions"]');

        if (quizzesElement) quizzesElement.textContent = stats.quizzesTaken;
        if (scoreElement) scoreElement.textContent = `${stats.averageScore}%`;
        if (questionsElement) questionsElement.textContent = stats.questionsAnswered;

        AppLogger.debug('STATS', 'Statistics updated successfully');
    } catch (error) {
        AppLogger.error('STATS', 'Failed to update statistics', error);
    }
}

function resetQuizState() {
    AppLogger.debug('QUIZ_STATE', 'Resetting quiz state');
    selectedAnswer = -1;
    timeRemaining = 30;
    isReviewMode = false;

    // Clear any active timers
    if (quizTimer) {
        clearInterval(quizTimer);
        quizTimer = null;
    }

    // Reset UI elements
    const answerButtons = document.querySelectorAll('.answer-btn');
    answerButtons.forEach(btn => {
        btn.classList.remove('selected', 'correct', 'incorrect');
        btn.disabled = false;
    });

    AppLogger.debug('QUIZ_STATE', 'Quiz state reset completed');
}

function loadSettings() {
    AppLogger.debug('SETTINGS', 'Loading settings screen');
    // Load settings - placeholder for now
    try {
        const settings = JSON.parse(localStorage.getItem('userSettings') || '{}');
        AppLogger.debug('SETTINGS', 'Settings loaded from localStorage', settings);
    } catch (error) {
        AppLogger.error('SETTINGS', 'Failed to load settings', error);
    }
}

function createLoadingContainer() {
    const container = document.createElement('div');
    container.id = 'loading-container';
    container.className = 'loading-container';
    container.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 9999;
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        display: none;
    `;
    document.body.appendChild(container);
    return container;
}

function showQuizError(message) {
    hideQuizLoading();
    showError(message);
}

// Initialize the application
async function initializeApp() {
    const initStart = performance.now();
    AppLogger.system('APP', 'Application initialization started', {
        userAgent: navigator.userAgent,
        screenSize: `${screen.width}x${screen.height}`,
        timestamp: new Date().toISOString()
    });

    // 🔥 CRITICAL FIX: Ensure DOM is fully loaded before attaching handlers
    AppLogger.debug('SETUP', 'Setting up all button click handlers');

    // Wait for DOM to be ready
    await new Promise(resolve => {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', resolve);
        } else {
            resolve();
        }
    });

    // 🔥 CRITICAL FIX: Set up Start Quiz button click handler with enhanced error handling
    function setupStartQuizButton() {
        const startQuizButton = document.getElementById('start-quiz-button');
        if (startQuizButton) {
            // Remove any existing handlers first
            startQuizButton.removeEventListener('click', startCustomQuiz);
            startQuizButton.onclick = null;

            // Add new handlers with error wrapping
            const safeStartCustomQuiz = function (event) {
                try {
                    event.preventDefault();
                    event.stopPropagation();
                    AppLogger.user('BUTTON_CLICK', 'Start Quiz button clicked');
                    startCustomQuiz();
                } catch (error) {
                    AppLogger.error('BUTTON_ERROR', 'Start Quiz button click failed', error);
                    showError('Failed to start quiz: ' + error.message);
                }
            };

            startQuizButton.addEventListener('click', safeStartCustomQuiz, { passive: false });
            startQuizButton.onclick = safeStartCustomQuiz; // Backup method

            // Ensure button is clickable
            startQuizButton.style.pointerEvents = 'auto';
            startQuizButton.style.zIndex = '1000';
            startQuizButton.disabled = false;

            AppLogger.info('SETUP', 'Start Quiz button handler attached with error handling');
            return true;
        }
        AppLogger.warn('SETUP', 'Start Quiz button not found in DOM');
        return false;
    }

    // 🔥 CRITICAL FIX: Set up Quick Quiz button click handler with enhanced error handling
    function setupQuickQuizButton() {
        const quickQuizButton = document.getElementById('quick-quiz-button');
        if (quickQuizButton) {
            // Remove any existing handlers first
            quickQuizButton.removeEventListener('click', startQuickQuiz);
            quickQuizButton.onclick = null;

            // Add new handlers with error wrapping
            const safeStartQuickQuiz = function (event) {
                try {
                    event.preventDefault();
                    event.stopPropagation();
                    AppLogger.user('BUTTON_CLICK', 'Quick Quiz button clicked');
                    startQuickQuiz();
                } catch (error) {
                    AppLogger.error('BUTTON_ERROR', 'Quick Quiz button click failed', error);
                    showError('Failed to start quick quiz: ' + error.message);
                }
            };

            quickQuizButton.addEventListener('click', safeStartQuickQuiz, { passive: false });
            quickQuizButton.onclick = safeStartQuickQuiz; // Backup method

            // Ensure button is clickable
            quickQuizButton.style.pointerEvents = 'auto';
            quickQuizButton.style.zIndex = '1000';
            quickQuizButton.disabled = false;

            AppLogger.info('SETUP', 'Quick Quiz button handler attached with error handling');
            return true;
        }
        AppLogger.warn('SETUP', 'Quick Quiz button not found in DOM');
        return false;
    }

    // 🔥 CRITICAL FIX: Set up navigation button handlers with enhanced error handling
    function setupNavigationButtons() {
        // Get all navigation buttons by class instead of specific IDs
        const navElements = document.querySelectorAll('.nav-item');
        let successCount = 0;

        navElements.forEach((navElement, index) => {
            if (navElement) {
                // Extract screen name from existing onclick attribute
                const onclickAttr = navElement.getAttribute('onclick');
                let screenName = 'home'; // default
                if (onclickAttr) {
                    const match = onclickAttr.match(/showScreen\('([^']+)'/);
                    if (match) {
                        screenName = match[1];
                    }
                }

                const buttonText = navElement.textContent.trim();
                
                // Remove existing handlers and onclick
                if (navElement._clickHandler) {
                    navElement.removeEventListener('click', navElement._clickHandler);
                }
                navElement.onclick = null;

                // Create safe click handler
                const safeNavHandler = function (event) {
                    try {
                        event.preventDefault();
                        event.stopPropagation();
                        AppLogger.user('NAVIGATION_CLICK', `${buttonText} navigation clicked`);
                        showScreen(screenName, navElement);
                    } catch (error) {
                        AppLogger.error('NAV_ERROR', `${buttonText} navigation failed`, error);
                        showError(`Failed to navigate to ${buttonText}: ` + error.message);
                    }
                };

                // Store handler reference for cleanup
                navElement._clickHandler = safeNavHandler;

                // Attach the event listener
                navElement.addEventListener('click', safeNavHandler, { passive: false });

                // Ensure button is clickable
                navElement.style.pointerEvents = 'auto';
                navElement.style.cursor = 'pointer';
                navElement.style.zIndex = '999';

                successCount++;
                AppLogger.debug('SETUP', `${buttonText} navigation handler attached for screen: ${screenName}`);
            }
        });

        AppLogger.info('SETUP', `Navigation handlers attached: ${successCount}/${navElements.length}`);
        return successCount > 0;
    }

    // Retry mechanism for button setup
    let retryCount = 0;
    const maxRetries = 5;

    function trySetupButtons() {
        const startQuizOk = setupStartQuizButton();
        const quickQuizOk = setupQuickQuizButton();
        const navOk = setupNavigationButtons();

        if (startQuizOk || quickQuizOk || navOk) {
            AppLogger.success('SETUP', 'Button handlers successfully attached');
            return true;
        }

        retryCount++;
        if (retryCount < maxRetries) {
            AppLogger.debug('SETUP', `Retrying button setup (${retryCount}/${maxRetries})`);
            setTimeout(trySetupButtons, 500);
        } else {
            AppLogger.error('SETUP', 'Failed to attach button handlers after max retries');
        }
        return false;
    }

    // Start the retry mechanism
    setTimeout(trySetupButtons, 100);

    // Check if we're in a reload scenario and try to restore from session storage first
    const isReload = performance.navigation.type === performance.navigation.TYPE_RELOAD;
    if (isReload) {
        AppLogger.info('SESSION', 'Page reload detected - attempting API key recovery');
        loadApiKeysFromSessionStorage();
    }

    // Stage 1: Load settings immediately
    AppLogger.info('INIT', 'Stage 1: Loading settings');
    if (typeof loadSettings === 'function') {
        const settingsStart = performance.now();
        await loadSettings();
        AppLogger.performance('SETTINGS', 'Settings loaded', { duration: performance.now() - settingsStart });
    } else {
        AppLogger.warning('INIT', 'loadSettings function not available');
    }

    // Load existing uploaded files
    AppLogger.info('INIT', 'Loading existing uploaded files');
    loadExistingFiles();

    // Stage 2: Setup auto-save and persistence after DOM is ready
    await new Promise(resolve => {
        if (document.readyState === 'complete' || document.readyState === 'interactive') {
            resolve();
        } else {
            document.addEventListener('DOMContentLoaded', resolve, { once: true });
        }
    });

    console.log('🔧 Stage 2: Setting up auto-save and persistence...');
    if (typeof setupAutoSave === 'function') setupAutoSave();
    if (typeof setupTopicAnalysis === 'function') setupTopicAnalysis();
    ensureApiKeyPersistence();
    updateApiKeyStatusIndicators();

    // Stage 2b: Validate settings load
    await new Promise(resolve => setTimeout(resolve, 100)); // Small delay for UI
    console.log('🎯 Stage 2b: Validating settings load...');
    const difficultySelect = document.getElementById('quiz-difficulty');
    const submodeSelect = document.getElementById('quiz-submode');
    const gameModeSelect = document.getElementById('quiz-game-mode');
    if (difficultySelect && submodeSelect && gameModeSelect) {
        console.log(`🔍 Current settings: difficulty=${difficultySelect.value}, submode=${submodeSelect.value}, gameMode=${gameModeSelect.value}`);
        if (typeof updateModeInfo === 'function') updateModeInfo();
        if (typeof updateGameModeInfo === 'function') updateGameModeInfo();
        if (typeof updateSubmodeInfo === 'function') updateSubmodeInfo();
        if (typeof updateDifficultyInfo === 'function') updateDifficultyInfo();
    }

    // Stage 3: Final verification and fallback
    await new Promise(resolve => setTimeout(resolve, 100));
    console.log('🔍 Stage 3: Final verification...');
    const savedSettings = localStorage.getItem('userSettings');
    if (savedSettings) {
        try {
            const settings = JSON.parse(savedSettings);
            console.log('🛠️ Force-applying localStorage settings as final fallback...');
            if (settings.default_difficulty && difficultySelect) {
                difficultySelect.value = settings.default_difficulty;
                console.log(`🔧 Force-set difficulty to: ${settings.default_difficulty}`);
            }
            if (settings.default_submode && submodeSelect) {
                submodeSelect.value = settings.default_submode;
                console.log(`🔧 Force-set question type to: ${settings.default_submode}`);
            }
            if (settings.default_game_mode && gameModeSelect) {
                gameModeSelect.value = settings.default_game_mode;
                console.log(`🔧 Force-set game mode to: ${settings.default_game_mode}`);
            }
            if (settings.default_quiz_mode) {
                const modeSelect = document.getElementById('quiz-mode');
                if (modeSelect) {
                    modeSelect.value = settings.default_quiz_mode;
                    console.log(`🔧 Force-set quiz mode to: ${settings.default_quiz_mode}`);
                }
            }
            if (settings.api_keys) {
                const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
                let keysRestored = 0;
                providers.forEach(provider => {
                    const input = document.getElementById(`${provider}-api-key`);
                    const savedKey = settings.api_keys[provider];
                    if (input && savedKey && savedKey.trim()) {
                        input.value = savedKey;
                        keysRestored++;
                        console.log(`🔧 Force-restored ${provider} API key`);
                    }
                });
                if (keysRestored > 0) {
                    updateProviderStatuses();
                    console.log(`✅ Force-restored ${keysRestored} API keys`);
                }
            }
            updateModeInfo();
            updateGameModeInfo();
            updateSubmodeInfo();
            updateDifficultyInfo();
            console.log('✅ Force-applied all settings from localStorage');
        } catch (e) {
            console.error('❌ Failed to parse localStorage settings:', e);
        }
    }
    console.log('✅ Settings persistence verification complete');

    // Stage 4: Ensure settings are saved to both backend and localStorage for persistence
    setTimeout(() => {
        console.log('💾 Stage 4: Ensuring settings persistence...');
        // ✅ NEW: Final API key status update
        setTimeout(() => {
            updateApiKeyStatusIndicators();

            // Save current state to ensure persistence across sessions
            setTimeout(() => {
                if (typeof saveSettings === 'function') {
                    const success = saveSettings();
                    if (success) {
                        console.log('💾 Settings successfully persisted on startup');

                        // ✅ NEW: Backup to session storage
                        setTimeout(() => {
                            saveApiKeysToSessionStorage();
                        }, 200);
                    } else {
                        console.log('⚠️ Settings persistence may have failed');
                    }
                }
            }, 200);
        }, 200);
    }, 300);

    console.log('✅ App initialization sequence started');

    // Initialize DeepSeek integration
    initializeDeepSeek();
}

// 🧠 DEEPSEEK INTEGRATION FUNCTIONS

function initializeDeepSeek() {
    console.log('🧠 Initializing DeepSeek integration...');

    // Check DeepSeek status
    if (pythonBridge && pythonBridge.getDeepSeekStatus) {
        pythonBridge.getDeepSeekStatus().then(statusJson => {
            try {
                const status = JSON.parse(statusJson);
                updateDeepSeekUI(status);
            } catch (e) {
                console.error('❌ Failed to parse DeepSeek status:', e);
                updateDeepSeekUI({ available: false, error: 'Status parsing failed' });
            }
        }).catch(error => {
            console.error('❌ DeepSeek status check failed:', error);
            updateDeepSeekUI({ available: false, error: 'Status check failed' });
        });
    } else {
        console.log('⚠️ DeepSeek methods not available in Python bridge');
        updateDeepSeekUI({ available: false, error: 'Bridge methods not available' });
    }
}

function updateDeepSeekUI(status) {
    const deepseekSection = document.getElementById('deepseek-section');
    const statusIndicator = document.querySelector('.status-indicator');
    const statusText = document.querySelector('.status-text');

    if (!deepseekSection || !statusIndicator || !statusText) {
        console.log('⚠️ DeepSeek UI elements not found');
        return;
    }

    if (status.available && status.ready) {
        // DeepSeek is ready - only show status, no button needed
        statusIndicator.textContent = '✅';
        statusIndicator.className = 'status-indicator ready';
        statusText.textContent = `Ready: ${status.thinking_model} + ${status.json_model}`;

        console.log('✅ DeepSeek pipeline ready');
    } else if (status.available && !status.ready) {
        // DeepSeek available but not ready
        statusIndicator.textContent = '⚠️';
        statusIndicator.className = 'status-indicator error';
        statusText.textContent = status.error || 'Pipeline not ready';

        console.log('⚠️ DeepSeek available but not ready:', status.error);
    } else {
        // DeepSeek not available
        deepseekSection.style.display = 'none';

        console.log('❌ DeepSeek not available:', status.error);
    }
}

function generateDeepSeekQuestion() {
    const deepSeekStart = performance.now();

    AppLogger.action('DEEPSEEK', 'DeepSeek question generation initiated');

    const topicInput = document.getElementById('quiz-topic');
    const difficultySelect = document.getElementById('quiz-difficulty');
    const submodeSelect = document.getElementById('quiz-submode');

    if (!topicInput || !difficultySelect || !submodeSelect) {
        AppLogger.error('DEEPSEEK', 'Required UI elements not found', {
            topicInput: !!topicInput,
            difficultySelect: !!difficultySelect,
            submodeSelect: !!submodeSelect
        });
        showError('Topic, difficulty, and question type inputs not found');
        return;
    }

    const topic = topicInput.value.trim();
    const difficulty = difficultySelect.value;
    const questionType = submodeSelect.value;

    if (!topic) {
        AppLogger.warn('DEEPSEEK', 'No topic provided for DeepSeek generation');
        showError('Please enter a topic first');
        return;
    }

    const deepSeekParams = { topic, difficulty, questionType };
    AppLogger.info('DEEPSEEK', 'DeepSeek generation parameters collected', deepSeekParams);

    // Show DeepSeek generation status
    showStatusDisplay(`🧠 DeepSeek AI thinking deeply about ${topic} (${questionType})...`, 'deepseek');

    AppLogger.debug('DEEPSEEK', `Generating DeepSeek question: ${topic} (${difficulty}) - Type: ${questionType}`);

    if (pythonBridge && pythonBridge.generateDeepSeekQuestion) {
        AppLogger.info('DEEPSEEK', 'Calling Python bridge for DeepSeek generation', deepSeekParams);

        pythonBridge.generateDeepSeekQuestion(topic, difficulty, questionType).then(resultJson => {
            try {
                const generationDuration = performance.now() - deepSeekStart;
                const result = JSON.parse(resultJson);

                AppLogger.performance('DEEPSEEK', 'DeepSeek generation completed', {
                    duration: generationDuration.toFixed(2),
                    success: result.success,
                    hasQuestion: !!result.question
                });

                if (result.success && result.question) {
                    AppLogger.success('DEEPSEEK', 'DeepSeek question generated successfully', {
                        topic,
                        difficulty,
                        questionType,
                        duration: generationDuration.toFixed(2)
                    });

                    // Display the generated question
                    displayDeepSeekQuestion(result.question);
                } else {
                    AppLogger.warn('DEEPSEEK', 'DeepSeek generation failed, initiating fallback', {
                        error: result.error,
                        hasFallback: !!result.fallback
                    });

                    if (result.fallback) {
                        // Fall back to regular quiz generation
                        showStatusDisplay('⚠️ DeepSeek unavailable, using standard generation...', 'warning');

                        AppLogger.info('DEEPSEEK', 'Executing fallback to standard quiz generation', {
                            originalDifficulty: difficulty,
                            fallbackDifficulty: 'hard'
                        });

                        // Remove expert difficulty temporarily for fallback
                        const originalDifficulty = difficulty;
                        difficultySelect.value = 'hard';
                        startCustomQuiz();
                        // Restore original difficulty
                        difficultySelect.value = originalDifficulty;
                    } else {
                        AppLogger.error('DEEPSEEK', 'DeepSeek generation failed without fallback', {
                            error: result.error
                        });
                        showError(result.error || 'DeepSeek generation failed');
                    }
                }
            } catch (e) {
                AppLogger.error('DEEPSEEK', 'Failed to parse DeepSeek response', {
                    error: e.message,
                    responsePreview: resultJson?.substring(0, 100)
                });
                showError('Failed to parse DeepSeek response');
            }
        }).catch(error => {
            const generationDuration = performance.now() - deepSeekStart;
            AppLogger.error('DEEPSEEK', 'DeepSeek generation error', {
                error: error.message,
                duration: generationDuration.toFixed(2),
                topic,
                difficulty,
                questionType
            });
            showError('DeepSeek generation failed');
        });
    } else {
        AppLogger.error('DEEPSEEK', 'DeepSeek generation method not available', {
            bridgeAvailable: !!pythonBridge,
            methodAvailable: !!(pythonBridge && pythonBridge.generateDeepSeekQuestion)
        });
        showError('DeepSeek not available');
    }
}

function displayDeepSeekQuestion(question) {
    // Switch to quiz screen
    showScreen('quiz');

    // Hide quiz setup and show quiz game
    document.getElementById('quiz-setup').style.display = 'none';
    document.getElementById('quiz-game').style.display = 'block';

    // Display the question
    handleQuestionReceived(question);

    // Update question number to indicate DeepSeek
    const questionNumber = document.getElementById('question-number');
    if (questionNumber) {
        questionNumber.textContent = '🧠 DeepSeek Expert Question';
    }
}

// 🧠 TOPIC ANALYSIS FUNCTIONS - Intelligent Question Type Recommendation

function setupTopicAnalysis() {
    /*
     Set up real-time topic analysis to intelligently guide question type selection.
     This provides a much more user-friendly experience by automatically enabling/disabling
     appropriate question types based on the topic the user enters.
     */
    const topicInput = document.getElementById('quiz-topic');
    const submodeSelect = document.getElementById('quiz-submode');

    if (!topicInput) {
        console.log('📝 Topic input not found - skipping topic analysis setup');
        return;
    }

    if (!submodeSelect) {
        console.log('📝 Question type select not found - skipping topic analysis setup');
        return;
    }

    console.log('🧠 Setting up intelligent topic analysis...');

    // Analyze topic on input with debouncing for better performance
    let analysisTimeout;
    topicInput.addEventListener('input', () => {
        clearTimeout(analysisTimeout);
        analysisTimeout = setTimeout(() => {
            const currentTopic = topicInput.value.trim();
            if (currentTopic.length >= 2) {
                console.log(`🧠 Analyzing topic: "${currentTopic}"`);
                console.log(`🔍 DEBUG: pythonBridge exists: ${!!pythonBridge}`);
                console.log(`🔍 DEBUG: analyzeTopic method exists: ${!!(pythonBridge && pythonBridge.analyzeTopic)}`);

                // CRITICAL FIX: Check if pythonBridge exists before calling
                if (pythonBridge && pythonBridge.analyzeTopic) {
                    console.log(`✅ Calling pythonBridge.analyzeTopic("${currentTopic}")`);
                    pythonBridge.analyzeTopic(currentTopic);
                } else {
                    console.warn('⚠️ Python bridge not available for topic analysis');
                    console.log(`🔍 DEBUG: pythonBridge object:`, pythonBridge);
                    if (pythonBridge) {
                        console.log(`🔍 DEBUG: Available methods:`, Object.getOwnPropertyNames(pythonBridge));
                    }
                }
            } else {
                // Reset to all enabled for short topics
                handleTopicProfileUpdated({
                    is_conceptual_possible: true,
                    is_numerical_possible: true,
                    confidence: 'low',
                    detected_type: 'unknown'
                });
            }
        }, 300); // 300ms debounce
    });

    // Also analyze on focus/paste
    topicInput.addEventListener('paste', () => {
        setTimeout(() => topicInput.dispatchEvent(new Event('input')), 10);
    });

    console.log('✅ Topic analysis event listeners configured');
}

function handleTopicProfileUpdated(profile) {
    /*
     Handle topic analysis results and adapt the UI accordingly.
     Enhanced with AI spell correction support.
     This is the core function that makes the interface intelligent.
     */
    try {
        console.log(`🧠 Topic profile received:`, profile);

        // Handle AI spell corrections first
        if (profile.spelling_corrected && profile.corrections_made && profile.corrections_made.length > 0) {
            console.log(`🤖 AI Spell corrections detected:`, profile.corrections_made);
            showSpellCorrectionFeedback(profile);
        }

        const submodeSelect = document.getElementById('quiz-submode');

        if (!submodeSelect) {
            console.log('⚠️ Question type select not found - UI adaptation skipped');
            return;
        }

        // Get the actual option elements
        const conceptualOption = submodeSelect.querySelector('option[value="conceptual"]');
        const numericalOption = submodeSelect.querySelector('option[value="numerical"]');
        const mixedOption = submodeSelect.querySelector('option[value="mixed"]');

        // Apply intelligent recommendations
        const shouldDisableNumerical = !profile.is_numerical_possible;
        const shouldDisableMixed = !profile.is_numerical_possible; // Mixed often requires numerical

        // Update option states
        if (numericalOption) {
            updateOptionState(numericalOption, !shouldDisableNumerical);
        }
        if (mixedOption) {
            updateOptionState(mixedOption, !shouldDisableMixed);
        }

        // Always keep conceptual enabled (conceptual questions work for any topic)
        if (conceptualOption) {
            updateOptionState(conceptualOption, true);
        }

        // 🧠 INTELLIGENT AUTO-SELECTION: Auto-select optimal question type based on topic analysis
        const currentValue = submodeSelect.value;
        let shouldAutoSelect = false;
        let newValue = currentValue;

        // Auto-select numerical for clearly numerical topics (like "atoms", "physics", "chemistry")
        console.log(`🔍 DEBUG: Checking auto-selection - detected_type: '${profile.detected_type}', confidence: '${profile.confidence}', current: '${currentValue}'`);

        if (profile.detected_type === 'numerical' && profile.confidence === 'high' && currentValue !== 'numerical') {
            newValue = 'numerical';
            shouldAutoSelect = true;
            console.log(`🧠 Auto-selected NUMERICAL for topic: ${profile.detected_type} (${profile.confidence} confidence)`);
        }
        // Also try medium confidence for clearly numerical topics like "atoms"
        else if (profile.detected_type === 'numerical' && profile.confidence === 'medium' && currentValue !== 'numerical') {
            newValue = 'numerical';
            shouldAutoSelect = true;
            console.log(`🧠 Auto-selected NUMERICAL (medium confidence) for topic: ${profile.detected_type} (${profile.confidence} confidence)`);
        }
        // Auto-select conceptual for clearly conceptual topics
        else if (profile.detected_type === 'conceptual' && profile.confidence === 'high' && currentValue !== 'conceptual') {
            newValue = 'conceptual';
            shouldAutoSelect = true;
            console.log(`🧠 Auto-selected CONCEPTUAL for topic: ${profile.detected_type} (${profile.confidence} confidence)`);
        }
        // Handle disabled options (fallback logic)
        else if ((currentValue === 'numerical' && shouldDisableNumerical) ||
            (currentValue === 'mixed' && shouldDisableMixed)) {
            newValue = 'conceptual';
            shouldAutoSelect = true;
            console.log(`🧠 Auto-switched to conceptual (fallback) for topic type: ${profile.detected_type}`);
        }

        // Apply the auto-selection
        if (shouldAutoSelect) {
            submodeSelect.value = newValue;

            // Update UI to reflect the change
            updateSubmodeInfo();

            // Show helpful feedback
            showTopicAnalysisFeedback(profile);

            // Save the automatic change
            if (typeof saveSettings === 'function') {
                saveSettings();
            }
        }

        // Update recommendation indicators
        updateTopicRecommendationIndicators(profile);

    } catch (error) {
        console.error('❌ Error handling topic profile:', error);
    }
}

function updateOptionState(option, enabled) {
    /* Update option visual state and functionality */
    if (!option) return;

    if (enabled) {
        option.disabled = false;
        option.classList.remove('disabled-option');
        option.classList.remove('topic-disabled');
        // Reset the text to original
        if (option.dataset.originalText) {
            option.textContent = option.dataset.originalText;
        }
    } else {
        option.disabled = true;
        option.classList.add('disabled-option');
        option.classList.add('topic-disabled');
        // Store original text and add disabled indicator
        if (!option.dataset.originalText) {
            option.dataset.originalText = option.textContent;
        }
        option.textContent = option.dataset.originalText + ' (Not suitable for this topic)';
    }
}

function updateTopicRecommendationIndicators(profile) {
    /* Add visual indicators showing recommended question types */
    const submodeSelect = document.getElementById('quiz-submode');
    if (!submodeSelect) return;

    const options = {
        'conceptual': submodeSelect.querySelector('option[value="conceptual"]'),
        'numerical': submodeSelect.querySelector('option[value="numerical"]'),
        'mixed': submodeSelect.querySelector('option[value="mixed"]')
    };

    // Remove all recommendation indicators first
    Object.values(options).forEach(option => {
        if (option) {
            option.classList.remove('topic-recommended', 'topic-optimal');
            // Reset to original text if we modified it for recommendations
            if (option.dataset.recommendedText) {
                option.textContent = option.dataset.originalText || option.textContent;
                delete option.dataset.recommendedText;
            }
        }
    });

    // Add recommendation indicators based on analysis
    if (profile.confidence === 'high') {
        const optimalType = profile.detected_type;

        if (optimalType === 'conceptual' && options.conceptual && !options.conceptual.disabled) {
            options.conceptual.classList.add('topic-optimal');
            if (!options.conceptual.dataset.originalText) {
                options.conceptual.dataset.originalText = options.conceptual.textContent;
            }
            options.conceptual.textContent = options.conceptual.dataset.originalText + ' ⭐ BEST MATCH';
            options.conceptual.dataset.recommendedText = true;
        } else if (optimalType === 'numerical' && options.numerical && !options.numerical.disabled) {
            options.numerical.classList.add('topic-optimal');
            if (!options.numerical.dataset.originalText) {
                options.numerical.dataset.originalText = options.numerical.textContent;
            }
            options.numerical.textContent = options.numerical.dataset.originalText + ' ⭐ BEST MATCH';
            options.numerical.dataset.recommendedText = true;

            if (options.conceptual && !options.conceptual.disabled) {
                options.conceptual.classList.add('topic-recommended');
            }
        } else if (optimalType === 'mixed' && options.mixed && !options.mixed.disabled) {
            options.mixed.classList.add('topic-optimal');
            if (!options.mixed.dataset.originalText) {
                options.mixed.dataset.originalText = options.mixed.textContent;
            }
            options.mixed.textContent = options.mixed.dataset.originalText + ' ⭐ BEST MATCH';
            options.mixed.dataset.recommendedText = true;
        }
    }
}

// Navigation functions
function showScreen(screenName, navElement) {
    const navigationStart = performance.now();
    const previousScreen = currentScreen;
    currentScreen = screenName;

    AppLogger.action('NAVIGATION', `Navigating from ${previousScreen} to ${screenName}`, {
        method: navElement ? 'nav-click' : 'programmatic',
        timestamp: Date.now()
    });

    try {
        // Hide all screens
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
            screen.style.display = 'none';
        });

        // Show selected screen
        const targetScreen = document.getElementById(`${screenName}-screen`);
        if (targetScreen) {
            targetScreen.classList.add('active');
            targetScreen.style.display = 'block';
            AppLogger.success('NAVIGATION', `Screen '${screenName}' activated successfully`);
        } else {
            AppLogger.error('NAVIGATION', `Screen '${screenName}-screen' not found in DOM`);
            showError(`Screen '${screenName}' not found`);
            return;
        }

        // Update navigation buttons
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });

        // If called from navigation, update the active nav item
        if (navElement) {
            if (navElement.classList.contains('nav-item')) {
                navElement.classList.add('active');
            } else {
                const navItem = navElement.closest('.nav-item');
                if (navItem) {
                    navItem.classList.add('active');
                }
            }
            AppLogger.debug('NAVIGATION', 'Navigation item updated', {
                element: navElement.textContent || navElement.innerText
            });
        }

        // Screen-specific initialization
        initializeScreen(screenName);

        const navigationDuration = performance.now() - navigationStart;
        AppLogger.performance('NAVIGATION', `Navigation completed in ${navigationDuration.toFixed(2)}ms`);

    } catch (error) {
        AppLogger.error('NAVIGATION', 'Navigation failed', {
            error: error.message,
            screenName: screenName,
            stack: error.stack
        });
        showError(`Navigation failed: ${error.message}`);
    }

    // Screen-specific initialization logic
    if (screenName === 'quiz') {
        AppLogger.info('QUIZ', 'Quiz screen initialized - setting up quiz interface');
        document.getElementById('quiz-setup').style.display = 'block';
        document.getElementById('quiz-game').style.display = 'none';
        document.getElementById('quiz-results').style.display = 'none';
        resetQuizState();
    }

    // If showing settings screen, load settings
    if (screenName === 'settings') {
        AppLogger.info('SETTINGS', 'Settings screen activated - loading current settings');
        loadSettings();
        updateProviderStatuses();
    }

    // If showing home screen, update stats
    if (screenName === 'home') {
        AppLogger.info('HOME', 'Home screen activated - updating statistics');
        updateStats();
    }

    // If showing train screen, load existing uploaded files and training configuration
    if (screenName === 'train') {
        AppLogger.info('TRAINING', 'Training screen activated - loading files and configuration');
        loadExistingFiles();
        // Only load training configuration when user actually navigates to train screen
        // and ensure Python bridge is ready
        setTimeout(async () => {
            if (pythonBridge && typeof pythonBridge.getTrainingConfiguration === 'function') {
                AppLogger.debug('TRAINING', 'Loading training configuration');
                await loadTrainingConfiguration();
            } else {
                AppLogger.warning('TRAINING', 'Python bridge not ready for training configuration, will retry');
                // Retry after a short delay
                setTimeout(async () => {
                    if (pythonBridge && typeof pythonBridge.getTrainingConfiguration === 'function') {
                        AppLogger.success('TRAINING', 'Training configuration loaded on retry');
                        await loadTrainingConfiguration();
                    } else {
                        AppLogger.error('TRAINING', 'Training configuration still unavailable after retry');
                    }
                }, 1000);
            }
        }, 100);
    }

    // If showing review screen, load question history
    if (screenName === 'review') {
        AppLogger.info('REVIEW', 'Review screen activated - loading question history');

        // Add a delay to ensure the screen is rendered first
        setTimeout(async () => {
            if (pythonBridge) {
                AppLogger.debug('REVIEW', 'Python bridge available - loading question history');
                try {
                    await loadQuestionHistory();
                    AppLogger.success('REVIEW', 'Question history loaded successfully');
                } catch (e) {
                    AppLogger.error('REVIEW', 'Failed to load question history', { error: e.message });
                }
            } else {
                AppLogger.error('REVIEW', 'Python bridge not available - using fallback');
                // Try to load anyway (will use fallback/mock data)
                await loadQuestionHistory();
            }
        }, 500); // Increased delay to ensure UI is ready

        currentScreen = screenName;
        const navigationDuration = performance.now() - navigationStart;

        // CRITICAL FIX: Check if pythonBridge exists before calling
        if (pythonBridge && pythonBridge.navigate) {
            pythonBridge.navigate(screenName);
            AppLogger.debug('NAVIGATION', 'Python bridge navigation call completed');
        } else {
            AppLogger.warning('NAVIGATION', 'Python bridge not available for navigation');
        }

        AppLogger.performance('NAVIGATION', `Screen transition completed: ${previousScreen} → ${screenName}`, {
            duration: `${navigationDuration.toFixed(2)}ms`,
            method: navElement ? 'user-click' : 'programmatic'
        });
    }

    // Theme toggle
    function toggleTheme() {
        const currentTheme = document.body.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        AppLogger.trackUserAction('Theme toggled', {
            from: currentTheme,
            to: newTheme,
            method: 'theme-toggle-button'
        });

        document.body.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);

        // Update toggle icon
        document.querySelector('.theme-toggle').textContent = newTheme === 'dark' ? '☀️' : '🌙';

        // Auto-save theme preference
        AppLogger.debug('SETTINGS', `Auto-saving theme change to: ${newTheme}`);
        setTimeout(saveSettings, 50);
    }

    // Reset quiz state
    function resetQuizState() {
        selectedAnswer = -1;
        currentQuestionState = null;
        isReviewMode = false;
        clearInterval(quizTimer);
        clearInterval(statusUpdateInterval);
        hideQuizElements();
        hideStatusDisplay();
    }

    // Hide quiz feedback elements
    function hideQuizElements() {
        const elements = [
            'feedback-container',
            'explanation-container',
            'navigation-buttons'
        ];

        elements.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.style.display = 'none';
        });
    }

    // Show/hide status display
    function showStatusDisplay(message, type = 'info') {
        const statusContainer = document.getElementById('status-display');
        if (statusContainer) {
            statusContainer.innerHTML = `
            <div class="status-message ${type}">
                <div class="status-icon">${getStatusIcon(type)}</div>
                <div class="status-text">${message}</div>
                <div class="status-spinner"></div>
            </div>
        `;
            statusContainer.style.display = 'block';
        }
    }

    function hideStatusDisplay() {
        const statusContainer = document.getElementById('status-display');
        if (statusContainer) {
            statusContainer.style.display = 'none';
        }
    }

    function getStatusIcon(type) {
        const icons = {
            'info': '🔄',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'turbo': '🚀',
            'gpu': '🎮',
            'api': '🌐',
            'cloud': '☁️',
            'network': '📡'
        };
        return icons[type] || '🔄';
    }

    // Quiz functions
    function startQuickQuiz() {
        AppLogger.trackUserAction('Quick Quiz started', { method: 'quick-start-button' });
        // Just navigate to quiz screen - DON'T bypass the setup!
        showScreen('quiz', null);
        // Let user configure their quiz settings
    }

    function startCustomQuiz() {
        const quizStartTime = performance.now();
        AppLogger.action('QUIZ_START', 'Custom quiz initiation started');

        try {
            // Validate required elements exist
            const requiredElements = [
                'quiz-topic', 'quiz-mode', 'quiz-game-mode',
                'quiz-submode', 'quiz-difficulty', 'quiz-questions'
            ];

            const missingElements = requiredElements.filter(id => !document.getElementById(id));
            if (missingElements.length > 0) {
                throw new Error(`Missing form elements: ${missingElements.join(', ')}`);
            }

            const topic = document.getElementById('quiz-topic').value || "General Knowledge";
            const mode = document.getElementById('quiz-mode').value;
            const gameMode = document.getElementById('quiz-game-mode').value;
            const submode = document.getElementById('quiz-submode').value;
            const difficulty = document.getElementById('quiz-difficulty').value;
            const numQuestions = parseInt(document.getElementById('quiz-questions').value);

            // Validate parameters
            if (!topic || !mode || !gameMode || !submode || !difficulty) {
                throw new Error('All quiz parameters must be selected');
            }

            if (isNaN(numQuestions) || numQuestions < 1 || numQuestions > 50) {
                throw new Error('Number of questions must be between 1 and 50');
            }

            const quizParams = { topic, mode, gameMode, submode, difficulty, numQuestions };
            AppLogger.action('QUIZ_PARAMS', 'Quiz parameters validated', quizParams);

            // 🧠 FIXED: Expert mode should start a proper quiz, not just generate one question
            AppLogger.info('QUIZ', `Starting ${difficulty} quiz: ${numQuestions} questions about "${topic}"`);

            // Expert mode will be handled by the backend automatically when difficulty=expert

            const params = {
                topic: topic,
                mode: mode,
                game_mode: gameMode,
                submode: submode,
                difficulty: difficulty,
                num_questions: numQuestions
            };

            // 🌊 Check if token streaming should be used
            const useStreaming = shouldUseTokenStreaming(difficulty, mode);
            AppLogger.debug('QUIZ', `Token streaming evaluation`, { useStreaming, difficulty, mode });

            if (useStreaming) {
                // Use new streaming functionality for live generation
                AppLogger.info('QUIZ', 'Starting streaming quiz generation');
                if (pythonBridge && pythonBridge.generateQuestionStreaming) {
                    AppLogger.debug('QUIZ', 'Calling Python bridge for streaming generation', { topic, difficulty });
                    pythonBridge.generateQuestionStreaming(topic, difficulty);
                    showQuizGame(); // Show quiz interface
                    return; // Streaming handles the rest
                } else {
                    AppLogger.warning('QUIZ', 'Streaming not available, falling back to regular generation');
                }
            }

            // Show appropriate generation status based on mode
            const questionTypeLabel = getQuestionTypeLabel(gameMode, submode);
            if (mode === 'offline') {
                AppLogger.info('QUIZ', `Starting LOCAL GPU generation for ${topic} (${questionTypeLabel})`);
                showStatusDisplay(`🎮 Starting LOCAL GPU generation for ${topic} (${questionTypeLabel})...`, 'gpu');
            } else {
                AppLogger.info('QUIZ', `Connecting to AI APIs for ${topic} (${questionTypeLabel})`);
                showStatusDisplay(`🌐 Connecting to AI APIs for ${topic} (${questionTypeLabel})...`, 'api');
            }

            // 🚀 CRITICAL FIX: Use non-blocking quiz start to prevent UI freezing
            if (pythonBridge && pythonBridge.startQuiz) {
                AppLogger.debug('QUIZ', 'Python bridge available, calling startQuiz', { paramsJson: JSON.stringify(params) });

                // Show loading immediately
                showQuizLoading('🧠 Generating your first question...');

                // Show quiz interface immediately to prevent UI blocking
                showQuizGame();

                // Start generation in background (non-blocking)
                try {
                    // 🌊 ENHANCED: Start token streaming if enabled
                    if (shouldUseTokenStreaming(mode, isTokenStreamingEnabled())) {
                        AppLogger.debug('QUIZ', 'Starting token streaming visualization', { topic, difficulty, submode });
                        createTokenStreamUI(topic, difficulty, submode);
                        startTokenStreamingSimulation(topic, difficulty, submode);
                    }

                    AppLogger.info('QUIZ', 'Calling Python bridge startQuiz method');
                    pythonBridge.startQuiz(JSON.stringify(params));

                    const quizSetupDuration = performance.now() - quizStartTime;
                    AppLogger.performance('QUIZ', 'Quiz setup completed', {
                        duration: `${quizSetupDuration.toFixed(2)}ms`,
                        ...quizParams
                    });

                } catch (error) {
                    AppLogger.error('QUIZ_START_ERROR', 'Quiz start error', {
                        error: error.message,
                        context: 'startQuiz call',
                        params: params
                    });
                    hideQuizLoading();
                    showError('Failed to start quiz: ' + error.message);
                }
            } else {
                AppLogger.error('QUIZ', 'Python bridge not available for starting quiz');
                showError('Cannot start quiz - Python bridge not available');
                return;
            }

            // Start status monitoring with proper mode
            AppLogger.debug('QUIZ', 'Starting status monitoring', { mode });
            startStatusMonitoring(mode);

        } catch (error) {
            AppLogger.error('QUIZ_START_ERROR', 'Failed to start custom quiz', {
                error: error.message,
                stack: error.stack,
                currentScreen: currentScreen
            });

            hideQuizLoading();
            showError('Failed to start quiz: ' + error.message);

            // Reset UI state
            const startButton = document.getElementById('start-quiz-button');
            if (startButton) {
                startButton.disabled = false;
                startButton.textContent = '⭐ START QUIZ';
            }
        }
    }

    function startStatusMonitoring(mode = 'auto') {
        let statusMessages = [];
        let bufferMessages = [];
        let statusType = 'info';

        // Different animations based on generation mode
        if (mode === 'offline') {
            // LOCAL GPU MODE - Show realistic GPU/hardware animations
            statusMessages = [
                '🎮 Initializing local GPU acceleration...',
                '⚡ Loading AI model into available VRAM...',
                '🚀 Configuring GPU for AI inference...',
                '🔥 Model loaded - generating with GPU assist...',
                '💨 Processing questions with local AI...',
                '⭐ Local generation ready!'
            ];
            bufferMessages = [
                '🎮 Using available GPU resources...',
                '⚡ Local AI processing in progress...',
                '🚀 Questions generating locally...'
            ];
            statusType = 'gpu';
        } else {
            // ONLINE API MODE - Show network/cloud animations  
            statusMessages = [
                '🌐 Connecting to cloud AI providers...',
                '🔗 Establishing secure API connections...',
                '📡 Sending requests to remote servers...',
                '☁️ AI models processing in the cloud...',
                '📦 Receiving generated questions...',
                '⭐ Download complete - questions ready!'
            ];
            bufferMessages = [
                '🌐 Fetching from cloud APIs...',
                '📡 Downloading AI-generated content...',
                '☁️ Cloud processing in progress...'
            ];
            statusType = 'api';
        }

        let messageIndex = 0;
        const interval = 1800; // Slightly slower for better readability
        statusUpdateInterval = setInterval(() => {
            if (messageIndex < statusMessages.length) {
                showStatusDisplay(statusMessages[messageIndex], statusType);
                messageIndex++;
            } else {
                // Cycle through appropriate buffer messages
                const bufferIndex = messageIndex % bufferMessages.length;
                let message = bufferMessages[bufferIndex];

                // For offline mode, show real GPU utilization frequently
                if (mode === 'offline' && Math.random() < 0.5) {
                    // Try to get actual GPU utilization with bulletproof error handling
                    try {
                        // Enhanced bridge readiness check
                        if (pythonBridge &&
                            pythonBridge.getGpuUtilization &&
                            typeof pythonBridge.getGpuUtilization === 'function') {

                            // Add timeout protection for the call
                            let gpuStatsStr;
                            try {
                                gpuStatsStr = pythonBridge.getGpuUtilization();
                            } catch (bridgeError) {
                                console.debug('Bridge call failed:', bridgeError);
                                throw new Error('Bridge call failed');
                            }

                            // Validate JSON string before parsing with enhanced checks
                            if (!gpuStatsStr || typeof gpuStatsStr !== 'string' || gpuStatsStr.length < 5) {
                                throw new Error('Invalid GPU stats response');
                            }

                            // Ensure it looks like JSON
                            const trimmed = gpuStatsStr.trim();
                            if (!trimmed.startsWith('{') || !trimmed.endsWith('}')) {
                                throw new Error(`Invalid JSON format: ${trimmed.substring(0, 50)}...`);
                            }

                            // Parse with error handling
                            let gpuStats;
                            try {
                                gpuStats = JSON.parse(trimmed);
                            } catch (parseError) {
                                console.debug('JSON parse error:', parseError, 'Raw response:', trimmed.substring(0, 100));
                                throw new Error(`JSON parse failed: ${parseError.message}`);
                            }

                            // Validate parsed object
                            if (!gpuStats || typeof gpuStats !== 'object') {
                                throw new Error('Parsed GPU stats is not an object');
                            }

                            if (gpuStats.available && gpuStats.gpu_utilization !== undefined) {
                                const gpuUsage = Math.round(gpuStats.gpu_utilization || 0);
                                const memUsage = Math.round(gpuStats.memory_utilization || 0);
                                const deviceName = gpuStats.device_name ? gpuStats.device_name.split(' ')[0] : 'GPU';

                                // Show detailed GPU stats
                                if (gpuStats.temperature_c && gpuStats.power_usage_w) {
                                    message = `🎮 ${deviceName}: ${gpuUsage}% GPU, ${memUsage}% VRAM, ${gpuStats.temperature_c}°C, ${Math.round(gpuStats.power_usage_w)}W`;
                                } else {
                                    message = `🎮 ${deviceName}: ${gpuUsage}% GPU utilization, ${memUsage}% VRAM`;
                                }

                                // Add status indicator
                                if (gpuUsage > 80) {
                                    message = `🔥 ${message} (High Load)`;
                                } else if (gpuUsage > 40) {
                                    message = `⚡ ${message} (Active)`;
                                } else {
                                    message = `💤 ${message} (Low Load)`;
                                }

                            } else if (gpuStats.available) {
                                const memUsage = Math.round(gpuStats.memory_utilization || 0);
                                message = `🎮 GPU Memory: ${memUsage}% (Shared with other apps)`;
                            } else {
                                // GPU not available or error
                                const errorMsg = gpuStats.error ? ` (${gpuStats.error})` : '';
                                message = `🎮 GPU monitoring unavailable${errorMsg} - CPU processing`;
                            }
                        } else {
                            // Fallback to educational messages about GPU sharing
                            const sharingMessages = [
                                'ℹ️ GPU shared with system - normal behavior',
                                '📺 Multiple apps using GPU - sharing resources',
                                '🎮 40-60% GPU usage is typical with other apps',
                                '⚡ Local AI processing with available GPU power'
                            ];
                            message = sharingMessages[Math.floor(Math.random() * sharingMessages.length)];
                        }
                    } catch (error) {
                        // Reduce console spam by using debug instead of warn for common errors
                        if (error.message.includes('Invalid GPU stats response') ||
                            error.message.includes('Bridge call failed')) {
                            console.debug('GPU stats temporarily unavailable:', error.message);
                        } else {
                            console.warn('Error getting GPU stats:', error.message || error);
                        }
                        // Use fallback message without spamming console
                        message = '🎮 Local AI processing active';
                    }
                }

                showStatusDisplay(message, statusType);
                messageIndex++;
            }
        }, interval);
    }

    function showQuizGame() {
        console.log('🎮 Showing quiz game interface...');

        const quizSetup = document.getElementById('quiz-setup');
        const quizGame = document.getElementById('quiz-game');
        const quizResults = document.getElementById('quiz-results');

        if (quizSetup) {
            quizSetup.style.display = 'none';
            console.log('✅ Quiz setup hidden');
        } else {
            console.error('❌ Quiz setup element not found');
        }

        if (quizGame) {
            quizGame.style.display = 'block';
            console.log('✅ Quiz game shown');

            // 🔍 ENHANCED: Check if key elements exist
            const questionText = document.getElementById('question-text');
            const optionsContainer = document.getElementById('options-container');
            const submitBtn = document.getElementById('submit-btn');

            console.log('🔍 Quiz game elements check:', {
                questionText: !!questionText,
                optionsContainer: !!optionsContainer,
                submitBtn: !!submitBtn
            });
        } else {
            console.error('❌ Quiz game element not found');
        }

        if (quizResults) {
            quizResults.style.display = 'none';
            console.log('✅ Quiz results hidden');
        } else {
            console.error('❌ Quiz results element not found');
        }

        resetQuizState();
    }

    function handleQuestionReceived(data) {
        console.log('🔥 DEBUG: handleQuestionReceived() called!');
        console.log('✅ Question received:', data);

        // 🔍 ENHANCED DEBUGGING: Log detailed question data
        console.log('🔍 Question data details:', {
            hasQuestion: !!data.question,
            hasOptions: !!data.options,
            optionsType: typeof data.options,
            optionsLength: data.options ? (Array.isArray(data.options) ? data.options.length : Object.keys(data.options).length) : 0,
            questionLength: data.question ? data.question.length : 0
        });

        // 🚀 CRITICAL FIX: Hide loading state when question arrives
        hideQuizLoading();

        // Hide status display once we receive a question
        hideStatusDisplay();
        clearInterval(statusUpdateInterval);

        // Ensure data exists
        if (!data) {
            console.error('No data received');
            return;
        }

        currentQuestionState = data;
        isReviewMode = data.review_mode || false;

        // Reset UI state
        hideQuizElements();
        clearInterval(quizTimer);

        // Update question number
        const questionNumber = document.getElementById('question-number');
        if (questionNumber) {
            questionNumber.textContent = `Question ${data.question_number} of ${data.total_questions}`;
        }

        // Update question text
        const questionText = document.getElementById('question-text');
        if (questionText) {
            console.log('🔍 Updating question text:', data.question.substring(0, 100) + '...');
            updateQuestionWithLatex(data.question, questionText).then(() => {
                console.log('✅ Question LaTeX rendered successfully');
                console.log('🔍 Question element content:', questionText.innerHTML.substring(0, 100) + '...');
            }).catch(err => {
                console.error('❌ Question LaTeX render error:', err);
                // Fallback to plain text
                questionText.innerHTML = data.question;
                console.log('🔄 Fallback to plain text applied');
            });
        } else {
            console.error('❌ Question text element not found!');
        }

        // Create option buttons
        const optionsContainer = document.getElementById('options-container');
        if (optionsContainer && data.options) {
            console.log('🔍 Creating options buttons...');
            optionsContainer.innerHTML = '';

            // 🚀 CRITICAL FIX: Handle both array and object formats for options
            let optionsArray = [];
            if (Array.isArray(data.options)) {
                optionsArray = data.options;
                console.log('✅ Options are in array format:', optionsArray.length, 'options');
            } else if (typeof data.options === 'object') {
                // Convert object format {A: "text", B: "text"} to array
                optionsArray = Object.values(data.options);
                console.log('✅ Options converted from object to array:', optionsArray.length, 'options');
            } else {
                console.error('❌ Invalid options format:', data.options);
                showError('Invalid question format received');
                return;
            }

            console.log('🔍 Options array:', optionsArray);

            optionsArray.forEach((option, index) => {
                console.log(`🔍 Creating option ${index + 1}:`, option.substring(0, 50) + '...');

                const button = document.createElement('button');
                button.className = 'option-button';

                // Create a span for the option letter and another for the content
                const letterSpan = document.createElement('span');
                letterSpan.className = 'option-letter';
                letterSpan.textContent = `${String.fromCharCode(65 + index)}. `;

                const contentSpan = document.createElement('span');
                contentSpan.className = 'option-content';
                // Process option text for LaTeX
                const processedOption = processLatexText(option);
                contentSpan.innerHTML = processedOption;

                button.appendChild(letterSpan);
                button.appendChild(contentSpan);
                button.onclick = (event) => selectAnswer(index, event);

                // If in review mode and the question was previously answered, highlight the answer
                if (isReviewMode && data.answered) {
                    if (index === data.user_answer) {
                        button.classList.add(data.is_correct ? 'correct-answer' : 'incorrect-answer');
                    }
                    if (index === data.correct_index && !data.is_correct) {
                        button.classList.add('correct-answer');
                    }
                    // FIXED: Don't disable buttons in review mode to allow interaction
                    // The original code had: button.disabled = true;
                    // This was preventing users from selecting options in review mode
                }

                optionsContainer.appendChild(button);
                console.log(`✅ Option ${index + 1} button created and added`);
            });

            console.log(`✅ All ${optionsArray.length} option buttons created successfully`);

            // Render LaTeX in all option buttons - wait a moment for DOM update
            setTimeout(() => {
                renderLatex(optionsContainer);
            }, 10);
        }

        // Handle review mode vs new question mode
        if (isReviewMode && data.answered) {
            // Show feedback and explanation immediately
            showAnswerFeedback({
                is_correct: data.is_correct,
                correct_index: data.correct_index,
                user_answer: data.user_answer,
                explanation: data.explanation,
                correct_answer_text: data.options[data.correct_index],
                feedback_message: data.is_correct ? "🎉 Correct!" : `❌ Incorrect. The correct answer was ${String.fromCharCode(65 + data.correct_index)}.`
            });
            showNavigationButtons();

            // Hide submit button in review mode
            const submitBtn = document.getElementById('submit-btn');
            if (submitBtn) submitBtn.style.display = 'none';

        } else {
            // New question mode - reset selection and start timer
            selectedAnswer = -1;
            startTimer();

            // Show submit button
            const submitBtn = document.getElementById('submit-btn');
            if (submitBtn) submitBtn.style.display = 'block';

            // Show brief "ready" status
            showStatusDisplay('⚡ Question ready!', 'success');
            setTimeout(hideStatusDisplay, 2000);
        }
    }

    function selectAnswer(index, event) {
        // Stop event propagation to prevent modal closing
        if (event) {
            event.stopPropagation();
            // Prevent default behavior as well
            event.preventDefault();
        }

        AppLogger.user('QUIZ_INTERACTION', `Answer selected: option ${index}`, {
            selectedIndex: index,
            previousSelection: selectedAnswer,
            currentScreen: currentScreen,
            isReviewMode: isReviewMode
        });

        // Don't allow selection in review mode if already answered
        if (isReviewMode) {
            AppLogger.debug('QUIZ_INTERACTION', 'Selection blocked - in review mode');
            return;
        }

        // Update selected answer
        selectedAnswer = index;

        // Update UI to show selection
        const optionButtons = document.querySelectorAll('.option-button');
        optionButtons.forEach((button, i) => {
            button.classList.remove('selected');
            if (i === index) {
                button.classList.add('selected');
            }
        });

        AppLogger.success('QUIZ_INTERACTION', `Answer ${index} selected successfully`);
    }

    // Missing function: Handle answer feedback from Python bridge
    function handleAnswerFeedback(feedbackData) {
        AppLogger.info('QUIZ_FEEDBACK', 'Answer feedback received', {
            isCorrect: feedbackData.is_correct,
            correctIndex: feedbackData.correct_index,
            userAnswer: feedbackData.user_answer
        });

        showAnswerFeedback(feedbackData);
    }

    // Missing function: Show answer feedback in UI
    function showAnswerFeedback(feedback) {
        AppLogger.info('UI', 'Displaying answer feedback', {
            isCorrect: feedback.is_correct,
            correctIndex: feedback.correct_index,
            userAnswer: feedback.user_answer
        });

        // Update option buttons to show correct/incorrect
        const optionButtons = document.querySelectorAll('.option-button');
        optionButtons.forEach((button, index) => {
            button.classList.remove('selected', 'correct-answer', 'incorrect-answer');

            if (index === feedback.correct_index) {
                button.classList.add('correct-answer');
            }

            if (index === feedback.user_answer && !feedback.is_correct) {
                button.classList.add('incorrect-answer');
            }
        });

        // Show feedback message
        const feedbackContainer = document.getElementById('feedback-container');
        if (feedbackContainer) {
            feedbackContainer.innerHTML = `
            <div class="feedback-message ${feedback.is_correct ? 'correct' : 'incorrect'}">
                <div class="feedback-icon">${feedback.is_correct ? '✅' : '❌'}</div>
                <div class="feedback-text">
                    ${feedback.is_correct ? 'Correct!' : 'Incorrect'}
                    ${feedback.explanation ? `<div class="explanation">${feedback.explanation}</div>` : ''}
                </div>
            </div>
        `;
            feedbackContainer.style.display = 'block';
        }
    }

    // Missing function: Handle quiz completion
    function handleQuizCompleted(completionData) {
        AppLogger.success('QUIZ', 'Quiz completed', {
            totalQuestions: completionData.total_questions,
            correctAnswers: completionData.correct_answers,
            score: completionData.score
        });

        // Show completion screen or results
        showScreen('results');

        // Display results
        const resultsContainer = document.getElementById('results-container');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
            <div class="quiz-results">
                <h2>Quiz Complete!</h2>
                <div class="score">Score: ${completionData.score}%</div>
                <div class="details">
                    ${completionData.correct_answers}/${completionData.total_questions} correct
                </div>
            </div>
        `;
        }
    }

    // Missing function: Handle errors from Python bridge
    function handleError(errorData) {
        AppLogger.error('PYTHON_BRIDGE', 'Error received from Python bridge', {
            error: errorData.error || errorData,
            context: errorData.context,
            timestamp: new Date().toISOString()
        });

        showError(errorData.error || errorData.message || 'An error occurred');
    }

    // Missing function: Update status from Python bridge
    function updateStatus(statusData) {
        AppLogger.info('STATUS_UPDATE', 'Status update received', {
            status: statusData.status || statusData,
            type: statusData.type,
            progress: statusData.progress
        });

        const statusMessage = statusData.status || statusData.message || statusData;
        const statusType = statusData.type || 'info';

        showStatusDisplay(statusMessage, statusType);
    }

    // Missing function: Handle streaming started
    function handleStreamingStarted(streamData) {
        AppLogger.info('STREAMING', 'Token streaming started', {
            sessionId: streamData.session_id,
            operation: streamData.operation
        });

        currentStreamSession = streamData.session_id;
        tokenStreamStats = {
            tokensReceived: 0,
            startTime: Date.now(),
            lastTokenTime: Date.now()
        };

        // Initialize streaming UI if needed
        initializeTokenStreamUI();
    }

    // Missing function: Handle token received during streaming
    function handleTokenReceived(tokenData) {
        if (!currentStreamSession || tokenData.session_id !== currentStreamSession) {
            return; // Ignore tokens from old sessions
        }

        tokenStreamStats.tokensReceived++;
        tokenStreamStats.lastTokenTime = Date.now();

        // Update streaming UI
        if (tokenStreamContainer) {
            const tokenElement = document.createElement('span');
            tokenElement.textContent = tokenData.token;
            tokenElement.className = 'stream-token';
            tokenStreamContainer.appendChild(tokenElement);
        }

        AppLogger.debug('STREAMING', `Token received: ${tokenData.token}`, {
            sessionId: tokenData.session_id,
            tokenCount: tokenStreamStats.tokensReceived
        });
    }

    // Missing function: Handle streaming completed
    function handleStreamingCompleted(completionData) {
        AppLogger.performance('STREAMING', 'Token streaming completed', {
            sessionId: completionData.session_id,
            totalTokens: tokenStreamStats.tokensReceived,
            duration: Date.now() - tokenStreamStats.startTime,
            tokensPerSecond: tokenStreamStats.tokensReceived / ((Date.now() - tokenStreamStats.startTime) / 1000)
        });

        currentStreamSession = null;

        // Finalize streaming UI
        if (tokenStreamContainer) {
            tokenStreamContainer.classList.add('streaming-complete');
        }
    }

    // Missing function: Initialize token streaming UI
    function initializeTokenStreamUI() {
        tokenStreamContainer = document.getElementById('token-stream-container');
        if (!tokenStreamContainer) {
            tokenStreamContainer = document.createElement('div');
            tokenStreamContainer.id = 'token-stream-container';
            tokenStreamContainer.className = 'token-stream';

            // Find appropriate parent container
            const questionContainer = document.getElementById('question-container');
            if (questionContainer) {
                questionContainer.appendChild(tokenStreamContainer);
            }
        }

        // Clear previous content
        tokenStreamContainer.innerHTML = '';
        tokenStreamContainer.classList.remove('streaming-complete');
    }

    // Missing function: Handle training progress
    function handleTrainingProgressStructured(progressData) {
        AppLogger.info('TRAINING', 'Training progress update', {
            epoch: progressData.epoch,
            step: progressData.step,
            loss: progressData.loss,
            progress: progressData.progress
        });

        // Update training UI if on training screen
        if (currentScreen === 'training') {
            updateTrainingProgressUI(progressData);
        }
    }

    // Missing function: Handle training status changes
    function handleTrainingStatusChanged(statusData) {
        AppLogger.info('TRAINING', 'Training status changed', {
            status: statusData.status,
            phase: statusData.phase
        });

        // Update training status UI
        const statusElement = document.getElementById('training-status');
        if (statusElement) {
            statusElement.textContent = statusData.status;
            statusElement.className = `training-status ${statusData.phase}`;
        }
    }

    // Missing function: Handle training metrics update
    function handleTrainingMetricsUpdate(metricsData) {
        AppLogger.performance('TRAINING', 'Training metrics updated', {
            accuracy: metricsData.accuracy,
            loss: metricsData.loss,
            learningRate: metricsData.learning_rate
        });

        // Update metrics display
        updateTrainingMetricsUI(metricsData);
    }

    // Missing function: Handle training config saved
    function handleTrainingConfigSaved(configData) {
        AppLogger.success('TRAINING', 'Training configuration saved', {
            configName: configData.name,
            parameters: configData.parameters
        });

        showStatusDisplay('Training configuration saved successfully', 'success');
    }

    // Helper function: Update training progress UI
    function updateTrainingProgressUI(progressData) {
        const progressBar = document.getElementById('training-progress-bar');
        const progressText = document.getElementById('training-progress-text');

        if (progressBar) {
            progressBar.style.width = `${progressData.progress}%`;
        }

        if (progressText) {
            progressText.textContent = `Epoch ${progressData.epoch}, Step ${progressData.step} - Loss: ${progressData.loss.toFixed(4)}`;
        }
    }

    // Helper function: Update training metrics UI
    function updateTrainingMetricsUI(metricsData) {
        const metricsContainer = document.getElementById('training-metrics');
        if (metricsContainer) {
            metricsContainer.innerHTML = `
            <div class="metric">
                <label>Accuracy:</label>
                <span>${(metricsData.accuracy * 100).toFixed(2)}%</span>
            </div>
            <div class="metric">
                <label>Loss:</label>
                <span>${metricsData.loss.toFixed(4)}</span>
            </div>
            <div class="metric">
                <label>Learning Rate:</label>
                <span>${metricsData.learning_rate.toExponential(2)}</span>
            </div>
        `;
        }
    }

    // Missing function: Load settings from backend and localStorage
    async function loadSettings() {
        AppLogger.info('SETTINGS', 'Loading user settings');

        try {
            // Try to load from Python bridge first
            if (pythonBridge && pythonBridge.getUserSettings) {
                const settingsJson = await pythonBridge.getUserSettings();
                const settings = JSON.parse(settingsJson);
                applySettingsToUI(settings);
                AppLogger.success('SETTINGS', 'Settings loaded from backend', { settingsCount: Object.keys(settings).length });
                return true;
            }

            // Fallback to localStorage
            const localSettings = localStorage.getItem('userSettings');
            if (localSettings) {
                const settings = JSON.parse(localSettings);
                applySettingsToUI(settings);
                AppLogger.success('SETTINGS', 'Settings loaded from localStorage', { settingsCount: Object.keys(settings).length });
                return true;
            }

            AppLogger.warn('SETTINGS', 'No settings found - using defaults');
            return false;
        } catch (error) {
            AppLogger.error('SETTINGS', 'Failed to load settings', { error: error.message });
            return false;
        }
    }

    // Missing function: Save settings to backend and localStorage
    function saveSettings() {
        AppLogger.info('SETTINGS', 'Saving user settings');

        try {
            const settings = collectSettingsFromUI();

            // Save to localStorage immediately
            localStorage.setItem('userSettings', JSON.stringify(settings));

            // Save to Python bridge if available
            if (pythonBridge && pythonBridge.saveUserSettings) {
                pythonBridge.saveUserSettings(JSON.stringify(settings));
            }

            AppLogger.success('SETTINGS', 'Settings saved successfully', { settingsCount: Object.keys(settings).length });
            return true;
        } catch (error) {
            AppLogger.error('SETTINGS', 'Failed to save settings', { error: error.message });
            return false;
        }
    }

    // Helper function: Apply settings to UI elements
    function applySettingsToUI(settings) {
        AppLogger.debug('SETTINGS', 'Applying settings to UI', settings);

        // Apply difficulty setting
        const difficultySelect = document.getElementById('quiz-difficulty');
        if (difficultySelect && settings.default_difficulty) {
            difficultySelect.value = settings.default_difficulty;
        }

        // Apply question type setting
        const submodeSelect = document.getElementById('quiz-submode');
        if (submodeSelect && settings.default_submode) {
            submodeSelect.value = settings.default_submode;
        }

        // Apply game mode setting
        const gameModeSelect = document.getElementById('quiz-game-mode');
        if (gameModeSelect && settings.default_game_mode) {
            gameModeSelect.value = settings.default_game_mode;
        }

        // Apply quiz mode setting
        const modeSelect = document.getElementById('quiz-mode');
        if (modeSelect && settings.default_quiz_mode) {
            modeSelect.value = settings.default_quiz_mode;
        }

        // Apply API keys
        if (settings.api_keys) {
            const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
            providers.forEach(provider => {
                const input = document.getElementById(`${provider}-api-key`);
                if (input && settings.api_keys[provider]) {
                    input.value = settings.api_keys[provider];
                }
            });
        }

        // Apply theme
        if (settings.theme) {
            document.body.setAttribute('data-theme', settings.theme);
        }
    }

    // Helper function: Collect settings from UI elements
    function collectSettingsFromUI() {
        const settings = {};

        // Collect basic settings
        const difficultySelect = document.getElementById('quiz-difficulty');
        if (difficultySelect) settings.default_difficulty = difficultySelect.value;

        const submodeSelect = document.getElementById('quiz-submode');
        if (submodeSelect) settings.default_submode = submodeSelect.value;

        const gameModeSelect = document.getElementById('quiz-game-mode');
        if (gameModeSelect) settings.default_game_mode = gameModeSelect.value;

        const modeSelect = document.getElementById('quiz-mode');
        if (modeSelect) settings.default_quiz_mode = modeSelect.value;

        // Collect API keys
        settings.api_keys = {};
        const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
        providers.forEach(provider => {
            const input = document.getElementById(`${provider}-api-key`);
            if (input && input.value.trim()) {
                settings.api_keys[provider] = input.value.trim();
            }
        });

        // Collect theme
        settings.theme = document.body.getAttribute('data-theme') || 'light';

        // Add timestamp
        settings.lastUpdated = new Date().toISOString();

        return settings;
    }
    function loadExistingFiles() {
        AppLogger.info('FILES', 'Loading existing uploaded files');

        if (pythonBridge && pythonBridge.getUploadedFiles) {
            pythonBridge.getUploadedFiles().then(filesJson => {
                try {
                    const files = JSON.parse(filesJson);
                    uploadedFiles = files;
                    updateFilesList();
                    AppLogger.success('FILES', 'Uploaded files loaded', { fileCount: files.length });
                } catch (error) {
                    AppLogger.error('FILES', 'Failed to parse uploaded files', { error: error.message });
                }
            }).catch(error => {
                AppLogger.error('FILES', 'Failed to load uploaded files', { error: error.message });
            });
        } else {
            AppLogger.warn('FILES', 'Python bridge not available for file loading');
        }
    }

    // Missing function: Setup auto-save functionality
    function setupAutoSave() {
        AppLogger.info('SETTINGS', 'Setting up auto-save functionality');

        // Auto-save on form changes
        const formElements = document.querySelectorAll('select, input[type="text"], input[type="password"]');
        formElements.forEach(element => {
            element.addEventListener('change', () => {
                AppLogger.debug('SETTINGS', `Auto-saving due to ${element.id} change`);
                setTimeout(saveSettings, 500); // Debounce saves
            });
        });

        // Auto-save on page unload
        window.addEventListener('beforeunload', () => {
            AppLogger.info('SETTINGS', 'Auto-saving before page unload');
            saveSettings();
        });

        AppLogger.success('SETTINGS', 'Auto-save functionality enabled');
    }

    // Missing function: Ensure API key persistence
    function ensureApiKeyPersistence() {
        AppLogger.info('API_KEYS', 'Ensuring API key persistence');

        // Save API keys to session storage as backup
        saveApiKeysToSessionStorage();

        // Set up periodic backup
        setInterval(() => {
            saveApiKeysToSessionStorage();
        }, 30000); // Every 30 seconds
    }

    // Helper function: Save API keys to session storage
    function saveApiKeysToSessionStorage() {
        try {
            const apiKeys = {};
            const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];

            providers.forEach(provider => {
                const input = document.getElementById(`${provider}-api-key`);
                if (input && input.value.trim()) {
                    apiKeys[provider] = input.value.trim();
                }
            });

            sessionStorage.setItem('apiKeysBackup', JSON.stringify(apiKeys));
            AppLogger.debug('API_KEYS', 'API keys backed up to session storage');
        } catch (error) {
            AppLogger.error('API_KEYS', 'Failed to backup API keys', { error: error.message });
        }
    }

    // Helper function: Load API keys from session storage
    function loadApiKeysFromSessionStorage() {
        try {
            const apiKeysBackup = sessionStorage.getItem('apiKeysBackup');
            if (apiKeysBackup) {
                const apiKeys = JSON.parse(apiKeysBackup);
                const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];

                providers.forEach(provider => {
                    const input = document.getElementById(`${provider}-api-key`);
                    if (input && apiKeys[provider]) {
                        input.value = apiKeys[provider];
                    }
                });

                AppLogger.success('API_KEYS', 'API keys restored from session storage');
                return true;
            }
        } catch (error) {
            AppLogger.error('API_KEYS', 'Failed to restore API keys from session storage', { error: error.message });
        }
        return false;
    }

    // Missing function: Update API key status indicators
    function updateApiKeyStatusIndicators() {
        AppLogger.debug('API_KEYS', 'Updating API key status indicators');

        const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
        providers.forEach(provider => {
            const input = document.getElementById(`${provider}-api-key`);
            const indicator = document.getElementById(`${provider}-status`);

            if (input && indicator) {
                const hasKey = input.value.trim().length > 0;
                indicator.textContent = hasKey ? '✅' : '❌';
                indicator.className = hasKey ? 'status-indicator ready' : 'status-indicator error';
            }
        });
    }

    // Missing function: Update provider statuses
    function updateProviderStatuses() {
        AppLogger.debug('PROVIDERS', 'Updating provider statuses');
        updateApiKeyStatusIndicators();

        // Additional provider-specific status updates can be added here
        if (pythonBridge && pythonBridge.checkProviderStatus) {
            const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
            providers.forEach(provider => {
                pythonBridge.checkProviderStatus(provider).then(statusJson => {
                    try {
                        const status = JSON.parse(statusJson);
                        updateProviderStatusUI(provider, status);
                    } catch (error) {
                        AppLogger.error('PROVIDERS', `Failed to parse ${provider} status`, { error: error.message });
                    }
                });
            });
        }
    }

    // Helper function: Update provider status UI
    function updateProviderStatusUI(provider, status) {
        const statusElement = document.getElementById(`${provider}-provider-status`);
        if (statusElement) {
            statusElement.textContent = status.available ? '🟢 Available' : '🔴 Unavailable';
            statusElement.className = status.available ? 'provider-status available' : 'provider-status unavailable';
        }
    }

    // Missing function: Update files list UI
    function updateFilesList() {
        const filesContainer = document.getElementById('uploaded-files-list');
        if (!filesContainer) return;

        if (uploadedFiles.length === 0) {
            filesContainer.innerHTML = '<p class="no-files">No files uploaded yet</p>';
            return;
        }

        const filesHTML = uploadedFiles.map(file => `
        <div class="file-item" data-filename="${file.name}">
            <span class="file-icon">📄</span>
            <span class="file-name">${file.name}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
            <button class="delete-file-btn" onclick="deleteFile('${file.name}')">🗑️</button>
        </div>
    `).join('');

        filesContainer.innerHTML = filesHTML;
        AppLogger.debug('FILES', 'Files list updated', { fileCount: uploadedFiles.length });
    }

    // Helper function: Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Missing function: Delete file
    function deleteFile(filename) {
        AppLogger.user('FILES', `File deletion requested: ${filename}`);

        if (pythonBridge && pythonBridge.deleteUploadedFile) {
            pythonBridge.deleteUploadedFile(filename).then(result => {
                if (result === 'success') {
                    uploadedFiles = uploadedFiles.filter(file => file.name !== filename);
                    updateFilesList();
                    AppLogger.success('FILES', `File deleted successfully: ${filename}`);
                } else {
                    AppLogger.error('FILES', `Failed to delete file: ${filename}`);
                    showError(`Failed to delete file: ${filename}`);
                }
            }).catch(error => {
                AppLogger.error('FILES', `Error deleting file: ${filename}`, { error: error.message });
                showError(`Error deleting file: ${filename}`);
            });
        }
    }

    // Missing function: Update mode info display
    function updateModeInfo() {
        AppLogger.debug('UI_UPDATE', 'Updating mode info display');

        const modeSelect = document.getElementById('quiz-mode');
        const modeInfo = document.getElementById('mode-info');

        if (!modeSelect || !modeInfo) return;

        const modeDescriptions = {
            'online': '🌐 Online mode uses cloud AI providers for question generation',
            'offline': '💻 Offline mode uses local models for question generation',
            'hybrid': '🔄 Hybrid mode combines online and offline capabilities'
        };

        const selectedMode = modeSelect.value;
        modeInfo.textContent = modeDescriptions[selectedMode] || 'Select a quiz mode';
        modeInfo.className = `mode-info ${selectedMode}`;
    }

    // Missing function: Update game mode info display
    function updateGameModeInfo() {
        AppLogger.debug('UI_UPDATE', 'Updating game mode info display');

        const gameModeSelect = document.getElementById('quiz-game-mode');
        const gameModeInfo = document.getElementById('game-mode-info');

        if (!gameModeSelect || !gameModeInfo) return;

        const gameModeDescriptions = {
            'standard': '📝 Standard quiz with multiple choice questions',
            'timed': '⏱️ Timed quiz with countdown for each question',
            'practice': '🎯 Practice mode with immediate feedback',
            'exam': '📋 Exam mode with final scoring only'
        };

        const selectedGameMode = gameModeSelect.value;
        gameModeInfo.textContent = gameModeDescriptions[selectedGameMode] || 'Select a game mode';
        gameModeInfo.className = `game-mode-info ${selectedGameMode}`;
    }

    // Missing function: Update submode info display
    function updateSubmodeInfo() {
        AppLogger.debug('UI_UPDATE', 'Updating submode info display');

        const submodeSelect = document.getElementById('quiz-submode');
        const submodeInfo = document.getElementById('submode-info');

        if (!submodeSelect || !submodeInfo) return;

        const submodeDescriptions = {
            'conceptual': '💭 Conceptual questions focus on understanding and theory',
            'numerical': '🔢 Numerical questions involve calculations and problem-solving',
            'mixed': '🎲 Mixed questions combine conceptual and numerical elements',
            'application': '🛠️ Application questions test practical usage',
            'analysis': '🔍 Analysis questions require critical thinking'
        };

        const selectedSubmode = submodeSelect.value;
        submodeInfo.textContent = submodeDescriptions[selectedSubmode] || 'Select a question type';
        submodeInfo.className = `submode-info ${selectedSubmode}`;
    }

    // Missing function: Update difficulty info display
    function updateDifficultyInfo() {
        AppLogger.debug('UI_UPDATE', 'Updating difficulty info display');

        const difficultySelect = document.getElementById('quiz-difficulty');
        const difficultyInfo = document.getElementById('difficulty-info');

        if (!difficultySelect || !difficultyInfo) return;

        const difficultyDescriptions = {
            'beginner': '🌱 Beginner level - Basic concepts and simple problems',
            'intermediate': '🌿 Intermediate level - Moderate complexity and depth',
            'advanced': '🌳 Advanced level - Complex problems requiring expertise',
            'expert': '🎓 Expert level - PhD-level questions with deep analysis',
            'phd': '🔬 PhD level - Research-grade questions with cutting-edge concepts'
        };

        const selectedDifficulty = difficultySelect.value;
        difficultyInfo.textContent = difficultyDescriptions[selectedDifficulty] || 'Select a difficulty level';
        difficultyInfo.className = `difficulty-info ${selectedDifficulty}`;
    }

    // Missing function: Load training configuration
    async function loadTrainingConfiguration() {
        AppLogger.info('TRAINING', 'Loading training configuration');

        try {
            if (pythonBridge && pythonBridge.getTrainingConfiguration) {
                const configJson = await pythonBridge.getTrainingConfiguration();
                const config = JSON.parse(configJson);

                // Apply configuration to UI
                applyTrainingConfigToUI(config);

                AppLogger.success('TRAINING', 'Training configuration loaded', {
                    configKeys: Object.keys(config).length
                });
                return true;
            } else {
                AppLogger.warn('TRAINING', 'Python bridge not available for training configuration');
                return false;
            }
        } catch (error) {
            AppLogger.error('TRAINING', 'Failed to load training configuration', {
                error: error.message
            });
            return false;
        }
    }

    // Helper function: Apply training configuration to UI
    function applyTrainingConfigToUI(config) {
        AppLogger.debug('TRAINING', 'Applying training configuration to UI', config);

        // Apply learning rate
        const learningRateInput = document.getElementById('learning-rate');
        if (learningRateInput && config.learning_rate) {
            learningRateInput.value = config.learning_rate;
        }

        // Apply batch size
        const batchSizeInput = document.getElementById('batch-size');
        if (batchSizeInput && config.batch_size) {
            batchSizeInput.value = config.batch_size;
        }

        // Apply epochs
        const epochsInput = document.getElementById('epochs');
        if (epochsInput && config.epochs) {
            epochsInput.value = config.epochs;
        }

        // Apply model selection
        const modelSelect = document.getElementById('training-model');
        if (modelSelect && config.model_name) {
            modelSelect.value = config.model_name;
        }

        // Apply other configuration options
        Object.keys(config).forEach(key => {
            const element = document.getElementById(`training-${key.replace('_', '-')}`);
            if (element && config[key] !== undefined) {
                if (element.type === 'checkbox') {
                    element.checked = config[key];
                } else {
                    element.value = config[key];
                }
            }
        });
    }

    // Missing function: Process LaTeX text for rendering
    function processLatexText(text) {
        if (!text) return '';

        // Simple LaTeX processing - replace common patterns
        return text
            .replace(/\$\$(.*?)\$\$/g, '<span class="latex-display">$1</span>')
            .replace(/\$(.*?)\$/g, '<span class="latex-inline">$1</span>')
            .replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '<span class="fraction"><span class="numerator">$1</span><span class="denominator">$2</span></span>')
            .replace(/\\sqrt\{([^}]+)\}/g, '<span class="sqrt">√<span class="radicand">$1</span></span>');
    }

    // Missing function: Render LaTeX in container
    function renderLatex(container) {
        if (!container) return;

        AppLogger.debug('LATEX', 'Rendering LaTeX in container', {
            containerId: container.id,
            hasLatexElements: container.querySelectorAll('.latex-inline, .latex-display').length > 0
        });

        // If MathJax is available, use it
        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([container]).then(() => {
                AppLogger.success('LATEX', 'MathJax rendering completed');
            }).catch(error => {
                AppLogger.error('LATEX', 'MathJax rendering failed', { error: error.message });
            });
        } else {
            AppLogger.debug('LATEX', 'MathJax not available - using basic LaTeX processing');
        }
    }

    // Missing function: Start timer for quiz questions
    function startTimer() {
        if (quizTimer) {
            clearInterval(quizTimer);
        }

        timeRemaining = 30; // Default 30 seconds
        const timerDisplay = document.getElementById('timer-display');

        AppLogger.info('QUIZ_TIMER', 'Timer started', { duration: timeRemaining });

        quizTimer = setInterval(() => {
            timeRemaining--;

            if (timerDisplay) {
                timerDisplay.textContent = `⏱️ ${timeRemaining}s`;
                timerDisplay.className = timeRemaining <= 10 ? 'timer-warning' : 'timer-normal';
            }

            if (timeRemaining <= 0) {
                clearInterval(quizTimer);
                AppLogger.warn('QUIZ_TIMER', 'Timer expired - auto-submitting answer');

                // Auto-submit or handle timeout
                if (selectedAnswer === -1) {
                    AppLogger.user('QUIZ_TIMEOUT', 'No answer selected when timer expired');
                    showError('Time expired! No answer was selected.');
                } else {
                    AppLogger.user('QUIZ_TIMEOUT', `Auto-submitting answer ${selectedAnswer} due to timeout`);
                    submitAnswer();
                }
            }
        }, 1000);
    }

    // Missing function: Submit answer
    function submitAnswer() {
        if (selectedAnswer === -1) {
            AppLogger.warn('QUIZ_SUBMIT', 'No answer selected for submission');
            showError('Please select an answer before submitting.');
            return;
        }

        AppLogger.action('QUIZ_SUBMIT', `Submitting answer: ${selectedAnswer}`, {
            selectedAnswer: selectedAnswer,
            timeRemaining: timeRemaining
        });

        // Clear timer
        if (quizTimer) {
            clearInterval(quizTimer);
            quizTimer = null;
        }

        // Send answer to Python bridge
        if (pythonBridge && pythonBridge.submitAnswer) {
            pythonBridge.submitAnswer(selectedAnswer);
            showStatusDisplay('Checking answer...', 'info');
        } else {
            AppLogger.error('QUIZ_SUBMIT', 'Python bridge not available for answer submission');
            showError('Unable to submit answer - connection issue');
        }
    }

    // Missing function: Show navigation buttons
    function showNavigationButtons() {
        const navContainer = document.getElementById('quiz-navigation');
        if (navContainer) {
            navContainer.innerHTML = `
            <button id="prev-question-btn" onclick="navigateQuestion('prev')">← Previous</button>
            <button id="next-question-btn" onclick="navigateQuestion('next')">Next →</button>
            <button id="finish-quiz-btn" onclick="finishQuiz()">Finish Quiz</button>
        `;
            navContainer.style.display = 'block';
        }
    }

    // Missing function: Navigate between questions
    function navigateQuestion(direction) {
        AppLogger.user('QUIZ_NAVIGATION', `Question navigation: ${direction}`);

        if (pythonBridge && pythonBridge.navigateQuestion) {
            pythonBridge.navigateQuestion(direction);
        }
    }

    // Missing function: Finish quiz
    function finishQuiz() {
        AppLogger.action('QUIZ_FINISH', 'Quiz finish requested by user');

        if (pythonBridge && pythonBridge.finishQuiz) {
            pythonBridge.finishQuiz();
            showStatusDisplay('Finishing quiz...', 'info');
        }
    }

    // Missing function: Start timer for quiz questions
    function startTimer() {
        if (quizTimer) {
            clearInterval(quizTimer);
        }

        timeRemaining = 30; // Default 30 seconds
        const timerDisplay = document.getElementById('timer-display');

        AppLogger.info('QUIZ_TIMER', 'Timer started', { duration: timeRemaining });

        quizTimer = setInterval(() => {
            timeRemaining--;

            if (timerDisplay) {
                timerDisplay.textContent = `⏱️ ${timeRemaining}s`;
                timerDisplay.className = timeRemaining <= 10 ? 'timer-warning' : 'timer-normal';
            }

            if (timeRemaining <= 0) {
                clearInterval(quizTimer);
                AppLogger.warn('QUIZ_TIMER', 'Timer expired - auto-submitting answer');

                // Auto-submit or handle timeout
                if (selectedAnswer === -1) {
                    AppLogger.user('QUIZ_TIMEOUT', 'No answer selected when timer expired');
                    showError('Time expired! No answer was selected.');
                } else {
                    AppLogger.user('QUIZ_TIMEOUT', `Auto-submitting answer ${selectedAnswer} due to timeout`);
                    submitAnswer();
                }
            }
        }, 1000);
    }

    // Missing function: Submit answer
    function submitAnswer() {
        if (selectedAnswer === -1) {
            AppLogger.warn('QUIZ_SUBMIT', 'No answer selected for submission');
            showError('Please select an answer before submitting.');
            return;
        }

        AppLogger.action('QUIZ_SUBMIT', `Submitting answer: ${selectedAnswer}`, {
            selectedAnswer: selectedAnswer,
            timeRemaining: timeRemaining
        });

        // Clear timer
        if (quizTimer) {
            clearInterval(quizTimer);
            quizTimer = null;
        }

        // Send answer to Python bridge
        if (pythonBridge && pythonBridge.submitAnswer) {
            pythonBridge.submitAnswer(selectedAnswer);
            showStatusDisplay('Checking answer...', 'info');
        } else {
            AppLogger.error('QUIZ_SUBMIT', 'Python bridge not available for answer submission');
            showError('Unable to submit answer - connection issue');
        }
    }
}

// Missing function: Show navigation buttons
function showNavigationButtons() {
    const navContainer = document.getElementById('quiz-navigation');
    if (navContainer) {
        navContainer.innerHTML = `
            <button id="prev-question-btn" onclick="navigateQuestion('prev')">← Previous</button>
            <button id="next-question-btn" onclick="navigateQuestion('next')">Next →</button>
            <button id="finish-quiz-btn" onclick="finishQuiz()">Finish Quiz</button>
        `;
        navContainer.style.display = 'block';
    }
}

// Missing function: Navigate between questions
function navigateQuestion(direction) {
    AppLogger.user('QUIZ_NAVIGATION', `Question navigation: ${direction}`);

    if (pythonBridge && pythonBridge.navigateQuestion) {
        pythonBridge.navigateQuestion(direction);
    }
}

// Missing function: Finish quiz
function finishQuiz() {
    AppLogger.action('QUIZ_FINISH', 'Quiz finish requested by user');

    if (pythonBridge && pythonBridge.finishQuiz) {
        pythonBridge.finishQuiz();
        showStatusDisplay('Finishing quiz...', 'info');
    }
}

// ==========================================
// REVIEW HISTORY FUNCTIONS
// ==========================================

/**
 * Load question history from the backend
 */
async function loadQuestionHistory(offset = 0, limit = 50) {
    try {
        AppLogger.info('REVIEW', 'Loading question history', { offset, limit });
        
        // Show loading indicator
        const loadingElement = document.getElementById('loading-history');
        const questionsListElement = document.getElementById('questions-list');
        const noQuestionsElement = document.getElementById('no-questions');
        
        if (loadingElement) loadingElement.style.display = 'block';
        if (questionsListElement) questionsListElement.style.display = 'none';
        if (noQuestionsElement) noQuestionsElement.style.display = 'none';
        
        let questions = [];
        
        if (pythonBridge && pythonBridge.getQuestionHistory) {
            try {
                const result = pythonBridge.getQuestionHistory(offset, limit);
                const data = JSON.parse(result);
                
                if (data.success) {
                    questions = data.questions || [];
                    AppLogger.success('REVIEW', `Loaded ${questions.length} questions from history`);
                } else {
                    AppLogger.error('REVIEW', 'Failed to load question history', { error: data.error });
                }
            } catch (e) {
                AppLogger.error('REVIEW', 'Error calling Python bridge for question history', { error: e.message });
            }
        }
        
        // Hide loading indicator
        if (loadingElement) loadingElement.style.display = 'none';
        
        if (questions.length === 0) {
            // Show no questions message
            if (noQuestionsElement) noQuestionsElement.style.display = 'block';
            AppLogger.info('REVIEW', 'No questions found in history');
        } else {
            // Display questions
            displayQuestionHistory(questions);
            if (questionsListElement) questionsListElement.style.display = 'block';
            
            // Update topic filter
            updateTopicFilter(questions);
        }
        
    } catch (error) {
        AppLogger.error('REVIEW', 'Error loading question history', { error: error.message });
        
        // Hide loading and show error
        const loadingElement = document.getElementById('loading-history');
        if (loadingElement) {
            loadingElement.innerHTML = '<p>❌ Error loading question history. Please try again.</p>';
        }
    }
}

/**
 * Display question history in the UI
 */
function displayQuestionHistory(questions) {
    const questionsListElement = document.getElementById('questions-list');
    if (!questionsListElement) return;
    
    questionsListElement.innerHTML = '';
    
    questions.forEach((question, index) => {
        const questionCard = createQuestionCard(question, index);
        questionsListElement.appendChild(questionCard);
    });
    
    AppLogger.debug('REVIEW', `Displayed ${questions.length} questions in history`);
}

/**
 * Create a question card element
 */
function createQuestionCard(question, index) {
    const card = document.createElement('div');
    card.className = 'question-card';
    card.style.cssText = `
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    `;
    
    // Get difficulty color
    const difficultyColors = {
        'easy': '#28a745',
        'medium': '#ffc107',
        'hard': '#dc3545',
        'expert': '#6f42c1'
    };
    
    const difficultyColor = difficultyColors[question.difficulty?.toLowerCase()] || '#6c757d';
    
    card.innerHTML = `
        <div class="question-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span class="question-topic" style="background: #e9ecef; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">
                ${question.topic || 'General'}
            </span>
            <span class="question-difficulty" style="background: ${difficultyColor}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">
                ${question.difficulty || 'Medium'}
            </span>
        </div>
        <div class="question-text" style="font-weight: bold; margin-bottom: 10px; color: #333;">
            ${question.question || 'No question text'}
        </div>
        <div class="question-options" style="margin-bottom: 10px;">
            ${(question.options || []).map((option, i) => `
                <div style="padding: 4px 0; color: ${i === question.correct ? '#28a745' : '#666'}; ${i === question.correct ? 'font-weight: bold;' : ''}">
                    ${String.fromCharCode(65 + i)}. ${option}
                </div>
            `).join('')}
        </div>
        <div class="question-meta" style="font-size: 12px; color: #666; display: flex; justify-content: space-between;">
            <span>ID: ${question.id || index}</span>
            <span>${question.timestamp || 'Unknown time'}</span>
        </div>
    `;
    
    // Add click handler to show question details
    card.addEventListener('click', () => {
        showQuestionModal(question);
    });
    
    // Add hover effects
    card.addEventListener('mouseenter', () => {
        card.style.transform = 'translateY(-2px)';
        card.style.boxShadow = '0 4px 8px rgba(0,0,0,0.15)';
    });
    
    card.addEventListener('mouseleave', () => {
        card.style.transform = 'translateY(0)';
        card.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
    });
    
    return card;
}

/**
 * Update topic filter dropdown with available topics
 */
function updateTopicFilter(questions) {
    const topicFilter = document.getElementById('topic-filter');
    if (!topicFilter) return;
    
    // Get unique topics
    const topics = [...new Set(questions.map(q => q.topic || 'General'))].sort();
    
    // Clear existing options (except "All Topics")
    topicFilter.innerHTML = '<option value="">All Topics</option>';
    
    // Add topic options
    topics.forEach(topic => {
        const option = document.createElement('option');
        option.value = topic;
        option.textContent = topic;
        topicFilter.appendChild(option);
    });
    
    AppLogger.debug('REVIEW', `Updated topic filter with ${topics.length} topics`);
}

/**
 * Search questions by text content
 */
function searchQuestions() {
    const searchTerm = document.getElementById('question-search')?.value?.toLowerCase() || '';
    AppLogger.debug('REVIEW', 'Searching questions', { searchTerm });
    
    if (pythonBridge && pythonBridge.searchQuestions) {
        try {
            const result = pythonBridge.searchQuestions(searchTerm);
            const data = JSON.parse(result);
            
            if (data.success) {
                displayQuestionHistory(data.questions || []);
                AppLogger.success('REVIEW', `Found ${data.questions?.length || 0} questions matching: ${searchTerm}`);
            }
        } catch (e) {
            AppLogger.error('REVIEW', 'Error searching questions', { error: e.message });
        }
    }
}

/**
 * Filter questions by topic
 */
function filterQuestionsByTopic() {
    const topic = document.getElementById('topic-filter')?.value || '';
    AppLogger.debug('REVIEW', 'Filtering questions by topic', { topic });
    
    if (pythonBridge && pythonBridge.filterQuestionsByTopic) {
        try {
            const result = pythonBridge.filterQuestionsByTopic(topic);
            const data = JSON.parse(result);
            
            if (data.success) {
                displayQuestionHistory(data.questions || []);
                AppLogger.success('REVIEW', `Filtered by topic: ${topic || 'All'}, found ${data.questions?.length || 0} questions`);
            }
        } catch (e) {
            AppLogger.error('REVIEW', 'Error filtering questions by topic', { error: e.message });
        }
    }
}

/**
 * Filter questions by difficulty
 */
function filterQuestionsByDifficulty() {
    const difficulty = document.getElementById('difficulty-filter')?.value || '';
    AppLogger.debug('REVIEW', 'Filtering questions by difficulty', { difficulty });
    
    if (pythonBridge && pythonBridge.filterQuestionsByDifficulty) {
        try {
            const result = pythonBridge.filterQuestionsByDifficulty(difficulty);
            const data = JSON.parse(result);
            
            if (data.success) {
                displayQuestionHistory(data.questions || []);
                AppLogger.success('REVIEW', `Filtered by difficulty: ${difficulty || 'All'}, found ${data.questions?.length || 0} questions`);
            }
        } catch (e) {
            AppLogger.error('REVIEW', 'Error filtering questions by difficulty', { error: e.message });
        }
    }
}

/**
 * Show question statistics
 */
function showQuestionStats() {
    AppLogger.info('REVIEW', 'Loading question statistics');
    
    if (pythonBridge && pythonBridge.getQuestionStatistics) {
        try {
            const result = pythonBridge.getQuestionStatistics();
            const data = JSON.parse(result);
            
            if (data.success) {
                displayQuestionStats(data);
                
                // Show stats section
                const statsElement = document.getElementById('question-stats');
                if (statsElement) {
                    statsElement.style.display = 'block';
                }
            }
        } catch (e) {
            AppLogger.error('REVIEW', 'Error loading question statistics', { error: e.message });
        }
    }
}

/**
 * Display question statistics
 */
function displayQuestionStats(data) {
    // Update total questions
    const totalElement = document.getElementById('total-questions-stat');
    if (totalElement) {
        totalElement.textContent = data.total_questions || 0;
    }
    
    // Update topics count
    const topicsElement = document.getElementById('topics-count-stat');
    if (topicsElement) {
        const topicsCount = Object.keys(data.topics || {}).length;
        topicsElement.textContent = topicsCount;
    }
    
    // Update expert questions count
    const expertElement = document.getElementById('expert-questions-stat');
    if (expertElement) {
        const expertCount = data.difficulties?.expert || 0;
        expertElement.textContent = expertCount;
    }
    
    AppLogger.success('REVIEW', 'Question statistics displayed');
}

/**
 * Show question detail modal
 */
function showQuestionModal(question) {
    const modal = document.getElementById('question-detail-modal');
    const modalContent = document.getElementById('modal-question-content');
    
    if (!modal || !modalContent) return;
    
    // Build modal content
    modalContent.innerHTML = `
        <div class="question-detail">
            <div class="question-meta-header" style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                <span class="badge" style="background: #e9ecef; padding: 8px 12px; border-radius: 6px;">
                    📚 ${question.topic || 'General'}
                </span>
                <span class="badge" style="background: #007bff; color: white; padding: 8px 12px; border-radius: 6px;">
                    ⭐ ${question.difficulty || 'Medium'}
                </span>
            </div>
            
            <div class="question-text" style="font-size: 18px; font-weight: bold; margin-bottom: 20px; line-height: 1.5;">
                ${question.question || 'No question text'}
            </div>
            
            <div class="question-options" style="margin-bottom: 20px;">
                <h4 style="margin-bottom: 10px;">Answer Options:</h4>
                ${(question.options || []).map((option, i) => `
                    <div class="option-item" style="
                        padding: 12px;
                        margin: 8px 0;
                        border-radius: 6px;
                        border: 2px solid ${i === question.correct ? '#28a745' : '#e9ecef'};
                        background: ${i === question.correct ? '#d4edda' : '#f8f9fa'};
                        font-weight: ${i === question.correct ? 'bold' : 'normal'};
                    ">
                        <strong>${String.fromCharCode(65 + i)}.</strong> ${option}
                        ${i === question.correct ? ' ✅ <em>(Correct Answer)</em>' : ''}
                    </div>
                `).join('')}
            </div>
            
            ${question.explanation ? `
                <div class="question-explanation" style="
                    background: #f0f8ff;
                    border-left: 4px solid #007bff;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 4px;
                ">
                    <h4 style="margin-bottom: 10px; color: #007bff;">💡 Explanation:</h4>
                    <p style="margin: 0; line-height: 1.6;">${question.explanation}</p>
                </div>
            ` : ''}
            
            <div class="question-footer" style="
                margin-top: 20px;
                padding-top: 15px;
                border-top: 1px solid #e9ecef;
                font-size: 14px;
                color: #666;
            ">
                <div style="display: flex; justify-content: space-between;">
                    <span>Question ID: ${question.id || 'Unknown'}</span>
                    <span>Generated: ${question.timestamp || 'Unknown time'}</span>
                </div>
            </div>
        </div>
    `;
    
    // Show modal
    modal.style.display = 'block';
    
    AppLogger.action('REVIEW', 'Question modal opened', { questionId: question.id });
}

/**
 * Close question detail modal
 */
function closeQuestionModal() {
    const modal = document.getElementById('question-detail-modal');
    if (modal) {
        modal.style.display = 'none';
    }
    
    AppLogger.action('REVIEW', 'Question modal closed');
}

/**
 * Nuclear option: Force load questions for debugging
 */
function nuclearLoadQuestions() {
    AppLogger.warn('REVIEW', 'NUCLEAR: Force loading questions');
    
    const debugElement = document.getElementById('nuclear-debug');
    if (debugElement) {
        debugElement.innerHTML = '🚀 NUCLEAR: Force loading questions...<br>';
    }
    
    if (pythonBridge) {
        try {
            // Test bridge connection
            const testResult = pythonBridge.testConnection('nuclear test');
            if (debugElement) {
                debugElement.innerHTML += `✅ Bridge test: ${testResult}<br>`;
            }
            
            // Try to get questions
            const historyResult = pythonBridge.getQuestionHistory(0, 100);
            const data = JSON.parse(historyResult);
            
            if (debugElement) {
                debugElement.innerHTML += `📊 History result: ${JSON.stringify(data, null, 2)}<br>`;
            }
            
            if (data.success && data.questions) {
                displayQuestionHistory(data.questions);
                if (debugElement) {
                    debugElement.innerHTML += `✅ Successfully loaded ${data.questions.length} questions!<br>`;
                }
            }
            
        } catch (e) {
            if (debugElement) {
                debugElement.innerHTML += `❌ Nuclear error: ${e.message}<br>`;
            }
            AppLogger.error('REVIEW', 'Nuclear load failed', { error: e.message });
        }
    } else {
        if (debugElement) {
            debugElement.innerHTML += '❌ Python bridge not available<br>';
        }
    }
}

/**
 * Nuclear option: Test display with mock data
 */
function nuclearTestDisplay() {
    AppLogger.warn('REVIEW', 'NUCLEAR: Testing display with mock data');
    
    const debugElement = document.getElementById('nuclear-debug');
    if (debugElement) {
        debugElement.innerHTML = '🧪 NUCLEAR: Testing display with mock data...<br>';
    }
    
    // Create mock questions
    const mockQuestions = [
        {
            id: 1,
            question: "What is the capital of France?",
            options: ["London", "Berlin", "Paris", "Madrid"],
            correct: 2,
            topic: "Geography",
            difficulty: "Easy",
            timestamp: new Date().toISOString(),
            explanation: "Paris is the capital and largest city of France."
        },
        {
            id: 2,
            question: "What is 2 + 2?",
            options: ["3", "4", "5", "6"],
            correct: 1,
            topic: "Mathematics",
            difficulty: "Easy",
            timestamp: new Date().toISOString(),
            explanation: "Basic arithmetic: 2 + 2 = 4"
        },
        {
            id: 3,
            question: "What is the chemical formula for water?",
            options: ["H2O", "CO2", "NaCl", "O2"],
            correct: 0,
            topic: "Chemistry",
            difficulty: "Medium",
            timestamp: new Date().toISOString(),
            explanation: "Water is composed of two hydrogen atoms and one oxygen atom: H2O"
        }
    ];
    
    // Display mock questions
    displayQuestionHistory(mockQuestions);
    updateTopicFilter(mockQuestions);
    
    // Hide loading/no questions elements
    const loadingElement = document.getElementById('loading-history');
    const noQuestionsElement = document.getElementById('no-questions');
    const questionsListElement = document.getElementById('questions-list');
    
    if (loadingElement) loadingElement.style.display = 'none';
    if (noQuestionsElement) noQuestionsElement.style.display = 'none';
    if (questionsListElement) questionsListElement.style.display = 'block';
    
    if (debugElement) {
        debugElement.innerHTML += `✅ Successfully displayed ${mockQuestions.length} mock questions!<br>`;
    }
    
    AppLogger.success('REVIEW', 'Nuclear test display completed', { mockQuestionsCount: mockQuestions.length });
}

// Make functions globally available
window.loadQuestionHistory = loadQuestionHistory;
window.searchQuestions = searchQuestions;
window.filterQuestionsByTopic = filterQuestionsByTopic;
window.filterQuestionsByDifficulty = filterQuestionsByDifficulty;
window.showQuestionStats = showQuestionStats;
window.closeQuestionModal = closeQuestionModal;
window.nuclearLoadQuestions = nuclearLoadQuestions;
window.nuclearTestDisplay = nuclearTestDisplay;
