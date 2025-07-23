// Global variables
let pythonBridge = null;
let currentScreen = 'home';
let selectedAnswer = -1;
let quizTimer = null;
let timeRemaining = 30;
let uploadedFiles = [];
let currentQuestionState = null;
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

// Initialize the app when QWebChannel is ready
new QWebChannel(qt.webChannelTransport, function(channel) {
    pythonBridge = channel.objects.pythonBridge;
    console.log("Python bridge initialized");
    
    // CRITICAL FIX: Validate bridge object before proceeding
    if (!pythonBridge) {
        console.error('❌ Failed to get pythonBridge from channel.objects');
        console.log('Available objects:', Object.keys(channel.objects));
        return;
    }

    // Connect Python signals to JavaScript handlers
    try {
        pythonBridge.questionReceived.connect(handleQuestionReceived);
        pythonBridge.answerFeedback.connect(handleAnswerFeedback);
        pythonBridge.quizCompleted.connect(handleQuizCompleted);
        pythonBridge.errorOccurred.connect(handleError);
        pythonBridge.updateStatus.connect(updateStatus);

        pythonBridge.topicProfileUpdated.connect(handleTopicProfileUpdated);

        if (pythonBridge.trainingProgressStructured) {
            pythonBridge.trainingProgressStructured.connect(handleTrainingProgressStructured);
        }
        if (pythonBridge.trainingStatusChanged) {
            pythonBridge.trainingStatusChanged.connect(handleTrainingStatusChanged);
        }
    if (pythonBridge.trainingMetricsUpdate) {
        pythonBridge.trainingMetricsUpdate.connect(handleTrainingMetricsUpdate);
    }
    if (pythonBridge.trainingConfigSaved) {
        pythonBridge.trainingConfigSaved.connect(handleTrainingConfigSaved);
    }

    if (pythonBridge.tokenReceived) {
        pythonBridge.tokenReceived.connect(handleTokenReceived);
    }
    if (pythonBridge.streamingStarted) {
        pythonBridge.streamingStarted.connect(handleStreamingStarted);
    }
    if (pythonBridge.streamingCompleted) {
        pythonBridge.streamingCompleted.connect(handleStreamingCompleted);
    }

    console.log("✅ Python signals connected to JavaScript handlers");
    
    } catch (error) {
        console.error('❌ Failed to connect Python bridge signals:', error);
        return;
    }
    
    // Small delay to ensure bridge is fully established before initializing app
    setTimeout(() => {
        initializeApp();
    }, 100);
});

// 🚀 CRITICAL FIX: Add missing error handling functions
function showError(message) {
    console.error('❌ Error:', message);

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
    console.log('🔄 Loading:', message);

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
    const loadingContainer = document.getElementById('loading-container');
    if (loadingContainer) {
        loadingContainer.style.display = 'none';
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
    console.log('🚀 Starting async Knowledge App initialization...');
    
    // CRITICAL FIX: Test bridge connectivity first
    if (!pythonBridge) {
        console.error('❌ Python bridge is null during initialization');
        showError('Failed to initialize app - bridge not available');
        return;
    }
    
    // Test bridge with the new test method
    try {
        console.log('🔗 Testing bridge connectivity...');
        if (typeof pythonBridge.testBridgeConnection === 'function') {
            const testResult = await pythonBridge.testBridgeConnection();
            const result = JSON.parse(testResult);
            if (result.success) {
                console.log('✅ Bridge connectivity test passed');
            } else {
                console.error('❌ Bridge connectivity test failed:', result.message);
                showError('Bridge connectivity test failed: ' + result.message);
                return;
            }
        } else {
            console.warn('⚠️ Bridge test method not available, continuing with initialization...');
        }
    } catch (error) {
        console.error('❌ Bridge connectivity test error:', error);
        showError('Bridge connectivity test error: ' + error.message);
        return;
    }
    
    // Show splash screen immediately
    showSplashScreen();
    
    try {
        // Check if we're in a reload scenario
        const isReload = performance.navigation.type === performance.navigation.TYPE_RELOAD;
        if (isReload) {
            console.log('🔄 Page reload detected - attempting API key recovery...');
            await loadApiKeysFromSessionStorage();
        }
        
        // Stage 1: Load settings asynchronously
        console.log('📋 Stage 1: Loading settings...');
        if (typeof loadSettings === 'function') {
            await loadSettings();
        }
        
        // Stage 2: Load files asynchronously
        console.log('📁 Stage 2: Loading existing files...');
        await loadExistingFiles();
        
        // Stage 3: Setup auto-save and persistence
        console.log('🔧 Stage 3: Setting up auto-save and persistence...');
        
        // Set up auto-save listeners
        if (typeof setupAutoSave === 'function') {
            setupAutoSave();
        }
        
        // Set up intelligent topic analysis
        if (typeof setupTopicAnalysis === 'function') {
            setupTopicAnalysis();
        }
        
        // Ensure API key persistence
        await ensureApiKeyPersistence();
        
        // Update status indicators
        updateApiKeyStatusIndicators();
        
        // Stage 4: Verify settings and apply fallbacks
        console.log('🎯 Stage 4: Verifying settings...');
        
        // Verify critical settings loaded correctly
        const difficultySelect = document.getElementById('quiz-difficulty');
        const submodeSelect = document.getElementById('quiz-submode');
        const gameModeSelect = document.getElementById('quiz-game-mode');
        
        if (difficultySelect && submodeSelect && gameModeSelect) {
            console.log(`🔍 Current settings: difficulty=${difficultySelect.value}, submode=${submodeSelect.value}, gameMode=${gameModeSelect.value}`);
            
            // Update info displays
            if (typeof updateModeInfo === 'function') updateModeInfo();
            if (typeof updateGameModeInfo === 'function') updateGameModeInfo();
            if (typeof updateSubmodeInfo === 'function') updateSubmodeInfo();
            if (typeof updateDifficultyInfo === 'function') updateDifficultyInfo();
        }
        
        // Stage 5: Final verification and persistence
        console.log('💾 Stage 5: Final verification and persistence...');
        
        // Force apply saved settings if needed
        const savedSettings = localStorage.getItem('userSettings');
        if (savedSettings) {
            try {
                const settings = JSON.parse(savedSettings);
                console.log('🛠️ Applying localStorage settings...');
                
                // Apply settings
                if (settings.default_difficulty && difficultySelect) {
                    difficultySelect.value = settings.default_difficulty;
                }
                if (settings.default_submode && submodeSelect) {
                    submodeSelect.value = settings.default_submode;
                }
                if (settings.default_game_mode && gameModeSelect) {
                    gameModeSelect.value = settings.default_game_mode;
                }
                
                // Update displays
                updateModeInfo();
                updateGameModeInfo();
                updateSubmodeInfo();
                updateDifficultyInfo();
                
            } catch (e) {
                console.error('❌ Failed to parse localStorage settings:', e);
            }
        }
        
        // Save current state
        if (typeof saveSettings === 'function') {
            const success = await saveSettings();
            if (success) {
                console.log('💾 Settings successfully persisted on startup');
                await saveApiKeysToSessionStorage();
            }
        }
        
        // Initialize DeepSeek integration
        console.log('🧠 Initializing DeepSeek integration...');
        await initializeDeepSeek();
        
        console.log('✅ All initialization complete!');
        
    } catch (error) {
        console.error('❌ Initialization error:', error);
        showError('Failed to initialize app: ' + error.message);
    } finally {
        // Always hide splash screen after initialization
        hideSplashScreen();
    }
}

function showSplashScreen() {
    let splash = document.getElementById('splash-screen');
    
    if (!splash) {
        splash = document.createElement('div');
        splash.id = 'splash-screen';
        splash.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(31, 41, 55, 0.95));
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 100000;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #a5b4fc;
            text-align: center;
            box-shadow: 0 0 60px rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(12px);
        `;
        
        // Create spinner
        const spinner = document.createElement('div');
        spinner.style.cssText = `
            width: 80px;
            height: 80px;
            border: 10px solid rgba(165, 180, 252, 0.3);
            border-top: 10px solid #6366f1;
            border-radius: 50%;
            animation: spin 1.2s linear infinite;
            margin-bottom: 40px;
        `;
        
        // Create text container
        const textContainer = document.createElement('div');
        textContainer.style.cssText = `
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
            max-width: 360px;
            padding: 0 30px;
        `;
        
        const title = document.createElement('h1');
        title.textContent = 'Knowledge App';
        title.style.cssText = `
            font-size: 4rem;
            font-weight: 900;
            margin: 0;
            letter-spacing: 3px;
            text-shadow: 0 6px 16px rgba(99, 102, 241, 0.8);
            color: #818cf8;
        `;
        
        const subtitle = document.createElement('p');
        subtitle.textContent = 'Modern Learning Platform';
        subtitle.style.cssText = `
            font-size: 1.75rem;
            margin: 0;
            color: #c7d2fe;
            font-weight: 600;
        `;
        
        const status = document.createElement('div');
        status.id = 'splash-status';
        status.textContent = 'Loading components...';
        status.style.cssText = `
            font-size: 1.25rem;
            margin-top: 25px;
            color: #a5b4fc;
            font-weight: 700;
            font-style: italic;
            text-shadow: 0 0 8px rgba(165, 180, 252, 0.7);
        `;
        
        textContainer.appendChild(title);
        textContainer.appendChild(subtitle);
        textContainer.appendChild(status);
        
        splash.appendChild(spinner);
        splash.appendChild(textContainer);
        
        // Add CSS animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: scale(0.95); }
                to { opacity: 1; transform: scale(1); }
            }
            #splash-screen {
                animation: fadeIn 0.8s ease-out;
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(splash);
    }
    
    splash.style.display = 'flex';
}

function hideSplashScreen() {
    console.log('🚀 Hiding all loading screens...');
    
    // Hide JavaScript splash screen
    const splash = document.getElementById('splash-screen');
    if (splash) {
        splash.style.display = 'none';
        splash.remove(); // Completely remove from DOM
        console.log('✅ JavaScript splash screen removed');
    }
    
    // ✅ CRITICAL FIX: Remove ALL loading overlays
    const loadingSelectors = [
        '[id*="loading"]',
        '[class*="loading"]', 
        '[id*="splash"]',
        '[class*="splash"]',
        '.loading-container',
        '.splash-container'
    ];
    
    loadingSelectors.forEach(selector => {
        const elements = document.querySelectorAll(selector);
        elements.forEach(el => {
            el.style.display = 'none';
            el.style.visibility = 'hidden';
            el.style.opacity = '0';
            el.style.zIndex = '-1';
            console.log('✅ Removed loading element:', el.id || el.className);
        });
    });
    
    // Clear any background styles that might be loading screens
    document.body.style.background = '';
    document.body.style.backgroundImage = '';
    
    // Force show main app content
    const mainElements = document.querySelectorAll('#app, main, .app-container, .screen');
    mainElements.forEach(el => {
        el.style.display = '';
        el.style.visibility = 'visible';
        el.style.opacity = '1';
        el.style.zIndex = '';
    });
    
    console.log('🎉 All loading screens cleared - app is now fully visible!');
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
    const topicInput = document.getElementById('quiz-topic');
    const difficultySelect = document.getElementById('quiz-difficulty');
    const submodeSelect = document.getElementById('quiz-submode');

    if (!topicInput || !difficultySelect || !submodeSelect) {
        showError('Topic, difficulty, and question type inputs not found');
        return;
    }

    const topic = topicInput.value.trim();
    const difficulty = difficultySelect.value;
    const questionType = submodeSelect.value;

    if (!topic) {
        showError('Please enter a topic first');
        return;
    }

    // Show DeepSeek generation status
    showStatusDisplay(`🧠 DeepSeek AI thinking deeply about ${topic} (${questionType})...`, 'deepseek');

    console.log(`🧠 Generating DeepSeek question: ${topic} (${difficulty}) - Type: ${questionType}`);

    if (pythonBridge && pythonBridge.generateDeepSeekQuestion) {
        pythonBridge.generateDeepSeekQuestion(topic, difficulty, questionType).then(resultJson => {
            try {
                const result = JSON.parse(resultJson);

                if (result.success && result.question) {
                    // Display the generated question
                    displayDeepSeekQuestion(result.question);
                    console.log('✅ DeepSeek question generated successfully');
                } else {
                    console.log('⚠️ DeepSeek generation failed, using fallback');
                    if (result.fallback) {
                        // Fall back to regular quiz generation
                        showStatusDisplay('⚠️ DeepSeek unavailable, using standard generation...', 'warning');
                        // Remove expert difficulty temporarily for fallback
                        const originalDifficulty = difficulty;
                        difficultySelect.value = 'hard';
                        startCustomQuiz();
                        // Restore original difficulty
                        difficultySelect.value = originalDifficulty;
                    } else {
                        showError(result.error || 'DeepSeek generation failed');
                    }
                }
            } catch (e) {
                console.error('❌ Failed to parse DeepSeek result:', e);
                showError('Failed to parse DeepSeek response');
            }
        }).catch(error => {
            console.error('❌ DeepSeek generation error:', error);
            showError('DeepSeek generation failed');
        });
    } else {
        console.error('❌ DeepSeek generation method not available');
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

        // 🛡️ CRITICAL UX FIX #16: Ask for user consent instead of forcefully changing UI
        // The old behavior hijacked user selections without consent, which was confusing
        if (shouldAutoSelect && newValue !== currentValue) {
            // Show a non-intrusive suggestion instead of forcing the change
            showQuestionTypeSuggestion(profile, newValue, currentValue);
        } else if ((currentValue === 'numerical' && shouldDisableNumerical) ||
                   (currentValue === 'mixed' && shouldDisableMixed)) {
            // Only force change when current selection becomes invalid
            submodeSelect.value = 'conceptual';
            updateSubmodeInfo();
            showTopicAnalysisFeedback(profile, 'forced_fallback');
            console.log(`🔄 Forced fallback to conceptual (current selection invalid)`);
        }
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
    // Hide all screens
    document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.remove('active');
    });
    
    // Show selected screen
    document.getElementById(`${screenName}-screen`).classList.add('active');
    
    // Update navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // If called from navigation, update the active nav item
    if (navElement) {
        navElement.closest('.nav-item').classList.add('active');
    }
    
    // If showing quiz screen, ensure setup is visible
    if (screenName === 'quiz') {
        document.getElementById('quiz-setup').style.display = 'block';
        document.getElementById('quiz-game').style.display = 'none';
        document.getElementById('quiz-results').style.display = 'none';
        resetQuizState();
    }
    
    // If showing settings screen, load settings
    if (screenName === 'settings') {
        loadSettings();
        updateProviderStatuses();
    }
    
    // If showing home screen, update stats
    if (screenName === 'home') {
        updateStats();
    }
    
    // If showing train screen, load existing uploaded files and training configuration
    if (screenName === 'train') {
        loadExistingFiles();
        // Only load training configuration when user actually navigates to train screen
        // and ensure Python bridge is ready
        setTimeout(async () => {
            if (pythonBridge && typeof pythonBridge.getTrainingConfiguration === 'function') {
                await loadTrainingConfiguration();
            } else {
                console.log('🔄 Python bridge not ready for training configuration, will retry...');
                // Retry after a short delay
                setTimeout(async () => {
                    if (pythonBridge && typeof pythonBridge.getTrainingConfiguration === 'function') {
                        await loadTrainingConfiguration();
                    }
                }, 1000);
            }
        }, 100);
    }
    
    // If showing review screen, load question history
    if (screenName === 'review') {
        console.log('🔧 Review screen activated in showScreen, loading question history...');
        
        // DEBUGGING: Test if bridge works at all
        console.log('🔧 Testing pythonBridge availability...');
        if (pythonBridge) {
            console.log('🔧 pythonBridge exists, testing getQuestionHistory...');
            try {
                const testResult = pythonBridge.getQuestionHistory();
                console.log('🔧 Test result:', testResult);
                console.log('🔧 Test result type:', typeof testResult);
            } catch (e) {
                console.error('🔧 Test bridge call failed:', e);
            }
        } else {
            console.error('🔧 pythonBridge does not exist!');
        }
        
        setTimeout(async () => {
            console.log('🔧 About to call loadQuestionHistory...');
            await loadQuestionHistory();
        }, 100);
    }
    
    currentScreen = screenName;
    // CRITICAL FIX: Check if pythonBridge exists before calling
    if (pythonBridge && pythonBridge.navigate) {
        pythonBridge.navigate(screenName);
    } else {
        console.warn('⚠️ Python bridge not available for navigation');
    }
}

// Theme toggle
function toggleTheme() {
    const currentTheme = document.body.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update toggle icon
    document.querySelector('.theme-toggle').textContent = newTheme === 'dark' ? '☀️' : '🌙';
    
    // Auto-save theme preference
    console.log(`🎨 Auto-saving theme change to: ${newTheme}`);
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
    // Just navigate to quiz screen - DON'T bypass the setup!
    showScreen('quiz', null);
    // Let user configure their quiz settings
}

function startCustomQuiz() {
    const topic = document.getElementById('quiz-topic').value || "General Knowledge";
    const mode = document.getElementById('quiz-mode').value;
    const gameMode = document.getElementById('quiz-game-mode').value;
    const submode = document.getElementById('quiz-submode').value;
    const difficulty = document.getElementById('quiz-difficulty').value;
    const numQuestions = parseInt(document.getElementById('quiz-questions').value);

    // 🧠 FIXED: Expert mode should start a proper quiz, not just generate one question
    console.log(`🚀 Starting ${difficulty} quiz: ${numQuestions} questions about "${topic}"`);

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
    console.log(`🌊 Token streaming enabled: ${useStreaming}`);

    if (useStreaming) {
        // Use new streaming functionality for live generation
        console.log('🌊 Starting streaming quiz generation...');
        if (pythonBridge && pythonBridge.generateQuestionStreaming) {
            pythonBridge.generateQuestionStreaming(topic, difficulty);
            showQuizGame(); // Show quiz interface
            return; // Streaming handles the rest
        } else {
            console.warn('⚠️ Streaming not available, falling back to regular generation');
        }
    }

    // Show appropriate generation status based on mode
    const questionTypeLabel = getQuestionTypeLabel(gameMode, submode);
    if (mode === 'offline') {
        showStatusDisplay(`🎮 Starting LOCAL GPU generation for ${topic} (${questionTypeLabel})...`, 'gpu');
    } else {
        showStatusDisplay(`🌐 Connecting to AI APIs for ${topic} (${questionTypeLabel})...`, 'api');
    }

    // 🚀 CRITICAL FIX: Use non-blocking quiz start to prevent UI freezing
    if (pythonBridge && pythonBridge.startQuiz) {
        console.log('🚀 Starting quiz with non-blocking generation...');

        // Show loading immediately
        showQuizLoading('🧠 Generating your first question...');

        // Show quiz interface immediately to prevent UI blocking
        showQuizGame();

        // Start generation in background (non-blocking)
        try {
            // 🌊 ENHANCED: Start token streaming if enabled
            if (shouldUseTokenStreaming(mode, isTokenStreamingEnabled())) {
                console.log('🌊 Starting token streaming visualization...');
                createTokenStreamUI(topic, difficulty, submode);
                startTokenStreamingSimulation(topic, difficulty, submode);
            }

            pythonBridge.startQuiz(JSON.stringify(params));
            console.log('✅ Quiz generation started in background - UI remains responsive');
        } catch (error) {
            console.error('❌ Failed to start quiz:', error);
            hideQuizLoading();
            showError('Failed to start quiz: ' + error.message);
        }
    } else {
        console.error('❌ Python bridge not available for starting quiz');
        showError('Cannot start quiz - Python bridge not available');
        return;
    }

    // Start status monitoring with proper mode
    startStatusMonitoring(mode);
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
    
    // FIXED: Allow selection in review mode for better UX
    // The original code had: if (isReviewMode) return;
    // This was preventing users from selecting options in review mode
    
    // Remove previous selection
    document.querySelectorAll('.option-button').forEach((btn, i) => {
        btn.classList.remove('selected');
    });
    
    // Add selection to clicked button
    document.querySelectorAll('.option-button')[index].classList.add('selected');
    selectedAnswer = index;
    
    console.log('🎯 Answer selected:', index, isReviewMode ? '(in review mode)' : '');
    
    // If in review mode, show feedback immediately when an option is selected
    if (isReviewMode && currentQuestionState) {
        const isCorrect = index === currentQuestionState.correct_index;
        
        // Highlight the selected answer
        document.querySelectorAll('.option-button')[index].classList.add(
            isCorrect ? 'correct-answer' : 'incorrect-answer'
        );
        
        // If incorrect, also highlight the correct answer
        if (!isCorrect && currentQuestionState.correct_index !== undefined) {
            document.querySelectorAll('.option-button')[currentQuestionState.correct_index].classList.add('correct-answer');
        }
        
        console.log('📝 Review mode feedback shown for selection');
    }
}

function submitAnswer() {
    if (selectedAnswer === -1) {
        alert('Please select an answer!');
        return;
    }
    
    clearInterval(quizTimer);
    
    // Show processing status
    showStatusDisplay('⚡ Processing your answer...', 'info');
    
    // Disable option buttons to prevent further selection
    document.querySelectorAll('.option-button').forEach(btn => {
        btn.disabled = true;
    });
    
    // Hide submit button
    const submitBtn = document.getElementById('submit-btn');
    if (submitBtn) submitBtn.style.display = 'none';
    
    // 🚀 CRITICAL FIX: Enhanced error handling for submit
    try {
        console.log('🎯 Submit answer called with selectedAnswer:', selectedAnswer);

        if (pythonBridge && pythonBridge.submitAnswer) {
            console.log('📤 Calling pythonBridge.submitAnswer with:', selectedAnswer);
            pythonBridge.submitAnswer(selectedAnswer);
            console.log('✅ Submit answer call completed');
        } else {
            console.error('❌ Python bridge not available for submitting answer');
            hideStatusDisplay();
            showError('Cannot submit answer - Python bridge not available');
        }

    } catch (error) {
        console.error('❌ Critical error in submitAnswer:', error);
        hideStatusDisplay();
        showError('Submit answer failed: ' + error.message);
    }
}

function handleAnswerFeedback(feedbackData) {
    try {
        console.log('📨 Answer feedback received:', feedbackData);

        // 🚀 CRITICAL FIX: Validate feedback data
        if (!feedbackData || typeof feedbackData !== 'object') {
            console.error('❌ Invalid feedback data:', feedbackData);
            showError('Invalid feedback data received');
            return;
        }

        // Hide processing status
        hideStatusDisplay();

        showAnswerFeedback(feedbackData);
        showNavigationButtons();

        // Show brief success message
        showStatusDisplay('✅ Answer processed!', 'success');
        setTimeout(hideStatusDisplay, 1500);

        console.log('✅ Answer feedback handled successfully');

    } catch (error) {
        console.error('❌ Critical error in handleAnswerFeedback:', error);
        hideStatusDisplay();
        showError('Failed to process answer feedback: ' + error.message);
    }
}

function showAnswerFeedback(feedbackData) {
    // Highlight options
    document.querySelectorAll('.option-button').forEach((btn, index) => {
        btn.classList.remove('correct-answer', 'incorrect-answer', 'selected');
        
        if (index === feedbackData.user_answer) {
            btn.classList.add(feedbackData.is_correct ? 'correct-answer' : 'incorrect-answer');
        }
        
        if (index === feedbackData.correct_index && !feedbackData.is_correct) {
            btn.classList.add('correct-answer');
        }
    });
    
    // Show feedback message
    const feedbackContainer = document.getElementById('feedback-container');
    if (feedbackContainer) {
        feedbackContainer.innerHTML = `
            <div class="feedback-message ${feedbackData.is_correct ? 'correct' : 'incorrect'}">
                ${feedbackData.feedback_message}
            </div>
        `;
        feedbackContainer.style.display = 'block';
    }
    
    // Show explanation
    const explanationContainer = document.getElementById('explanation-container');
    if (explanationContainer) {
        if (feedbackData.explanation && feedbackData.explanation.trim() !== '' && 
            feedbackData.explanation !== 'No explanation available.') {
            updateExplanationWithLatex(feedbackData.explanation, explanationContainer);
        } else {
            // Hide explanation container if no valid explanation
            explanationContainer.style.display = 'none';
        }
    }
}

function showNavigationButtons() {
    const navigationContainer = document.getElementById('navigation-buttons');
    if (navigationContainer) {
        navigationContainer.style.display = 'flex';
    }
}

function skipQuestion() {
    clearInterval(quizTimer);
    showStatusDisplay('⚡ Skipping question...', 'warning');
    // CRITICAL FIX: Check if pythonBridge exists before calling
    if (pythonBridge && pythonBridge.submitAnswer) {
        pythonBridge.submitAnswer(-1);
    } else {
        console.error('❌ Python bridge not available for skipping question');
        showTemporaryMessage('❌ Cannot skip question - Python bridge not available', 'error');
    }
}

function startTimer() {
    timeRemaining = 30;
    clearInterval(quizTimer);
    
    quizTimer = setInterval(() => {
        timeRemaining--;
        document.getElementById('quiz-timer').textContent = `Time: ${timeRemaining}s`;
        
        if (timeRemaining <= 0) {
            clearInterval(quizTimer);
            submitAnswer();
        }
    }, 1000);
}

// Navigation functions - INSTANT since questions are pre-generated
function showPreviousQuestion() {
    // CRITICAL FIX: Check if pythonBridge exists before calling
    if (pythonBridge && pythonBridge.showPreviousQuestion) {
        pythonBridge.showPreviousQuestion(); // INSTANT - no loading delay
    } else {
        console.error('❌ Python bridge not available for navigation');
        showTemporaryMessage('❌ Cannot navigate - Python bridge not available', 'error');
    }
}

function showNextQuestion() {
    // CRITICAL FIX: Check if pythonBridge exists before calling
    if (pythonBridge && pythonBridge.showNextQuestion) {
        pythonBridge.showNextQuestion(); // INSTANT - no loading delay
    } else {
        console.error('❌ Python bridge not available for navigation');
        showTemporaryMessage('❌ Cannot navigate - Python bridge not available', 'error');
    }
}

function loadNextNewQuestion() {
    // CRITICAL FIX: Check if pythonBridge exists before calling
    if (pythonBridge && pythonBridge.loadNextQuestion) {
        pythonBridge.loadNextQuestion(); // INSTANT - no loading delay
    } else {
        console.error('❌ Python bridge not available for loading questions');
        showTemporaryMessage('❌ Cannot load questions - Python bridge not available', 'error');
    }
}

function handleQuizCompleted(results) {
    console.log('Quiz completed:', results);
    
    // Hide any status displays
    hideStatusDisplay();
    clearInterval(statusUpdateInterval);
    
    // Ensure results exist
    if (!results) {
        console.error('No results received');
        return;
    }
    
    // Hide game, show results
    const quizGame = document.getElementById('quiz-game');
    const quizResults = document.getElementById('quiz-results');
    
    if (quizGame) quizGame.style.display = 'none';
    if (quizResults) quizResults.style.display = 'block';
    
    // Update results
    const scorePercentage = document.getElementById('score-percentage');
    const scoreText = document.getElementById('score-text');
    
    if (scorePercentage) {
        scorePercentage.textContent = `${Math.round(results.percentage || 0)}%`;
    }
    
    if (scoreText) {
        scoreText.textContent = `You scored ${results.score || 0} out of ${results.total || 0}`;
    }
    
    // Update stats (simplified for now)
    updateStats();
}

function resetQuiz() {
    document.getElementById('quiz-setup').style.display = 'block';
    document.getElementById('quiz-game').style.display = 'none';
    document.getElementById('quiz-results').style.display = 'none';
    resetQuizState();
}

// Training functions
function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    
    const uploadArea = event.currentTarget;
    uploadArea.classList.remove('drag-over');
    
    const files = event.dataTransfer.files;
    handleFiles(files);
}

function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    
    event.currentTarget.classList.add('drag-over');
}

function handleFileSelect(event) {
    const files = event.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    // 🔧 FIX: Process files asynchronously to prevent UI freeze
    Array.from(files).forEach(async (file) => {
        if (isValidFile(file)) {
            try {
                // Show upload progress for large files
                if (file.size > 10 * 1024 * 1024) { // 10MB threshold
                    showNotification(`📤 Uploading large file: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)}MB)`, 'info');
                }

                // Upload file to backend with chunked processing
                await uploadFileToBackend(file);

            } catch (error) {
                console.error(`Failed to upload ${file.name}:`, error);
                showNotification(`❌ Failed to upload ${file.name}: ${error.message}`, 'error');
            }
        }
    });

    // Refresh training UI after processing
    setTimeout(() => {
        if (uploadedFiles.length > 0) {
            loadTrainingConfiguration();
        }
    }, 100);
}

function isValidFile(file) {
    const validTypes = ['.pdf', '.txt', '.docx'];
    const extension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

    // 🔧 FIX: Add file size validation to prevent UI freeze
    const maxSize = 100 * 1024 * 1024; // 100MB limit
    if (file.size > maxSize) {
        showNotification(`❌ File too large: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum size: 100MB`, 'error');
        return false;
    }

    return validTypes.includes(extension);
}

// 🔧 FIX: Add chunked file upload function to prevent UI freeze
async function uploadFileToBackend(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = async function(e) {
            try {
                // Convert to base64 (remove data URL prefix)
                const base64Data = e.target.result.split(',')[1];

                if (!pythonBridge || !pythonBridge.uploadFile) {
                    throw new Error('Backend upload not available');
                }

                // Call backend upload method
                const result = await pythonBridge.uploadFile(file.name, base64Data);

                let uploadResult;
                try {
                    uploadResult = JSON.parse(result);
                } catch (parseError) {
                    throw new Error('Invalid response from backend');
                }

                if (uploadResult.success) {
                    // Add to uploaded files list only after successful backend upload
                    uploadedFiles.push({
                        name: file.name,
                        size: file.size,
                        path: uploadResult.file_path
                    });
                    showNotification(`✅ Uploaded: ${file.name}`, 'success');
                    resolve(uploadResult);
                } else {
                    throw new Error(uploadResult.error || 'Upload failed');
                }

            } catch (error) {
                reject(error);
            }
        };

        reader.onerror = function() {
            reject(new Error('Failed to read file'));
        };

        // 🔧 FIX: Use readAsDataURL but with progress tracking for large files
        reader.readAsDataURL(file);
    });
}



function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function removeFile(filename) {
    try {
        console.log('🗑️ Removing file:', filename);

        // Show immediate feedback
        showTemporaryMessage(`🔄 Removing ${filename}...`, 'info');

        // Remove from backend first
        if (pythonBridge && pythonBridge.deleteUploadedFile) {
            const result = await pythonBridge.deleteUploadedFile(filename);

            let deleteResult;
            try {
                deleteResult = JSON.parse(result);
            } catch (e) {
                throw new Error('Invalid response from backend');
            }

            if (deleteResult.success) {
                console.log('✅ File deleted from backend:', deleteResult);
                showTemporaryMessage(`✅ ${filename} removed successfully`, 'success');
            } else {
                console.warn('⚠️ Backend deletion failed:', deleteResult.error);
                showTemporaryMessage(`⚠️ Backend deletion failed: ${deleteResult.error}`, 'warning');
                // Continue with frontend removal even if backend fails
            }
        } else {
            console.warn('⚠️ Backend deletion not available');
        }

        // Remove from frontend list
        const initialCount = uploadedFiles.length;
        uploadedFiles = uploadedFiles.filter(file => file.name !== filename);
        const finalCount = uploadedFiles.length;

        console.log(`📊 File list updated: ${initialCount} -> ${finalCount} files`);

        // Force reload files from backend to ensure UI is in sync
        console.log('🔄 Reloading files from backend to sync UI...');
        await loadExistingFiles();

        // Refresh training UI with debug info
        console.log(`📊 After removal - uploadedFiles count: ${uploadedFiles.length}`);
        console.log('📋 Current files:', uploadedFiles.map(f => f.name));

        // Force refresh the training UI by resetting the loaded flag
        trainingConfigurationLoaded = false;
        loadTrainingConfiguration();

    } catch (error) {
        console.error('❌ File removal failed:', error);
        showTemporaryMessage(`❌ Failed to remove ${filename}: ${error.message}`, 'error');

        // Still try to remove from frontend list as fallback
        uploadedFiles = uploadedFiles.filter(file => file.name !== filename);
        trainingConfigurationLoaded = false;
        loadTrainingConfiguration();
    }
}

async function loadExistingFiles() {
    /**
     * 🔍 Load existing uploaded files from backend and display them in UI
     */
    try {
        // CRITICAL FIX: Add null checks for all pythonBridge.log calls
        if (pythonBridge && pythonBridge.log) {
            pythonBridge.log('📚 Loading existing uploaded files...');
            pythonBridge.log('🖥️ Current active screen: ' + document.querySelector('.screen.active')?.id);
            pythonBridge.log('🎯 Train screen element: ' + document.getElementById('train-screen'));
            pythonBridge.log('📄 File list element: ' + document.getElementById('file-list'));
        }
        
        if (pythonBridge && pythonBridge.getUploadedFiles) {
            // Handle the Promise returned by PyQt WebChannel
            pythonBridge.getUploadedFiles().then(filesData => {
                pythonBridge.log('📋 Received files data type: ' + typeof filesData);
                pythonBridge.log('📋 Received files data (first 200 chars): ' + String(filesData).substring(0, 200));
                pythonBridge.log('📋 Raw filesData object keys: ' + (typeof filesData === 'object' ? Object.keys(filesData) : 'N/A'));
                pythonBridge.log('📋 Contains JSON_STRING prefix? ' + String(filesData).includes('JSON_STRING:'));
                
                if (filesData) {
                try {
                    let existingFiles;
                    
                    // Handle PyQt's automatic JSON conversion
                    if (typeof filesData === 'object' && Array.isArray(filesData)) {
                        // PyQt already converted JSON to JavaScript array - use directly!
                        pythonBridge.log('🔧 PyQt auto-converted JSON to array - using directly');
                        existingFiles = filesData;
                    } else if (typeof filesData === 'string' && filesData.startsWith('JSON_STRING:')) {
                        // Remove the bypass prefix and parse
                        const jsonString = filesData.substring('JSON_STRING:'.length);
                        pythonBridge.log('🔧 Detected JSON_STRING prefix, extracting pure JSON');
                        pythonBridge.log('🔧 JSON string to parse: ' + jsonString.substring(0, 200));
                        existingFiles = JSON.parse(jsonString);
                    } else if (typeof filesData === 'object') {
                        // PyQt returned an object - check if it has array-like properties
                        pythonBridge.log('🔧 Checking if object is array-like...');
                        const keys = Object.keys(filesData);
                        if (keys.every(key => !isNaN(key))) {
                            // Object with numeric keys - convert to array
                            existingFiles = Object.values(filesData);
                            pythonBridge.log('🔧 Converted object with numeric keys to array');
                        } else {
                            pythonBridge.log('❌ Unexpected object format: ' + JSON.stringify(filesData));
                            return;
                        }
                    } else if (typeof filesData === 'string') {
                        // Regular JSON string
                        pythonBridge.log('🔧 Parsing regular JSON string');
                        existingFiles = JSON.parse(filesData);
                    } else {
                        pythonBridge.log('❌ Unexpected data type: ' + typeof filesData);
                        return;
                    }
                    pythonBridge.log('🔍 Parsed result type: ' + typeof existingFiles);
                    pythonBridge.log('🔍 Parsed result is array: ' + Array.isArray(existingFiles));
                    pythonBridge.log('🔍 Parsed result length: ' + (Array.isArray(existingFiles) ? existingFiles.length : 'N/A'));
                    pythonBridge.log('🔍 First item: ' + (existingFiles && existingFiles[0] ? JSON.stringify(existingFiles[0]) : 'N/A'));
                    
                    if (!Array.isArray(existingFiles)) {
                        console.error('❌ Expected array but got:', typeof existingFiles);
                        return;
                    }
                    
                    console.log(`📚 Found ${existingFiles.length} existing files`);

                    // Clear current list and load files
                    uploadedFiles = [];
                    
                    // 🔥 CRITICAL FIX: Load files asynchronously to prevent UI freezing
                    async function loadFilesAsync() {
                        pythonBridge.log(`🚀 Starting async file loading for ${existingFiles.length} files...`);

                        const batchSize = 3; // Process files in small batches
                        const totalFiles = existingFiles.length;

                        for (let i = 0; i < totalFiles; i += batchSize) {
                            const batch = existingFiles.slice(i, i + batchSize);

                            // Process current batch
                            batch.forEach((fileInfo, batchIndex) => {
                                const globalIndex = i + batchIndex;
                                pythonBridge.log(`📁 Loading file ${globalIndex + 1}: ${fileInfo.name}`);
                                const fileObj = {
                                    name: fileInfo.name,
                                    size: fileInfo.size,
                                    path: fileInfo.path
                                };
                                uploadedFiles.push(fileObj);
                            });

                            // Yield control back to UI thread after each batch
                            await new Promise(resolve => setTimeout(resolve, 0));

                            // Update progress
                            const processed = Math.min(i + batchSize, totalFiles);
                            pythonBridge.log(`⚡ Processed ${processed}/${totalFiles} files...`);
                        }

                        pythonBridge.log('🔚 Loaded files. Total: ' + uploadedFiles.length);

                        // Refresh training UI if files exist
                        if (uploadedFiles.length > 0) {
                            loadTrainingConfiguration();
                            pythonBridge.log(`✅ Training button enabled with ${uploadedFiles.length} files`);
                        }

                        pythonBridge.log(`✅ Loaded ${uploadedFiles.length} existing files into UI`);
                    }

                    // Start async loading
                    loadFilesAsync().catch(error => {
                        pythonBridge.log('❌ Error in async file loading: ' + error);
                    });

                    // Refresh training UI after processing
                    setTimeout(() => {
                        if (uploadedFiles.length > 0) {
                            loadTrainingConfiguration();
                        }
                    }, 100);
                } catch (parseError) {
                    pythonBridge.log('❌ Failed to parse existing files data: ' + parseError);
                }
                } else {
                    pythonBridge.log('📝 No existing files data returned from backend');
                }
            }).catch(error => {
                pythonBridge.log('❌ Error getting uploaded files: ' + error);
            });
        } else {
            pythonBridge.log('⚠️ Python bridge not available for loading files');
        }
        
    } catch (error) {
        pythonBridge.log('❌ Error loading existing files: ' + error);
    }
}

// 🚀 Phase 2: Enhanced Training with Complete User Experience
function startTraining() {
    try {
        console.log("🚀 Phase 2: Starting enhanced training setup...");
        if (pythonBridge && pythonBridge.log) {
            pythonBridge.log("🚀 Phase 2: Starting enhanced training setup...");
        }
        
        showScreen('train');
        
        // Training configuration is now loaded when user navigates to train screen
        console.log('🚀 Training setup complete - configuration loaded on screen navigation');
        
    } catch (error) {
        console.error("❌ Training setup failed:", error);
        if (pythonBridge && pythonBridge.log) {
            pythonBridge.log("❌ Training setup failed: " + error);
        }
        showTemporaryMessage('❌ Failed to start training setup: ' + error.message, 'error');
    }
}

// Load and populate training configuration
let trainingConfigurationLoaded = false;
let trainingConfigurationLoading = false;

async function loadTrainingConfiguration() {
    // Prevent multiple rapid calls
    if (trainingConfigurationLoading || trainingConfigurationLoaded) {
        console.log('🔄 Training configuration already loading or loaded, skipping...');
        return;
    }
    
    trainingConfigurationLoading = true;
    
    // CRITICAL FIX: Check if Python bridge and methods are available
    if (!pythonBridge) {
        console.error('❌ Python bridge not available for training configuration');
        trainingConfigurationLoading = false;
        showTrainingError('Python bridge not ready. Please try again in a moment.');
        return;
    }
    
    if (!pythonBridge.getTrainingConfiguration || !pythonBridge.getAvailableBaseModels || !pythonBridge.getTrainingPresets) {
        console.error('❌ Training configuration methods not available');
        trainingConfigurationLoading = false;
        showTrainingError('Training methods not ready. Please try again in a moment.');
        return;
    }
    
    try {
        // Safely parse each JSON response with individual error handling
        let config = {};
        try {
            const configData = await pythonBridge.getTrainingConfiguration();
            
            let configJsonString = configData;
            if (configData && typeof configData === 'string' && configData.startsWith('JSON_STRING:')) {
                configJsonString = configData.substring('JSON_STRING:'.length);
            }
            
            if (configJsonString && typeof configJsonString === 'string' && configJsonString.trim().startsWith('{')) {
                config = JSON.parse(configJsonString);
            } else {
                console.warn('Invalid config data, using fallback');
                config = {
                    selected_files: [],
                    adapter_name: "fallback_adapter",
                    base_model: "microsoft/DialoGPT-small",
                    training_preset: "standard_training"
                };
            }
        } catch (configError) {
            console.error('Config parsing failed:', configError);
            config = {
                selected_files: [],
                adapter_name: "fallback_adapter",
                base_model: "microsoft/DialoGPT-small", 
                training_preset: "standard_training"
            };
        }
        
        let baseModels = [];
        try {
            const baseModelsData = await pythonBridge.getAvailableBaseModels();
            
            let modelsJsonString = baseModelsData;
            if (baseModelsData && typeof baseModelsData === 'string' && baseModelsData.startsWith('JSON_STRING:')) {
                modelsJsonString = baseModelsData.substring('JSON_STRING:'.length);
            }
            
            if (modelsJsonString && typeof modelsJsonString === 'string' && modelsJsonString.trim().startsWith('[')) {
                baseModels = JSON.parse(modelsJsonString);
            } else {
                console.warn('Invalid base models data, using fallback');
                baseModels = [{
                    id: "microsoft/DialoGPT-small",
                    name: "DialoGPT Small",
                    description: "Default model",
                    size: "small",
                    recommended: true
                }];
            }
        } catch (modelsError) {
            console.error('Base models parsing failed:', modelsError);
            baseModels = [{
                id: "microsoft/DialoGPT-small",
                name: "DialoGPT Small", 
                description: "Default model",
                size: "small",
                recommended: true
            }];
        }
        
        let presets = [];
        try {
            const presetsData = await pythonBridge.getTrainingPresets();
            console.log('Raw presets data type:', typeof presetsData);
            console.log('Raw presets data length:', presetsData ? presetsData.length : 'null/undefined');
            console.log('Raw presets data:', presetsData);
            
            let presetsJsonString = presetsData;
            if (presetsData && typeof presetsData === 'string' && presetsData.startsWith('JSON_STRING:')) {
                presetsJsonString = presetsData.substring('JSON_STRING:'.length);
            }
            
            if (presetsJsonString && typeof presetsJsonString === 'string' && presetsJsonString.trim().startsWith('[')) {
                presets = JSON.parse(presetsJsonString);
            } else {
                console.warn('Invalid presets data, using fallback');
                presets = [{
                    id: "standard_training",
                    name: "Standard Training",
                    description: "Default training",
                    estimated_time: "15-45 minutes",
                    recommended: true,
                    config: {epochs: 2, batch_size: 4, learning_rate: 0.0002}
                }];
            }
        } catch (presetsError) {
            console.error('Presets parsing failed:', presetsError);
            presets = [{
                id: "standard_training",
                name: "Standard Training",
                description: "Default training",
                estimated_time: "15-45 minutes", 
                recommended: true,
                config: {epochs: 2, batch_size: 4, learning_rate: 0.0002}
            }];
        }
        
        let uploadedFiles = [];
        try {
            const uploadedFilesData = await pythonBridge.getUploadedFiles();
            console.log('Raw uploaded files data:', uploadedFilesData);
            if (uploadedFilesData) {
                if (typeof uploadedFilesData === 'string') {
                    if (uploadedFilesData.startsWith('JSON_STRING:')) {
                        const jsonString = uploadedFilesData.substring('JSON_STRING:'.length);
                        uploadedFiles = JSON.parse(jsonString);
                    } else {
                        uploadedFiles = JSON.parse(uploadedFilesData);
                    }
                } else if (Array.isArray(uploadedFilesData)) {
                    uploadedFiles = uploadedFilesData;
                }
            }
        } catch (filesError) {
            console.error('Uploaded files parsing failed:', filesError);
            uploadedFiles = [];
        }
        
        console.log('Final parsed data:', {config, baseModels, presets, uploadedFiles});
        populateTrainingUI(config, baseModels, presets, uploadedFiles);
        
        trainingConfigurationLoaded = true;
        trainingConfigurationLoading = false;
        
    } catch (error) {
        console.error('Failed to load training configuration:', error);
        showTrainingError('Failed to load training configuration: ' + error.message);
        trainingConfigurationLoading = false;
    }
}

// Populate training UI with configuration options
function populateTrainingUI(config, baseModels, presets, uploadedFiles) {
    const trainContent = document.getElementById('train-content');
    if (!trainContent) {
        console.error('❌ Train content element not found');
        trainingConfigurationLoading = false;
        return;
    }
    
    trainContent.innerHTML = `
        <div class="training-simple">
            <h3>🚀 Start Training</h3>
            
            ${uploadedFiles.length > 0 ? `
                <div class="files-list">
                    ${uploadedFiles.map(file => `
                        <div class="file-item">
                            <label>
                                <input type="checkbox" name="selected_files" value="${file.name}" checked>
                                📄 ${file.name}
                            </label>
                            <button onclick="removeFile('${file.name}')" class="remove-btn">×</button>
                        </div>
                    `).join('')}
                </div>
                
                <div class="training-buttons">
                    <button onclick="startEnhancedTraining()" class="btn-primary" id="start-training-btn">
                        🚀 Start Enhanced Training
                    </button>
                    <button onclick="processDocumentsWithDeepSeek()" class="btn-secondary" id="deepseek-process-btn">
                        🤖 Process with DeepSeek 14B
                    </button>
                </div>
            ` : `
                <p class="no-files">📤 Upload some documents first to start training</p>
            `}
            
            <div id="training-progress" style="display: none;">
                <div class="progress-simple">
                    <div class="progress-bar" id="progress-bar"></div>
                    <span id="progress-text">Training...</span>
                </div>
            </div>
        </div>
    `;
    
}

// Utility function to sanitize strings and remove null bytes
function sanitizeString(str) {
    if (typeof str !== 'string') return str;
    // Enhanced sanitization: remove all null bytes and control characters
    return str.replace(/[\x00-\x1F\x7F-\x9F]/g, '').trim();
}

// Utility function to sanitize object recursively
function sanitizeObject(obj) {
    if (obj === null || obj === undefined) return obj;
    
    if (typeof obj === 'string') {
        return sanitizeString(obj);
    }
    
    if (Array.isArray(obj)) {
        return obj.map(item => sanitizeObject(item));
    }
    
    if (typeof obj === 'object') {
        const sanitized = {};
        for (const [key, value] of Object.entries(obj)) {
            sanitized[sanitizeString(key)] = sanitizeObject(value);
        }
        return sanitized;
    }
    
    return obj;
}

// 🔧 FIX: Renamed to reflect actual complexity - Enhanced Training Pipeline
window.startEnhancedTraining = function() {
    try {
        const selectedFiles = Array.from(document.querySelectorAll('input[name="selected_files"]:checked')).map(cb => sanitizeString(cb.value));

        if (selectedFiles.length === 0) {
            showNotification('❌ Please select at least one file', 'error');
            return;
        }

        // Enhanced training configuration (reflects actual backend complexity)
        const config = {
            selected_files: selectedFiles,
            adapter_name: sanitizeString(`trained_model_${Date.now()}`),
            base_model: sanitizeString("microsoft/DialoGPT-small"),
            training_preset: sanitizeString("enhanced_pipeline"),
            pipeline_stages: ["document_processing", "data_generation", "model_training", "validation"]
        };

        console.log('🚀 Starting Enhanced Training Pipeline with files:', selectedFiles);
        showNotification('🚀 Starting Enhanced Training Pipeline - this may take several minutes', 'info');

        // CRITICAL FIX: Sanitize the entire config object and remove null bytes
        const sanitizedConfig = sanitizeObject(config);
        const cleanJsonString = JSON.stringify(sanitizedConfig);
        pythonBridge.startModelTraining(cleanJsonString);

        // Show enhanced progress with stage indicators
        const progressDiv = document.getElementById('training-progress');
        const startBtn = document.getElementById('start-training-btn');

        if (progressDiv) {
            progressDiv.style.display = 'block';
            progressDiv.innerHTML = `
                <div class="training-stages">
                    <div class="stage active">📄 Document Processing</div>
                    <div class="stage">🤖 Data Generation</div>
                    <div class="stage">🔧 Model Training</div>
                    <div class="stage">✅ Validation</div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
            `;
        }

        if (startBtn) {
            startBtn.disabled = true;
            startBtn.textContent = '🔄 Enhanced Training in Progress...';
        }

    } catch (error) {
        console.error('Failed to start enhanced training:', error);
        showNotification('❌ Enhanced Training failed: ' + error.message, 'error');
    }
};

// 🔧 FIX: Keep backward compatibility alias
window.startSimpleTraining = window.startEnhancedTraining;

// Simple training complete handler
function handleTrainingComplete() {
    const progressDiv = document.getElementById('training-progress');
    const startBtn = document.getElementById('start-training-btn');
    
    if (progressDiv) progressDiv.style.display = 'none';
    if (startBtn) {
        startBtn.disabled = false;
        startBtn.textContent = '🚀 Start Training';
    }
    
    showNotification('🎉 Training completed!', 'success');
}

// Show notification
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed; top: 20px; right: 20px; z-index: 1000;
        padding: 12px 20px; border-radius: 8px; color: white;
        background: ${type === 'error' ? '#f44336' : '#4caf50'};
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        animation: slideIn 0.3s ease; font-size: 14px;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 4000);
}

// 🤖 DeepSeek Document Processing Function
window.processDocumentsWithDeepSeek = async function() {
    try {
        console.log('🤖 Starting DeepSeek document processing...');

        // Get selected files
        const selectedFiles = Array.from(document.querySelectorAll('input[name="selected_files"]:checked')).map(cb => sanitizeString(cb.value));

        if (selectedFiles.length === 0) {
            showNotification('❌ Please select at least one file for processing', 'error');
            return;
        }

        // Disable the button and show processing state
        const processBtn = document.getElementById('deepseek-process-btn');
        if (processBtn) {
            processBtn.disabled = true;
            processBtn.textContent = '🔄 Processing...';
        }

        // Show progress
        updateDeepSeekProgress(0, selectedFiles.length, '🤖 Initializing DeepSeek 14B model...');

        // Call backend to start processing
        const result = await pythonBridge.processDocumentsWithDeepSeek(JSON.stringify({
            selected_files: selectedFiles,
            config: {
                chunk_size: 1200,
                enable_cross_referencing: true,
                output_formats: ['json', 'xml', 'csv'],
                deepseek_model: 'deepseek-r1:14b',
                processing_timeout: 300,
                enable_validation: true,
                max_chunks_per_file: 50
            }
        }));

        console.log('✅ DeepSeek processing completed:', result);

        // Show results
        displayProcessingResults(result);

        // Reset button
        if (processBtn) {
            processBtn.disabled = false;
            processBtn.textContent = '🤖 Process with DeepSeek 14B';
        }

    } catch (error) {
        console.error('❌ DeepSeek processing failed:', error);
        showNotification('❌ DeepSeek processing failed: ' + error.message, 'error');

        // Reset button
        const processBtn = document.getElementById('deepseek-process-btn');
        if (processBtn) {
            processBtn.disabled = false;
            processBtn.textContent = '🤖 Process with DeepSeek 14B';
        }
    }
};

// Update DeepSeek processing progress
function updateDeepSeekProgress(current, total, message) {
    console.log(`🔄 DeepSeek Progress: ${message} (${current}/${total})`);

    // Show progress in the UI
    const progressDiv = document.getElementById('training-progress');
    const progressText = document.getElementById('progress-text');
    const progressBar = document.getElementById('progress-bar');

    if (progressDiv) progressDiv.style.display = 'block';
    if (progressText) progressText.textContent = message;
    if (progressBar) {
        const percentage = total > 0 ? (current / total) * 100 : 0;
        progressBar.style.width = `${percentage}%`;
    }
}

// Display DeepSeek processing results
function displayProcessingResults(resultJson) {
    try {
        const result = typeof resultJson === 'string' ? JSON.parse(resultJson) : resultJson;

        if (result.success) {
            if (result.status === 'processing_started') {
                // Processing started in background
                showNotification(`🤖 DeepSeek processing started for ${result.message}`, 'info');
                updateDeepSeekProgress(0, 1, `🔄 ${result.message}`);
            } else {
                // Processing completed
                showNotification('✅ DeepSeek processing completed successfully!', 'success');

                // Hide progress
                const progressDiv = document.getElementById('training-progress');
                if (progressDiv) progressDiv.style.display = 'none';

                // Show detailed results
                console.log('📊 Processing Results:', result);

                if (result.training_data) {
                    const concepts = result.training_data.concepts || [];
                    const relationships = result.training_data.relationships || [];
                    const examples = result.training_data.training_examples || [];

                    console.log(`📚 Extracted ${concepts.length} concepts, ${relationships.length} relationships, ${examples.length} training examples`);
                }
            }
        } else {
            showNotification(`❌ DeepSeek processing failed: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('❌ Failed to parse processing results:', error);
        showNotification('❌ Failed to parse processing results', 'error');
    }
}

// Handle DeepSeek completion from background thread
function handleDeepSeekCompletion(result) {
    console.log('✅ DeepSeek background processing completed:', result);
    displayProcessingResults(result);

    // Reset button
    const processBtn = document.getElementById('deepseek-process-btn');
    if (processBtn) {
        processBtn.disabled = false;
        processBtn.textContent = '🤖 Process with DeepSeek 14B';
    }
}

// Handle DeepSeek error from background thread
function handleDeepSeekError(error) {
    console.error('❌ DeepSeek background processing failed:', error);
    showNotification(`❌ DeepSeek processing failed: ${error.error || 'Unknown error'}`, 'error');

    // Reset button
    const processBtn = document.getElementById('deepseek-process-btn');
    if (processBtn) {
        processBtn.disabled = false;
        processBtn.textContent = '🤖 Process with DeepSeek 14B';
    }

    // Hide progress
    const progressDiv = document.getElementById('training-progress');
    if (progressDiv) progressDiv.style.display = 'none';
}

// Simple training signal handlers
function handleTrainingStatusChanged(status) {
    switch (status) {
        case 'completed':
            handleTrainingComplete();
            break;
        case 'error':
        case 'cancelled':
            handleTrainingComplete();
            showNotification('❌ Training stopped', 'error');
            break;
    }
}

// 🚀 Missing Training Handler Functions - CRITICAL FIX
function handleTrainingProgressStructured(progressData) {
    try {
        console.log('🔥 Training progress received:', progressData);
        
        // Handle structured progress data
        if (progressData && typeof progressData === 'object') {
            const stage = progressData.stage || 'unknown';
            const progress = progressData.progress || 0;
            const message = progressData.message || 'Processing...';
            
            // Update UI with progress information
            updateStatus(`Training: ${stage} - ${message} (${progress}%)`);
            
            // Show notification for major milestones
            if (progress === 100 || stage === 'completed') {
                showNotification(`✅ ${stage} completed`, 'success');
            }
        }
    } catch (error) {
        console.error('❌ Error handling training progress:', error);
    }
}

function handleTrainingMetricsUpdate(metricsData) {
    try {
        console.log('📊 Training metrics received:', metricsData);
        
        // Handle training metrics updates
        if (metricsData && typeof metricsData === 'object') {
            const loss = metricsData.loss || 0;
            const accuracy = metricsData.accuracy || 0;
            const epoch = metricsData.epoch || 0;
            const step = metricsData.step || 0;
            
            // Update training UI if available
            const metricsElement = document.getElementById('training-metrics');
            if (metricsElement) {
                metricsElement.innerHTML = `
                    <div>Epoch: ${epoch}</div>
                    <div>Step: ${step}</div>
                    <div>Loss: ${loss.toFixed(4)}</div>
                    <div>Accuracy: ${(accuracy * 100).toFixed(2)}%</div>
                `;
            }
        }
    } catch (error) {
        console.error('❌ Error handling training metrics:', error);
    }
}

function handleTrainingConfigSaved(configData) {
    try {
        console.log('💾 Training config saved:', configData);
        
        // Handle training configuration save confirmation
        if (configData && typeof configData === 'object') {
            const success = configData.success !== false;
            const message = configData.message || 'Configuration saved';
            
            if (success) {
                showNotification(`✅ ${message}`, 'success');
            } else {
                showNotification(`❌ ${message}`, 'error');
            }
        }
    } catch (error) {
        console.error('❌ Error handling training config save:', error);
    }
}



// Settings functions
async function saveSettings() {
    try {
        // 🔒 SECURITY FIX #16: Separate API keys from settings
        // First, securely store API keys using the secure manager
        const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
        for (const provider of providers) {
            const input = document.getElementById(`${provider}-api-key`);
            if (input && input.value.trim()) {
                await window.secureApiKeyManager.store(provider, input.value.trim());
            }
        }
        
        // Collect non-sensitive settings from the UI
        const settings = {
            theme: document.body.getAttribute('data-theme') || 'light',
            default_timer: parseInt(document.getElementById('default-timer')?.value) || 30,
            show_answers: document.getElementById('show-answers')?.checked || true,
            
            // Quiz Preferences - NOW SAVED! 🎯
            default_game_mode: document.getElementById('quiz-game-mode')?.value || 'casual',
            default_difficulty: document.getElementById('quiz-difficulty')?.value || 'medium',
            default_quiz_mode: document.getElementById('quiz-mode')?.value || 'offline',
            default_submode: document.getElementById('quiz-submode')?.value || 'mixed',
            
            // 🔒 SECURITY FIX #16: No API keys in settings object
            // API keys are now stored securely on the server
            api_keys: {}
        };
        
        console.log('Saving settings:', {
            ...settings,
            api_keys: '🔒 Stored securely on server'
        });
        
        let saveSuccess = false;
        
        // Save to Python backend first
        if (pythonBridge && pythonBridge.saveUserSettings) {
            try {
                const settingsJson = JSON.stringify(settings);
                pythonBridge.saveUserSettings(settingsJson);
                console.log('✅ Settings saved to Python backend');
                pythonBridge.log('✅ JS: Settings saved to Python backend');
                saveSuccess = true;
            } catch (error) {
                console.error('❌ Failed to save to Python backend:', error);
                pythonBridge.log(`❌ JS: Failed to save to Python backend: ${error}`);
            }
        } else {
            console.log('⚠️ Python bridge not available for saving');
            // CRITICAL FIX: Don't try to log when bridge is not available
        }
        
        // ALWAYS save to localStorage as backup (without API keys)
        try {
            // Make sure we're not storing API keys in localStorage
            const settingsForStorage = { ...settings };
            if (settingsForStorage.api_keys) {
                delete settingsForStorage.api_keys;
            }
            
            localStorage.setItem('userSettings', JSON.stringify(settingsForStorage));
            console.log('✅ Settings saved to localStorage as backup (no API keys)');
            pythonBridge.log('✅ JS: Settings saved to localStorage as backup (no API keys)');
            saveSuccess = true;
        } catch (storageError) {
            console.error('❌ Failed to save to localStorage:', storageError);
            pythonBridge.log(`❌ JS: Failed to save to localStorage: ${storageError}`);
        }
        
        // Update provider statuses
        updateProviderStatuses();
        
        // Show success message
        if (saveSuccess) {
            showTemporaryMessage('✅ Settings saved successfully!', 'success');
        } else {
            showTemporaryMessage('❌ Failed to save settings!', 'error');
        }
        
        return saveSuccess;
        
    } catch (error) {
        console.error('❌ Critical error in saveSettings:', error);
        pythonBridge.log(`❌ JS: Critical error in saveSettings: ${error}`);
        showTemporaryMessage('❌ Failed to save settings!', 'error');
        return false;
    }
}

async function loadSettings() {
    try {
        console.log('🔄 Loading settings...');
        // CRITICAL FIX: Check if pythonBridge exists before logging
        if (pythonBridge && pythonBridge.log) {
            pythonBridge.log('🔄 JS: Loading settings...');
        }
        
        let settings = null;
        
        // 🚀 PRIORITY: Try localStorage FIRST due to PyQt serialization issues
        const savedSettings = localStorage.getItem('userSettings');
        if (savedSettings) {
            try {
                settings = JSON.parse(savedSettings);
                console.log('✅ Loaded settings from localStorage (primary method)');
                // CRITICAL FIX: Check if pythonBridge exists before logging
                if (pythonBridge && pythonBridge.log) {
                    pythonBridge.log('✅ JS: Loaded settings from localStorage (primary method)');
                }
                
                // 🔒 SECURITY FIX #16: API keys are now stored securely on server
                // Initialize empty API keys object - actual keys will be loaded securely
                settings.api_keys = {};
                
                // Load secure API key indicators (not the actual keys)
                try {
                    // Check which providers have securely stored keys
                    const providers = await window.secureApiKeyManager.list();
                    if (providers && providers.length > 0) {
                        console.log(`🔒 Found ${providers.length} securely stored API keys`);
                        
                        // Mark that these providers have keys (without exposing the actual keys)
                        providers.forEach(provider => {
                            // Just mark that a key exists, don't store the actual key
                            settings.api_keys[provider] = '••••••••••••••••••••••••';
                        });
                    }
                } catch (secureError) {
                    console.error('❌ Failed to check secure API keys:', secureError);
                }
            } catch (error) {
                console.error('❌ Failed to parse localStorage settings:', error);
                pythonBridge.log(`❌ JS: Failed to parse localStorage settings: ${error}`);
                settings = null; // Will try backend as fallback
            }
        }
        
        // Only try backend if localStorage failed
        if (!settings && pythonBridge && pythonBridge.getUserSettings) {
            console.log('🔗 localStorage failed, trying Python backend as fallback...');
            pythonBridge.log('🔗 JS: localStorage failed, trying Python backend as fallback...');
            try {
                const settingsData = pythonBridge.getUserSettings();
                console.log('📋 Received settings data:', typeof settingsData, settingsData ? 'has data' : 'no data');
                pythonBridge.log(`📋 JS: Received settings data: ${typeof settingsData}, ${settingsData ? 'has data' : 'no data'}`);
                
                if (settingsData) {
                    try {
                        // ENHANCED JSON parsing to handle PyQt serialization issues
                        if (typeof settingsData === 'string') {
                            let jsonString = settingsData;
                            
                            // Handle special JSON_STRING prefix that bypasses PyQt conversion
                            if (settingsData.startsWith('JSON_STRING:')) {
                                jsonString = settingsData.substring('JSON_STRING:'.length);
                                console.log('🔧 Detected JSON_STRING prefix, extracting pure JSON');
                                pythonBridge.log('🔧 JS: Detected JSON_STRING prefix, extracting pure JSON');
                            }
                            
                            settings = JSON.parse(jsonString);
                            console.log('✅ Successfully parsed settings JSON string from backend');
                            pythonBridge.log('✅ JS: Successfully parsed settings JSON string from backend');
                        } else if (typeof settingsData === 'object' && settingsData !== null) {
                            // Check if it's an empty object (PyQt serialization issue)
                            const keys = Object.keys(settingsData);
                            if (keys.length === 0) {
                                // Silent fallback to defaults - no console spam
                                settings = null; // Use defaults
                            } else {
                                // PyQt converted JSON to object - validate structure
                                if (settingsData.api_keys && typeof settingsData.api_keys === 'object') {
                                    settings = settingsData;
                                } else {
                                    // Invalid structure - use defaults
                                    settings = null;
                                }
                            }
                        } else {
                            // Invalid data type - use defaults silently
                            settings = null;
                        }
                        
                        // CRITICAL: Validate that we actually have a valid settings object
                        if (settings && typeof settings === 'object') {
                            // IMPORTANT: Don't replace the entire settings object!
                            // Just ensure api_keys exists - preserve all other properties
                            if (!settings.api_keys || typeof settings.api_keys !== 'object') {
                                console.log('⚠️ Missing or invalid api_keys - adding empty api_keys while preserving other settings');
                                pythonBridge.log('⚠️ JS: Missing or invalid api_keys - adding empty api_keys while preserving other settings');
                                settings.api_keys = {
                                    openai: '',
                                    anthropic: '',
                                    gemini: '',
                                    groq: '',
                                    openrouter: ''
                                };
                            }
                            
                            // Ensure all required properties exist with defaults (don't overwrite existing values)
                            if (!settings.default_game_mode) settings.default_game_mode = 'casual';
                            if (!settings.default_difficulty) settings.default_difficulty = 'medium';
                            if (!settings.default_submode) settings.default_submode = 'mixed';
                            if (!settings.default_quiz_mode) settings.default_quiz_mode = 'auto';
                            if (!settings.theme) settings.theme = 'light';
                            if (typeof settings.default_timer === 'undefined') settings.default_timer = 30;
                            if (typeof settings.show_answers === 'undefined') settings.show_answers = true;
                        }
                        
                    } catch (parseError) {
                        console.error('❌ Failed to parse settings JSON:', parseError);
                        pythonBridge.log(`❌ JS: Failed to parse settings JSON: ${parseError}`);
                        console.error('Raw settings data:', settingsData);
                        settings = null;
                    }
                } else {
                    console.log('⚠️ No settings data returned from Python backend');
                    pythonBridge.log('⚠️ JS: No settings data returned from Python backend');
                }
            } catch (bridgeError) {
                console.error('❌ Error calling Python bridge getUserSettings:', bridgeError);
                pythonBridge.log(`❌ JS: Error calling Python bridge getUserSettings: ${bridgeError}`);
            }
        }
        
        // Last resort: create defaults if both localStorage and backend failed
        
        // If still no settings, create defaults
        if (!settings) {
            console.log('🔧 Creating default settings...');
            pythonBridge.log('🔧 JS: Creating default settings...');
            settings = {
                theme: 'light',
                default_timer: 30,
                show_answers: true,
                default_game_mode: 'casual',
                default_difficulty: 'medium',
                default_quiz_mode: 'offline',
                default_submode: 'mixed',
                api_keys: {
                    openai: '',
                    anthropic: '',
                    gemini: '',
                    groq: '',
                    openrouter: ''
                }
            };
        }
        
        if (settings) {
            console.log('🎛️ Applying settings to UI...');
            pythonBridge.log('🎛️ JS: Applying settings to UI...');
            
            // DEBUG: Log all properties of the settings object  
            const settingsKeys = Object.keys(settings);
            pythonBridge.log(`🔍 JS: Settings object keys: ${settingsKeys.join(', ')}`);
            pythonBridge.log(`🔍 JS: Settings object type: ${typeof settings}`);
            pythonBridge.log(`🔍 JS: Has api_keys property: ${!!settings.api_keys}`);
            pythonBridge.log(`🔍 JS: api_keys type: ${typeof settings.api_keys}`);
            
            // Load API keys - ROBUST APPROACH
            if (settings.api_keys && typeof settings.api_keys === 'object') {
                console.log('🔑 Loading API keys...');
                pythonBridge.log('🔑 JS: Loading API keys...');
                
                const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
                let keysLoaded = 0;
                
                providers.forEach(provider => {
                    const input = document.getElementById(`${provider}-api-key`);
                    if (input) {
                        const apiKey = settings.api_keys[provider] || '';
                        
                        if (apiKey && apiKey.trim() !== '' && apiKey !== '***') {
                            input.value = apiKey;
                            keysLoaded++;
                            console.log(`✅ Loaded ${provider} API key`);
                            pythonBridge.log(`✅ JS: Loaded ${provider} API key`);
                        } else {
                            input.value = '';
                            console.log(`⚠️ ${provider} API key is empty, masked, or missing`);
                            pythonBridge.log(`⚠️ JS: ${provider} API key is empty, masked, or missing`);
                        }
                    } else {
                        console.error(`❌ Could not find input element for ${provider}-api-key`);
                        pythonBridge.log(`❌ JS: Could not find input element for ${provider}-api-key`);
                    }
                });
                
                console.log(`🔑 Total API keys loaded: ${keysLoaded}/${providers.length}`);
                pythonBridge.log(`🔑 JS: Total API keys loaded: ${keysLoaded}/${providers.length}`);
                
                // Update provider statuses after loading keys
                updateProviderStatuses();
            } else {
                console.log('⚠️ No valid api_keys found in settings object');
                pythonBridge.log('⚠️ JS: No valid api_keys found in settings object');
                // Clear all API key inputs
                const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
                providers.forEach(provider => {
                    const input = document.getElementById(`${provider}-api-key`);
                    if (input) input.value = '';
                });
            }
            
            // Load theme settings
            if (settings.theme) {
                document.body.setAttribute('data-theme', settings.theme);
                const darkModeToggle = document.getElementById('dark-mode-toggle');
                if (darkModeToggle) {
                    darkModeToggle.checked = settings.theme === 'dark';
                }
                console.log(`🎨 Theme set to: ${settings.theme}`);
                pythonBridge.log(`🎨 JS: Theme set to: ${settings.theme}`);
                
                // Update the theme toggle button text
                const themeToggle = document.querySelector('.theme-toggle');
                if (themeToggle) {
                    themeToggle.textContent = settings.theme === 'dark' ? '☀️' : '🌙';
                }
            }
            
            // Load quiz mode settings (casual/serious)
            if (settings.default_game_mode) {
                const gameModeSelect = document.getElementById('quiz-game-mode');
                if (gameModeSelect) {
                    gameModeSelect.value = settings.default_game_mode;
                    console.log(`🎮 Game mode set to: ${settings.default_game_mode}`);
                    pythonBridge.log(`🎮 JS: Game mode set to: ${settings.default_game_mode}`);
                }
            }
            
            // Load quiz difficulty settings
            if (settings.default_difficulty) {
                const difficultySelect = document.getElementById('quiz-difficulty');
                if (difficultySelect) {
                    difficultySelect.value = settings.default_difficulty;
                    console.log(`⚡ Difficulty set to: ${settings.default_difficulty}`);
                    pythonBridge.log(`⚡ JS: Difficulty set to: ${settings.default_difficulty}`);
                }
            }
            
            // Load quiz mode (auto/offline/online)
            if (settings.default_quiz_mode) {
                const modeSelect = document.getElementById('quiz-mode');
                if (modeSelect) {
                    modeSelect.value = settings.default_quiz_mode;
                    console.log(`🌐 Quiz mode set to: ${settings.default_quiz_mode}`);
                    pythonBridge.log(`🌐 JS: Quiz mode set to: ${settings.default_quiz_mode}`);
                    // Update mode info after setting the value
                    updateModeInfo();
                }
            } else if (typeof settings.offline_mode === 'boolean') {
                // Legacy support for old boolean offline_mode setting
                const modeSelect = document.getElementById('quiz-mode');
                if (modeSelect) {
                    modeSelect.value = settings.offline_mode ? 'offline' : 'online';
                    console.log(`🌐 Quiz mode set to: ${settings.offline_mode ? 'offline' : 'online'} (legacy)`);
                    pythonBridge.log(`🌐 JS: Quiz mode set to: ${settings.offline_mode ? 'offline' : 'online'} (legacy)`);
                    updateModeInfo();
                }
            }
            
            // Load question type settings
            if (settings.default_submode) {
                const submodeSelect = document.getElementById('quiz-submode');
                if (submodeSelect) {
                    submodeSelect.value = settings.default_submode;
                    console.log(`📊 Question type set to: ${settings.default_submode}`);
                    pythonBridge.log(`📊 JS: Question type set to: ${settings.default_submode}`);
                }
            }
            
            // Load timer settings
            if (settings.default_timer) {
                const timerInput = document.getElementById('default-timer');
                if (timerInput) timerInput.value = settings.default_timer;
                console.log(`⏰ Timer set to: ${settings.default_timer}s`);
                pythonBridge.log(`⏰ JS: Timer set to: ${settings.default_timer}s`);
            }
            
            // Load show answers setting
            if (typeof settings.show_answers === 'boolean') {
                const showAnswersCheckbox = document.getElementById('show-answers');
                if (showAnswersCheckbox) showAnswersCheckbox.checked = settings.show_answers;
                console.log(`📝 Show answers: ${settings.show_answers}`);
                pythonBridge.log(`📝 JS: Show answers: ${settings.show_answers}`);
            }
            
            console.log('✅ Settings loaded and applied successfully');
            pythonBridge.log('✅ JS: Settings loaded and applied successfully');
        } else {
            console.log('⚠️ No settings to load - using defaults');
            pythonBridge.log('⚠️ JS: No settings to load - using defaults');
        }
    } catch (error) {
        console.error('❌ Error loading settings:', error);
        pythonBridge.log(`❌ JS: Error loading settings: ${error}`);
        console.error('Error stack:', error.stack);
    }
}

function updateProviderStatuses() {
    const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
    
    providers.forEach(provider => {
        const input = document.getElementById(`${provider}-api-key`);
        const statusElement = document.getElementById(`${provider}-status`);
        const card = input?.closest('.api-provider-card');
        
        if (input && statusElement) {
            const hasKey = input.value.trim().length > 0;
            
            if (hasKey) {
                statusElement.textContent = '✅';
                statusElement.className = 'provider-status available';
                if (card) {
                    card.classList.remove('error');
                    card.classList.add('connected');
                }
            } else {
                statusElement.textContent = '❌';
                statusElement.className = 'provider-status unavailable';
                if (card) {
                    card.classList.remove('connected', 'error');
                }
            }
        }
    });
}

function testAllProviders() {
    const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
    let testsRunning = 0;
    
    providers.forEach(provider => {
        const input = document.getElementById(`${provider}-api-key`);
        const statusElement = document.getElementById(`${provider}-status`);
        const card = input?.closest('.api-provider-card');
        
        if (input && input.value.trim()) {
            testsRunning++;
            
            // Show testing status
            statusElement.textContent = '🧪';
            statusElement.className = 'provider-status testing';
            if (card) {
                card.classList.remove('connected', 'error');
                card.classList.add('testing');
            }
            
            // Test the API key (mock test for now)
            setTimeout(() => {
                testApiKey(provider, input.value.trim());
            }, Math.random() * 2000 + 1000); // Random delay 1-3 seconds
        }
    });
    
    if (testsRunning === 0) {
        showTemporaryMessage('⚠️ No API keys to test. Please enter your API keys first.', 'warning');
    } else {
        showTemporaryMessage(`🧪 Testing ${testsRunning} API provider(s)...`, 'info');
    }
}

function testApiKey(provider, apiKey) {
    const statusElement = document.getElementById(`${provider}-status`);
    const card = statusElement?.closest('.api-provider-card');
    
    console.log(`🧪 Testing ${provider} API key...`);
    
    // First, validate format
    const isValidFormat = validateApiKeyFormat(provider, apiKey);
    if (!isValidFormat) {
        statusElement.textContent = '❌';
        statusElement.className = 'provider-status unavailable';
        if (card) {
            card.classList.remove('testing', 'connected');
            card.classList.add('error');
        }
        
        showTemporaryMessage(`❌ ${provider.toUpperCase()}: Invalid API key format`, 'error');
        return;
    }
    
    // If Python bridge is available, test with backend
    if (pythonBridge && pythonBridge.testApiKey) {
        try {
            pythonBridge.testApiKey(provider, apiKey);
        } catch (error) {
            console.error(`Failed to test ${provider} API:`, error);
            fallbackToFormatValidation(provider, apiKey, statusElement, card);
        }
    } else {
        console.log(`⚠️ Python bridge not available, using format validation only`);
        fallbackToFormatValidation(provider, apiKey, statusElement, card);
    }
}

function fallbackToFormatValidation(provider, apiKey, statusElement, card) {
    const isValid = validateApiKeyFormat(provider, apiKey);
    
    setTimeout(() => {
        if (isValid) {
            statusElement.textContent = '✅';
            statusElement.className = 'provider-status available';
            if (card) {
                card.classList.remove('testing', 'error');
                card.classList.add('connected');
            }
            showTemporaryMessage(`✅ ${provider.toUpperCase()}: Format valid (not tested with API)`, 'warning');
        } else {
            statusElement.textContent = '❌';
            statusElement.className = 'provider-status unavailable';
            if (card) {
                card.classList.remove('testing', 'connected');
                card.classList.add('error');
            }
            showTemporaryMessage(`❌ ${provider.toUpperCase()}: Invalid format`, 'error');
        }
    }, 500);
}

// Handle API test results from Python backend
function handleApiTestResult(provider, success, message) {
    const statusElement = document.getElementById(`${provider}-status`);
    const card = statusElement?.closest('.api-provider-card');
    
    console.log(`🧪 API test result for ${provider}:`, success ? 'SUCCESS' : 'FAILED', message);
    
    if (success) {
        statusElement.textContent = '✅';
        statusElement.className = 'provider-status available';
        if (card) {
            card.classList.remove('testing', 'error');
            card.classList.add('connected');
        }
        showApiTestDialog(provider, true, message);
    } else {
        statusElement.textContent = '❌';
        statusElement.className = 'provider-status unavailable';
        if (card) {
            card.classList.remove('testing', 'connected');
            card.classList.add('error');
        }
        showApiTestDialog(provider, false, message);
    }
}

function showApiTestDialog(provider, success, message) {
    const title = success ? 
        `🎉 ${provider.toUpperCase()} API Test Successful!` : 
        `❌ ${provider.toUpperCase()} API Test Failed`;
    
    const body = success ?
        `Your ${provider.toUpperCase()} API key is working correctly and ready to use for generating questions.` :
        `API Test Error: ${message}\n\nPlease check your API key and try again.`;
    
    const emoji = success ? '🚀' : '⚠️';
    
    // Create modal dialog
    const modal = document.createElement('div');
    modal.className = 'api-test-modal';
    modal.innerHTML = `
        <div class="api-test-modal-content ${success ? 'success' : 'error'}">
            <div class="api-test-modal-header">
                <span class="api-test-modal-emoji">${emoji}</span>
                <h3>${title}</h3>
            </div>
            <div class="api-test-modal-body">
                <p>${body.replace(/\n/g, '<br>')}</p>
                ${success ? 
                    '<div class="api-test-success-details">✅ Ready for question generation<br>✅ API connection verified</div>' :
                    '<div class="api-test-error-details">💡 Tip: Make sure your API key is valid and has proper permissions</div>'
                }
            </div>
            <div class="api-test-modal-footer">
                <button class="btn btn-primary" onclick="closeApiTestDialog()">Got it!</button>
            </div>
        </div>
        <div class="api-test-modal-backdrop" onclick="closeApiTestDialog()"></div>
    `;
    
    // Add modal styles
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    
    document.body.appendChild(modal);
    
    // Auto-close after 10 seconds for success, or keep open for errors
    if (success) {
        setTimeout(() => {
            if (document.body.contains(modal)) {
                closeApiTestDialog();
            }
        }, 10000);
    }
}

function closeApiTestDialog() {
    const modal = document.querySelector('.api-test-modal');
    if (modal) {
        modal.remove();
    }
}

function validateApiKeyFormat(provider, apiKey) {
    // Basic format validation - FIXED to allow hyphens and underscores
    const formats = {
        openai: /^sk-[a-zA-Z0-9_-]+$/,
        anthropic: /^sk-ant-[a-zA-Z0-9_-]+$/,
        gemini: /^[a-zA-Z0-9_-]+$/,
        groq: /^gsk_[a-zA-Z0-9_-]+$/,
        openrouter: /^sk-or-[a-zA-Z0-9_-]+$/  // Fixed to allow v1- format
    };
    
    console.log(`🔍 Validating ${provider} API key format:`, apiKey.substring(0, 15) + '...');
    const isValid = formats[provider] ? formats[provider].test(apiKey) : apiKey.length > 10;
    console.log(`✅ Format validation result for ${provider}:`, isValid);
    
    return isValid;
}

function clearAllApiKeys() {
    if (confirm('Are you sure you want to clear all API keys? This action cannot be undone.')) {
        const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
        
        providers.forEach(provider => {
            const input = document.getElementById(`${provider}-api-key`);
            if (input) {
                input.value = '';
            }
        });
        
        // ✅ ENHANCED: Immediately save after clearing to persist changes
        const saveSuccess = saveSettings();
        if (saveSuccess) {
            updateProviderStatuses();
            showTemporaryMessage('🗑️ All API keys cleared and saved', 'success');
            
            // Clear session storage as well
            clearApiKeysFromSessionStorage();
            
            console.log('✅ API keys cleared from all storage locations');
            pythonBridge.log('✅ JS: API keys cleared from all storage locations');
        } else {
            showTemporaryMessage('❌ Failed to save cleared API keys', 'error');
        }
    }
}

// 🔒 SECURITY FIX #16: Secure API key management (no client-side storage)
function saveApiKeysToSessionStorage() {
    try {
        const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
        let savedCount = 0;
        
        // Use secure API key manager instead of session storage
        providers.forEach(async provider => {
            const input = document.getElementById(`${provider}-api-key`);
            if (input && input.value.trim()) {
                // Store API key securely on the server
                const success = await window.secureApiKeyManager.store(provider, input.value.trim());
                if (success) {
                    savedCount++;
                    console.log(`🔒 ${provider} API key securely stored on server`);
                }
            }
        });
        
        if (savedCount > 0) {
            console.log(`🔒 ${savedCount} API keys securely backed up`);
            return true;
        }
    } catch (error) {
        console.error('❌ Failed to securely save API keys:', error);
    }
    return false;
}

async function loadApiKeysFromSessionStorage() {
    try {
        // Use secure API key manager to check which providers have keys
        const providers = await window.secureApiKeyManager.list();
        if (providers && providers.length > 0) {
            // Update UI to show which providers have keys (without exposing actual keys)
            let keysIndicated = 0;
            
            for (const provider of providers) {
                const input = document.getElementById(`${provider}-api-key`);
                if (input && !input.value.trim()) {
                    // Don't load actual key value, just indicate it exists with placeholder
                    input.placeholder = '••••••••••••••••••••••••••••••••••••••••';
                    input.classList.add('has-secure-key');
                    keysIndicated++;
                }
            }
            
            if (keysIndicated > 0) {
                updateProviderStatuses();
                console.log(`🔒 Indicated ${keysIndicated} securely stored API keys`);
                if (pythonBridge && pythonBridge.log) {
                    pythonBridge.log(`🔒 JS: Indicated ${keysIndicated} securely stored API keys`);
                }
            }
        }
    } catch (error) {
        console.error('❌ Failed to check for securely stored API keys:', error);
    }
    return false;
}

async function clearApiKeysFromSessionStorage() {
    try {
        // Use secure API key manager to clear all keys
        const success = await window.secureApiKeyManager.clear();
        if (success) {
            console.log('🗑️ Cleared all securely stored API keys');
            
            // Also clear any UI indicators
            const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
            providers.forEach(provider => {
                const input = document.getElementById(`${provider}-api-key`);
                if (input) {
                    input.classList.remove('has-secure-key');
                    if (!input.value.trim()) {
                        input.placeholder = 'Enter API key';
                    }
                }
            });
        }
    } catch (error) {
        console.error('❌ Failed to clear secure API keys:', error);
    }
}

// ✅ NEW: Advanced Reset Functionality - Two integrated approaches
function resetApiKeysAdvanced() {
    const modal = document.createElement('div');
    modal.className = 'reset-modal-overlay';
    modal.innerHTML = `
        <div class="reset-modal">
            <h3>🔄 Reset API Keys</h3>
            <p>Choose your reset method:</p>
            <div class="reset-options">
                <button class="btn btn-warning" onclick="softResetApiKeys()">
                    🔄 Soft Reset
                    <small>Clear current session, keep saved keys</small>
                </button>
                <button class="btn btn-danger" onclick="hardResetApiKeys()">
                    🗑️ Hard Reset
                    <small>Permanently delete all saved API keys</small>
                </button>
            </div>
            <button class="btn btn-secondary" onclick="closeResetModal()">Cancel</button>
        </div>
    `;
    
    // Add modal styles
    const style = document.createElement('style');
    style.textContent = `
        .reset-modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        }
        .reset-modal {
            background: var(--bg-color, white);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            max-width: 400px;
            text-align: center;
        }
        .reset-options {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin: 1.5rem 0;
        }
        .reset-options .btn {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
            padding: 1rem;
        }
        .reset-options .btn small {
            opacity: 0.8;
            font-size: 0.8rem;
        }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(modal);
}

function softResetApiKeys() {
    // Clear current UI inputs only, don't delete saved keys
    const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
    
    providers.forEach(provider => {
        const input = document.getElementById(`${provider}-api-key`);
        if (input) {
            input.value = '';
        }
    });
    
    clearApiKeysFromSessionStorage();
    updateProviderStatuses();
    closeResetModal();
    showTemporaryMessage('🔄 Soft reset complete - reload page to restore saved keys', 'info');
}

function hardResetApiKeys() {
    if (confirm('⚠️ This will permanently delete ALL saved API keys. Are you absolutely sure?')) {
        clearAllApiKeys();
        
        // Also clear localStorage specifically
        try {
            const settings = JSON.parse(localStorage.getItem('userSettings') || '{}');
            if (settings.api_keys) {
                const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
                providers.forEach(provider => {
                    settings.api_keys[provider] = '';
                });
                localStorage.setItem('userSettings', JSON.stringify(settings));
            }
        } catch (error) {
            console.error('Failed to clear localStorage API keys:', error);
        }
        
        closeResetModal();
        showTemporaryMessage('🗑️ Hard reset complete - all API keys permanently deleted', 'warning');
    }
}

function closeResetModal() {
    const modal = document.querySelector('.reset-modal-overlay');
    if (modal) {
        modal.remove();
    }
    const style = document.querySelector('style[data-reset-modal]');
    if (style) {
        style.remove();
    }
}

// 🔒 SECURITY FIX #16: Ensure API key persistence using secure storage
async function ensureApiKeyPersistence() {
    try {
        // Check for securely stored API keys
        const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
        let emptyInputs = false;
        
        // Check if any API key inputs are empty
        providers.forEach(provider => {
            const input = document.getElementById(`${provider}-api-key`);
            if (input && !input.value.trim()) {
                emptyInputs = true;
            }
        });
        
        // If any inputs are empty, check for securely stored keys
        if (emptyInputs) {
            try {
                // Get list of providers with securely stored keys
                const secureProviders = await window.secureApiKeyManager.list();
                if (secureProviders && secureProviders.length > 0) {
                    let indicatedKeys = 0;
                    
                    // Update UI to indicate which providers have securely stored keys
                    secureProviders.forEach(provider => {
                        const input = document.getElementById(`${provider}-api-key`);
                        if (input && !input.value.trim()) {
                            // Don't expose actual key, just indicate it exists
                            input.placeholder = '••••••••••••••••••••••••••••••••••••••••';
                            input.classList.add('has-secure-key');
                            indicatedKeys++;
                        }
                    });
                    
                    if (indicatedKeys > 0) {
                        updateProviderStatuses();
                        console.log(`🔒 Indicated ${indicatedKeys} securely stored API keys`);
                        if (pythonBridge && pythonBridge.log) {
                            pythonBridge.log(`🔒 JS: Indicated ${indicatedKeys} securely stored API keys`);
                        }
                    }
                }
            } catch (error) {
                console.error('❌ Failed to check secure API keys:', error);
            }
        }
        
    } catch (error) {
        console.error('❌ Error ensuring API key persistence:', error);
    }
}

// ✅ ENHANCED: API Key validation and health check
function validateAllApiKeys() {
    const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
    const results = {
        valid: 0,
        invalid: 0,
        empty: 0,
        total: providers.length
    };
    
    providers.forEach(provider => {
        const input = document.getElementById(`${provider}-api-key`);
        if (input && input.value.trim()) {
            const isValid = validateApiKeyFormat(provider, input.value.trim());
            if (isValid) {
                results.valid++;
            } else {
                results.invalid++;
            }
        } else {
            results.empty++;
        }
    });
    
    console.log('🔍 API Key Validation Results:', results);
    
    // Show summary
    const message = `📊 API Keys: ${results.valid} valid, ${results.invalid} invalid, ${results.empty} empty`;
    showTemporaryMessage(message, results.invalid > 0 ? 'warning' : 'info');
    
    return results;
}

// Utility functions
function updateStatus(message) {
    // Could show a toast notification
    console.log('Status:', message);
    
    // Show in status display if quiz is active
    if (currentScreen === 'quiz' && document.getElementById('quiz-game').style.display !== 'none') {
        showStatusDisplay(message, 'info');
        setTimeout(hideStatusDisplay, 3000);
    }
    
    // For training messages, also show temporary message on any screen
    if (message.toLowerCase().includes('training') || message.includes('🚀') || message.includes('📚') || message.includes('🔄') || message.includes('✅')) {
        showTemporaryMessage(message, 'info');
        
        // If training completed successfully, re-enable training button
        if (message.toLowerCase().includes('training completed') || message.includes('✅ Training completed')) {
            const trainButton = document.querySelector('#train-screen .btn-primary');
                        if (trainButton) {
                trainButton.disabled = false;
                trainButton.textContent = 'Start Training';
            }
        }
    }
}

function handleError(error) {
    console.error('❌ Error:', error);

    // 🚀 CRITICAL FIX: Hide loading and show proper error
    hideQuizLoading();

    // Show error using the new error system
    showError(error);

    // Also show in status display for compatibility
    showStatusDisplay(`❌ Error: ${error}`, 'error');
    setTimeout(hideStatusDisplay, 5000);
    
    // If this is a training error, re-enable the training button
    if (error.toLowerCase().includes('training')) {
        const trainButton = document.querySelector('#train-screen .btn-primary');
        if (trainButton) {
            trainButton.disabled = false;
            trainButton.textContent = 'Start Training';
        }
        // Also show a temporary message
        showTemporaryMessage(`❌ ${error}`, 'error');
    }
    
    clearInterval(statusUpdateInterval);
}

function updateStats() {
    try {
        console.log('📊 Updating dashboard stats...');
        
        // Get user settings which contains answered_questions_history
        let settings = null;
        
        // Try to load from Python backend first
        if (pythonBridge && pythonBridge.getUserSettings) {
            const settingsData = pythonBridge.getUserSettings();
            if (settingsData) {
                try {
                    if (typeof settingsData === 'string') {
                        settings = JSON.parse(settingsData);
                    } else if (typeof settingsData === 'object') {
                        settings = settingsData;
                    }
                } catch (parseError) {
                    console.error('Failed to parse settings for stats:', parseError);
                    settings = null;
                }
            }
        }
        
        // Fallback to localStorage
        if (!settings) {
            const settingsStr = localStorage.getItem('userSettings');
            if (settingsStr) {
                try {
                    settings = JSON.parse(settingsStr);
                } catch (error) {
                    console.error('Failed to parse localStorage settings for stats:', error);
                }
            }
        }
        
        // Calculate stats from history
        let totalQuizzes = 0;
        let totalQuestionsAnswered = 0;
        let totalCorrect = 0;
        let averageScore = 0;
        
        if (settings && settings.answered_questions_history && Array.isArray(settings.answered_questions_history)) {
            const history = settings.answered_questions_history;
            
            // Count questions and correct answers
            totalQuestionsAnswered = history.length;
            totalCorrect = history.filter(q => q.is_correct === true).length;
            
            // Estimate total quizzes (rough estimate: questions / average 10 questions per quiz)
            totalQuizzes = Math.max(1, Math.ceil(totalQuestionsAnswered / 10));
            
            // Calculate average score
            if (totalQuestionsAnswered > 0) {
                averageScore = Math.round((totalCorrect / totalQuestionsAnswered) * 100);
            }
            
            console.log(`📊 Stats calculated: ${totalQuizzes} quizzes, ${totalQuestionsAnswered} questions, ${totalCorrect} correct (${averageScore}%)`);
        } else {
            console.log('📊 No history found, showing zero stats');
        }
        
        // Update the dashboard stat cards
        const statCards = document.querySelectorAll('.stat-card h3');
        if (statCards.length >= 3) {
            statCards[0].textContent = totalQuizzes.toString();
            statCards[1].textContent = averageScore + '%';
            statCards[2].textContent = totalQuestionsAnswered.toString();
            
            console.log('✅ Dashboard stats updated successfully');
        } else {
            console.error('❌ Could not find all stat card elements');
        }
        
    } catch (error) {
        console.error('❌ Error updating stats:', error);
        
        // Fallback to show zeros if there's an error
        const statCards = document.querySelectorAll('.stat-card h3');
        if (statCards.length >= 3) {
            statCards[0].textContent = '0';
            statCards[1].textContent = '0%';
            statCards[2].textContent = '0';
        }
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (event) => {
    if (currentScreen === 'quiz' && document.getElementById('quiz-game').style.display !== 'none') {
        if (event.key >= '1' && event.key <= '4') {
            selectAnswer(parseInt(event.key) - 1);
        } else if (event.key === 'Enter') {
            if (!isReviewMode) {
                submitAnswer();
            }
        } else if (event.key === 'ArrowLeft') {
            showPreviousQuestion();
        } else if (event.key === 'ArrowRight') {
            showNextQuestion();
        } else if (event.key === ' ') {
            event.preventDefault();
            if (isReviewMode) {
                loadNextNewQuestion();
            }
        }
    }
}); 

// API and Mode functions
function updateModeInfo() {
    const mode = document.getElementById('quiz-mode').value;
    const modeInfo = document.getElementById('mode-info');
    const apiStatus = document.getElementById('api-status');
    
    if (mode === 'online') {
        modeInfo.innerHTML = '<small>🌐 Using cloud APIs for generation</small>';
        apiStatus.style.display = 'block';
        checkApiStatus();
    } else if (mode === 'offline') {
        modeInfo.innerHTML = '<small>🚀 Local generation with available GPU resources</small>';
        apiStatus.style.display = 'none';
    } else {
        modeInfo.innerHTML = '<small>🤖 Auto-selecting best available method</small>';
        apiStatus.style.display = 'none';
    }
}

function updateGameModeInfo() {
    const gameMode = document.getElementById('quiz-game-mode').value;
    const gameModeInfo = document.getElementById('game-mode-info');
    
    if (!gameModeInfo) return;
    
    if (gameMode === 'serious') {
        gameModeInfo.innerHTML = '<small>⏱️ <strong>Serious Mode:</strong> 5-minute timer, focused UI, aggressive difficulty adjustment</small>';
    } else {
        gameModeInfo.innerHTML = '<small>🎵 <strong>Casual Mode:</strong> Relaxed learning with background music and no time pressure</small>';
    }
    
    // Update submode info when game mode changes
    updateSubmodeInfo();
}

function updateSubmodeInfo() {
    const gameMode = document.getElementById('quiz-game-mode').value;
    const submode = document.getElementById('quiz-submode').value;
    const infoDiv = document.getElementById('submode-info');
    
    if (!infoDiv) return;
    
    const questionTypeLabel = getQuestionTypeLabel(gameMode, submode);
    const submodeDescriptions = {
        'mixed': {
            'casual': '🔀 Engaging mix of visual math problems and interactive concepts',
            'serious': '🔀 Balanced combination of calculations and theoretical analysis'
        },
        'numerical': {
            'casual': '📊 Fun math problems with visual elements and real-world examples',
            'serious': '📊 Pure mathematical calculations, formulas, and problem-solving'
        },
        'conceptual': {
            'casual': '🧠 Interactive theory with engaging explanations and examples',
            'serious': '🧠 Deep theoretical analysis and advanced conceptual understanding'
        }
    };
    
    const description = submodeDescriptions[submode][gameMode];
    infoDiv.innerHTML = `<small>${description} → <strong>${questionTypeLabel}</strong></small>`;
}

function updateDifficultyInfo() {
    const difficultySelect = document.getElementById('quiz-difficulty');
    const difficultyInfo = document.getElementById('difficulty-info');
    const deepseekSection = document.getElementById('deepseek-section');

    if (!difficultySelect || !difficultyInfo) return;

    const difficulty = difficultySelect.value;
    const difficultyDescriptions = {
        'easy': '🟢 Basic understanding and factual recall',
        'medium': '🟡 Analytical thinking and moderate reasoning',
        'hard': '🔴 Advanced analysis and complex problem-solving',
        'expert': '🔥💀 PhD-LEVEL ENGINEERING BRUTALITY - Research-grade complexity requiring domain expertise 💀🔥 (Auto-DeepSeek)'
    };

    const description = difficultyDescriptions[difficulty] || difficultyDescriptions['medium'];
    difficultyInfo.innerHTML = `<small>${description}</small>`;

    // Show DeepSeek info section when expert mode is selected (for status display only)
    if (deepseekSection) {
        if (difficulty === 'expert') {
            // Re-check DeepSeek status when expert mode is selected
            if (pythonBridge && pythonBridge.getDeepSeekStatus) {
                pythonBridge.getDeepSeekStatus().then(statusJson => {
                    try {
                        const status = JSON.parse(statusJson);
                        if (status.available && status.ready) {
                            deepseekSection.style.display = 'block';
                        }
                    } catch (e) {
                        console.error('❌ Failed to parse DeepSeek status:', e);
                    }
                });
            }
        } else {
            deepseekSection.style.display = 'none';
        }
    }
}

function getQuestionTypeLabel(gameMode, submode) {
    const questionTypes = {
        'casual': {
            'mixed': 'Visual Mixed',
            'numerical': 'Visual Calculation',
            'conceptual': 'Visual Concept'
        },
        'serious': {
            'mixed': 'Mixed Analysis',
            'numerical': 'Pure Calculation',
            'conceptual': 'Theory Analysis'
        }
    };
    
    return questionTypes[gameMode][submode] || 'Mixed';
}

function checkApiStatus() {
    // Request API status from Python backend
    if (pythonBridge && pythonBridge.getApiStatus) {
        pythonBridge.getApiStatus();
    } else {
        // Fallback - show generic info
        updateApiProviders([
            {name: 'OpenAI', status: 'unknown'},
            {name: 'Anthropic', status: 'unknown'},
            {name: 'Gemini', status: 'unknown'},
            {name: 'Groq', status: 'unknown'},
            {name: 'OpenRouter', status: 'unknown'}
        ]);
    }
}

function updateApiProviders(providers) {
    const container = document.getElementById('api-providers');
    if (!container) return;
    
    const statusIcons = {
        'available': '✅',
        'unavailable': '❌',
        'unknown': '❓'
    };
    
    container.innerHTML = providers.map(provider => 
        `<span class="api-provider ${provider.status}">
            ${statusIcons[provider.status] || '❓'} ${provider.name}
        </span>`
    ).join(' ');
}

// LaTeX rendering functions
function renderLatex(element) {
    /* Render LaTeX in the given element using MathJax */
    console.log('🔧 MathJax status check:', {
        mathJaxExists: !!window.MathJax,
        hasTypesetPromise: !!(window.MathJax && window.MathJax.typesetPromise),
        hasStartup: !!(window.MathJax && window.MathJax.startup),
        startupReady: !!(window.MathJax && window.MathJax.startup && window.MathJax.startup.promise)
    });
    
    if (window.MathJax && window.MathJax.typesetPromise) {
        console.log('🧮 Calling MathJax.typesetPromise...');
        return window.MathJax.typesetPromise([element]).then(() => {
            console.log('✅ MathJax typesetPromise resolved successfully');
        }).catch(function (err) {
            console.error('❌ MathJax typesetPromise failed:', err);
            throw err;
        });
    } else if (window.MathJax && window.MathJax.Hub) {
        // Fallback for MathJax v2
        console.log('🔄 Using MathJax v2 fallback...');
        return new Promise((resolve) => {
            window.MathJax.Hub.Queue(["Typeset", window.MathJax.Hub, element]);
            window.MathJax.Hub.Queue(resolve);
        });
    } else {
        console.warn('⚠️ MathJax not loaded or typesetPromise not available, LaTeX will not render');
        console.log('🔍 Available MathJax properties:', window.MathJax ? Object.keys(window.MathJax) : 'None');
        return Promise.resolve();
    }
}

function renderLatexWithRetry(element, maxRetries = 3, delay = 500) {
    /**
     * Render LaTeX with retry logic and timing fixes for generated questions
     */
    return new Promise((resolve, reject) => {
        let attempts = 0;

        function attemptRender() {
            attempts++;
            console.log(`🔧 LaTeX render attempt ${attempts}/${maxRetries}`);

            if (!window.MathJax || !window.MathJax.typesetPromise) {
                if (attempts < maxRetries) {
                    console.log(`⏳ MathJax not ready, retrying in ${delay}ms...`);
                    setTimeout(attemptRender, delay);
                    return;
                } else {
                    console.warn('⚠️ MathJax not available after retries, using offline fallback');
                    renderLatexOffline(element).then(resolve).catch(reject);
                    return;
                }
            }

            // Pre-render check for LaTeX content
            const hasLatex = element.innerHTML.includes('$') ||
                           element.innerHTML.includes('\\') ||
                           element.innerHTML.includes('MathJax');

            if (!hasLatex) {
                console.log('✅ No LaTeX content detected, skipping render');
                resolve();
                return;
            }

            window.MathJax.typesetPromise([element]).then(() => {
                console.log(`✅ MathJax rendering completed successfully on attempt ${attempts}`);

                // Verify rendering actually worked
                const mathElements = element.querySelectorAll('.MathJax, mjx-container');
                if (mathElements.length > 0) {
                    console.log(`✅ Found ${mathElements.length} rendered math elements`);
                    resolve();
                } else if (attempts < maxRetries) {
                    console.warn(`⚠️ No rendered math elements found, retrying...`);
                    const retryDelay = delay * Math.pow(1.5, attempts - 1);
                    setTimeout(attemptRender, retryDelay);
                } else {
                    console.warn('⚠️ MathJax rendering verification failed, using offline fallback');
                    renderLatexOffline(element).then(resolve).catch(reject);
                }
            }).catch(error => {
                console.error(`❌ MathJax rendering failed on attempt ${attempts}:`, error);

                if (attempts < maxRetries) {
                    // Wait before retry with exponential backoff
                    const retryDelay = delay * Math.pow(2, attempts - 1);
                    console.log(`⏳ Retrying in ${retryDelay}ms...`);
                    setTimeout(attemptRender, retryDelay);
                } else {
                    console.warn('⚠️ MathJax failed after all retries, using offline fallback');
                    renderLatexOffline(element).then(resolve).catch(reject);
                }
            });
        }

        // Add small initial delay to allow DOM to settle
        setTimeout(attemptRender, 100);
    });
}

function processLatexText(text) {
    /* Process text to improve LaTeX rendering with PhD-level scientific terminology */
    if (!text) return text;
    
    // Handle scientific notation first (e.g., 10^-5, 2.5 x 10^23)
    text = text.replace(/(\d+(?:\.\d+)?)\s*[xX×]\s*10\^(-?\d+)/g, '$1 \\times 10^{$2}');
    text = text.replace(/\b10\^(-?\d+)/g, '10^{$1}');
    
    // 🔥 PhD-LEVEL: Handle advanced material science compounds and alloys
    text = text.replace(/\bGdFeCo\b/g, '$\\text{GdFeCo}$');
    text = text.replace(/\bFe3O4\b/g, '$\\text{Fe}_3\\text{O}_4$');
    text = text.replace(/\bFe2O3\b/g, '$\\text{Fe}_2\\text{O}_3$');
    text = text.replace(/\bCuO\b/g, '$\\text{CuO}$');
    text = text.replace(/\bTiO2\b/g, '$\\text{TiO}_2$');
    text = text.replace(/\bSiO2\b/g, '$\\text{SiO}_2$');
    text = text.replace(/\bAl2O3\b/g, '$\\text{Al}_2\\text{O}_3$');
    text = text.replace(/\bNiFe\b/g, '$\\text{NiFe}$');
    text = text.replace(/\bCoFe\b/g, '$\\text{CoFe}$');
    text = text.replace(/\bCoPt\b/g, '$\\text{CoPt}$');
    text = text.replace(/\bFePt\b/g, '$\\text{FePt}$');
    
    // 🔥 PhD-LEVEL: Advanced physics and chemistry compounds
    text = text.replace(/\bMnO2\b/g, '$\\text{MnO}_2$');
    text = text.replace(/\bCaCO3\b/g, '$\\text{CaCO}_3$');
    text = text.replace(/\bNaCl\b/g, '$\\text{NaCl}$');
    text = text.replace(/\bKCl\b/g, '$\\text{KCl}$');
    text = text.replace(/\bH2SO4\b/g, '$\\text{H}_2\\text{SO}_4$');
    text = text.replace(/\bHCl\b/g, '$\\text{HCl}$');
    text = text.replace(/\bNaOH\b/g, '$\\text{NaOH}$');
    text = text.replace(/\bCu2O\b/g, '$\\text{Cu}_2\\text{O}$');
    
    // 🔥 PhD-LEVEL: Handle crystallographic planes and directions
    text = text.replace(/\[(\d{3})\]/g, '$[$$1]$');
    text = text.replace(/\((\d{3})\)/g, '$($$1)$');
    
    // 🔥 PhD-LEVEL: Temperature expressions with Kelvin and Celsius
    text = text.replace(/(\d+(?:\.\d+)?)\s*K\b/g, '$$$1\\text{ K}$$');
    text = text.replace(/(\d+(?:\.\d+)?)\s*°C\b/g, '$$$1^{\\circ}\\text{C}$$');
    text = text.replace(/(\d+(?:\.\d+)?)\s*°F\b/g, '$$$1^{\\circ}\\text{F}$$');
    
    // 🔥 PhD-LEVEL: Magnetic and electrical units
    text = text.replace(/(\d+(?:\.\d+)?)\s*Tesla\b/g, '$$$1\\text{ T}$$');
    text = text.replace(/(\d+(?:\.\d+)?)\s*Gauss\b/g, '$$$1\\text{ G}$$');
    text = text.replace(/(\d+(?:\.\d+)?)\s*Oe\b/g, '$$$1\\text{ Oe}$$'); // Oersted
    text = text.replace(/(\d+(?:\.\d+)?)\s*A\/m\b/g, '$$$1\\text{ A/m}$$');
    text = text.replace(/(\d+(?:\.\d+)?)\s*J\/kg·K\b/g, '$$$1\\text{ J/(kg·K)}$$');
    text = text.replace(/(\d+(?:\.\d+)?)\s*eV\b/g, '$$$1\\text{ eV}$$');
    text = text.replace(/(\d+(?:\.\d+)?)\s*meV\b/g, '$$$1\\text{ meV}$$');
    
    // 🔥 PhD-LEVEL: Energy and power units  
    text = text.replace(/(\d+(?:\.\d+)?)\s*J\/mol\b/g, '$$$1\\text{ J/mol}$$');
    text = text.replace(/(\d+(?:\.\d+)?)\s*kJ\/mol\b/g, '$$$1\\text{ kJ/mol}$$');
    text = text.replace(/(\d+(?:\.\d+)?)\s*erg\b/g, '$$$1\\text{ erg}$$');
    text = text.replace(/(\d+(?:\.\d+)?)\s*W\/m²\b/g, '$$$1\\text{ W/m}^2$$');
    
    // Convert common math expressions to LaTeX
    text = text.replace(/\b(\d+)\/(\d+)\b/g, '\\frac{$1}{$2}'); // Convert fractions like 5/5 to \frac{5}{5}
    text = text.replace(/\b(\d+)\*(\d+)\b/g, '$1 \\times $2'); // Convert multiplication like 8*4 to 8 \times 4
    
    // Handle powers with negative exponents (e.g., x^-2, a^-n)
    text = text.replace(/\b(\w+)\^(-?\d+)\b/g, '$1^{$2}'); // Convert powers like x^2 or x^-2 to x^{2} or x^{-2}
    text = text.replace(/\b([A-Za-z]+)(\d+)\b/g, '$1_{$2}'); // Convert subscripts like H2 to H_{2}
    
    // 🔥 PhD-LEVEL: Handle Greek letters in technical contexts
    text = text.replace(/\bDelta\b/g, '$\\Delta$');
    text = text.replace(/\btheta\b/g, '$\\theta$');
    text = text.replace(/\bTheta\b/g, '$\\Theta$');
    text = text.replace(/\balpha\b/g, '$\\alpha$');
    text = text.replace(/\bAlpha\b/g, '$\\Alpha$');
    text = text.replace(/\bbeta\b/g, '$\\beta$');
    text = text.replace(/\bBeta\b/g, '$\\Beta$');
    text = text.replace(/\bgamma\b/g, '$\\gamma$');
    text = text.replace(/\bGamma\b/g, '$\\Gamma$');
    text = text.replace(/\bdelta\b/g, '$\\delta$');
    text = text.replace(/\bepsilon\b/g, '$\\epsilon$');
    text = text.replace(/\bEpsilon\b/g, '$\\Epsilon$');
    text = text.replace(/\blambda\b/g, '$\\lambda$');
    text = text.replace(/\bLambda\b/g, '$\\Lambda$');
    text = text.replace(/\bmu\b/g, '$\\mu$');
    text = text.replace(/\bnu\b/g, '$\\nu$');
    text = text.replace(/\bxi\b/g, '$\\xi$');
    text = text.replace(/\bXi\b/g, '$\\Xi$');
    text = text.replace(/\bomicron\b/g, '$\\omicron$');
    text = text.replace(/\bpi\b/g, '$\\pi$');
    text = text.replace(/\bPi\b/g, '$\\Pi$');
    text = text.replace(/\brho\b/g, '$\\rho$');
    text = text.replace(/\bRho\b/g, '$\\Rho$');
    text = text.replace(/\bsigma\b/g, '$\\sigma$');
    text = text.replace(/\bSigma\b/g, '$\\Sigma$');
    text = text.replace(/\btau\b/g, '$\\tau$');
    text = text.replace(/\bTau\b/g, '$\\Tau$');
    text = text.replace(/\bupsilon\b/g, '$\\upsilon$');
    text = text.replace(/\bUpsilon\b/g, '$\\Upsilon$');
    text = text.replace(/\bphi\b/g, '$\\phi$');
    text = text.replace(/\bPhi\b/g, '$\\Phi$');
    text = text.replace(/\bchi\b/g, '$\\chi$');
    text = text.replace(/\bChi\b/g, '$\\Chi$');
    text = text.replace(/\bpsi\b/g, '$\\psi$');
    text = text.replace(/\bPsi\b/g, '$\\Psi$');
    text = text.replace(/\bomega\b/g, '$\\omega$');
    text = text.replace(/\bOmega\b/g, '$\\Omega$');
    
    // Wrap mathematical expressions in $ delimiters if not already wrapped
    text = text.replace(/(?<!\$)((?:\d+(?:\.\d+)?)\s*\\times\s*10\^{-?\d+})(?!\$)/g, '$$1$');
    text = text.replace(/(?<!\$)(10\^{-?\d+})(?!\$)/g, '$$1$');
    text = text.replace(/(?<!\$)\\frac\{[^}]+\}\{[^}]+\}(?!\$)/g, '$$$&$$');
    text = text.replace(/(?<!\$)(\w+\^{-?\d+})(?!\$)/g, '$$1$');
    text = text.replace(/(?<!\$)(\w+_{[^}]+})(?!\$)/g, '$$1$');
    
    // Handle common chemical formulas
    text = text.replace(/\bH2O\b/g, '$\\text{H}_2\\text{O}$');
    text = text.replace(/\bCO2\b/g, '$\\text{CO}_2$');
    text = text.replace(/\bNH3\b/g, '$\\text{NH}_3$');
    text = text.replace(/\bCH4\b/g, '$\\text{CH}_4$');
    text = text.replace(/\bO2\b/g, '$\\text{O}_2$');
    text = text.replace(/\bN2\b/g, '$\\text{N}_2$');
    text = text.replace(/\bCl2\b/g, '$\\text{Cl}_2$');
    text = text.replace(/\bBr2\b/g, '$\\text{Br}_2$');
    text = text.replace(/\bI2\b/g, '$\\text{I}_2$');
    
    // Handle degree symbol
    text = text.replace(/(\d+)\s*degrees?/g, '$1^{\\circ}$');
    text = text.replace(/(\d+)°/g, '$1^{\\circ}$');
    
    // Handle inequalities
    text = text.replace(/\s<=\s/g, ' $\\leq$ ');
    text = text.replace(/\s>=\s/g, ' $\\geq$ ');
    text = text.replace(/\s!=\s/g, ' $\\neq$ ');
    text = text.replace(/\s<<\s/g, ' $\\ll$ ');
    text = text.replace(/\s>>\s/g, ' $\\gg$ ');
    text = text.replace(/\s~\s/g, ' $\\sim$ ');
    text = text.replace(/\s±\s/g, ' $\\pm$ ');
    text = text.replace(/\s∝\s/g, ' $\\propto$ ');
    
    // Handle square roots
    text = text.replace(/sqrt\(([^)]+)\)/g, '$\\sqrt{$1}$');
    
    // 🔥 PhD-LEVEL: Handle partial derivatives and gradients
    text = text.replace(/∂([a-zA-Z])/g, '$\\partial $1$');
    text = text.replace(/∇([a-zA-Z])/g, '$\\nabla $1$');
    text = text.replace(/∆([a-zA-Z])/g, '$\\Delta $1$');
    
    // 🔥 PhD-LEVEL: Handle integrals and summations
    text = text.replace(/∫/g, '$\\int$');
    text = text.replace(/∑/g, '$\\sum$');
    text = text.replace(/∏/g, '$\\prod$');
    
    return text;
}

function updateQuestionWithLatex(questionText, element) {
    /* Update question text with LaTeX processing and rendering */
    console.log('🔍 Original question text:', questionText);
    const processedText = processLatexText(questionText);
    console.log('🔬 Processed question text:', processedText);
    
    // Add visual LaTeX processing indicator
    const hasLatex = processedText.includes('$') || processedText.includes('\\');
    if (hasLatex) {
        element.innerHTML = processedText + ' <span style="color: #3b82f6; font-size: 0.8em;">⚡ LaTeX Processing...</span>';
    } else {
        element.innerHTML = processedText;
    }
    
    // Render LaTeX after setting the content
    console.log('🧮 Attempting MathJax rendering...');
    return renderLatex(element).then(() => {
        console.log('✅ MathJax rendering completed successfully');
        // Remove processing indicator
        if (hasLatex) {
            element.innerHTML = processedText;
            // Re-render to apply the processed LaTeX
            return renderLatex(element);
        }
    }).catch(err => {
        console.error('❌ MathJax rendering failed:', err);
        console.log('🔧 MathJax status:', window.MathJax ? 'Loaded' : 'Not loaded');
        // Remove processing indicator even on error
        if (hasLatex) {
            element.innerHTML = processedText + ' <span style="color: #ef4444; font-size: 0.8em;">❌ LaTeX Error</span>';
        }
    });
}

function updateExplanationWithLatex(explanationText, element) {
    /* Update explanation text with LaTeX processing and rendering */
    const processedText = processLatexText(explanationText);
    element.innerHTML = `
        <div class="explanation">
            <h4>💡 Explanation:</h4>
            <p>${processedText}</p>
        </div>
    `;
    element.style.display = 'block';
    
    // Render LaTeX after setting the content
    return renderLatex(element);
}

// Initialize MathJax when it's ready
document.addEventListener('DOMContentLoaded', function() {
    // Wait for MathJax to be ready
    if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
        window.MathJax.startup.promise.then(() => {
            console.log('✅ MathJax loaded and ready for LaTeX rendering');
        }).catch((error) => {
            console.warn('MathJax initialization error:', error);
        });
    } else {
        // MathJax not ready yet, wait for it
        const checkMathJax = setInterval(() => {
            if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
                clearInterval(checkMathJax);
                window.MathJax.startup.promise.then(() => {
                    console.log('✅ MathJax loaded and ready for LaTeX rendering');
                }).catch((error) => {
                    console.warn('MathJax initialization error:', error);
                });
            }
        }, 100);
        
        // Stop checking after 10 seconds
        setTimeout(() => clearInterval(checkMathJax), 10000);
    }
});

// Auto-save settings when API keys are entered
function setupAutoSave() {
    const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
    
    providers.forEach(provider => {
        const input = document.getElementById(`${provider}-api-key`);
        if (input) {
            // Update provider status on input changes
            input.addEventListener('input', updateProviderStatuses);
            input.addEventListener('blur', updateProviderStatuses);
            
            // 🔒 SECURITY FIX #16: Save on blur with secure storage
            input.addEventListener('blur', async () => {
                console.log(`Auto-saving settings after ${provider} API key change`);
                
                // Securely store API key if it has a value
                if (input.value.trim()) {
                    await window.secureApiKeyManager.store(provider, input.value.trim());
                    console.log(`🔒 ${provider} API key securely stored`);
                }
                
                // Save other settings
                await saveSettings();
            });
            
            // 🔒 SECURITY FIX #16: Save on Enter key with secure storage
            input.addEventListener('keypress', async (e) => {
                if (e.key === 'Enter') {
                    console.log(`Auto-saving settings after Enter in ${provider} API key`);
                    
                    // Securely store API key if it has a value
                    if (input.value.trim()) {
                        await window.secureApiKeyManager.store(provider, input.value.trim());
                        console.log(`🔒 ${provider} API key securely stored`);
                    }
                    
                    // Save other settings
                    await saveSettings();
                }
            });
            
            // 🔒 SECURITY FIX #16: Debounced secure storage while typing
            input.addEventListener('input', () => {
                // Debounced secure storage
                clearTimeout(input.secureBackupTimer);
                input.secureBackupTimer = setTimeout(async () => {
                    if (input.value.trim()) {
                        await window.secureApiKeyManager.store(provider, input.value.trim());
                        console.log(`🔒 ${provider} API key securely stored (debounced)`);
                    }
                }, 1500); // Secure backup after 1.5 seconds of no typing
            });
        }
    });
    
    // Auto-save other settings changes too
    const settingsInputs = [
        'quiz-game-mode', 'quiz-difficulty', 'quiz-mode', 'quiz-submode',
        'default-timer', 'show-answers'
    ];
    
    settingsInputs.forEach(inputId => {
        const input = document.getElementById(inputId);
        if (input) {
            input.addEventListener('change', () => {
                console.log(`Auto-saving settings after ${inputId} change`);
                setTimeout(saveSettings, 100); // Small delay to ensure UI updates
            });
        }
    });
    
    // ✅ NEW: Set up periodic API key persistence check
    setInterval(() => {
        ensureApiKeyPersistence();
    }, 30000); // Check every 30 seconds
    
    // Set up theme toggle auto-save (already handled in toggleTheme function)
    console.log('✅ Auto-save listeners set up for all settings with enhanced session storage');
}

function showTemporaryMessage(message, type = 'info') {
    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `temp-message ${type}`;
    messageDiv.textContent = message;
    
    // Style the message
    Object.assign(messageDiv.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '12px 24px',
        borderRadius: '8px',
        fontWeight: '600',
        zIndex: '10000',
        opacity: '0',
        transform: 'translateX(100%)',
        transition: 'all 0.3s ease'
    });
    
    // Set colors based on type
    const colors = {
        success: { bg: 'rgba(16, 185, 129, 0.9)', color: 'white' },
        error: { bg: 'rgba(239, 68, 68, 0.9)', color: 'white' },
        warning: { bg: 'rgba(245, 158, 11, 0.9)', color: 'white' },
        info: { bg: 'rgba(59, 130, 246, 0.9)', color: 'white' }
    };
    
    const color = colors[type] || colors.info;
    messageDiv.style.backgroundColor = color.bg;
    messageDiv.style.color = color.color;
    
    document.body.appendChild(messageDiv);
    
    // Animate in
    setTimeout(() => {
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after delay
    setTimeout(() => {
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.parentNode.removeChild(messageDiv);
            }
        }, 300);
    }, 3000);
}

// ✅ NEW: Update API key status indicators
function updateApiKeyStatusIndicators() {
    try {
        const persistenceStatus = document.getElementById('api-persistence-status');
        const sessionStatus = document.getElementById('api-session-status');
        
        if (persistenceStatus) {
            // Check if localStorage has API keys
            const savedSettings = localStorage.getItem('userSettings');
            let hasStoredKeys = false;
            
            if (savedSettings) {
                try {
                    const settings = JSON.parse(savedSettings);
                    if (settings.api_keys) {
                        const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
                        hasStoredKeys = providers.some(provider => 
                            settings.api_keys[provider] && settings.api_keys[provider].trim()
                        );
                    }
                } catch (error) {
                    console.error('Error checking stored API keys:', error);
                }
            }
            
            persistenceStatus.innerHTML = hasStoredKeys 
                ? '💾 Auto-save: Keys Stored' 
                : '💾 Auto-save: No Keys';
            persistenceStatus.style.color = hasStoredKeys ? '#10b981' : '#6b7280';
        }
        
        if (sessionStatus) {
            // Check if secure API key manager has stored keys
            let hasSecureKeys = false;
            
            try {
                // Use async IIFE to handle the async call
                (async () => {
                    const providers = await window.secureApiKeyManager.list();
                    hasSecureKeys = providers && providers.length > 0;
                    
                    sessionStatus.innerHTML = hasSecureKeys 
                        ? '🔒 Secure: Keys Stored' 
                        : '🔒 Secure: Ready';
                    sessionStatus.style.color = hasSecureKeys ? '#10b981' : '#6b7280';
                })();
            } catch (error) {
                console.error('Error checking secure API keys:', error);
                sessionStatus.innerHTML = '🔒 Secure: Ready';
                sessionStatus.style.color = '#6b7280';
            }
        }
        
    } catch (error) {
        console.error('❌ Error updating API key status indicators:', error);
    }
}

function showSpellCorrectionFeedback(profile) {
    /* Show AI spell correction feedback with smart UI suggestions */
    try {
        if (!profile.ui_feedback || !profile.ui_feedback.show_correction_notice) {
                        return;
                    }
        
        const originalTopic = profile.original_topic || '';
        const correctedTopic = profile.corrected_topic || '';
        const corrections = profile.corrections_made || [];
        
        if (corrections.length === 0) {
            return;
        }
        
        // Create a sophisticated correction notification
        let correctionText = corrections.join(', ');
        let message = `🤖 AI Correction: ${correctionText}`;
        
        // If it's a single correction, make it more conversational
        if (corrections.length === 1) {
            const correction = corrections[0];
            if (correction.includes('→')) {
                const [from, to] = correction.split(' → ');
                message = `🤖 Did you mean "${to}"? (auto-corrected "${from}")`;
            }
        } else {
            message = `🤖 AI corrected ${corrections.length} spelling issues: ${correctionText}`;
        }
        
        // Show correction with option to accept
        showTemporaryMessage(message, 'info');
        
        // Auto-update the topic input field with corrected version
        const topicInput = document.getElementById('quiz-topic');
        if (topicInput && correctedTopic && correctedTopic !== originalTopic) {
            // Store original for undo if needed
            topicInput.dataset.originalValue = originalTopic;
            topicInput.dataset.correctedValue = correctedTopic;
            
            // Apply correction with visual feedback
            setTimeout(() => {
                topicInput.value = correctedTopic;
                topicInput.style.background = 'rgba(16, 185, 129, 0.1)'; // Green highlight
                topicInput.style.borderColor = 'rgba(16, 185, 129, 0.5)';
                
                // Remove highlight after a moment
                setTimeout(() => {
                    topicInput.style.background = '';
                    topicInput.style.borderColor = '';
                }, 2000);
            }, 500);
            
            console.log(`🤖 Auto-applied correction: "${originalTopic}" → "${correctedTopic}"`);
        }
        
                } catch (error) {
        console.error('❌ Error showing spell correction feedback:', error);
    }
}

function showTopicAnalysisFeedback(profile) {
    /* Show subtle feedback about the topic analysis enhanced with AI awareness */
    try {
        if (profile.confidence === 'high' && profile.detected_type !== 'unknown') {
            let message = '';
            let icon = '🧠';
            
            // Include spell correction context if available
            const correctionContext = profile.spelling_corrected ? ' (after AI spell correction)' : '';
            
            switch (profile.detected_type) {
                case 'conceptual':
                    message = `Perfect for conceptual questions! This topic focuses on understanding and theory${correctionContext}.`;
                    icon = '💭';
                    break;
                case 'numerical':
                    message = `Great for numerical questions! This topic involves calculations and quantitative analysis${correctionContext}.`;
                    icon = '🔢';
                    break;
                case 'mixed':
                    message = `Excellent topic! Both conceptual and numerical questions work well here${correctionContext}.`;
                    icon = '🎯';
                    break;
            }
            
            if (message) {
                // Slight delay if spell correction was also shown
                const delay = profile.spelling_corrected ? 3500 : 0;
                setTimeout(() => {
                    showTemporaryMessage(`${icon} ${message}`, 'info');
                }, delay);
            }
        }
        
        // Also show confidence level for debugging in development
        if (profile.confidence === 'medium') {
            console.log(`🧠 Medium confidence analysis for "${profile.original_topic || profile.corrected_topic}"`);
        }
        
    } catch (error) {
        console.error('❌ Error showing topic analysis feedback:', error);
    }
}

function showQuestionTypeSuggestion(profile, suggestedType, currentType) {
    /**
     * 🛡️ CRITICAL UX FIX #16: Show user-friendly suggestion instead of hijacking UI
     * 
     * This replaces the old behavior where the UI would forcefully change
     * user selections without consent, which was confusing and jarring.
     */
    try {
        const typeNames = {
            'conceptual': 'Conceptual',
            'numerical': 'Numerical', 
            'mixed': 'Mixed'
        };
        
        const suggestedName = typeNames[suggestedType] || suggestedType;
        const currentName = typeNames[currentType] || currentType;
        
        let confidence_text = '';
        if (profile.confidence === 'high') {
            confidence_text = 'strongly suggests';
        } else if (profile.confidence === 'medium') {
            confidence_text = 'suggests';
        } else {
            confidence_text = 'might work better with';
        }
        
        const message = `💡 AI Analysis: This topic ${confidence_text} "${suggestedName}" questions. ` +
                       `Currently set to "${currentName}". Would you like to switch?`;
        
        // Create a non-intrusive notification with action buttons
        showActionableNotification(message, [
            {
                text: `Switch to ${suggestedName}`,
                action: () => {
                    const submodeSelect = document.getElementById('quiz-submode');
                    if (submodeSelect) {
                        submodeSelect.value = suggestedType;
                        updateSubmodeInfo();
                        showTemporaryMessage(`✅ Switched to ${suggestedName} questions`, 'success');
                        console.log(`👤 User accepted suggestion: ${currentType} → ${suggestedType}`);
                    }
                },
                primary: true
            },
            {
                text: `Keep ${currentName}`,
                action: () => {
                    showTemporaryMessage(`👍 Keeping ${currentName} questions`, 'info');
                    console.log(`👤 User declined suggestion: staying with ${currentType}`);
                },
                primary: false
            }
        ], 'suggestion', 8000); // 8 second timeout
        
    } catch (error) {
        console.error('❌ Error showing question type suggestion:', error);
    }
}

function showActionableNotification(message, actions, type = 'info', timeout = 5000) {
    /**
     * Show a non-intrusive notification with action buttons
     * This respects user autonomy while providing helpful suggestions
     */
    try {
        // Remove any existing actionable notifications
        const existing = document.querySelector('.actionable-notification');
        if (existing) {
            existing.remove();
        }
        
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `actionable-notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-message">${message}</div>
                <div class="notification-actions">
                    ${actions.map((action, index) => 
                        `<button class="btn ${action.primary ? 'btn-primary' : 'btn-secondary'} btn-sm notification-btn" 
                                data-action="${index}">${action.text}</button>`
                    ).join('')}
                </div>
            </div>
            <button class="notification-close">&times;</button>
        `;
        
        // Add event listeners
        notification.querySelectorAll('.notification-btn').forEach((btn, index) => {
            btn.addEventListener('click', () => {
                if (actions[index] && actions[index].action) {
                    actions[index].action();
                }
                notification.remove();
            });
        });
        
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });
        
        // Add to DOM
        document.body.appendChild(notification);
        
        // Auto-remove after timeout
        if (timeout > 0) {
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, timeout);
        }
        
        // Animate in
        requestAnimationFrame(() => {
            notification.classList.add('show');
        });
        
    } catch (error) {
        console.error('❌ Error showing actionable notification:', error);
        // Fallback to simple message
        showTemporaryMessage(message, type);
    }
}

// 🏆 QUESTION HISTORY & REVIEW FUNCTIONALITY

let questionHistoryData = [];
let filteredQuestions = [];
let currentStats = {};

async function loadQuestionHistory() {
    /* Load all question history from the database - ASYNC VERSION */
    console.log('📚 Loading question history asynchronously...');

    // Show enhanced loading state with progress indicator
    const loadingElement = document.getElementById('loading-history');
    const noQuestionsElement = document.getElementById('no-questions');
    const questionsListElement = document.getElementById('questions-list');

    if (loadingElement) {
        loadingElement.style.display = 'block';
        loadingElement.innerHTML = `
            <div class="loading-spinner">
                <div class="spinner"></div>
                <p>Loading question history...</p>
            </div>
        `;
    }
    if (noQuestionsElement) noQuestionsElement.style.display = 'none';
    if (questionsListElement) questionsListElement.innerHTML = '';

    try {
        // FIXED: Asynchronous bridge call with timeout and retry logic
        if (pythonBridge && typeof pythonBridge.getQuestionHistory === 'function') {
            let result;

            // Add timeout and retry mechanism
            const maxRetries = 3;
            const timeout = 10000; // 10 seconds

            for (let attempt = 1; attempt <= maxRetries; attempt++) {
                try {
                    console.log(`📡 Attempt ${attempt}/${maxRetries} to load question history...`);

                    // Wrap synchronous call in Promise with timeout
                    result = await new Promise((resolve, reject) => {
                        const timeoutId = setTimeout(() => {
                            reject(new Error(`Request timeout after ${timeout}ms`));
                        }, timeout);

                        try {
                            const bridgeResult = pythonBridge.getQuestionHistory();
                            clearTimeout(timeoutId);
                            resolve(bridgeResult);
                        } catch (error) {
                            clearTimeout(timeoutId);
                            reject(error);
                        }
                    });

                    break; // Success, exit retry loop

                } catch (bridgeError) {
                    console.error(`❌ Bridge call attempt ${attempt} failed:`, bridgeError);

                    if (attempt === maxRetries) {
                        throw new Error(`Failed after ${maxRetries} attempts: ${bridgeError.message}`);
                    }

                    // Show retry message
                    if (loadingElement) {
                        loadingElement.innerHTML = `
                            <div class="loading-spinner">
                                <div class="spinner"></div>
                                <p>Retrying... (${attempt}/${maxRetries})</p>
                            </div>
                        `;
                    }

                    // Wait before retry
                    await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
                }
            }
            
            // FIXED: Handle the case where result might be undefined or null
            if (result === undefined || result === null) {
                console.warn('⚠️ Bridge returned undefined/null result, treating as empty response');
                result = '{"success": true, "questions": [], "total_count": 0}';
            }
            
            // FIXED: Robust JSON parsing
            let data;
            if (typeof result === 'string' && result.length > 0) {
                try {
                    data = JSON.parse(result);
                } catch (parseError) {
                    console.error('❌ JSON parse failed:', parseError, 'Raw:', result);
                    // Fallback to empty state
                    data = { success: true, questions: [], total_count: 0 };
                }
            } else if (typeof result === 'object' && result !== null) {
                data = result;
            } else {
                console.warn('⚠️ Unexpected result type:', typeof result, 'Falling back to empty state');
                data = { success: true, questions: [], total_count: 0 };
            }
            
            // FIXED: Ensure data structure is valid
            if (!data || typeof data !== 'object') {
                console.warn('⚠️ Invalid data structure, using fallback');
                data = { success: true, questions: [], total_count: 0 };
            }
            
            // FIXED: More robust success checking
            if (data.success === true || (data.questions && Array.isArray(data.questions))) {
                questionHistoryData = data.questions || [];
                filteredQuestions = [...questionHistoryData];
                displayQuestions(filteredQuestions);
                updateTopicFilterOptions();
                
                console.log(`✅ Loaded ${questionHistoryData.length} questions from history`);
                showTemporaryMessage(`📚 Loaded ${questionHistoryData.length} questions from history`, 'success');
            } else {
                console.warn('⚠️ Backend indicated failure, showing empty state');
                showNoQuestionsMessage();
                showTemporaryMessage(`📚 No questions available yet. Generate some questions first!`, 'info');
            }
        } else {
            console.warn('⚠️ Python bridge not available for review functionality');
            showNoQuestionsMessage();
            showUserFriendlyError('Connection Error', 'Unable to connect to the backend. Please refresh the page and try again.', 'warning');
        }
    } catch (error) {
        console.error('❌ Error loading question history:', error);
        showNoQuestionsMessage();

        // Enhanced error handling with specific messages
        let errorTitle = 'Loading Error';
        let errorMessage = 'Failed to load question history. ';
        let errorType = 'error';

        if (error.message.includes('timeout')) {
            errorTitle = 'Connection Timeout';
            errorMessage = 'The request took too long. Please check your connection and try again.';
            errorType = 'warning';
        } else if (error.message.includes('Failed after')) {
            errorTitle = 'Connection Failed';
            errorMessage = 'Unable to reach the backend after multiple attempts. Please refresh the page.';
            errorType = 'error';
        } else if (error.message.includes('JSON')) {
            errorTitle = 'Data Error';
            errorMessage = 'Received invalid data from the backend. Please try again.';
            errorType = 'warning';
        } else {
            errorMessage += 'Please try refreshing the page or contact support if the problem persists.';
        }

        showUserFriendlyError(errorTitle, errorMessage, errorType);
    }

    // Hide loading state
    if (loadingElement) loadingElement.style.display = 'none';
}

function displayQuestions(questions) {
    /* Display questions in the review interface - SECURE VERSION */
    const questionsList = document.getElementById('questions-list');
    const noQuestions = document.getElementById('no-questions');

    if (!questions || questions.length === 0) {
        showNoQuestionsMessage();
        return;
    }

    noQuestions.style.display = 'none';

    // SECURITY FIX: Use secure DOM manipulation instead of innerHTML
    questionsList.innerHTML = ''; // Clear existing content

    questions.forEach(question => {
        // Validate and sanitize question data
        if (!question || !question.id) {
            console.warn('⚠️ Skipping invalid question:', question);
            return;
        }

        const questionCard = document.createElement('div');
        questionCard.className = 'question-card';
        questionCard.onclick = () => showQuestionDetail(escapeHtml(question.id));

        // Create header section
        const header = document.createElement('div');
        header.className = 'question-card-header';

        const meta = document.createElement('div');
        meta.className = 'question-meta';

        // Safely create difficulty badge
        const difficultyBadge = document.createElement('span');
        difficultyBadge.className = `difficulty-badge difficulty-${escapeHtml(question.difficulty || 'unknown')}`;
        difficultyBadge.textContent = `${getDifficultyIcon(question.difficulty)} ${question.difficulty || 'Unknown'}`;

        // Safely create topic badge
        const topicBadge = document.createElement('span');
        topicBadge.className = 'topic-badge';
        topicBadge.textContent = question.topic || 'Unknown Topic';

        // Safely create type badge
        const typeBadge = document.createElement('span');
        typeBadge.className = 'type-badge';
        typeBadge.textContent = `${getTypeIcon(question.question_type)} ${question.question_type || 'Unknown'}`;

        meta.appendChild(difficultyBadge);
        meta.appendChild(topicBadge);
        meta.appendChild(typeBadge);

        // Create date section
        const dateDiv = document.createElement('div');
        dateDiv.className = 'question-date';
        dateDiv.textContent = formatDate(question.generated_at);

        header.appendChild(meta);
        header.appendChild(dateDiv);

        // Create preview section
        const preview = document.createElement('div');
        preview.className = 'question-preview';

        const questionTitle = document.createElement('h4');
        questionTitle.textContent = truncateText(question.question || 'No question text', 150);

        const stats = document.createElement('div');
        stats.className = 'question-stats';

        const statsSpan = document.createElement('span');
        if (question.times_answered > 0) {
            statsSpan.className = 'accuracy-stat';
            statsSpan.textContent = `📊 ${Math.round((question.accuracy || 0) * 100)}% accuracy (${question.times_answered} attempts)`;
        } else {
            statsSpan.className = 'never-attempted';
            statsSpan.textContent = '💫 Never attempted';
        }

        stats.appendChild(statsSpan);
        preview.appendChild(questionTitle);
        preview.appendChild(stats);

        // Create actions section
        const actions = document.createElement('div');
        actions.className = 'question-actions';

        const practiceBtn = document.createElement('button');
        practiceBtn.className = 'btn-small btn-primary';
        practiceBtn.textContent = 'Practice';
        practiceBtn.onclick = (event) => {
            event.stopPropagation();
            practiceQuestion(escapeHtml(question.id));
        };

        const detailsBtn = document.createElement('button');
        detailsBtn.className = 'btn-small btn-secondary';
        detailsBtn.textContent = 'Details';
        detailsBtn.onclick = (event) => {
            event.stopPropagation();
            showQuestionDetail(escapeHtml(question.id));
        };

        actions.appendChild(practiceBtn);
        actions.appendChild(detailsBtn);

        // Assemble the card
        questionCard.appendChild(header);
        questionCard.appendChild(preview);
        questionCard.appendChild(actions);

        questionsList.appendChild(questionCard);
    });
}

// SECURITY UTILITY FUNCTIONS
function escapeHtml(unsafe) {
    /* Escape HTML to prevent XSS attacks */
    if (typeof unsafe !== 'string') {
        return String(unsafe || '');
    }
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function showUserFriendlyError(title, message, type = 'error') {
    /* Show user-friendly error messages with modal or toast */
    console.error(`${title}: ${message}`);

    // Create error modal
    const modal = document.createElement('div');
    modal.className = 'error-modal-overlay';
    modal.innerHTML = `
        <div class="error-modal ${type}">
            <div class="error-header">
                <h3>${escapeHtml(title)}</h3>
                <button class="close-btn" onclick="this.closest('.error-modal-overlay').remove()">×</button>
            </div>
            <div class="error-body">
                <p>${escapeHtml(message)}</p>
            </div>
            <div class="error-footer">
                <button class="btn btn-primary" onclick="closeApiTestDialog()">Got it!</button>
            </div>
        </div>
        <div class="error-modal-backdrop" onclick="closeApiTestDialog()"></div>
    `;

    document.body.appendChild(modal);

    // Auto-close after 10 seconds for warnings
    if (type === 'warning') {
        setTimeout(() => {
            if (modal.parentNode) {
                modal.remove();
            }
        }, 10000);
    }

    // Also show temporary message
    showTemporaryMessage(`${title}: ${message}`, type);
}

function showNoQuestionsMessage() {
    /* Show message when no questions are available */
    document.getElementById('no-questions').style.display = 'block';
    document.getElementById('questions-list').innerHTML = '';
}

function getDifficultyIcon(difficulty) {
    /* Get icon for difficulty level */
    const icons = {
        'easy': '🟢',
        'medium': '🟡', 
        'hard': '🔴',
        'expert': '🔥💀'
    };
    return icons[difficulty] || '⚪';
}

function getTypeIcon(questionType) {
    /* Get icon for question type */
    const icons = {
        'numerical': '📊',
        'conceptual': '🧠',
        'mixed': '🔀'
    };
    return icons[questionType] || '❓';
}

function formatDate(dateString) {
    /* Format date for display */
    try {
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    } catch (error) {
        return dateString;
    }
}

function truncateText(text, maxLength) {
    /* Truncate text to specified length */
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function updateTopicFilterOptions() {
    /* Update topic filter dropdown with available topics */
    const topicFilter = document.getElementById('topic-filter');
    const topics = [...new Set(questionHistoryData.map(q => q.topic))].sort();
    
    // Clear existing options (except "All Topics")
    topicFilter.innerHTML = '<option value="">All Topics</option>';
    
    // Add topic options
    topics.forEach(topic => {
        const option = document.createElement('option');
        option.value = topic;
        option.textContent = topic;
        topicFilter.appendChild(option);
    });
}

function searchQuestions() {
    /* Search questions by content */
    const searchTerm = document.getElementById('question-search').value.toLowerCase();
    
    if (!searchTerm.trim()) {
        filteredQuestions = [...questionHistoryData];
        displayQuestions(filteredQuestions);
        return;
    }
    
    try {
        // CRITICAL FIX: Use consistent pythonBridge reference
        if (pythonBridge && pythonBridge.searchQuestions) {
            const result = pythonBridge.searchQuestions(searchTerm);
            
            // CRITICAL FIX: Handle both JSON string and object responses from PyQt
            let data;
            if (typeof result === 'string') {
                data = JSON.parse(result);
            } else if (typeof result === 'object' && result !== null) {
                data = result;
            } else {
                throw new Error('Invalid search response type: ' + typeof result);
            }
            
            if (data.success) {
                filteredQuestions = data.questions || [];
                displayQuestions(filteredQuestions);
                
                console.log(`🔍 Found ${filteredQuestions.length} questions matching: ${searchTerm}`);
                
                if (filteredQuestions.length === 0) {
                    showTemporaryMessage('🔍 No questions found matching your search', 'info');
                }
            } else {
                console.error('❌ Search failed:', data.error);
                showTemporaryMessage('❌ Search failed', 'error');
            }
        }
    } catch (error) {
        console.error('❌ Error searching questions:', error);
        showTemporaryMessage('❌ Error searching questions', 'error');
    }
}

function filterQuestionsByTopic() {
    /* Filter questions by selected topic */
    const selectedTopic = document.getElementById('topic-filter').value;
    
    if (!selectedTopic) {
        filteredQuestions = [...questionHistoryData];
        displayQuestions(filteredQuestions);
        return;
    }
    
    try {
        // CRITICAL FIX: Use consistent pythonBridge reference
        if (pythonBridge && pythonBridge.getQuestionsByTopic) {
            const result = pythonBridge.getQuestionsByTopic(selectedTopic);
            
            // CRITICAL FIX: Handle both JSON string and object responses from PyQt
            let data;
            if (typeof result === 'string') {
                data = JSON.parse(result);
            } else if (typeof result === 'object' && result !== null) {
                data = result;
            } else {
                throw new Error('Invalid topic filter response type: ' + typeof result);
            }
            
            if (data.success) {
                filteredQuestions = data.questions || [];
                displayQuestions(filteredQuestions);
                
                console.log(`📚 Filtered to ${filteredQuestions.length} questions for topic: ${selectedTopic}`);
            } else {
                console.error('❌ Topic filter failed:', data.error);
                showTemporaryMessage('❌ Topic filter failed', 'error');
            }
        }
    } catch (error) {
        console.error('❌ Error filtering by topic:', error);
        showTemporaryMessage('❌ Error filtering by topic', 'error');
    }
}

function filterQuestionsByDifficulty() {
    /* Filter questions by selected difficulty */
    const selectedDifficulty = document.getElementById('difficulty-filter').value;
    
    if (!selectedDifficulty) {
        filteredQuestions = [...questionHistoryData];
        displayQuestions(filteredQuestions);
        return;
    }
    
    try {
        // CRITICAL FIX: Use consistent pythonBridge reference
        if (pythonBridge && pythonBridge.getQuestionsByDifficulty) {
            const result = pythonBridge.getQuestionsByDifficulty(selectedDifficulty);
            
            // CRITICAL FIX: Handle both JSON string and object responses from PyQt
            let data;
            if (typeof result === 'string') {
                data = JSON.parse(result);
            } else if (typeof result === 'object' && result !== null) {
                data = result;
            } else {
                throw new Error('Invalid difficulty filter response type: ' + typeof result);
            }
            
            if (data.success) {
                filteredQuestions = data.questions || [];
                displayQuestions(filteredQuestions);
                
                console.log(`📊 Filtered to ${filteredQuestions.length} questions for difficulty: ${selectedDifficulty}`);
            } else {
                console.error('❌ Difficulty filter failed:', data.error);
                showTemporaryMessage('❌ Difficulty filter failed', 'error');
            }
        }
    } catch (error) {
        console.error('❌ Error filtering by difficulty:', error);
        showTemporaryMessage('❌ Error filtering by difficulty', 'error');
    }
}

function showQuestionStats() {
    /* Show comprehensive question statistics */
    console.log('📊 Loading question statistics...');
    
    try {
        // CRITICAL FIX: Use consistent pythonBridge reference
        if (pythonBridge && pythonBridge.getQuestionStatistics) {
            const result = pythonBridge.getQuestionStatistics();
            
            // CRITICAL FIX: Handle both JSON string and object responses from PyQt
            let data;
            if (typeof result === 'string') {
                data = JSON.parse(result);
            } else if (typeof result === 'object' && result !== null) {
                data = result;
            } else {
                throw new Error('Invalid statistics response type: ' + typeof result);
            }
            
            if (data.success) {
                currentStats = data.statistics;
                displayQuestionStats(currentStats);
                
                // Toggle stats display
                const statsDiv = document.getElementById('question-stats');
                statsDiv.style.display = statsDiv.style.display === 'none' ? 'block' : 'none';
                
                console.log('✅ Loaded question statistics:', currentStats);
            } else {
                console.error('❌ Failed to load statistics:', data.error);
                showTemporaryMessage('❌ Failed to load statistics', 'error');
            }
        }
    } catch (error) {
        console.error('❌ Error loading statistics:', error);
        showTemporaryMessage('❌ Error loading statistics', 'error');
    }
}

function displayQuestionStats(stats) {
    /* Display question statistics */
    document.getElementById('total-questions-stat').textContent = stats.total_questions || 0;
    
    const topicsCount = Object.keys(stats.by_topic || {}).length;
    document.getElementById('topics-count-stat').textContent = topicsCount;
    
    const expertCount = stats.by_difficulty?.expert || 0;
    document.getElementById('expert-questions-stat').textContent = expertCount;
}

function showQuestionDetail(questionId) {
    /* Show detailed view of a question */
    const question = questionHistoryData.find(q => q.id === questionId);
    if (!question) {
        showTemporaryMessage('❌ Question not found', 'error');
        return;
    }
    
    const modalContent = document.getElementById('modal-question-content');
    modalContent.innerHTML = `
        <div class="question-detail">
            <div class="question-meta-detail">
                <span class="difficulty-badge difficulty-${question.difficulty}">${getDifficultyIcon(question.difficulty)} ${question.difficulty}</span>
                <span class="topic-badge">${question.topic}</span>
                <span class="type-badge">${getTypeIcon(question.question_type)} ${question.question_type}</span>
                <span class="date-badge">📅 ${formatDate(question.generated_at)}</span>
            </div>
            
            <div class="question-content-detail">
                <h3>${question.question}</h3>
                
                <div class="options-detail">
                    <h4>📝 Options:</h4>
                    <ul class="options-list">
                        ${question.options.map((option, index) => `
                            <li class="option-item ${index === question.correct_index ? 'correct-option' : ''}">
                                <span class="option-letter">${String.fromCharCode(65 + index)}</span>
                                <span class="option-text">${option}</span>
                                ${index === question.correct_index ? '<span class="correct-indicator">✅ Correct</span>' : ''}
                            </li>
                        `).join('')}
                    </ul>
                </div>
                
                <div class="explanation-detail">
                    <h4>💡 Explanation:</h4>
                    <p>${question.explanation}</p>
                </div>
                
                ${question.times_answered > 0 ? `
                    <div class="performance-detail">
                        <h4>📊 Your Performance:</h4>
                        <p>Attempted ${question.times_answered} time(s) with ${Math.round(question.accuracy * 100)}% accuracy</p>
                        <div class="accuracy-bar">
                            <div class="accuracy-fill" style="width: ${question.accuracy * 100}%"></div>
                        </div>
                    </div>
                ` : ''}
            </div>
            
            <div class="question-actions-detail">
                <button class="btn btn-primary" onclick="practiceQuestion('${question.id}')">🎯 Practice This Question</button>
                <button class="btn btn-secondary" onclick="closeQuestionModal()">Close</button>
            </div>
        </div>
    `;
    
    // Show modal
    document.getElementById('question-detail-modal').style.display = 'flex';
    
    // Process LaTeX in the modal content
    updateQuestionWithLatex(question.question, modalContent.querySelector('h3'));
    question.options.forEach((option, index) => {
        const optionElement = modalContent.querySelector(`.option-text:nth-of-type(${index + 1})`);
        if (optionElement) {
            updateQuestionWithLatex(option, optionElement);
        }
    });
    updateExplanationWithLatex(question.explanation, modalContent.querySelector('.explanation-detail p'));
}

function closeQuestionModal() {
    /* Close the question detail modal */
    document.getElementById('question-detail-modal').style.display = 'none';
}

// Add event listener to close modal when clicking outside
document.addEventListener('DOMContentLoaded', function() {
    const modal = document.getElementById('question-detail-modal');
    
    // Close modal when clicking outside the modal content
    modal.addEventListener('click', function(event) {
        // If the click is directly on the modal background (not on its children)
        if (event.target === modal) {
            closeQuestionModal();
        }
    });
});

async function practiceQuestion(questionId) {
    try {
        // Call backend to get practice question data
        const response = await pythonBridge.practiceQuestion(questionId);
        const data = JSON.parse(response);

        if (!data.success) {
            alert("Error loading practice question: " + (data.error || "Unknown error"));
            return;
        }

        // Initialize quiz state for practice mode
        currentQuiz = {
            topic: data.topic || "Practice",
            mode: "practice",
            game_mode: "review",
            submode: "practice",
            difficulty: data.difficulty || "medium",
            num_questions: 1,
            score: 0,
            total_answered: 0,
            practice_mode: true,
            original_question_id: questionId
        };

        quizQuestions = [data];
        currentQuestionIndex = 0;

        // Show quiz UI
        showScreen('quiz');
        showQuizGame();

        // Display the practice question
        handleQuestionReceived(data);

        showTemporaryMessage(`🎯 Practice mode: ${data.topic || 'Unknown'} question`, 'info');

    } catch (error) {
        console.error("Failed to load practice question:", error);
        alert("Failed to load practice question. See console for details.");
    }
}

// Initialize review screen when it's first shown
document.addEventListener('DOMContentLoaded', function() {
    console.log('🔧 DOMContentLoaded - Setting up review screen handlers');
    
    // Load question history when review screen is first accessed
    const reviewNavButton = document.querySelector('button[onclick*="review"]');
    console.log('🔧 Review nav button found:', !!reviewNavButton);
    
    if (reviewNavButton) {
        const originalHandler = reviewNavButton.onclick;
        reviewNavButton.onclick = function(event) {
            console.log('🔧 Review button clicked');
            
            // Call original handler first
            if (originalHandler) {
                originalHandler.call(this, event);
            }
            
            // Load question history if review screen is shown
            setTimeout(() => {
                const reviewScreen = document.getElementById('review-screen');
                console.log('🔧 Review screen active?', reviewScreen?.classList.contains('active'));
                
                if (reviewScreen && reviewScreen.classList.contains('active')) {
                    console.log('🔧 Loading question history...');
                    loadQuestionHistory();
                }
            }, 200);
        };
    }
});

// Also add to the showScreen function to ensure it's called
function showScreenWrapper(screenName, navElement) {
    console.log('🔧 Showing screen:', screenName);
    
    // Call the original showScreen function (if it exists)
    if (typeof showScreen === 'function') {
        showScreen(screenName, navElement);
    }
    
    // If it's the review screen, load question history
    if (screenName === 'review') {
        console.log('🔧 Review screen detected, loading question history...');
        setTimeout(() => {
            loadQuestionHistory();
        }, 100);
    }
}

// 🌊 Token Streaming Functions
function isTokenStreamingEnabled() {
    const checkbox = document.getElementById('token-streaming-enabled');
    return checkbox && checkbox.checked;
}

function shouldUseTokenStreaming(difficulty, mode) {
    const tokenStreamingEnabled = document.getElementById('token-streaming-enabled');
    const userEnabled = tokenStreamingEnabled ? tokenStreamingEnabled.checked : false;

    console.log(`🌊 Token streaming check: checkbox=${!!tokenStreamingEnabled}, enabled=${userEnabled}, difficulty=${difficulty}, mode=${mode}`);

    // 🔥 FIXED: Enable token streaming for offline mode with DeepSeek
    if (mode === 'offline' && (difficulty === 'expert' || difficulty === 'hard')) {
        console.log('🌊 ENABLED: Token streaming for offline DeepSeek generation');
        return true;
    }

    // Also enable if user explicitly enabled it
    if (userEnabled) {
        console.log('🌊 ENABLED: Token streaming by user preference');
        return true;
    }

    return false;
}

// 🚀 NEW: Streaming handlers for live token updates
function handleStreamingStarted(message) {
    console.log('🌊 Streaming started:', message);
    
    // Initialize streaming UI
    initializeStreamingUI();
    
    // Show streaming status
    showStatusDisplay(message, 'streaming');
    
    // Reset streaming stats
    tokenStreamStats = {
        tokensReceived: 0,
        startTime: Date.now(),
        lastTokenTime: Date.now()
    };
}

function handleTokenReceived(token) {
    console.log('🌊 Token received:', token);
    
    // Update streaming stats
    tokenStreamStats.tokensReceived++;
    tokenStreamStats.lastTokenTime = Date.now();
    
    // Display token in streaming container
    displayStreamingToken(token);
    
    // Update streaming stats display
    updateStreamingStats();
}

function handleStreamingCompleted(questionData) {
    console.log('✅ Streaming completed:', questionData);
    
    // Hide streaming UI
    hideStreamingUI();
    
    // Display the final question
    handleQuestionReceived(questionData);
    
    // Show completion stats
    showStreamingCompletionStats();
}

function initializeStreamingUI() {
    // Create or show streaming container
    let streamingContainer = document.getElementById('streaming-container');
    if (!streamingContainer) {
        streamingContainer = createStreamingContainer();
    }
    
    streamingContainer.style.display = 'block';
    streamingContainer.innerHTML = `
        <div class="streaming-header">
            <h3>🌊 Live Generation</h3>
            <div class="streaming-stats">
                <span id="token-count">0 tokens</span>
                <span id="streaming-speed">0 tok/s</span>
            </div>
        </div>
        <div class="streaming-content" id="streaming-content">
            <div class="streaming-cursor">▋</div>
        </div>
    `;
    
    tokenStreamContainer = document.getElementById('streaming-content');
}

function createStreamingContainer() {
    const container = document.createElement('div');
    container.id = 'streaming-container';
    container.className = 'streaming-container';
    container.style.cssText = `
        position: fixed;
        top: 20%;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 800px;
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #007bff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        z-index: 10000;
        font-family: 'Courier New', monospace;
        backdrop-filter: blur(10px);
        display: none;
    `;
    
    document.body.appendChild(container);
    return container;
}

function displayStreamingToken(token) {
    if (!tokenStreamContainer) return;
    
    // Remove cursor
    const cursor = tokenStreamContainer.querySelector('.streaming-cursor');
    if (cursor) cursor.remove();
    
    // Add token
    const tokenSpan = document.createElement('span');
    tokenSpan.textContent = token;
    tokenSpan.className = 'streaming-token';
    tokenSpan.style.cssText = `
        animation: tokenFadeIn 0.3s ease-in;
        color: #007bff;
    `;
    
    tokenStreamContainer.appendChild(tokenSpan);
    
    // Add new cursor
    const newCursor = document.createElement('div');
    newCursor.className = 'streaming-cursor';
    newCursor.textContent = '▋';
    newCursor.style.cssText = `
        display: inline-block;
        animation: cursorBlink 1s infinite;
        color: #007bff;
        margin-left: 2px;
    `;
    
    tokenStreamContainer.appendChild(newCursor);
    
    // Auto-scroll to bottom
    tokenStreamContainer.scrollTop = tokenStreamContainer.scrollHeight;
}

function updateStreamingStats() {
    const tokenCountElement = document.getElementById('token-count');
    const tokenSpeedElement = document.getElementById('streaming-speed');
    const progressFill = document.getElementById('stream-progress-fill');

    if (tokenCountElement) {
        tokenCountElement.textContent = tokenStreamStats.tokensReceived;
    }

    if (tokenSpeedElement && tokenStreamStats.startTime) {
        const elapsed = (Date.now() - tokenStreamStats.startTime) / 1000;
        const speed = elapsed > 0 ? (tokenStreamStats.tokensReceived / elapsed).toFixed(1) : '0';
        tokenSpeedElement.textContent = speed;
    }

    // Update progress bar (estimate based on typical question length)
    if (progressFill) {
        const estimatedTotalTokens = 100; // Rough estimate
        const progress = Math.min((tokenStreamStats.tokensReceived / estimatedTotalTokens) * 100, 95);
        progressFill.style.width = progress + '%';
    }
}

function hideStreamingUI() {
    const streamingContainer = document.getElementById('streaming-container');
    if (streamingContainer) {
        streamingContainer.style.display = 'none';
    }
}

function showStreamingCompletionStats() {
    if (tokenStreamStats.startTime) {
        const elapsed = (Date.now() - tokenStreamStats.startTime) / 1000;
        const avgSpeed = (tokenStreamStats.tokensReceived / elapsed).toFixed(1);
        
        showStatusDisplay(
            `✅ Generation complete! ${tokenStreamStats.tokensReceived} tokens in ${elapsed.toFixed(1)}s (${avgSpeed} tok/s)`,
            'success'
        );
    }
}

// Update the quiz generation to use streaming when enabled
function startCustomQuizWithStreaming() {
    const topic = document.getElementById('quiz-topic').value.trim();
    const difficulty = document.getElementById('quiz-difficulty').value;
    
    if (!topic) {
        showError('Please enter a topic for the quiz.');
        return;
    }
    
    if (!pythonBridge) {
        showError('Python bridge not initialized.');
        return;
    }
    
    // Check if streaming is enabled
    if (isTokenStreamingEnabled()) {
        console.log('🌊 Starting streaming quiz generation');
        pythonBridge.generateQuestionStreaming(topic, difficulty);
    } else {
        console.log('📝 Starting regular quiz generation');
        pythonBridge.generateQuestion(topic, difficulty);
    }
    
    // Show quiz game screen
    showQuizGame();
}

// 🚀 EMERGENCY FALLBACK FUNCTIONS
window.closeTokenStreamingDialog = function() {
    console.log('🚀 Closing token streaming dialog');
    const dialog = document.querySelector('.token-streaming-dialog');
    if (dialog) {
        dialog.style.display = 'none';
    }

    // Hide any streaming overlays
    const overlay = document.querySelector('.streaming-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }

    // Also close the token stream UI if it exists
    if (typeof closeTokenStreamUI === 'function') {
        closeTokenStreamUI();
    }
};

window.generateQuestionNormally = function(topic, difficulty, questionType) {
    console.log(`🚀 FALLBACK: Generating question normally - ${topic} (${difficulty}) ${questionType}`);

    // Close streaming dialog first
    if (window.closeTokenStreamingDialog) {
        window.closeTokenStreamingDialog();
    }

    // Trigger normal generation using the proper quiz flow
    setTimeout(() => {
        console.log('🚀 FALLBACK: Starting normal quiz generation');

        // Set the form values to match the streaming parameters
        const topicInput = document.getElementById('quiz-topic');
        const difficultySelect = document.getElementById('quiz-difficulty');
        const submodeSelect = document.getElementById('quiz-submode');

        if (topicInput) topicInput.value = topic;
        if (difficultySelect) difficultySelect.value = difficulty;
        if (submodeSelect) submodeSelect.value = questionType;

        // Use the proper quiz generation function
        startCustomQuiz();
    }, 100);
};

// Define the missing generateQuestion function as an alias to startCustomQuiz
function generateQuestion() {
    console.log('🚀 generateQuestion() called - redirecting to startCustomQuiz()');
    startCustomQuiz();
}

function startTokenStreamingVisualization(topic, difficulty, questionType) {
    console.log(`🌊 Starting token streaming for: ${topic} (${difficulty}) - ${questionType}`);

    if (!pythonBridge || !pythonBridge.startTokenStreaming) {
        console.error('❌ Token streaming not available');
        showError('Token streaming not available');
        return;
    }

    // Reset stats
    tokenStreamStats = {
        tokensReceived: 0,
        startTime: Date.now(),
        lastTokenTime: Date.now()
    };

    // Create streaming UI
    createTokenStreamUI(topic, difficulty, questionType);

    // Add timeout mechanism (30 seconds for connection)
    const streamingTimeout = setTimeout(() => {
        console.log('⏰ Streaming timeout - falling back to normal generation');
        closeTokenStreamUI();
        generateQuestion();
    }, 30000);

    // Start streaming
    pythonBridge.startTokenStreaming(topic, difficulty, questionType).then(resultJson => {
        try {
            const result = JSON.parse(resultJson);

            if (result.success) {
                currentStreamSession = result.session_id;
                console.log(`✅ Token streaming started - Session: ${currentStreamSession}`);

                // Clear connection timeout and set longer timeout for generation
                clearTimeout(streamingTimeout);
                setTimeout(() => {
                    console.log('⏰ Extended streaming timeout - falling back to normal generation');
                    closeTokenStreamUI();
                    generateQuestion();
                }, 120000); // 2 minutes for full generation
            } else {
                clearTimeout(streamingTimeout);
                console.error('❌ Failed to start token streaming:', result.error);
                showError('Failed to start token streaming: ' + result.error);
                closeTokenStreamUI();
            }
        } catch (e) {
            clearTimeout(streamingTimeout);
            console.error('❌ Failed to parse streaming result:', e);
            showError('Failed to start token streaming');
            closeTokenStreamUI();
        }
    }).catch(error => {
        clearTimeout(streamingTimeout);
        console.error('❌ Token streaming error:', error);
        showError('Token streaming failed');
        closeTokenStreamUI();
    });
}

function createTokenStreamUI(topic, difficulty, questionType) {
    // Remove existing container if any
    if (tokenStreamContainer) {
        tokenStreamContainer.remove();
    }

    // Create container
    tokenStreamContainer = document.createElement('div');
    tokenStreamContainer.className = 'token-stream-container';

    tokenStreamContainer.innerHTML = `
        <div class="token-stream-header">
            <div class="token-stream-title">
                🌊 AI Thinking: ${topic} (${difficulty})
            </div>
            <button class="token-stream-close" onclick="closeTokenStreamUI()">
                ✕ Close
            </button>
        </div>

        <div class="token-stream-content" id="token-stream-content">
            <div class="token-chunk thinking">🤔 Initializing AI generation...</div>
        </div>

        <div class="stream-progress-bar">
            <div class="stream-progress-fill" id="stream-progress-fill"></div>
        </div>

        <div class="token-stream-stats">
            <div class="token-counter">
                📊 Tokens: <span id="token-count">0</span>
            </div>
            <div class="token-speed">
                ⚡ Speed: <span id="token-speed">0</span> tokens/sec
            </div>
        </div>
    `;

    document.body.appendChild(tokenStreamContainer);

    // Add fade-in animation
    setTimeout(() => {
        tokenStreamContainer.style.opacity = '1';
    }, 10);
}

function closeTokenStreamUI() {
    if (tokenStreamContainer) {
        tokenStreamContainer.style.opacity = '0';
        setTimeout(() => {
            if (tokenStreamContainer) {
                tokenStreamContainer.remove();
                tokenStreamContainer = null;
            }
        }, 300);
    }

    // Stop streaming session if active
    if (currentStreamSession && pythonBridge && pythonBridge.stopTokenStreaming) {
        pythonBridge.stopTokenStreaming(currentStreamSession);
        currentStreamSession = null;
    }
}

function handleTokenStreamChunk(sessionId, tokenChunk) {
    if (sessionId !== currentStreamSession || !tokenStreamContainer) {
        return;
    }

    const content = document.getElementById('token-stream-content');
    if (!content) return;

    // Update stats
    tokenStreamStats.tokensReceived++;
    tokenStreamStats.lastTokenTime = Date.now();

    // Determine token type for styling
    let tokenClass = 'token-chunk';
    if (tokenChunk.includes('Analyzing') || tokenChunk.includes('Considering') || tokenChunk.includes('Planning')) {
        tokenClass += ' thinking';
    } else if (tokenChunk.includes('?') || tokenChunk.startsWith('What') || tokenChunk.startsWith('How')) {
        tokenClass += ' question';
    } else if (tokenChunk.startsWith('A)') || tokenChunk.startsWith('B)') || tokenChunk.startsWith('C)') || tokenChunk.startsWith('D)')) {
        tokenClass += ' option';
    } else if (tokenChunk.includes('correct') || tokenChunk.includes('answer') || tokenChunk.includes('explanation')) {
        tokenClass += ' explanation';
    } else if (tokenChunk.includes('🎯') || tokenChunk.includes('📝') || tokenChunk.includes('✅')) {
        tokenClass += ' special';
    }

    // Create token element
    const tokenElement = document.createElement('span');
    tokenElement.className = tokenClass;
    tokenElement.textContent = tokenChunk + ' ';

    // Add to content
    content.appendChild(tokenElement);

    // Auto-scroll to bottom
    content.scrollTop = content.scrollHeight;

    // Update UI stats
    updateTokenStreamStats();
}

function updateTokenStreamStats() {
    const tokenCountElement = document.getElementById('token-count');
    const tokenSpeedElement = document.getElementById('token-speed');
    const progressFill = document.getElementById('stream-progress-fill');

    if (tokenCountElement) {
        tokenCountElement.textContent = tokenStreamStats.tokensReceived;
    }

    if (tokenSpeedElement && tokenStreamStats.startTime) {
        const elapsed = (Date.now() - tokenStreamStats.startTime) / 1000;
        const speed = elapsed > 0 ? (tokenStreamStats.tokensReceived / elapsed).toFixed(1) : '0';
        tokenSpeedElement.textContent = speed;
    }

    // Update progress bar (estimate based on typical question length)
    if (progressFill) {
        const estimatedTotalTokens = 100; // Rough estimate
        const progress = Math.min((tokenStreamStats.tokensReceived / estimatedTotalTokens) * 100, 95);
        progressFill.style.width = progress + '%';
    }
}

function handleTokenStreamCompleted(sessionId, finalQuestion) {
    if (sessionId !== currentStreamSession) {
        return;
    }

    console.log('✅ Token streaming completed:', finalQuestion);

    // Update progress to 100%
    const progressFill = document.getElementById('stream-progress-fill');
    if (progressFill) {
        progressFill.style.width = '100%';
    }

    // Add completion message
    const content = document.getElementById('token-stream-content');
    if (content) {
        const completionElement = document.createElement('div');
        completionElement.className = 'token-chunk special';
        completionElement.innerHTML = '<br><br>✅ Generation complete! Loading question...';
        content.appendChild(completionElement);
        content.scrollTop = content.scrollHeight;
    }

    // 🔧 FIX: Use requestAnimationFrame to prevent UI thread blocking
    requestAnimationFrame(() => {
        // Close streaming UI after a short delay using non-blocking approach
        setTimeout(() => {
            console.log('🔄 Closing streaming UI and displaying question...');

            // Close streaming UI first
            closeTokenStreamUI();

            // 🔧 FIX: Use requestAnimationFrame again to ensure UI updates are processed
            requestAnimationFrame(() => {
                // Display the final question with proper error handling
                if (finalQuestion && finalQuestion.question) {
                    console.log('📋 Displaying streamed question:', finalQuestion);
                    displayGeneratedQuestion(finalQuestion);
                } else {
                    console.log('⚠️ No valid question data from streaming, falling back to normal generation');
                    // Fallback to normal generation if streaming didn't produce a valid question
                    generateQuestion();
                }
            });
        }, 1500);
    });

    currentStreamSession = null;
}

function handleTokenStreamError(sessionId, errorMessage) {
    if (sessionId !== currentStreamSession) {
        return;
    }

    console.error('❌ Token streaming error:', errorMessage);

    // Show error in stream
    const content = document.getElementById('token-stream-content');
    if (content) {
        const errorElement = document.createElement('div');
        errorElement.className = 'token-chunk';
        errorElement.style.color = '#ef4444';
        errorElement.innerHTML = `<br><br>❌ Error: ${errorMessage}<br>🚀 Falling back to normal generation...`;
        content.appendChild(errorElement);
        content.scrollTop = content.scrollHeight;
    }

    // 🔧 FIX: Use requestAnimationFrame to prevent UI blocking during error handling
    requestAnimationFrame(() => {
        // Close after delay and fallback to normal generation
        setTimeout(() => {
            console.log('🚀 Streaming error - falling back to normal generation');
            closeTokenStreamUI();

            // Fallback to normal generation
            requestAnimationFrame(() => {
                generateQuestion();
            });
        }, 2000);
    });

    currentStreamSession = null;
}

function displayGeneratedQuestion(questionData) {
    console.log('📋 Displaying generated question:', questionData);

    // 🔧 FIX: Ensure we have valid question data
    if (!questionData || !questionData.question) {
        console.error('❌ Invalid question data received:', questionData);
        showError('Invalid question data received from streaming');
        return;
    }

    try {
        // Hide any loading displays
        hideQuizLoading();
        hideStatusDisplay();

        // Show success message briefly
        showStatusDisplay('✅ Question generated successfully!', 'success');

        // 🔧 FIX: Use requestAnimationFrame to prevent UI blocking during question display
        requestAnimationFrame(() => {
            try {
                // Ensure we're on the quiz screen
                showScreen('quiz');
                showQuizGame();

                // Use the existing question handler with proper error handling
                handleQuestionReceived(questionData);

                console.log('✅ Question displayed successfully via streaming');
            } catch (displayError) {
                console.error('❌ Error displaying question:', displayError);
                showError('Failed to display generated question: ' + displayError.message);
            }
        });

    } catch (error) {
        console.error('❌ Error in displayGeneratedQuestion:', error);
        showError('Failed to process generated question: ' + error.message);
    }
}

function startTokenStreamingSimulation(topic, difficulty, questionType) {
    console.log(`🌊 Starting ENHANCED token streaming simulation for: ${topic} (${difficulty}) - ${questionType}`);

    // Reset stats
    tokenStreamStats = {
        tokensReceived: 0,
        startTime: Date.now(),
        lastTokenTime: Date.now()
    };

    // 🚫 REMOVED: No hardcoded sample tokens - all content must be AI generated
    // Sample tokens removed to ensure pure AI generation without fallback content
    const sampleTokens = {
        numerical: ["Generating", "numerical", "question", "with", "AI..."],
        conceptual: ["Generating", "conceptual", "question", "with", "AI..."]
    };

    const tokens = sampleTokens[questionType] || sampleTokens.numerical;
    let tokenIndex = 0;

    // Start streaming tokens
    const streamInterval = setInterval(() => {
        if (tokenIndex >= tokens.length) {
            clearInterval(streamInterval);
            console.log('🌊 Token streaming simulation completed');

            // Close token stream UI after a delay
            setTimeout(() => {
                closeTokenStreamUI();
            }, 2000);
            return;
        }

        const token = tokens[tokenIndex];
        displayStreamingToken(token);
        tokenStreamStats.tokensReceived++;
        tokenStreamStats.lastTokenTime = Date.now();

        updateStreamingStats();
        tokenIndex++;

    }, 200 + Math.random() * 300); // Variable delay for realistic streaming
}



// ✅ EMERGENCY FIX: Force clear all loading screens
function forceHideAllLoadingScreens() {
    console.log('🚨 EMERGENCY: Force clearing all loading screens...');
    
    // Remove by ID patterns
    const idPatterns = ['loading', 'splash', 'spinner', 'init'];
    idPatterns.forEach(pattern => {
        const elements = document.querySelectorAll(`[id*="${pattern}"]`);
        elements.forEach(el => {
            el.style.display = 'none !important';
            el.remove();
        });
    });
    
    // Remove by class patterns  
    const classPatterns = ['loading', 'splash', 'spinner', 'overlay'];
    classPatterns.forEach(pattern => {
        const elements = document.querySelectorAll(`[class*="${pattern}"]`);
        elements.forEach(el => {
            el.style.display = 'none !important';
            el.style.zIndex = '-9999';
        });
    });
    
    // Clear body backgrounds
    document.body.style.background = 'var(--bg-color, #ffffff)';
    document.body.style.backgroundImage = 'none';
    
    // Force show app
    const app = document.querySelector('#app, main, .app-container');
    if (app) {
        app.style.display = 'block !important';
        app.style.visibility = 'visible !important';
        app.style.opacity = '1 !important';
    }
    
    console.log('🎉 Emergency clear complete!');
}

// Auto-call force clear after 3 seconds as backup
setTimeout(forceHideAllLoadingScreens, 3000);
