/**
 * API Key Persistence Fix for Knowledge App
 * 
 * This script ensures API keys are properly saved and persisted
 * across app sessions. It fixes the issue where API keys get deleted.
 */

console.log('🔧 Loading API Key Persistence Fix...');

// Enhanced API key saving function
function saveApiKeysSecurely() {
    console.log('💾 Saving API keys securely...');
    
    const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
    const apiKeys = {};
    let hasKeys = false;
    
    // Collect all API keys from UI inputs
    providers.forEach(provider => {
        const input = document.getElementById(`${provider}-api-key`);
        if (input) {
            const value = input.value.trim();
            apiKeys[provider] = value;
            if (value) {
                hasKeys = true;
                // 🛡️ CRITICAL SECURITY FIX: Use secure logging for API key detection
                const redactedKey = value.length > 8 ? `${value.substring(0, 4)}****${value.substring(value.length - 4)}` : '***';
                console.log(`✅ Found ${provider} API key: ${redactedKey}`);
            }
        }
    });
    
    if (!hasKeys) {
        console.log('⚠️ No API keys found to save');
        return false;
    }
    
    // Get current settings from localStorage
    let currentSettings = {};
    try {
        const savedSettings = localStorage.getItem('userSettings');
        if (savedSettings) {
            currentSettings = JSON.parse(savedSettings);
        }
    } catch (e) {
        console.error('❌ Failed to load current settings from localStorage:', e);
    }
    
    // Ensure proper structure
    if (!currentSettings.api_keys) {
        currentSettings.api_keys = {};
    }
    
    // Update API keys
    Object.assign(currentSettings.api_keys, apiKeys);
    
    // Add metadata for tracking
    currentSettings.last_api_key_update = new Date().toISOString();
    currentSettings.api_key_save_count = (currentSettings.api_key_save_count || 0) + 1;
    
    // Save to localStorage with error handling
    try {
        localStorage.setItem('userSettings', JSON.stringify(currentSettings));
        console.log('✅ API keys saved to localStorage');
        
        // Also save to sessionStorage as backup
        sessionStorage.setItem('apiKeysBackup', JSON.stringify(apiKeys));
        console.log('✅ API keys backed up to sessionStorage');
        
    } catch (e) {
        console.error('❌ Failed to save to localStorage:', e);
        return false;
    }
    
    // Save to backend if available
    if (typeof pythonBridge !== 'undefined' && pythonBridge.saveUserSettings) {
        try {
            const result = pythonBridge.saveUserSettings(JSON.stringify(currentSettings));
            console.log('✅ API keys saved to backend:', result);
        } catch (e) {
            console.error('❌ Failed to save to backend:', e);
        }
    } else {
        console.log('⚠️ Python bridge not available for backend save');
    }
    
    // Verify persistence after a short delay
    setTimeout(() => {
        verifyApiKeyPersistence(apiKeys);
    }, 1000);
    
    return true;
}

// Verify that API keys were saved correctly
function verifyApiKeyPersistence(originalKeys) {
    console.log('🔍 Verifying API key persistence...');
    
    try {
        const savedSettings = localStorage.getItem('userSettings');
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            const savedKeys = settings.api_keys || {};
            
            let verified = 0;
            let total = 0;
            
            Object.keys(originalKeys).forEach(provider => {
                if (originalKeys[provider]) {
                    total++;
                    if (savedKeys[provider] === originalKeys[provider]) {
                        verified++;
                        console.log(`✅ ${provider}: Verified`);
                    } else {
                        console.log(`❌ ${provider}: Mismatch`);
                    }
                }
            });
            
            if (verified === total && total > 0) {
                console.log(`🎉 All ${verified} API keys verified successfully!`);
                showApiKeyStatus('success', `${verified} API keys saved successfully`);
            } else {
                console.log(`⚠️ Only ${verified}/${total} API keys verified`);
                showApiKeyStatus('warning', `Only ${verified}/${total} API keys saved correctly`);
            }
        }
    } catch (e) {
        console.error('❌ Verification failed:', e);
        showApiKeyStatus('error', 'Failed to verify API key persistence');
    }
}

// Restore API keys from backup if main storage fails
function restoreApiKeysFromBackup() {
    console.log('🔄 Attempting to restore API keys from backup...');
    
    try {
        const backup = sessionStorage.getItem('apiKeysBackup');
        if (backup) {
            const apiKeys = JSON.parse(backup);
            let restored = 0;
            
            Object.keys(apiKeys).forEach(provider => {
                const input = document.getElementById(`${provider}-api-key`);
                if (input && apiKeys[provider] && !input.value.trim()) {
                    input.value = apiKeys[provider];
                    restored++;
                    // 🛡️ CRITICAL SECURITY FIX: Use secure logging for API key restoration
                    const redactedKey = apiKeys[provider].length > 8 ? `${apiKeys[provider].substring(0, 4)}****${apiKeys[provider].substring(apiKeys[provider].length - 4)}` : '***';
                    console.log(`🔄 Restored ${provider} API key from backup: ${redactedKey}`);
                }
            });
            
            if (restored > 0) {
                console.log(`✅ Restored ${restored} API keys from backup`);
                showApiKeyStatus('info', `Restored ${restored} API keys from backup`);
                return true;
            }
        }
    } catch (e) {
        console.error('❌ Failed to restore from backup:', e);
    }
    
    return false;
}

// Show API key status message to user
function showApiKeyStatus(type, message) {
    // Create or update status element
    let statusElement = document.getElementById('api-key-status');
    if (!statusElement) {
        statusElement = document.createElement('div');
        statusElement.id = 'api-key-status';
        statusElement.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            z-index: 10000;
            max-width: 300px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        `;
        document.body.appendChild(statusElement);
    }
    
    // Set colors based on type
    const colors = {
        success: '#28a745',
        warning: '#ffc107',
        error: '#dc3545',
        info: '#17a2b8'
    };
    
    statusElement.style.backgroundColor = colors[type] || colors.info;
    statusElement.textContent = message;
    statusElement.style.display = 'block';
    
    // Auto-hide after 1 second (fast dismissal to prevent UI blocking)
    setTimeout(() => {
        if (statusElement && statusElement.parentNode) {
            statusElement.style.opacity = '0';
            statusElement.style.transition = 'opacity 0.3s ease-out';
            setTimeout(() => {
                if (statusElement && statusElement.parentNode) {
                    statusElement.parentNode.removeChild(statusElement);
                }
            }, 300);
        }
    }, 1000);
}

// Enhanced event listeners for API key inputs
function setupApiKeyEventListeners() {
    console.log('🔧 Setting up enhanced API key event listeners...');
    
    const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
    
    providers.forEach(provider => {
        const input = document.getElementById(`${provider}-api-key`);
        if (input) {
            // Remove existing listeners to avoid duplicates
            input.removeEventListener('blur', handleApiKeyBlur);
            input.removeEventListener('keypress', handleApiKeyKeypress);
            input.removeEventListener('input', handleApiKeyInput);
            
            // Add enhanced listeners
            input.addEventListener('blur', handleApiKeyBlur);
            input.addEventListener('keypress', handleApiKeyKeypress);
            input.addEventListener('input', handleApiKeyInput);
            
            console.log(`✅ Enhanced listeners added for ${provider}`);
        }
    });
}

// Event handlers
function handleApiKeyBlur(event) {
    const provider = event.target.id.replace('-api-key', '');
    if (event.target.value.trim()) {
        console.log(`🔑 Auto-saving ${provider} API key (blur event)`);
        saveApiKeysSecurely();
    }
}

function handleApiKeyKeypress(event) {
    if (event.key === 'Enter') {
        const provider = event.target.id.replace('-api-key', '');
        if (event.target.value.trim()) {
            console.log(`🔑 Auto-saving ${provider} API key (Enter pressed)`);
            saveApiKeysSecurely();
        }
    }
}

function handleApiKeyInput(event) {
    // Debounced auto-save while typing
    clearTimeout(event.target.autoSaveTimer);
    event.target.autoSaveTimer = setTimeout(() => {
        if (event.target.value.trim()) {
            const provider = event.target.id.replace('-api-key', '');
            console.log(`🔑 Auto-saving ${provider} API key (input debounced)`);
            saveApiKeysSecurely();
        }
    }, 2000); // Save 2 seconds after user stops typing
}

// Load API keys on page load
function loadApiKeysOnStartup() {
    console.log('🔄 Loading API keys on startup...');
    
    try {
        const savedSettings = localStorage.getItem('userSettings');
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            const apiKeys = settings.api_keys || {};
            
            let loaded = 0;
            Object.keys(apiKeys).forEach(provider => {
                const input = document.getElementById(`${provider}-api-key`);
                if (input && apiKeys[provider] && apiKeys[provider].trim()) {
                    input.value = apiKeys[provider];
                    loaded++;
                    // 🛡️ CRITICAL SECURITY FIX: Use secure logging for API key loading
                    const redactedKey = apiKeys[provider].length > 8 ? `${apiKeys[provider].substring(0, 4)}****${apiKeys[provider].substring(apiKeys[provider].length - 4)}` : '***';
                    console.log(`✅ Loaded ${provider} API key: ${redactedKey}`);
                }
            });
            
            if (loaded > 0) {
                console.log(`✅ Loaded ${loaded} API keys from localStorage`);
                // 🚀 PERFORMANCE: Don't show notification for normal loading to prevent UI blocking
                // showApiKeyStatus('success', `Loaded ${loaded} saved API keys`);
            }
        } else {
            // Try to restore from backup
            restoreApiKeysFromBackup();
        }
    } catch (e) {
        console.error('❌ Failed to load API keys on startup:', e);
        restoreApiKeysFromBackup();
    }
}

// Initialize the fix when DOM is ready
function initializeApiKeyPersistenceFix() {
    console.log('🚀 Initializing API Key Persistence Fix...');

    // 🚀 PERFORMANCE FIX: Defer heavy operations to prevent UI blocking
    setTimeout(() => {
        try {
            // Load existing API keys asynchronously
            loadApiKeysOnStartup();

            // Setup enhanced event listeners
            setupApiKeyEventListeners();

            console.log('✅ API Key Persistence Fix initialized successfully');
        } catch (e) {
            console.error('❌ API Key Persistence Fix initialization failed:', e);
        }
    }, 100);

    // Make functions globally available immediately (lightweight)
    window.saveApiKeysSecurely = saveApiKeysSecurely;
    window.restoreApiKeysFromBackup = restoreApiKeysFromBackup;
    window.verifyApiKeyPersistence = verifyApiKeyPersistence;

    // 🚀 QUICK NOTIFICATION: Show and hide immediately to prevent hanging
    showApiKeyStatus('success', 'API Keys Ready');
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApiKeyPersistenceFix);
} else {
    // DOM is already ready
    initializeApiKeyPersistenceFix();
}

// Also initialize after a short delay to ensure all elements are loaded
setTimeout(initializeApiKeyPersistenceFix, 1000);

console.log('✅ API Key Persistence Fix script loaded');
