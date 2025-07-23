
// DEPRECATED: This file has been replaced by api_key_persistence_fix.js
// Please use api_key_persistence_fix.js instead as it provides more comprehensive functionality

console.warn('⚠️ enhanced_api_key_saver.js is deprecated - please use api_key_persistence_fix.js instead');

// Redirect to the main API key persistence implementation
if (typeof window !== 'undefined' && window.saveApiKeysSecurely) {
    // Main implementation is already loaded
    console.log('✅ Main API key persistence already available');
} else {
    // Simple fallback implementation until main loads
    function saveApiKeysProperly() {
        console.log('🔧 Fallback API Key Saver (please load api_key_persistence_fix.js for full functionality)');
        
        const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
        const apiKeys = {};
        let hasKeys = false;
        
        // Collect all API keys from UI
        providers.forEach(provider => {
            const input = document.getElementById(`${provider}-api-key`);
            if (input && input.value.trim()) {
                apiKeys[provider] = input.value.trim();
                hasKeys = true;
            }
        });
        
        if (hasKeys) {
            try {
                // Get current settings
                const currentSettings = JSON.parse(localStorage.getItem('userSettings') || '{}');
                if (!currentSettings.api_keys) {
                    currentSettings.api_keys = {};
                }
                
                // Update API keys
                Object.assign(currentSettings.api_keys, apiKeys);
                
                // Save to localStorage
                localStorage.setItem('userSettings', JSON.stringify(currentSettings));
                console.log('✅ API keys saved (fallback implementation)');
                
                return true;
            } catch (e) {
                console.error('❌ Failed to save API keys (fallback):', e);
                return false;
            }
        } else {
            console.log('⚠️ No API keys to save');
            return false;
        }
    }
    
    // Make available globally
    if (typeof window !== 'undefined') {
        window.saveApiKeysProperly = saveApiKeysProperly;
    }
}
