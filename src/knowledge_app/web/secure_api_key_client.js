/**
 * 🔒 SECURITY FIX #16: Secure Client-Side API Key Manager
 * 
 * Replaces insecure localStorage API key storage with secure server-side management.
 * This client-side module communicates with the secure backend for all API key operations.
 * 
 * SECURITY FEATURES:
 * - No client-side storage of API keys
 * - All keys stored encrypted on server-side
 * - Secure communication with backend
 * - Input validation and sanitization
 * - Audit logging
 */

class SecureApiKeyClient {
    constructor() {
        this.supportedProviders = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
        this.initialized = false;
        this.initializationPromise = null;
        
        console.log('🔒 SecureApiKeyClient initialized - no client-side storage');
    }
    
    async initialize() {
        if (this.initialized) return true;
        if (this.initializationPromise) return this.initializationPromise;
        
        this.initializationPromise = this._performInitialization();
        return this.initializationPromise;
    }
    
    async _performInitialization() {
        try {
            // Verify backend connection
            if (!pythonBridge) {
                throw new Error('Python bridge not available');
            }
            
            // Test secure API key functionality
            const testResult = await this._callSecureMethod('listSecureApiProviders', []);
            if (!testResult.success) {
                throw new Error('Secure API key backend not available');
            }
            
            this.initialized = true;
            console.log('✅ SecureApiKeyClient initialized successfully');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize SecureApiKeyClient:', error);
            this.initialized = false;
            throw error;
        }
    }
    
    async storeApiKey(provider, apiKey) {
        /**
         * 🔒 Securely store an API key on the server
         * @param {string} provider - API provider name
         * @param {string} apiKey - API key to store
         * @returns {Promise<boolean>} Success status
         */
        try {
            await this.initialize();
            
            if (!this.supportedProviders.includes(provider)) {
                throw new Error(`Unsupported provider: ${provider}`);
            }
            
            if (!apiKey || typeof apiKey !== 'string' || apiKey.trim().length < 10) {
                throw new Error('Invalid API key format');
            }
            
            // Sanitize inputs
            const cleanProvider = this._sanitizeProvider(provider);
            const cleanApiKey = this._sanitizeApiKey(apiKey);
            
            const result = await this._callSecureMethod('storeSecureApiKey', [cleanProvider, cleanApiKey]);
            
            if (result.success) {
                console.log(`🔒 API key stored securely for ${provider}`);
                
                // Update UI status
                this._updateProviderStatus(provider, true);
                
                // Trigger settings save (without API keys in client-side storage)
                this._triggerSecureSettingsSave();
                
                return true;
            } else {
                console.error(`❌ Failed to store API key for ${provider}:`, result.error);
                return false;
            }
            
        } catch (error) {
            console.error(`❌ Error storing API key for ${provider}:`, error);
            return false;
        }
    }
    
    async hasApiKey(provider) {
        /**
         * 🔒 Check if provider has a stored API key (without retrieving it)
         * @param {string} provider - API provider name
         * @returns {Promise<boolean>} Whether key exists
         */
        try {
            await this.initialize();
            
            const result = await this._callSecureMethod('getSecureApiKey', [provider]);
            return result.success && result.has_key;
            
        } catch (error) {
            console.error(`❌ Error checking API key for ${provider}:`, error);
            return false;
        }
    }
    
    async removeApiKey(provider) {
        /**
         * 🔒 Remove stored API key for provider
         * @param {string} provider - API provider name
         * @returns {Promise<boolean>} Success status
         */
        try {
            await this.initialize();
            
            const result = await this._callSecureMethod('removeSecureApiKey', [provider]);
            
            if (result.success) {
                console.log(`🗑️ API key removed for ${provider}`);
                
                // Update UI status
                this._updateProviderStatus(provider, false);
                
                return true;
            } else {
                console.error(`❌ Failed to remove API key for ${provider}:`, result.error);
                return false;
            }
            
        } catch (error) {
            console.error(`❌ Error removing API key for ${provider}:`, error);
            return false;
        }
    }
    
    async listProvidersWithKeys() {
        /**
         * 🔒 Get list of providers with stored API keys
         * @returns {Promise<string[]>} List of provider names
         */
        try {
            await this.initialize();
            
            const result = await this._callSecureMethod('listSecureApiProviders', []);
            
            if (result.success) {
                return result.providers || [];
            } else {
                console.error('❌ Failed to list API providers:', result.error);
                return [];
            }
            
        } catch (error) {
            console.error('❌ Error listing API providers:', error);
            return [];
        }
    }
    
    async clearAllApiKeys() {
        /**
         * 🔒 Remove all stored API keys
         * @returns {Promise<boolean>} Success status
         */
        try {
            await this.initialize();
            
            const providers = await this.listProvidersWithKeys();
            let allSuccess = true;
            
            for (const provider of providers) {
                const success = await this.removeApiKey(provider);
                if (!success) allSuccess = false;
            }
            
            if (allSuccess) {
                console.log('🗑️ All API keys cleared successfully');
                this._updateAllProviderStatuses(false);
            }
            
            return allSuccess;
            
        } catch (error) {
            console.error('❌ Error clearing all API keys:', error);
            return false;
        }
    }
    
    async loadApiKeysToUI() {
        /**
         * 🔒 Load API key status to UI (without exposing actual keys)
         * This replaces the insecure localStorage loading
         */
        try {
            await this.initialize();
            
            const providers = await this.listProvidersWithKeys();
            
            // Update UI to show which providers have keys (without showing the keys)
            this.supportedProviders.forEach(provider => {
                const hasKey = providers.includes(provider);
                this._updateProviderStatus(provider, hasKey);
                
                // Clear any input fields (security measure)
                const input = document.getElementById(`${provider}-api-key`);
                if (input) {
                    input.value = hasKey ? '••••••••••••••••' : '';
                    input.placeholder = hasKey ? 'API key stored securely' : `Enter ${provider} API key`;
                }
            });
            
            console.log(`🔒 Loaded secure API key status for ${providers.length} providers`);
            
        } catch (error) {
            console.error('❌ Error loading API key status:', error);
        }
    }
    
    _sanitizeProvider(provider) {
        // Remove any potentially dangerous characters
        return provider.toLowerCase().replace(/[^a-z0-9_-]/g, '');
    }
    
    _sanitizeApiKey(apiKey) {
        // Remove dangerous characters but preserve key structure
        return apiKey.trim().replace(/[\x00-\x1f\x7f-\x9f<>"']/g, '');
    }
    
    async _callSecureMethod(methodName, args) {
        return new Promise((resolve, reject) => {
            try {
                const result = pythonBridge[methodName](...args);
                
                // Handle both direct results and JSON strings
                if (typeof result === 'string') {
                    resolve(JSON.parse(result));
                } else {
                    resolve(result);
                }
            } catch (error) {
                reject(error);
            }
        });
    }
    
    _updateProviderStatus(provider, hasKey) {
        // Update provider status indicators in UI
        const statusElement = document.querySelector(`[data-provider="${provider}"] .status-indicator`);
        if (statusElement) {
            statusElement.textContent = hasKey ? '✅' : '❌';
            statusElement.className = `status-indicator ${hasKey ? 'ready' : 'error'}`;
        }
        
        // Update provider status text
        const statusText = document.querySelector(`[data-provider="${provider}"] .status-text`);
        if (statusText) {
            statusText.textContent = hasKey ? 'API key stored securely' : 'No API key';
        }
    }
    
    _updateAllProviderStatuses(hasKeys) {
        this.supportedProviders.forEach(provider => {
            this._updateProviderStatus(provider, hasKeys);
        });
    }
    
    _triggerSecureSettingsSave() {
        // Trigger settings save without including API keys in client-side data
        if (typeof saveSettings === 'function') {
            // This will now save settings without API keys to localStorage
            // API keys are handled securely on the server
            saveSettings();
        }
    }
}

// Global secure API key client instance
let secureApiKeyClient = null;

function getSecureApiKeyClient() {
    if (!secureApiKeyClient) {
        secureApiKeyClient = new SecureApiKeyClient();
    }
    return secureApiKeyClient;
}

// 🔒 SECURITY FIX #16: Replace insecure API key functions
window.secureApiKeyManager = {
    store: async (provider, apiKey) => {
        const client = getSecureApiKeyClient();
        return await client.storeApiKey(provider, apiKey);
    },
    
    has: async (provider) => {
        const client = getSecureApiKeyClient();
        return await client.hasApiKey(provider);
    },
    
    remove: async (provider) => {
        const client = getSecureApiKeyClient();
        return await client.removeApiKey(provider);
    },
    
    list: async () => {
        const client = getSecureApiKeyClient();
        return await client.listProvidersWithKeys();
    },
    
    clear: async () => {
        const client = getSecureApiKeyClient();
        return await client.clearAllApiKeys();
    },
    
    loadToUI: async () => {
        const client = getSecureApiKeyClient();
        return await client.loadApiKeysToUI();
    }
};

console.log('🔒 Secure API Key Client loaded - localStorage API key storage disabled');
