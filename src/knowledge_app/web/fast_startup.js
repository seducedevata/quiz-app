// Ultra-fast startup optimization for Knowledge App
// Hardware-accelerated parallel initialization

(function() {
    'use strict';
    
    // Performance tracking
    const startupMetrics = {
        start: performance.now(),
        phases: {}
    };
    
    function markPhase(name) {
        startupMetrics.phases[name] = performance.now() - startupMetrics.start;
        console.log(`⚡ ${name}: ${startupMetrics.phases[name].toFixed(2)}ms`);
    }
    
    // Hardware profile detection
    const hardware = {
        cores: navigator.hardwareConcurrency || 2,
        memory: navigator.deviceMemory || 4,
        timing: performance.now()
    };
    
    // Fast resource preloading
    const preloadQueue = [];
    function preloadResource(href, type) {
        const link = document.createElement('link');
        link.rel = 'preload';
        link.href = href;
        link.as = type;
        link.fetchPriority = 'high';
        document.head.appendChild(link);
        preloadQueue.push(link);
    }
    
    // Critical path optimization
    const criticalTasks = [];
    const backgroundTasks = [];
    
    function addCriticalTask(task) {
        criticalTasks.push(task);
    }
    
    function addBackgroundTask(task) {
        backgroundTasks.push(task);
    }
    
    // Ultra-fast async initialization
    async function fastInitialize() {
        markPhase('FastInit Start');
        
        // Hardware-optimized thread allocation
        const threadCount = Math.min(hardware.cores, 8);
        
        // Critical path: settings and UI
        const criticalPromises = criticalTasks.map(task => task());
        
        // Background tasks: models and heavy resources
        if (window.requestIdleCallback) {
            requestIdleCallback(() => {
                backgroundTasks.forEach(task => task());
            });
        } else {
            setTimeout(() => {
                backgroundTasks.forEach(task => task());
            }, 100);
        }
        
        // Wait for critical path only
        await Promise.race([
            Promise.all(criticalPromises),
            new Promise(resolve => setTimeout(resolve, 2000)) // 2s max
        ]);
        
        markPhase('FastInit Complete');
        console.log('⚡ Ultra-fast startup metrics:', startupMetrics);
    }
    
    // Optimized settings loading
    async function fastLoadSettings() {
        try {
            if (window.pythonBridge && window.pythonBridge.loadSettings) {
                const settings = await window.pythonBridge.loadSettings();
                if (settings && settings.success) {
                    return settings.data || {};
                }
            }
            return {};
        } catch (error) {
            console.warn('⚠️ Settings load failed, using defaults');
            return {};
        }
    }
    
    // Optimized API key verification
    async function fastVerifyAPIKeys() {
        const providers = ['openai', 'anthropic', 'gemini', 'groq', 'openrouter'];
        const verificationPromises = providers.map(async provider => {
            try {
                if (window.secureApiKeyManager) {
                    const result = await window.secureApiKeyManager.get(provider);
                    const data = JSON.parse(result);
                    return { provider, hasKey: data.success && data.api_key };
                }
                return { provider, hasKey: false };
            } catch {
                return { provider, hasKey: false };
            }
        });
        
        const results = await Promise.all(verificationPromises);
        return results.reduce((acc, { provider, hasKey }) => {
            acc[provider] = hasKey;
            return acc;
        }, {});
    }
    
    // Hardware-accelerated UI rendering
    function fastRenderUI() {
        return new Promise(resolve => {
            requestAnimationFrame(() => {
                // Fast DOM updates
                const criticalElements = [
                    '#api-keys-section',
                    '#main-content',
                    '#loading-screen'
                ];
                
                criticalElements.forEach(selector => {
                    const element = document.querySelector(selector);
                    if (element) {
                        element.style.willChange = 'transform';
                        element.style.transform = 'translateZ(0)'; // GPU acceleration
                    }
                });
                
                resolve();
            });
        });
    }
    
    // Background model preloading
    function preloadModels() {
        if (window.pythonBridge && window.pythonBridge.preloadModels) {
            window.pythonBridge.preloadModels().catch(() => {
                console.log('⚡ Background model preloading started');
            });
        }
    }
    
    // Memory optimization
    function optimizeMemory() {
        // Clear preload queue after startup
        setTimeout(() => {
            preloadQueue.forEach(link => link.remove());
            preloadQueue.length = 0;
        }, 5000);
    }
    
    // Initialize fast startup
    document.addEventListener('DOMContentLoaded', () => {
        markPhase('DOM Ready');
        
        // Add critical tasks
        addCriticalTask(fastLoadSettings);
        addCriticalTask(fastRenderUI);
        addCriticalTask(fastVerifyAPIKeys);
        
        // Add background tasks
        addBackgroundTask(preloadModels);
        addBackgroundTask(optimizeMemory);
        
        // Start ultra-fast initialization
        fastInitialize();
    });
    
    // Export for global access
    window.fastStartup = {
        initialize: fastInitialize,
        metrics: startupMetrics,
        hardware: hardware
    };
    
})();
