// Quick test script to debug frontend-backend communication
// Run this in the browser developer console

console.log("🔍 Testing frontend-backend bridge...");

// Test 1: Check if pythonBridge exists
if (window.pythonBridge) {
    console.log("✅ pythonBridge exists:", window.pythonBridge);
    
    // Test 2: Check if injected API key status exists
    if (window.injectedApiKeyStatus) {
        console.log("✅ injectedApiKeyStatus exists:", window.injectedApiKeyStatus);
    } else {
        console.log("❌ injectedApiKeyStatus not found");
    }
    
    // Test 3: Try to call a bridge method
    try {
        if (window.pythonBridge.getSecureApiKey) {
            console.log("✅ getSecureApiKey method available");
            window.pythonBridge.getSecureApiKey("all", function(result) {
                console.log("🔧 API key status from bridge:", result);
            });
        } else {
            console.log("❌ getSecureApiKey method not available");
        }
    } catch (error) {
        console.error("❌ Error calling bridge method:", error);
    }
    
    // Test 4: Try to call loadSettings
    if (window.loadSettings) {
        console.log("✅ loadSettings function available");
        try {
            window.loadSettings();
            console.log("✅ loadSettings called successfully");
        } catch (error) {
            console.error("❌ Error calling loadSettings:", error);
        }
    } else {
        console.log("❌ loadSettings function not available");
    }
    
} else {
    console.log("❌ pythonBridge not found");
}

// Test 5: Check current settings in localStorage
const savedSettings = localStorage.getItem('userSettings');
if (savedSettings) {
    try {
        const settings = JSON.parse(savedSettings);
        console.log("🔧 Current localStorage settings:", settings);
        if (settings.api_keys) {
            console.log("🔧 API keys in settings:", Object.keys(settings.api_keys));
        }
    } catch (error) {
        console.error("❌ Error parsing localStorage settings:", error);
    }
} else {
    console.log("❌ No userSettings in localStorage");
}

console.log("🏁 Bridge test completed");
