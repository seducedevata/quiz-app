/**
 * Debug Enhancement for Token Streaming
 * This script adds comprehensive logging to track token streaming flow
 */

// Add enhanced debugging to the token stream handler
const originalHandleTokenStreamChunk = window.handleTokenStreamChunk;
window.handleTokenStreamChunk = function(sessionId, tokenChunk) {
    console.log(`🔥 ENHANCED DEBUG - Token Stream Handler Called:`);
    console.log(`  sessionId: ${sessionId}`);
    console.log(`  currentStreamSession: ${window.currentStreamSession}`);
    console.log(`  tokenChunk: "${tokenChunk}"`);
    console.log(`  tokenChunk length: ${tokenChunk.length}`);
    
    // Check if DOM elements exist
    const content = document.getElementById('token-stream-content');
    const streamingContainer = document.getElementById('streaming-container');
    
    console.log(`  token-stream-content exists: ${!!content}`);
    console.log(`  streaming-container exists: ${!!streamingContainer}`);
    
    if (content) {
        console.log(`  content.innerHTML before: "${content.innerHTML.slice(0, 100)}..."`);
    }
    
    // Call original function
    if (originalHandleTokenStreamChunk) {
        const result = originalHandleTokenStreamChunk.call(this, sessionId, tokenChunk);
        
        if (content) {
            console.log(`  content.innerHTML after: "${content.innerHTML.slice(-100)}"`);
            console.log(`  content.children.length: ${content.children.length}`);
        }
        
        return result;
    } else {
        console.warn('⚠️ Original handleTokenStreamChunk not found!');
    }
};

// Add debugging to UI creation functions
const originalCreateTokenStreamUI = window.createTokenStreamUI;
window.createTokenStreamUI = function(topic, difficulty, questionType) {
    console.log(`🔥 ENHANCED DEBUG - createTokenStreamUI called:`);
    console.log(`  topic: ${topic}`);
    console.log(`  difficulty: ${difficulty}`);
    console.log(`  questionType: ${questionType}`);
    
    if (originalCreateTokenStreamUI) {
        const result = originalCreateTokenStreamUI.call(this, topic, difficulty, questionType);
        
        // Check if UI was created
        const content = document.getElementById('token-stream-content');
        const container = window.tokenStreamContainer;
        
        console.log(`  UI created successfully: ${!!container}`);
        console.log(`  token-stream-content exists: ${!!content}`);
        console.log(`  container in DOM: ${container && document.body.contains(container)}`);
        console.log(`  container visibility: ${container ? container.style.display : 'N/A'}`);
        
        return result;
    } else {
        console.warn('⚠️ Original createTokenStreamUI not found!');
    }
};

// Add debugging to streaming conditions
const originalShouldUseTokenStreaming = window.shouldUseTokenStreaming;
window.shouldUseTokenStreaming = function(difficulty, mode) {
    console.log(`🔥 ENHANCED DEBUG - shouldUseTokenStreaming called:`);
    console.log(`  difficulty: ${difficulty}`);
    console.log(`  mode: ${mode}`);
    
    const checkbox = document.getElementById('token-streaming-enabled');
    const userEnabled = checkbox ? checkbox.checked : false;
    const isReasoningModel = window.detectReasoningModel ? window.detectReasoningModel() : false;
    const currentModel = window.getCurrentModelName ? window.getCurrentModelName() : 'unknown';
    
    console.log(`  checkbox exists: ${!!checkbox}`);
    console.log(`  userEnabled: ${userEnabled}`);
    console.log(`  isReasoningModel: ${isReasoningModel}`);
    console.log(`  currentModel: ${currentModel}`);
    
    if (originalShouldUseTokenStreaming) {
        const result = originalShouldUseTokenStreaming.call(this, difficulty, mode);
        console.log(`  shouldUseTokenStreaming result: ${result}`);
        return result;
    } else {
        console.warn('⚠️ Original shouldUseTokenStreaming not found!');
        return false;
    }
};

// Add debugging to startTokenStreamingVisualization
const originalStartTokenStreamingVisualization = window.startTokenStreamingVisualization;
window.startTokenStreamingVisualization = function(topic, difficulty, questionType) {
    console.log(`🔥 ENHANCED DEBUG - startTokenStreamingVisualization called:`);
    console.log(`  topic: ${topic}`);
    console.log(`  difficulty: ${difficulty}`);
    console.log(`  questionType: ${questionType}`);
    
    // Check pythonBridge availability
    console.log(`  pythonBridge exists: ${!!window.pythonBridge}`);
    console.log(`  startTokenStreaming method: ${!!(window.pythonBridge && window.pythonBridge.startTokenStreaming)}`);
    
    if (originalStartTokenStreamingVisualization) {
        return originalStartTokenStreamingVisualization.call(this, topic, difficulty, questionType);
    } else {
        console.warn('⚠️ Original startTokenStreamingVisualization not found!');
    }
};

// Monitor pythonBridge signal connections
if (window.pythonBridge) {
    console.log(`🔥 ENHANCED DEBUG - pythonBridge signals:`);
    console.log(`  tokenReceived: ${!!window.pythonBridge.tokenReceived}`);
    console.log(`  streamingStarted: ${!!window.pythonBridge.streamingStarted}`);
    console.log(`  streamingCompleted: ${!!window.pythonBridge.streamingCompleted}`);
} else {
    console.log(`🔥 ENHANCED DEBUG - pythonBridge not available yet`);
}

console.log(`🔥 Enhanced token streaming debug logging activated!`);
