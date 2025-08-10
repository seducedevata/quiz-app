#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const PortManager = require('./port-manager');

async function startAPI() {
    const portManager = new PortManager();
    
    console.log('🔧 Starting Knowledge App Express API Server...\n');
    
    try {
        console.log('📡 Starting Express API server on port 8000...');
        console.log('🐍 Python Bridge compatibility enabled');
        
        // Start Express API server
        const apiProcess = spawn('node', ['server.js'], {
            stdio: 'inherit',
            shell: true,
            cwd: path.join(process.cwd(), 'api-server')
        });
        
        // Setup graceful shutdown
        portManager.setupGracefulShutdown(null, 8000);
        
        apiProcess.on('error', (error) => {
            console.error('❌ Failed to start Express API server:', error);
            process.exit(1);
        });
        
        apiProcess.on('exit', (code) => {
            console.log(`📡 Express API server exited with code ${code}`);
            process.exit(code);
        });
        
    } catch (error) {
        console.error('❌ Failed to start Express API server:', error);
        process.exit(1);
    }
}

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught Exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

startAPI();
