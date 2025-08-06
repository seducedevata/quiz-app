#!/usr/bin/env node

const { spawn } = require('child_process');
const PortManager = require('./port-manager');

async function startDev() {
    const portManager = new PortManager();
    
    console.log('🚀 Starting Knowledge App Development Server...\n');
    
    try {
        // Clear ports before starting
        await portManager.clearAllPorts();
        
        console.log('📦 Starting Next.js development server...');
        
        // Start Next.js dev server
        const nextProcess = spawn('npm', ['run', 'dev:next'], {
            stdio: 'inherit',
            shell: true,
            cwd: process.cwd()
        });
        
        // Setup graceful shutdown
        portManager.setupGracefulShutdown(null, 3000);
        
        nextProcess.on('error', (error) => {
            console.error('❌ Failed to start Next.js server:', error);
            process.exit(1);
        });
        
        nextProcess.on('exit', (code) => {
            console.log(`📦 Next.js server exited with code ${code}`);
            process.exit(code);
        });
        
    } catch (error) {
        console.error('❌ Failed to start development server:', error);
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

startDev();