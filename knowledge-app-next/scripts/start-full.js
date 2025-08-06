#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const PortManager = require('./port-manager');

async function startFull() {
    const portManager = new PortManager();
    
    console.log('🚀 Starting Complete Knowledge App Stack...\n');
    
    try {
        // Clear all ports before starting
        await portManager.clearAllPorts();
        
        console.log('📡 Starting API server...');
        
        // Start API server first
        const apiProcess = spawn('node', ['server.js'], {
            stdio: ['inherit', 'pipe', 'pipe'],
            shell: true,
            cwd: path.join(process.cwd(), 'api-server')
        });
        
        // Log API server output with prefix
        apiProcess.stdout.on('data', (data) => {
            process.stdout.write(`[API] ${data}`);
        });
        
        apiProcess.stderr.on('data', (data) => {
            process.stderr.write(`[API] ${data}`);
        });
        
        // Wait for API server to start
        await new Promise((resolve) => {
            apiProcess.stdout.on('data', (data) => {
                if (data.toString().includes('API server running on port 8000')) {
                    resolve();
                }
            });
            
            // Fallback timeout
            setTimeout(resolve, 3000);
        });
        
        console.log('📦 Starting Next.js development server...');
        
        // Start Next.js dev server
        const nextProcess = spawn('npm', ['run', 'dev:next'], {
            stdio: ['inherit', 'pipe', 'pipe'],
            shell: true,
            cwd: process.cwd()
        });
        
        // Log Next.js output with prefix
        nextProcess.stdout.on('data', (data) => {
            process.stdout.write(`[NEXT] ${data}`);
        });
        
        nextProcess.stderr.on('data', (data) => {
            process.stderr.write(`[NEXT] ${data}`);
        });
        
        // Setup graceful shutdown for both processes
        const shutdown = async (signal) => {
            console.log(`\n🛑 Received ${signal}, shutting down both servers...`);
            
            // Kill both processes
            if (nextProcess && !nextProcess.killed) {
                nextProcess.kill('SIGTERM');
            }
            if (apiProcess && !apiProcess.killed) {
                apiProcess.kill('SIGTERM');
            }
            
            // Clear ports
            await portManager.clearAllPorts();
            console.log('✅ Complete shutdown finished');
            process.exit(0);
        };
        
        process.on('SIGTERM', () => shutdown('SIGTERM'));
        process.on('SIGINT', () => shutdown('SIGINT'));
        process.on('SIGQUIT', () => shutdown('SIGQUIT'));
        
        // Handle process exits
        apiProcess.on('exit', (code) => {
            console.log(`📡 API server exited with code ${code}`);
            if (code !== 0) {
                nextProcess.kill();
                process.exit(code);
            }
        });
        
        nextProcess.on('exit', (code) => {
            console.log(`📦 Next.js server exited with code ${code}`);
            if (code !== 0) {
                apiProcess.kill();
                process.exit(code);
            }
        });
        
        console.log('\n✅ Both servers started successfully!');
        console.log('🌐 Next.js App: http://localhost:3000');
        console.log('📡 API Server: http://localhost:8000');
        console.log('\n💡 Press Ctrl+C to stop both servers\n');
        
    } catch (error) {
        console.error('❌ Failed to start servers:', error);
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

startFull();