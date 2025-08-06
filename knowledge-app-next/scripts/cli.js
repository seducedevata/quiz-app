#!/usr/bin/env node

const PortManager = require('./port-manager');

async function main() {
    const portManager = new PortManager();
    const command = process.argv[2];
    
    switch (command) {
        case 'clear':
        case 'clear-ports':
            await portManager.clearAllPorts();
            break;
            
        case 'check':
            console.log('🔍 Checking port availability...');
            for (const port of portManager.ports) {
                const available = await portManager.isPortAvailable(port);
                console.log(`Port ${port}: ${available ? '✅ Available' : '❌ In use'}`);
            }
            break;
            
        case 'find':
            const startPort = parseInt(process.argv[3]) || 3000;
            try {
                const availablePort = await portManager.findAvailablePort(startPort);
                console.log(`✅ Available port found: ${availablePort}`);
            } catch (error) {
                console.error('❌ No available port found');
            }
            break;
            
        default:
            console.log(`
🔧 Knowledge App Port Manager

Usage:
  node scripts/cli.js clear        - Clear all configured ports
  node scripts/cli.js check        - Check port availability
  node scripts/cli.js find [port]  - Find next available port

Examples:
  node scripts/cli.js clear
  node scripts/cli.js check
  node scripts/cli.js find 3000
            `);
    }
}

main().catch(console.error);