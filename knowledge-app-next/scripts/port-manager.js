const net = require('net');
const { exec } = require('child_process');

class PortManager {
  constructor() {
    this.activeServers = {};
  }

  async isPortInUse(port) {
    return new Promise((resolve) => {
      const server = net.createServer();
      server.once('error', (err) => {
        if (err.code === 'EADDRINUSE') {
          resolve(true);
        } else {
          resolve(false);
        }
      });
      server.once('listening', () => {
        server.close();
        resolve(false);
      });
      server.listen(port);
    });
  }

  async killPort(port) {
    console.log(`Attempting to kill any process on port ${port}...`);
    return new Promise((resolve, reject) => {
      exec(`netstat -ano | findstr :${port}`, (error, stdout, stderr) => {
        if (error) {
          console.log(`No process found on port ${port} or error: ${error.message}`);
          return resolve();
        }
        if (stderr) {
          console.warn(`Stderr while checking port ${port}: ${stderr}`);
        }

        const lines = stdout.trim().split('\n');
        if (lines.length > 0 && lines[0] !== '') {
          const pidMatch = lines[0].match(/\s(\d+)$/);
          if (pidMatch) {
            const pid = pidMatch[1];
            exec(`taskkill /PID ${pid} /F`, (killError, killStdout, killStderr) => {
              if (killError) {
                console.error(`Error killing process ${pid} on port ${port}: ${killError.message}`);
                return reject(killError);
              }
              console.log(`Successfully killed process ${pid} on port ${port}.`);
              resolve();
            });
          } else {
            console.log(`Could not find PID for port ${port}.`);
            resolve();
          }
        } else {
          console.log(`Port ${port} is not in use.`);
          resolve();
        }
      });
    });
  }

  setupGracefulShutdown(server, port) {
    const exitHandler = async () => {
      console.log(`
Server on port ${port} is shutting down...`);
      if (server) {
        server.close(() => {
          console.log(`Server on port ${port} closed.`);
          process.exit(0);
        });
      } else {
        process.exit(0);
      }
    };

    process.on('exit', exitHandler);
    process.on('SIGINT', exitHandler); // Ctrl+C
    process.on('SIGTERM', exitHandler);
    process.on('uncaughtException', (err) => {
      console.error('Uncaught Exception:', err);
      exitHandler();
    });
  }
}

module.exports = PortManager;
