@echo off
echo 🚀 Starting Knowledge App...
echo.

REM Clear any existing processes on ports
echo 🧹 Clearing ports...
node scripts/port-manager.js

REM Start the full application stack
echo 📦 Starting application stack...
npm run dev:full

pause