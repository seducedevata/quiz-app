@echo off
echo 🚀 Starting Full Knowledge App (Next.js + Python Bridge)
echo.

REM Start the Python bridge server in a new window
echo 🐍 Starting Python Bridge Server...
start "Python Bridge Server" cmd /k "cd /d "%~dp0" && start-bridge.bat"

REM Wait a moment for the bridge to start
timeout /t 3 /nobreak >nul

REM Start the Next.js development server
echo ⚛️ Starting Next.js Development Server...
echo.
echo 🌐 Next.js will be available at: http://localhost:3000
echo 🐍 Python Bridge available at: http://localhost:8000
echo.
echo Both servers are starting... Please wait a moment.
echo.

npm run dev