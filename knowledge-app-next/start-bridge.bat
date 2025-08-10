@echo off
echo 🚀 Starting Next.js Bridge Server...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

REM Install required packages if not already installed
echo 📦 Installing required Python packages...
pip install flask flask-cors flask-socketio python-socketio

REM Change to the correct directory
cd /d "%~dp0"

REM Start the bridge server
echo 🌐 Starting bridge server on http://localhost:8000
echo 📡 WebSocket available for real-time communication
echo 🔗 CORS enabled for Next.js frontend
echo.
echo Press Ctrl+C to stop the server
echo.

python api-server/bridge-server.py

pause