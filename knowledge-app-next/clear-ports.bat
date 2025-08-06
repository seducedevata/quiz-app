@echo off
echo 🧹 Clearing all ports used by Knowledge App...
echo.

node -e "const PortManager = require('./scripts/port-manager'); new PortManager().clearAllPorts().then(() => console.log('✅ All ports cleared!'));"

echo.
echo ✅ Port clearing complete!
pause