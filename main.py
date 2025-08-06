#!/usr/bin/env python3
"""
Knowledge App - Pure QtWebEngine Main Entry Point
Clean, modern web-based UI with zero QtWidgets bloatware

🔧 LOGGING OPTIMIZATION:
- Console logging set to WARNING level to reduce noise
- File logging maintains DEBUG level for troubleshooting
- Repetitive status checks converted from INFO to DEBUG
- JS messages and API calls reduced to DEBUG level
"""

import sys
import os
import logging
import warnings
from pathlib import Path
import threading
from backend import ipc_server # Import the IPC server

# ========== COMPREHENSIVE LOGGING SYSTEM ==========
class ComprehensiveFilter(logging.Filter):
    """Filter that captures ALL frontend/backend interactions"""
    
    def filter(self, record):
        message = record.getMessage().lower()
        
        # ALWAYS show these critical categories
        if any(keyword in message for keyword in [
            'user', 'action', 'navigation', 'bridge', 'client', 'js log', 
            'review', 'quiz', 'button', 'click', 'screen', 'showing',
            'loading', 'filtering', 'history', 'error', 'critical',
            'warning', 'failed', 'success', 'system', 'starting'
        ]):
            return True
            
        # Always show ERROR, WARNING and above
        if record.levelno >= logging.WARNING:
            return True
            
        # Show INFO level for important modules
        if record.levelno == logging.INFO and any(module in record.name for module in [
            'webengine_app', 'Bridge', 'knowledge_app', 'main'
        ]):
            return True
            
        # Suppress only DEBUG from noisy third-party modules
        if record.levelno == logging.DEBUG and any(module in record.name for module in [
            'urllib3', 'requests', 'asyncio', 'websockets'
        ]):
            return False
            
        return True

# Setup comprehensive console logging - capture EVERYTHING
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.addFilter(ComprehensiveFilter())
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

# Ensure UTF-8 encoding for Windows terminal
import codecs
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Setup file logging for debugging (captures everything)
try:
    file_handler = logging.FileHandler('app_debug.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
except:
    # Fallback without file logging if it fails
    file_handler = None

# Configure root logger
if file_handler:
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler, file_handler]
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler]
    )

# Log critical startup
    logging.info("Knowledge App starting...")# Suppress all third-party noise
for module in ['urllib3', 'requests', 'asyncio', 'websockets', 'PyQt5', 'qt', 'pdfminer', 'httpx']:
    logging.getLogger(module).setLevel(logging.CRITICAL)
    logging.getLogger(module).propagate = False

# Create specialized loggers for critical events
critical_logger = logging.getLogger('CRITICAL')
user_logger = logging.getLogger('USER')
system_logger = logging.getLogger('SYSTEM')

# Set these to always show important events
for logger in [critical_logger, user_logger, system_logger]:
    logger.setLevel(logging.INFO)

# Enable full console output and store original streams
_original_stdout = sys.stdout
_original_stderr = sys.stderr
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Suppress all Qt debug messages by setting Qt logging rules
os.environ['QT_LOGGING_RULES'] = '*.debug=false;*.info=false;*.warning=false;qt.*=false'




os.environ['QTWEBENGINE_REMOTE_DEBUGGING'] = '0'

# SUPPRESS FAISS GPU WARNINGS IMMEDIATELY AT STARTUP
warnings.filterwarnings("ignore", message=r".*Failed to load GPU Faiss.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*GpuIndexIVFFlat.*not defined.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*FAISS.*", category=UserWarning)

# SUPPRESS PDF MINING WARNINGS - these spam the console during PDF processing
warnings.filterwarnings("ignore", message=r".*Cannot set gray.*color.*", category=UserWarning)
warnings.filterwarnings("ignore", module="pdfminer.*")

# Set pdfminer logging to ERROR to suppress noisy warnings
import logging
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('pdfminer.pdfinterp').setLevel(logging.ERROR)
logging.getLogger('pdfminer.converter').setLevel(logging.ERROR)
logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)

# SUPPRESS OTHER NOISY LOGGERS IMMEDIATELY
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.ERROR)
logging.getLogger('websockets').setLevel(logging.ERROR)

# Suppress Qt related loggers
logging.getLogger('PyQt5').setLevel(logging.ERROR)
logging.getLogger('qt').setLevel(logging.ERROR)

# CRITICAL FIX: Configure NLTK data path before any imports use it
try:
    import nltk
    import os
    # Add common NLTK data locations
    nltk_paths = [
        os.path.expanduser('~/nltk_data'),
        'C:/Users/ADMIN/nltk_data',
        os.path.join(os.path.dirname(__file__), 'nltk_data'),
    ]
    for path in nltk_paths:
        if os.path.exists(path) and path not in nltk.data.path:
            nltk.data.path.insert(0, path)
            print(f"✅ Added NLTK data path: {path}")
except ImportError:
    pass

# Add src directory to path for imports
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# ========== CRITICAL EVENT LOGGING ==========
def log_critical(message):
    """Log critical events that user should see"""
    print(f"🔥 CRITICAL: {message}")
    logging.critical(f"CRITICAL: {message}")

def log_user_action(message):
    """Log important user actions"""
    print(f"👤 USER: {message}")
    logging.info(f"USER_ACTION: {message}")

def log_system_event(message):
    """Log important system events"""
    print(f"⚙️ SYSTEM: {message}")
    logging.info(f"SYSTEM: {message}")

def log_error(message):
    """Log errors that need attention"""
    print(f"❌ ERROR: {message}")
    logging.error(f"ERROR: {message}")

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        from PyQt5.QtCore import QUrl, QCoreApplication, Qt
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        from PyQt5.QtWebChannel import QWebChannel
        return True
    except ImportError as e:
        log_error(f"PyQt5 import failed: {e}")
        return False


def main():
    """Main application entry point - Pure QtWebEngine with Critical Logging"""
    try:
        log_system_event("Knowledge App initializing...")
        
        # Start the IPC server in a separate thread
        ipc_thread = threading.Thread(target=ipc_server.start_ipc_server, daemon=True)
        ipc_thread.start()
        log_system_event("IPC server thread started")

        # Check dependencies
        log_system_event("Checking PyQt5 dependencies...")
        if not check_dependencies():
            log_error("PyQt5 dependencies check failed")
            return 1
        log_system_event("PyQt5 dependencies verified")
        
        # CRITICAL: Set Qt attributes BEFORE creating QApplication for WebEngine
        log_system_event("Configuring Qt attributes for WebEngine...")
        from PyQt5.QtCore import QCoreApplication, Qt, QDir, QStandardPaths

        # Fix QtWebEngine cache issues and startup problems
        QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
        QCoreApplication.setAttribute(Qt.AA_DisableWindowContextHelpButton)
        QCoreApplication.setAttribute(Qt.AA_SynthesizeTouchForUnhandledMouseEvents, False)
        QCoreApplication.setAttribute(Qt.AA_SynthesizeMouseForUnhandledTouchEvents, False)
        QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        
        # 🔧 BUTTON FIX: Add mouse tracking attributes for better click detection
        QCoreApplication.setAttribute(Qt.AA_SynthesizeTouchForUnhandledMouseEvents, False)
        log_system_event("Qt attributes configured for button click detection")
        
        # Import and run the QtWebEngine app 
        log_system_event("Importing application modules...")
        from PyQt5.QtWidgets import QApplication
        from knowledge_app.webengine_app import KnowledgeAppWebEngine
        
        # Create QApplication (required for QWebEngineView widgets)
        log_system_event("Creating Qt Application...")
        app = QApplication(sys.argv)
        app.setApplicationName("Knowledge App")
        app.setApplicationVersion("1.0.0")
        
        # 🔧 BUTTON FIX: Store startup time for debugging
        app.setProperty("startTime", str(threading.Thread.ident))
        
        # Add crash detection for Qt events
        def qt_message_handler(mode, context, message):
            if mode == 3:  # QtCriticalMsg
                log_critical(f"Qt CRITICAL: {message}")
            elif mode == 2:  # QtWarningMsg and it's about crashes
                if "crash" in message.lower() or "abort" in message.lower():
                    log_error(f"Qt WARNING (potential crash): {message}")
            # 🔧 BUTTON FIX: Log mouse/click related messages
            elif "click" in message.lower() or "mouse" in message.lower() or "button" in message.lower():
                log_user_action(f"Qt MOUSE/BUTTON: {message}")
        
        # Install the message handler (only if available)
        try:
            from PyQt5.QtCore import qInstallMessageHandler
            qInstallMessageHandler(qt_message_handler)
            log_system_event("Qt message handler installed for crash detection")
        except:
            log_error("Failed to install Qt message handler")
        
        # CRITICAL FIX: Configure QtWebEngine to reduce cache errors
        log_system_event("Configuring WebEngine cache settings...")
        try:
            from PyQt5.QtWebEngineWidgets import QWebEngineProfile, QWebEngineSettings

            # Create a writable cache directory in the app folder
            cache_dir = Path(__file__).parent / "cache" / "webengine"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Set the cache directory for the default profile
            profile = QWebEngineProfile.defaultProfile()
            profile.setCachePath(str(cache_dir))
            profile.setPersistentStoragePath(str(cache_dir / "storage"))

            # Disable ALL caching to prevent permission issues
            profile.setHttpCacheType(QWebEngineProfile.NoCache)
            profile.setHttpCacheMaximumSize(0)

            # Disable other problematic features
            settings = profile.settings()
            settings.setAttribute(QWebEngineSettings.PluginsEnabled, False)
            settings.setAttribute(QWebEngineSettings.JavascriptCanOpenWindows, False)
            settings.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebGLEnabled, False)
            settings.setAttribute(QWebEngineSettings.Accelerated2dCanvasEnabled, False)
            
            log_system_event("WebEngine cache and security settings configured ✅")

        except Exception as e:
            log_error(f"WebEngine configuration failed: {e}")
        
        # 🚀 Start the main application UI
        log_system_event("Starting main application UI...")

        try:
            from knowledge_app.loading_screen import AnimatedLoadingScreen
            log_system_event("Loading screen module imported ✅")

            # Create and show splash screen
            splash = AnimatedLoadingScreen()
            log_system_event("Splash screen created")

            splash.show()
            splash.raise_()
            splash.activateWindow()
            log_user_action("Splash screen displayed to user")

            # Start the loading process (this will create the main window)
            splash.start_loading()
            log_system_event("Application loading process started")

            log_critical("🚀 Knowledge App running - Ready for user interaction")
            
            # Check if the main window is actually visible
            window_count = len(app.allWindows())
            log_system_event(f"Active windows count: {window_count}")
            
            for i, window in enumerate(app.allWindows()):
                window_name = window.objectName() or 'Unnamed'
                visible = window.isVisible()
                log_system_event(f"Window {i}: {window_name} - Visible: {visible}")
            
            # Add exception handling around the event loop
            try:
                log_system_event("Starting Qt event loop...")
                result = app.exec_()
                
                if result == 0:
                    log_user_action("Application closed normally by user")
                else:
                    log_error(f"Application exited with error code: {result}")
                
                return result
                
            except Exception as app_error:
                log_critical(f"EXCEPTION in Qt event loop: {app_error}")
                import traceback
                traceback.print_exc()
                return 1

        except Exception as e:
            log_critical(f"Application launch failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
    except KeyboardInterrupt:
        log_user_action("Application interrupted by user (Ctrl+C)")
        return 0
    except Exception as e:
        log_critical(f"STARTUP ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
