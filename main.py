#!/usr/bin/env python3
"""
Knowledge App - Pure QtWebEngine Main Entry Point
Clean, modern web-based UI with zero QtWidgets bloatware
"""

import sys
import os
import logging
import warnings
from pathlib import Path

# 🔥 ENABLE FULL CONSOLE OUTPUT FOR DEBUGGING
# Allow all output to see detailed logging

# Store original streams
_original_stdout = sys.stdout
_original_stderr = sys.stderr

# Keep original streams for debugging
sys.stdout = _original_stdout
sys.stderr = _original_stderr

# Suppress all Qt debug messages by setting Qt logging rules
os.environ['QT_LOGGING_RULES'] = '*.debug=false;*.info=false;*.warning=false;qt.*=false'

# Suppress Qt WebEngine messages and fix Windows issues
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--disable-logging --disable-gpu-sandbox --no-sandbox --disable-dev-shm-usage --log-level=3 --disable-features=VizDisplayCompositor --disable-extensions --disable-plugins --disable-web-security --allow-running-insecure-content'
os.environ['QTWEBENGINE_DISABLE_SANDBOX'] = '1'
os.environ['QTWEBENGINE_REMOTE_DEBUGGING'] = '0'

# SUPPRESS FAISS GPU WARNINGS IMMEDIATELY AT STARTUP
warnings.filterwarnings("ignore", message=r".*Failed to load GPU Faiss.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*GpuIndexIVFFlat.*not defined.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*FAISS.*", category=UserWarning)

# Add src directory to path for imports
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Fix Unicode logging issues for Windows
def setup_debug_logging():
    """Setup detailed logging with Windows Unicode compatibility"""

    # Clear any existing handlers first
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Enable detailed logging
    root_logger.setLevel(logging.DEBUG)

    # Console handler with UTF-8 encoding support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Use ASCII-safe formatter to avoid Unicode issues
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Set encoding to handle Unicode properly
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass  # Fallback silently

    root_logger.addHandler(console_handler)

    # File handler for detailed logs with UTF-8 encoding
    try:
        file_handler = logging.FileHandler('knowledge_app_debug.log', mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except:
        # Fallback without file logging if it fails
        pass

    # Enable specific modules for debugging
    debug_modules = [
        'src.knowledge_app.core.mcq_manager',
        'src.knowledge_app.core.unified_inference_manager',
        'src.knowledge_app.webengine_app'
    ]
    for module_name in debug_modules:
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(logging.DEBUG)
        module_logger.propagate = True

# Setup debug logging immediately
setup_debug_logging()

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        from PyQt5.QtCore import QUrl, QCoreApplication, Qt
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        from PyQt5.QtWebChannel import QWebChannel
        return True
    except ImportError as e:
        _original_stderr.write(f"IMPORT ERROR: {e}\n")
        return False


def main():
    """Main application entry point - Pure QtWebEngine - SILENT"""
    try:
        
        # Check dependencies
        if not check_dependencies():
            return 1
        
        # CRITICAL: Set Qt attributes BEFORE creating QApplication for WebEngine
        from PyQt5.QtCore import QCoreApplication, Qt, QDir, QStandardPaths

        # Fix QtWebEngine cache issues and startup problems
        QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
        QCoreApplication.setAttribute(Qt.AA_DisableWindowContextHelpButton)
        QCoreApplication.setAttribute(Qt.AA_SynthesizeTouchForUnhandledMouseEvents, False)
        QCoreApplication.setAttribute(Qt.AA_SynthesizeMouseForUnhandledTouchEvents, False)
        QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        
        # Import and run the QtWebEngine app 
        from PyQt5.QtWidgets import QApplication
        from knowledge_app.webengine_app import KnowledgeAppWebEngine
        
        # Create QApplication (required for QWebEngineView widgets)
        app = QApplication(sys.argv)
        app.setApplicationName("Knowledge App")
        app.setApplicationVersion("1.0.0")
        
        # CRITICAL FIX: Configure QtWebEngine to reduce cache errors
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

        except Exception as e:
            # Silent fallback - continue without cache fixes
            print(f"⚠️ WebEngine config warning: {e}")
            pass
        
        # 🚀 DEBUG: Add verbose output to see what's happening
        print("🔧 Starting Knowledge App with splash screen...")

        try:
            from knowledge_app.loading_screen import AnimatedLoadingScreen
            print("✅ Imported AnimatedLoadingScreen")

            # Create and show splash screen
            splash = AnimatedLoadingScreen()
            print("✅ Created splash screen")

            splash.show()
            splash.raise_()
            splash.activateWindow()
            print("✅ Showing splash screen")

            # Start the loading process (this will create the main window)
            splash.start_loading()
            print("✅ Started loading process")

            print("🚀 Running application...")
            result = app.exec_()
            print(f"🏁 App finished with exit code: {result}")
            return result

        except Exception as e:
            print(f"❌ App launch failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        # Show critical startup errors only
        _original_stderr.write(f"CRITICAL ERROR: {e}\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


