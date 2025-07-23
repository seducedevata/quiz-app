"""
Qt Warning Suppressor - Suppress Qt-level warnings

This module provides utilities to suppress warnings at the Qt level,
including CSS property warnings that can't be suppressed with Python's
warning system.

Features:
- Suppress Qt CSS property warnings
- Suppress Qt debug output
- Custom message handler for Qt
- Environment variable configuration
"""

import os
import sys
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Qt message types (from Qt documentation)
QT_DEBUG_MSG = 0
QT_WARNING_MSG = 1
QT_CRITICAL_MSG = 2
QT_FATAL_MSG = 3
QT_INFO_MSG = 4


class QtWarningSupressor:
    """Suppress Qt-level warnings and debug messages"""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.original_handler: Optional[Callable] = None
        self.suppressed_messages = []

        # Messages to suppress (case-insensitive)
        self.suppress_patterns = [
            "unknown property",
            "unknown css property",
            "css property",
            "box-shadow",
            "text-shadow",
            "transform",
            "transition",
            "animation",
            "filter",
            "backdrop-filter",
            "clip-path",
            "mask",
            "perspective",
            "qpainter",
            "painter not active",
            "qpaintdevice",
            "qpaintengine",
            "painting",
            "qstylesheet",
            "style sheet",
            "qstyle",
        ]

    def qt_message_handler(self, msg_type, context, message):
        """Custom Qt message handler to suppress unwanted messages"""
        message_lower = message.lower()

        # Check if message should be suppressed
        should_suppress = any(pattern in message_lower for pattern in self.suppress_patterns)

        if should_suppress:
            self.suppressed_messages.append((msg_type, message))
            if self.debug_mode:
                logger.debug(f"Suppressed Qt message: {message}")
            return  # Don't print the message

        # For non-suppressed messages, use default behavior
        if msg_type == QT_DEBUG_MSG and not self.debug_mode:
            return  # Suppress debug messages in non-debug mode
        elif msg_type == QT_WARNING_MSG:
            if self.debug_mode:
                logger.warning(f"Qt Warning: {message}")
        elif msg_type == QT_CRITICAL_MSG:
            logger.error(f"Qt Critical: {message}")
        elif msg_type == QT_FATAL_MSG:
            logger.critical(f"Qt Fatal: {message}")
        elif msg_type == QT_INFO_MSG:
            if self.debug_mode:
                logger.info(f"Qt Info: {message}")

    def install(self):
        """Install the custom Qt message handler"""
        try:
            from PyQt5.QtCore import qInstallMessageHandler

            # Install our custom handler
            self.original_handler = qInstallMessageHandler(self.qt_message_handler)

            if not self.debug_mode:
                logger.debug("Qt warning suppression installed")

            return True

        except ImportError:
            logger.warning("PyQt5 not available, cannot install Qt message handler")
            return False
        except Exception as e:
            logger.error(f"Failed to install Qt message handler: {e}")
            return False

    def uninstall(self):
        """Restore the original Qt message handler"""
        try:
            from PyQt5.QtCore import qInstallMessageHandler

            if self.original_handler:
                qInstallMessageHandler(self.original_handler)
                self.original_handler = None

                if not self.debug_mode:
                    logger.debug("Qt warning suppression uninstalled")

            return True

        except Exception as e:
            logger.error(f"Failed to uninstall Qt message handler: {e}")
            return False

    def get_suppression_stats(self):
        """Get statistics about suppressed messages"""
        stats = {}
        for msg_type, message in self.suppressed_messages:
            type_name = {
                QT_DEBUG_MSG: "Debug",
                QT_WARNING_MSG: "Warning",
                QT_CRITICAL_MSG: "Critical",
                QT_FATAL_MSG: "Fatal",
                QT_INFO_MSG: "Info",
            }.get(msg_type, "Unknown")

            stats[type_name] = stats.get(type_name, 0) + 1

        return {
            "total_suppressed": len(self.suppressed_messages),
            "by_type": stats,
            "debug_mode": self.debug_mode,
        }


def setup_qt_warning_suppression(debug_mode: bool = False) -> QtWarningSupressor:
    """
    Set up Qt-level warning suppression.

    Args:
        debug_mode: If True, warnings will be logged for debugging

    Returns:
        QtWarningSupressor instance
    """
    # Set Qt environment variables for cleaner output
    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.xcb=false"
    os.environ["QT_ASSUME_STDERR_HAS_CONSOLE"] = "1"

    # Create and install suppressor
    suppressor = QtWarningSupressor(debug_mode=debug_mode)

    if suppressor.install():
        logger.info("✅ Qt warning suppression enabled")
    else:
        logger.warning("⚠️ Failed to enable Qt warning suppression")

    return suppressor


def setup_environment_for_clean_qt_output():
    """Set up environment variables for clean Qt output"""
    qt_env_vars = {
        # Disable Qt debug output
        "QT_LOGGING_RULES": "*.debug=false;qt.qpa.xcb=false;qt.qpa.input=false",
        # Assume stderr has console (prevents some warnings)
        "QT_ASSUME_STDERR_HAS_CONSOLE": "1",
        # Disable Qt warnings about deprecated features
        "QT_DEPRECATED_WARNINGS": "0",
        # Disable Qt accessibility warnings
        "QT_ACCESSIBILITY": "0",
        # Disable Qt style warnings
        "QT_STYLE_OVERRIDE": "",
        # Disable Qt platform plugin warnings
        "QT_QPA_PLATFORM_PLUGIN_PATH": "",
    }

    for var, value in qt_env_vars.items():
        if var not in os.environ:
            os.environ[var] = value

    logger.debug("Qt environment configured for clean output")


# Global suppressor instance
_global_qt_suppressor: Optional[QtWarningSupressor] = None


def get_global_qt_suppressor() -> Optional[QtWarningSupressor]:
    """Get the global Qt warning suppressor instance"""
    return _global_qt_suppressor


def install_global_qt_warning_suppression(debug_mode: bool = False) -> bool:
    """Install global Qt warning suppression"""
    global _global_qt_suppressor

    if _global_qt_suppressor is None:
        # Set up environment first
        setup_environment_for_clean_qt_output()

        # Create and install suppressor
        _global_qt_suppressor = setup_qt_warning_suppression(debug_mode)
        return True

    return False


def uninstall_global_qt_warning_suppression() -> bool:
    """Uninstall global Qt warning suppression"""
    global _global_qt_suppressor

    if _global_qt_suppressor:
        success = _global_qt_suppressor.uninstall()
        _global_qt_suppressor = None
        return success

    return False


# Auto-setup when module is imported (can be disabled by setting environment variable)
if os.getenv("DISABLE_AUTO_QT_WARNING_SUPPRESSION") != "1":
    try:
        # Only auto-setup if PyQt5 is available
        import PyQt5

        debug_mode = os.getenv("KNOWLEDGE_APP_DEBUG_WARNINGS", "false").lower() == "true"
        install_global_qt_warning_suppression(debug_mode=debug_mode)
    except ImportError:
        # PyQt5 not available, skip auto-setup
        pass
    except Exception as e:
        logger.warning(f"Failed to auto-setup Qt warning suppression: {e}")