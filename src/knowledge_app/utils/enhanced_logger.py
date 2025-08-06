"""
Enhanced Logging System for Knowledge App
Provides detailed user action tracking for both frontend and backend
"""

import logging
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import traceback

class UserActionLogger:
    """Enhanced logger specifically for user actions and system events"""

    def __init__(self, log_file: str = "user_actions.log"):
        self.log_file = log_file
        self.session_id = str(int(time.time()))
        self.action_counter = 0
        self.lock = threading.Lock()

        # Create dedicated loggers
        self.user_logger = logging.getLogger('USER_ACTION')
        self.system_logger = logging.getLogger('SYSTEM_EVENT')
        self.api_logger = logging.getLogger('API_CALL')
        self.frontend_logger = logging.getLogger('FRONTEND')
        self.backend_logger = logging.getLogger('BACKEND')

        self._setup_loggers()

        # Log session start
        self.log_system_event("APPLICATION_STARTED", {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "user": "current_user"
        })

    def _setup_loggers(self):
        """Setup enhanced logging configuration"""

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )

        console_formatter = logging.Formatter(
            '🔥 %(levelname)s [%(name)s]: %(message)s'
        )

        # File handler for detailed logs
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)

        # Console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)

        # Setup all loggers
        for logger in [self.user_logger, self.system_logger, self.api_logger,
                      self.frontend_logger, self.backend_logger]:
            logger.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.propagate = False

    def log_user_action(self, action: str, details: Dict[str, Any] = None, source: str = "unknown"):
        """Log user actions with detailed context"""
        with self.lock:
            self.action_counter += 1

            log_data = {
                "session_id": self.session_id,
                "action_id": self.action_counter,
                "action": action,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "details": details or {}
            }

            message = f"USER ACTION: {action} | Source: {source}"
            if details:
                message += f" | Details: {json.dumps(details, default=str)}"

            self.user_logger.info(message)

            # Also log to console with emoji for visibility
            print(f"👤 USER: {action} ({source})")

    def log_system_event(self, event: str, details: Dict[str, Any] = None):
        """Log system events"""
        log_data = {
            "session_id": self.session_id,
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }

        message = f"SYSTEM EVENT: {event}"
        if details:
            message += f" | Details: {json.dumps(details, default=str)}"

        self.system_logger.info(message)
        print(f"⚙️ SYSTEM: {event}")

    def log_api_call(self, endpoint: str, method: str = "GET", params: Dict[str, Any] = None,
                    response_status: int = None, duration: float = None):
        """Log API calls with timing information"""
        log_data = {
            "session_id": self.session_id,
            "endpoint": endpoint,
            "method": method,
            "params": params or {},
            "response_status": response_status,
            "duration_ms": duration,
            "timestamp": datetime.now().isoformat()
        }

        message = f"API CALL: {method} {endpoint}"
        if response_status:
            message += f" | Status: {response_status}"
        if duration:
            message += f" | Duration: {duration:.2f}ms"
        if params:
            message += f" | Params: {json.dumps(params, default=str)[:200]}"

        self.api_logger.info(message)
        print(f"🌐 API: {method} {endpoint} ({response_status or 'pending'})")

    def log_frontend_event(self, event: str, element: str = None, details: Dict[str, Any] = None):
        """Log frontend events like clicks, navigation, etc."""
        log_data = {
            "session_id": self.session_id,
            "event": event,
            "element": element,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }

        message = f"FRONTEND EVENT: {event}"
        if element:
            message += f" | Element: {element}"
        if details:
            message += f" | Details: {json.dumps(details, default=str)}"

        self.frontend_logger.info(message)
        print(f"🖥️ FRONTEND: {event} ({element or 'unknown element'})")

    def log_backend_operation(self, operation: str, module: str = None, details: Dict[str, Any] = None,
                            duration: float = None, success: bool = True):
        """Log backend operations"""
        log_data = {
            "session_id": self.session_id,
            "operation": operation,
            "module": module,
            "success": success,
            "duration_ms": duration,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }

        message = f"BACKEND OPERATION: {operation}"
        if module:
            message += f" | Module: {module}"
        if duration:
            message += f" | Duration: {duration:.2f}ms"
        message += f" | Success: {success}"
        if details:
            message += f" | Details: {json.dumps(details, default=str)[:200]}"

        self.backend_logger.info(message)
        status_emoji = "✅" if success else "❌"
        print(f"🔧 BACKEND: {status_emoji} {operation} ({module or 'core'})")

    def log_error(self, error: Exception, context: str = None, details: Dict[str, Any] = None):
        """Log errors with full context"""
        error_data = {
            "session_id": self.session_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }

        message = f"ERROR: {type(error).__name__}: {str(error)}"
        if context:
            message += f" | Context: {context}"

        self.system_logger.error(message)
        print(f"❌ ERROR: {str(error)} ({context or 'unknown context'})")

    def log_navigation(self, from_screen: str, to_screen: str, trigger: str = "user_click"):
        """Log navigation events"""
        self.log_frontend_event("NAVIGATION", "screen_change", {
            "from": from_screen,
            "to": to_screen,
            "trigger": trigger
        })

        self.log_user_action("NAVIGATE", {
            "from_screen": from_screen,
            "to_screen": to_screen,
            "trigger": trigger
        }, source="navigation")

# Global logger instance
_global_logger = None

def get_enhanced_logger() -> UserActionLogger:
    """Get the global enhanced logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = UserActionLogger()
    return _global_logger

# Convenience functions for easy access
def log_user_action(action: str, details: Dict[str, Any] = None, source: str = "unknown"):
    """Convenience function to log user actions"""
    get_enhanced_logger().log_user_action(action, details, source)

def log_system_event(event: str, details: Dict[str, Any] = None):
    """Convenience function to log system events"""
    get_enhanced_logger().log_system_event(event, details)

def log_api_call(endpoint: str, method: str = "GET", params: Dict[str, Any] = None,
                response_status: int = None, duration: float = None):
    """Convenience function to log API calls"""
    get_enhanced_logger().log_api_call(endpoint, method, params, response_status, duration)

def log_frontend_event(event: str, element: str = None, details: Dict[str, Any] = None):
    """Convenience function to log frontend events"""
    get_enhanced_logger().log_frontend_event(event, element, details)

def log_backend_operation(operation: str, module: str = None, details: Dict[str, Any] = None,
                        duration: float = None, success: bool = True):
    """Convenience function to log backend operations"""
    get_enhanced_logger().log_backend_operation(operation, module, details, duration, success)

def log_error(error: Exception, context: str = None, details: Dict[str, Any] = None):
    """Convenience function to log errors"""
    get_enhanced_logger().log_error(error, context, details)

def log_navigation(from_screen: str, to_screen: str, trigger: str = "user_click"):
    """Convenience function to log navigation"""
    get_enhanced_logger().log_navigation(from_screen, to_screen, trigger)
