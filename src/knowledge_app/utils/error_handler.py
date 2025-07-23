"""
Enhanced error handling system with intelligent logging and recovery
Pure logging-based implementation - no QtWidgets bloatware
"""

from .async_converter import async_time_sleep


from .async_converter import async_time_sleep


import logging
import sys
import traceback
from enum import Enum
from typing import Optional, Dict, Any, Callable
from functools import wraps
import time
import json
import threading
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorCategory:
    """Error categories for better error handling"""

    UI = "ui"
    NETWORK = "network"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    GPU = "gpu"
    MEMORY = "memory"
    MODEL = "model"
    SYSTEM = "system"
    INITIALIZATION = "initialization"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


class ErrorSeverity:
    """Error severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorHandler:
    """Centralized error handling system with thread-safe logging"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ErrorHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.error_log_path = None
        self.error_stats_path = None
        self.error_callbacks: Dict[str, Callable] = {}
        self.last_error = None
        self.error_stats: Dict[str, Dict[str, int]] = {}
        self._recovery_handlers: Dict[str, Callable] = {}
        self._max_retries = 3
        self._retry_delays = [1, 5, 15]  # Seconds between retries
        self._retry_counts: Dict[str, int] = {}  # Track retries per error type
        self._retry_lock = threading.Lock()
        # Thread-safe logging lock for enterprise-grade error handling
        self._log_lock = threading.Lock()
        # Professional error manager for user-friendly error handling
        self._professional_error_manager = None
        # 🛡️ SECURITY FIX: Recursion depth tracking to prevent infinite recursion
        self._recursion_depth = 0
        self._max_recursion_depth = 5

    def initialize(self, log_dir: Path) -> None:
        """Initialize error handler with log directory"""
        try:
            self.error_log_path = log_dir / "error.log"
            self.error_stats_path = log_dir / "error_stats.json"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Load existing error stats
            self._load_error_stats()

        except Exception as e:
            logger.error(f"Failed to initialize error handler: {e}")

    def set_professional_error_manager(self, error_manager):
        """Set the professional error manager for user-friendly error handling"""
        self._professional_error_manager = error_manager
        logger.info("Professional error manager integrated")

    def _load_error_stats(self):
        """Load error statistics from file"""
        try:
            if self.error_stats_path and self.error_stats_path.exists():
                with open(self.error_stats_path, "r") as f:
                    self.error_stats = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load error stats: {e}")
            self.error_stats = {}

    def _save_error_stats(self):
        """Save error statistics to file"""
        try:
            if self.error_stats_path:
                with open(self.error_stats_path, "w") as f:
                    json.dump(self.error_stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save error stats: {e}")

    def categorize_error(self, error: Exception) -> str:
        """Categorize an error based on its type and context"""
        error_type = type(error).__name__.lower()

        if any(net_err in error_type for net_err in ["socket", "http", "url", "connection"]):
            return ErrorCategory.NETWORK
        elif any(db_err in error_type for db_err in ["sql", "db", "database"]):
            return ErrorCategory.DATABASE
        elif any(fs_err in error_type for fs_err in ["file", "io", "os", "path"]):
            return ErrorCategory.FILE_SYSTEM
        elif any(gpu_err in error_type for gpu_err in ["cuda", "gpu", "device"]):
            return ErrorCategory.GPU
        elif any(mem_err in error_type for mem_err in ["memory", "allocation"]):
            return ErrorCategory.MEMORY
        elif any(model_err in error_type for model_err in ["model", "tensor", "parameter"]):
            return ErrorCategory.MODEL
        elif "qt" in error_type or "ui" in error_type:
            return ErrorCategory.UI
        elif any(sys_err in error_type for sys_err in ["system", "os", "environment"]):
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.UNKNOWN

    def register_recovery_handler(self, category: str, handler: Callable) -> None:
        """Register a recovery handler for a specific error category"""
        self._recovery_handlers[category] = handler

    def attempt_recovery(self, error: Exception, category: str) -> bool:
        """Attempt to recover from an error"""
        try:
            if category not in self._recovery_handlers:
                return False

            with self._retry_lock:
                retry_count = self._retry_counts.get(category, 0)
                if retry_count >= self._max_retries:
                    logger.warning(f"Max retries ({self._max_retries}) reached for {category}")
                    return False

                self._retry_counts[category] = retry_count + 1

            # Call recovery handler
            handler = self._recovery_handlers[category]
            success = handler(error)

            if success:
                with self._retry_lock:
                    self._retry_counts[category] = 0
                logger.info(f"Successfully recovered from {category} error")
                return True

            return False

        except Exception as e:
            logger.error(f"Error during recovery attempt: {e}")
            return False

    def reset_retry_count(self, category: str) -> None:
        """Reset retry count for a category"""
        with self._retry_lock:
            self._retry_counts[category] = 0

    def handle_error(
        self,
        error: Exception,
        context: str = "",
        severity: str = ErrorSeverity.ERROR,
        show_dialog: bool = True,
        retry_count: int = 0,
    ) -> None:
        """Handle an error with recovery attempts and thread-safe logging"""
        # 🛡️ SECURITY FIX: Prevent infinite recursion in error handler
        self._recursion_depth += 1
        if self._recursion_depth > self._max_recursion_depth:
            logger.critical(f"💥 CRITICAL: Error handler recursion depth exceeded ({self._recursion_depth}). Breaking recursion to prevent stack overflow.")
            self._recursion_depth = 0  # Reset counter
            return

        try:
            # Get error category
            category = self.categorize_error(error)

            # Thread-safe error logging and statistics update
            with self._log_lock:
                # Log error details
                self.last_error = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": "".join(
                        traceback.format_exception(type(error), error, error.__traceback__)
                    ),
                    "context": context,
                    "severity": severity,
                    "category": category,
                    "timestamp": datetime.now().isoformat(),
                }

                # Update error statistics
                if category not in self.error_stats:
                    self.error_stats[category] = {"count": 0, "last_occurrence": None}
                self.error_stats[category]["count"] += 1
                self.error_stats[category]["last_occurrence"] = datetime.now().isoformat()

                # Log error
                self._log_error(category, context, error, severity)

                # Write to error log file
                self._write_to_error_log()

                # Save error statistics to file
                self._save_error_stats()

            # Attempt recovery if appropriate
            if severity != ErrorSeverity.CRITICAL and retry_count < self._max_retries:
                if category in self._recovery_handlers:
                    if self.attempt_recovery(error, category):
                        return

                    # Wait before retry
                    if retry_count < len(self._retry_delays):
                        await async_time_await async_time_await async_time_sleep(self._retry_delays[retry_count])

                    # 🛡️ SECURITY FIX: Check recursion depth before recursive call
                    if self._recursion_depth < self._max_recursion_depth:
                        self.handle_error(error, context, severity, show_dialog, retry_count + 1)
                    else:
                        logger.error(f"❌ Max recursion depth reached, stopping retry attempts for {context}")
                    return

            # Call registered error callbacks
            for callback in self.error_callbacks.values():
                try:
                    callback(self.last_error)
                except Exception as cb_error:
                    logger.error(f"Error callback failed: {cb_error}")

        except Exception as e:
            logger.error(f"Error in error handler: {e}")
        finally:
            # 🛡️ SECURITY FIX: Always decrement recursion depth
            self._recursion_depth = max(0, self._recursion_depth - 1)

    def _log_error(self, category: str, context: str, error: Exception, severity: str):
        """Log error with appropriate severity level"""
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"💥 CRITICAL [{category}] in {context}: {error}")
            logger.critical(self.last_error["traceback"])
        elif severity == ErrorSeverity.ERROR:
            logger.error(f"❌ ERROR [{category}] in {context}: {error}")
        elif severity == ErrorSeverity.WARNING:
            logger.warning(f"⚠️ WARNING [{category}] in {context}: {error}")
        else:
            logger.info(f"ℹ️ INFO [{category}] in {context}: {error}")

    def _write_to_error_log(self):
        """Write error details to log file"""
        if self.error_log_path and self.last_error:
            try:
                with open(self.error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Timestamp: {self.last_error['timestamp']}\n")
                    f.write(f"Category: {self.last_error['category']}\n")
                    f.write(f"Severity: {self.last_error['severity']}\n")
                    f.write(f"Context: {self.last_error['context']}\n")
                    f.write(f"Error: {self.last_error['type']}: {self.last_error['message']}\n")
                    f.write(f"Traceback:\n{self.last_error['traceback']}\n")
            except Exception as e:
                logger.error(f"Failed to write to error log: {e}")



    def register_error_callback(self, name: str, callback: Callable) -> None:
        """Register a callback to be called when errors occur"""
        self.error_callbacks[name] = callback

    def unregister_error_callback(self, name: str) -> None:
        """Unregister an error callback"""
        self.error_callbacks.pop(name, None)

    def get_last_error(self) -> Optional[Dict[str, Any]]:
        """Get information about the last error"""
        return self.last_error

    def clear_last_error(self) -> None:
        """Clear the last error information"""
        self.last_error = None

    def get_error_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get error statistics"""
        return self.error_stats

    def clear_error_stats(self) -> None:
        """Clear error statistics"""
        self.error_stats = {}
        self._save_error_stats()


# Global error handler instance
error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance"""
    return error_handler


def handle_exception(exctype: Type[BaseException], value: BaseException, tb) -> None:
    """Global exception handler for unhandled exceptions"""
    try:
        if issubclass(exctype, KeyboardInterrupt):
            # Handle keyboard interrupt specially
            sys.__excepthook__(exctype, value, tb)
            return

        error_handler.handle_error(
            value, context="Unhandled exception", severity=ErrorSeverity.CRITICAL, show_dialog=True
        )
    except Exception as e:
        # If error handling fails, use sys.__excepthook__ as last resort
        print(f"Error handler failed: {e}", file=sys.stderr)
        sys.__excepthook__(exctype, value, tb)


# Install global exception handler
sys.excepthook = handle_exception