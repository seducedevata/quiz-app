"""
Centralized logging configuration for Knowledge App

This module provides a unified logging system for the entire application,
supporting different log levels, handlers, and configurations.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Log level constants
LOG_LEVEL_DEBUG = logging.DEBUG
LOG_LEVEL_INFO = logging.INFO
LOG_LEVEL_WARNING = logging.WARNING
LOG_LEVEL_ERROR = logging.ERROR

# Default logging format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ULTRA_DETAILED_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(levelname)8s] %(name)s:%(lineno)d - %(funcName)s() - %(message)s"
)
DETAILED_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)


class LogManager:
    """Centralized logging management system"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.log_dir = None
        self.log_level = logging.INFO
        self.log_format = DEFAULT_LOG_FORMAT
        self.initialized = False
        self.handlers: List[logging.Handler] = []

    def initialize(
        self, log_dir: Path, config: Optional[Dict[str, Any]] = None, env: str = "production"
    ) -> None:
        """
        Initialize logging system

        Args:
            log_dir: Directory for log files
            config: Optional configuration dictionary
            env: Environment ("production", "development", "testing")
        """
        try:
            # Convert to absolute path and resolve any symlinks
            self.log_dir = Path(log_dir).resolve()
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Clean up any existing handlers
            self._remove_existing_handlers()

            # Update settings from config
            if config:
                self.log_level = getattr(logging, config.get("log_level", "INFO"))
                self.log_format = config.get("log_format", self.log_format)

            # Create handlers based on environment
            self._setup_handlers(env)

            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(self.log_level)

            # Add handlers to root logger
            for handler in self.handlers:
                root_logger.addHandler(handler)

            # Mark as initialized
            self.initialized = True

            # Log initialization
            logging.info("Logging system initialized")
            logging.info(f"Log directory: {str(self.log_dir)}")
            logging.info(f"Log level: {logging.getLevelName(self.log_level)}")
            logging.info(f"Environment: {env}")

        except Exception as e:
            # If logging setup fails, print to stderr as last resort
            print(f"Failed to initialize logging: {e}", file=sys.stderr)
            raise

    def _remove_existing_handlers(self) -> None:
        """Remove existing handlers from root logger"""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    def _setup_handlers(self, env: str) -> None:
        """Set up comprehensive handlers with ultra-detailed logging"""
        self.handlers = []

        # Ultra-detailed console handler for debugging
        console_level = LOG_LEVEL_DEBUG if env != "production" else LOG_LEVEL_INFO
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(logging.Formatter(ULTRA_DETAILED_LOG_FORMAT))
        self.handlers.append(console_handler)

        # Main application log with ultra details
        main_log = self.log_dir / "app_detailed.log"
        file_handler = logging.handlers.RotatingFileHandler(
            str(main_log),
            maxBytes=50 * 1024 * 1024,  # 50MB for detailed logs
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(ULTRA_DETAILED_LOG_FORMAT))
        self.handlers.append(file_handler)

        # Online MCQ generation specific log
        online_log = self.log_dir / "online_mcq.log"
        online_handler = logging.handlers.RotatingFileHandler(
            str(online_log),
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=5,
            encoding="utf-8",
        )
        online_handler.setFormatter(logging.Formatter(ULTRA_DETAILED_LOG_FORMAT))
        online_handler.addFilter(self._create_module_filter("online_mcq"))
        self.handlers.append(online_handler)

        # Offline MCQ generation specific log
        offline_log = self.log_dir / "offline_mcq.log"
        offline_handler = logging.handlers.RotatingFileHandler(
            str(offline_log),
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=5,
            encoding="utf-8",
        )
        offline_handler.setFormatter(logging.Formatter(ULTRA_DETAILED_LOG_FORMAT))
        offline_handler.addFilter(self._create_module_filter("offline_mcq"))
        self.handlers.append(offline_handler)

        # LaTeX rendering specific log
        latex_log = self.log_dir / "latex_rendering.log"
        latex_handler = logging.handlers.RotatingFileHandler(
            str(latex_log),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,
            encoding="utf-8",
        )
        latex_handler.setFormatter(logging.Formatter(ULTRA_DETAILED_LOG_FORMAT))
        latex_handler.addFilter(self._create_module_filter("latex"))
        self.handlers.append(latex_handler)

        # API calls and network specific log
        api_log = self.log_dir / "api_calls.log"
        api_handler = logging.handlers.RotatingFileHandler(
            str(api_log),
            maxBytes=30 * 1024 * 1024,  # 30MB for API details
            backupCount=5,
            encoding="utf-8",
        )
        api_handler.setFormatter(logging.Formatter(ULTRA_DETAILED_LOG_FORMAT))
        api_handler.addFilter(self._create_module_filter("api"))
        self.handlers.append(api_handler)

        # Error log file handler with stack traces
        error_log = self.log_dir / "error_detailed.log"
        error_handler = logging.handlers.RotatingFileHandler(
            str(error_log),
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(LOG_LEVEL_ERROR)
        error_handler.setFormatter(logging.Formatter(ULTRA_DETAILED_LOG_FORMAT))
        self.handlers.append(error_handler)

        # Performance monitoring log
        perf_log = self.log_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            str(perf_log),
            maxBytes=15 * 1024 * 1024,  # 15MB
            backupCount=3,
            encoding="utf-8",
        )
        perf_handler.setFormatter(logging.Formatter(ULTRA_DETAILED_LOG_FORMAT))
        perf_handler.addFilter(self._create_module_filter("performance"))
        self.handlers.append(perf_handler)

    def _create_module_filter(self, module_keyword: str):
        """Create a filter for specific modules/components"""
        class ModuleFilter(logging.Filter):
            def __init__(self, keyword):
                super().__init__()
                self.keyword = keyword.lower()
                
            def filter(self, record):
                # Include records that contain the keyword in logger name or message
                return (self.keyword in record.name.lower() or 
                       self.keyword in record.getMessage().lower())
        
        return ModuleFilter(module_keyword)

    def set_level(self, level: str) -> None:
        """Set logging level"""
        if not self.initialized:
            return

        try:
            numeric_level = getattr(logging, level.upper())
            logging.getLogger().setLevel(numeric_level)
            self.log_level = numeric_level
            logging.info(f"Log level changed to: {level}")
        except (AttributeError, TypeError) as e:
            logging.error(f"Invalid log level: {level}")

    def get_log_files(self) -> Dict[str, Path]:
        """Get paths to log files"""
        if not self.log_dir:
            return {}

        return {
            "main": self.log_dir / "app_detailed.log",
            "error": self.log_dir / "error_detailed.log",
            "debug": self.log_dir / "online_mcq.log",
        }

    def archive_logs(self) -> Optional[Path]:
        """Archive current logs to timestamped directory"""
        if not self.initialized or not self.log_dir:
            return None

        try:
            # Create archive directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = self.log_dir / "archive" / timestamp
            archive_dir.mkdir(parents=True, exist_ok=True)

            # Move current logs to archive
            for log_file in self.get_log_files().values():
                if log_file.exists():
                    target = archive_dir / log_file.name
                    log_file.rename(target)

            logging.info(f"Logs archived to: {archive_dir}")
            return archive_dir

        except Exception as e:
            logging.error(f"Failed to archive logs: {e}")
            return None

    def cleanup_old_logs(self, max_age_days: int = 30) -> None:
        """Clean up old log archives"""
        if not self.initialized or not self.log_dir:
            return

        try:
            archive_dir = self.log_dir / "archive"
            if not archive_dir.exists():
                return

            cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)

            for item in archive_dir.iterdir():
                if item.is_dir() and item.stat().st_mtime < cutoff:
                    for log_file in item.glob("*.log"):
                        log_file.unlink()
                    item.rmdir()

            logging.info(f"Cleaned up log archives older than {max_age_days} days")

        except Exception as e:
            logging.error(f"Failed to clean up old logs: {e}")


def get_log_manager() -> LogManager:
    """Get the singleton LogManager instance"""
    return LogManager()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging for the application

    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file path
    """
    # Create logs directory if needed
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log file specified
    if log_file:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                delay=True,  # Don't open file until first log
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.error(f"Failed to setup file logging: {e}")

    # Set logging level for specific modules
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

    # Log startup message
    root_logger.info("Logging system initialized")