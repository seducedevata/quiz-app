"""
Comprehensive logging configuration for Knowledge App
Provides structured logging with multiple handlers and formatters
"""

import logging
import logging.handlers
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class StructuredFormatter(logging.Formatter):
    """Custom formatter that adds structured data to log records"""
    
    def format(self, record):
        # Add structured data if available
        if hasattr(record, 'data') and record.data:
            # Convert data to JSON string for structured logging
            try:
                data_str = json.dumps(record.data, default=str, separators=(',', ':'))
                record.structured_data = f" | DATA: {data_str}"
            except (TypeError, ValueError):
                record.structured_data = f" | DATA: {str(record.data)}"
        else:
            record.structured_data = ""
        
        return super().format(record)

class ColoredConsoleFormatter(StructuredFormatter):
    """Console formatter with colors for different log levels"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        return super().format(record)

class KnowledgeAppLogger:
    """Main logging manager for Knowledge App"""
    
    def __init__(self, logs_dir: str = "logs", app_name: str = "knowledge_app"):
        self.logs_dir = Path(logs_dir)
        self.app_name = app_name
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create date-based log file names
        self.date_str = datetime.now().strftime('%Y%m%d')
        self.datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Track logging statistics - initialize before setup_logging
        self.stats = {
            'total_logs': 0,
            'logs_by_level': {},
            'logs_by_category': {},
            'start_time': datetime.now(),
            'session_id': f"{app_name}_{self.datetime_str}"
        }
        
        # Initialize logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        
        # Create formatters
        detailed_formatter = StructuredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-20s:%(lineno)-4d | %(message)s%(structured_data)s'
        )
        
        simple_formatter = StructuredFormatter(
            '%(asctime)s | %(levelname)-8s | %(message)s%(structured_data)s'
        )
        
        console_formatter = ColoredConsoleFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
        )
        
        # Create handlers
        handlers = []
        
        # 1. Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        handlers.append(('console', console_handler))
        
        # 2. Main application log file
        main_log_file = self.logs_dir / f"{self.app_name}_{self.date_str}.log"
        main_handler = logging.FileHandler(main_log_file, encoding='utf-8')
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(detailed_formatter)
        handlers.append(('main', main_handler))
        
        # 3. Error-only log file
        error_log_file = self.logs_dir / f"errors_{self.date_str}.log"
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        handlers.append(('error', error_handler))
        
        # 4. User actions log file
        user_log_file = self.logs_dir / f"user_actions_{self.date_str}.log"
        user_handler = logging.FileHandler(user_log_file, encoding='utf-8')
        user_handler.setLevel(logging.INFO)
        user_handler.setFormatter(detailed_formatter)
        
        # Filter for user actions only
        user_handler.addFilter(lambda record: 'USER_ACTION' in record.name or 'UI_EVENT' in record.name)
        handlers.append(('user', user_handler))
        
        # 5. Performance log file
        perf_log_file = self.logs_dir / f"performance_{self.date_str}.log"
        perf_handler = logging.FileHandler(perf_log_file, encoding='utf-8')
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(detailed_formatter)
        
        # Filter for performance logs only
        perf_handler.addFilter(lambda record: 'PERFORMANCE' in record.name)
        handlers.append(('performance', perf_handler))
        
        # 6. Rotating file handler for long-running sessions
        rotating_log_file = self.logs_dir / f"{self.app_name}_rotating.log"
        rotating_handler = logging.handlers.RotatingFileHandler(
            rotating_log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        rotating_handler.setLevel(logging.DEBUG)
        rotating_handler.setFormatter(simple_formatter)
        handlers.append(('rotating', rotating_handler))
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add our handlers
        for name, handler in handlers:
            root_logger.addHandler(handler)
        
        # Store handler references
        self.handlers = dict(handlers)
        
        # Log the initialization
        init_logger = logging.getLogger('LOGGING_INIT')
        init_logger.info(f"Logging system initialized", extra={
            'data': {
                'session_id': self.stats['session_id'],
                'logs_dir': str(self.logs_dir),
                'handlers_count': len(handlers),
                'log_files': {
                    'main': str(main_log_file),
                    'error': str(error_log_file),
                    'user': str(user_log_file),
                    'performance': str(perf_log_file),
                    'rotating': str(rotating_log_file)
                }
            }
        })
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name"""
        logger = logging.getLogger(name)
        
        # Add custom logging methods
        def log_with_data(level, message, **kwargs):
            logger.log(level, message, extra={'data': kwargs})
        
        logger.log_data = log_with_data
        logger.debug_data = lambda msg, **kwargs: log_with_data(logging.DEBUG, msg, **kwargs)
        logger.info_data = lambda msg, **kwargs: log_with_data(logging.INFO, msg, **kwargs)
        logger.warning_data = lambda msg, **kwargs: log_with_data(logging.WARNING, msg, **kwargs)
        logger.error_data = lambda msg, **kwargs: log_with_data(logging.ERROR, msg, **kwargs)
        logger.critical_data = lambda msg, **kwargs: log_with_data(logging.CRITICAL, msg, **kwargs)
        
        return logger
    
    def update_stats(self, level: str, category: str = None):
        """Update logging statistics"""
        self.stats['total_logs'] += 1
        
        if level not in self.stats['logs_by_level']:
            self.stats['logs_by_level'][level] = 0
        self.stats['logs_by_level'][level] += 1
        
        if category:
            if category not in self.stats['logs_by_category']:
                self.stats['logs_by_category'][category] = 0
            self.stats['logs_by_category'][category] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current logging statistics"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'logs_per_minute': self.stats['total_logs'] / (uptime / 60) if uptime > 0 else 0,
            'log_files': {
                name: {
                    'path': handler.baseFilename if hasattr(handler, 'baseFilename') else 'N/A',
                    'level': handler.level,
                    'size': Path(handler.baseFilename).stat().st_size if hasattr(handler, 'baseFilename') and Path(handler.baseFilename).exists() else 0
                }
                for name, handler in self.handlers.items()
                if hasattr(handler, 'baseFilename')
            }
        }
    
    def export_session_logs(self, export_file: Optional[str] = None) -> str:
        """Export current session logs to a JSON file"""
        if not export_file:
            export_file = self.logs_dir / f"session_export_{self.stats['session_id']}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'session_info': self.get_stats(),
            'log_files': {}
        }
        
        # Read recent entries from each log file
        for name, handler in self.handlers.items():
            if hasattr(handler, 'baseFilename'):
                log_file = Path(handler.baseFilename)
                if log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            # Get last 100 lines
                            export_data['log_files'][name] = {
                                'file_path': str(log_file),
                                'total_lines': len(lines),
                                'recent_entries': lines[-100:] if len(lines) > 100 else lines
                            }
                    except Exception as e:
                        export_data['log_files'][name] = {
                            'file_path': str(log_file),
                            'error': str(e)
                        }
        
        # Write export file
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return str(export_file)
    
    def cleanup_old_logs(self, days_to_keep: int = 7):
        """Clean up log files older than specified days"""
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        cleaned_files = []
        for log_file in self.logs_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_date:
                try:
                    log_file.unlink()
                    cleaned_files.append(str(log_file))
                except Exception as e:
                    logging.getLogger('LOG_CLEANUP').error(f"Failed to delete {log_file}: {e}")
        
        if cleaned_files:
            logging.getLogger('LOG_CLEANUP').info(f"Cleaned up {len(cleaned_files)} old log files", extra={
                'data': {'cleaned_files': cleaned_files}
            })
        
        return cleaned_files

# Global logger instance
_app_logger = None

def get_app_logger() -> KnowledgeAppLogger:
    """Get the global app logger instance"""
    global _app_logger
    if _app_logger is None:
        _app_logger = KnowledgeAppLogger()
    return _app_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger with structured data support"""
    return get_app_logger().get_logger(name)

# Convenience functions for common logging patterns
def log_user_action(action: str, **kwargs):
    """Log a user action with structured data"""
    logger = get_logger('USER_ACTION')
    # Avoid parameter conflict by using action_type instead of action
    logger.info_data(f"User action: {action}", action_type=action, **kwargs)

def log_system_event(event: str, **kwargs):
    """Log a system event with structured data"""
    logger = get_logger('SYSTEM')
    logger.info_data(f"System event: {event}", event_type=event, **kwargs)

def log_performance(operation: str, duration_ms: float, success: bool = True, **kwargs):
    """Log a performance metric"""
    logger = get_logger('PERFORMANCE')
    status = "SUCCESS" if success else "FAILED"
    logger.info_data(f"Performance: {operation} - {duration_ms:.2f}ms ({status})", 
                    operation=operation, duration_ms=duration_ms, success=success, **kwargs)

def log_error(message: str, **kwargs):
    """Log an error with structured data"""
    logger = get_logger('ERROR')
    logger.error_data(f"Error: {message}", error_message=message, **kwargs)

def log_bridge_call(method: str, duration_ms: float = None, success: bool = True, **kwargs):
    """Log a bridge method call"""
    logger = get_logger('BRIDGE')
    status = "SUCCESS" if success else "FAILED"
    duration_str = f" - {duration_ms:.2f}ms" if duration_ms is not None else ""
    logger.info_data(f"Bridge call: {method}{duration_str} ({status})", 
                    method_name=method, duration_ms=duration_ms, success=success, **kwargs)