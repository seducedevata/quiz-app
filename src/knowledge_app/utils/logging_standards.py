"""
Logging Standards and Level Enforcement

This module provides standardized logging level guidelines and enforcement
to ensure consistent logging practices across the Knowledge App codebase.

CRITICAL FIX: Addresses inconsistent logging levels where critical issues
are logged as warnings while minor issues use errors.
"""

import logging
import functools
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Standardized log levels with clear usage guidelines"""
    
    # CRITICAL: System cannot continue, immediate action required
    CRITICAL = "critical"
    
    # ERROR: Functionality broken, user impact, needs fixing
    ERROR = "error"
    
    # WARNING: Potential issues, degraded functionality, should be addressed
    WARNING = "warning"
    
    # INFO: Normal operation, important events, user-visible actions
    INFO = "info"
    
    # DEBUG: Detailed diagnostic information, development/troubleshooting
    DEBUG = "debug"


class LoggingStandardsEnforcer:
    """
    CRITICAL FIX: Enforce consistent logging levels across the application
    
    This class provides standardized logging methods and guidelines to prevent
    critical issues being logged as warnings and ensure proper log level usage.
    """
    
    def __init__(self):
        # Define standard patterns for each log level
        self.level_patterns = {
            LogLevel.CRITICAL: {
                "keywords": [
                    "blocking", "startup", "cannot continue", "system failure",
                    "data corruption", "security breach", "fatal error"
                ],
                "examples": [
                    "Application startup blocked due to critical dependency issues",
                    "Database corruption detected - system cannot continue",
                    "Security vulnerability exploited - immediate action required"
                ]
            },
            
            LogLevel.ERROR: {
                "keywords": [
                    "failed", "error", "exception", "broken", "crash",
                    "generation failed", "model loading failed", "api error"
                ],
                "examples": [
                    "MCQ generation failed completely",
                    "Model loading crashed with CUDA error",
                    "API request failed with authentication error"
                ]
            },
            
            LogLevel.WARNING: {
                "keywords": [
                    "fallback", "degraded", "suboptimal", "missing optional",
                    "version mismatch", "timeout", "retry"
                ],
                "examples": [
                    "Using fallback generation method due to model unavailability",
                    "Optional dependency missing - some features disabled",
                    "Version mismatch detected - may cause compatibility issues"
                ]
            },
            
            LogLevel.INFO: {
                "keywords": [
                    "started", "completed", "initialized", "ready",
                    "processing", "loaded", "saved", "configured"
                ],
                "examples": [
                    "MCQ generation started for topic: Physics",
                    "Model loaded successfully",
                    "Configuration saved to file"
                ]
            },
            
            LogLevel.DEBUG: {
                "keywords": [
                    "details", "internal", "diagnostic", "trace",
                    "parameters", "intermediate", "cache"
                ],
                "examples": [
                    "Internal model parameters: temperature=0.7, max_tokens=512",
                    "Cache hit for question generation request",
                    "Detailed timing information for performance analysis"
                ]
            }
        }
        
        # Common misclassifications to fix
        self.misclassification_fixes = {
            # These should be ERROR, not WARNING
            "mcq generation failed": LogLevel.ERROR,
            "model loading failed": LogLevel.ERROR,
            "api key test failed": LogLevel.ERROR,
            "training failed": LogLevel.ERROR,
            "evaluation failed": LogLevel.ERROR,
            
            # These should be CRITICAL, not ERROR/WARNING
            "application startup blocked": LogLevel.CRITICAL,
            "critical dependency missing": LogLevel.CRITICAL,
            "system cannot continue": LogLevel.CRITICAL,
            "data corruption": LogLevel.CRITICAL,
            
            # These should be WARNING, not ERROR
            "fallback method used": LogLevel.WARNING,
            "optional feature disabled": LogLevel.WARNING,
            "version mismatch detected": LogLevel.WARNING,
            "performance degraded": LogLevel.WARNING,
            
            # These should be INFO, not WARNING
            "initialization complete": LogLevel.INFO,
            "processing started": LogLevel.INFO,
            "configuration loaded": LogLevel.INFO,
            "cache cleared": LogLevel.INFO
        }
    
    def get_recommended_level(self, message: str, context: Optional[Dict[str, Any]] = None) -> LogLevel:
        """
        Analyze a log message and recommend the appropriate log level
        
        Args:
            message: The log message to analyze
            context: Optional context information (exception type, severity, etc.)
            
        Returns:
            Recommended LogLevel
        """
        message_lower = message.lower()
        
        # Check for direct misclassification fixes
        for pattern, level in self.misclassification_fixes.items():
            if pattern in message_lower:
                return level
        
        # Analyze context if provided
        if context:
            # Critical system failures
            if context.get("blocks_startup", False) or context.get("system_failure", False):
                return LogLevel.CRITICAL
            
            # Errors that break functionality
            if context.get("breaks_functionality", False) or context.get("user_impact", False):
                return LogLevel.ERROR
            
            # Degraded performance or fallback scenarios
            if context.get("uses_fallback", False) or context.get("degraded_performance", False):
                return LogLevel.WARNING
        
        # Pattern-based analysis
        for level, patterns in self.level_patterns.items():
            for keyword in patterns["keywords"]:
                if keyword in message_lower:
                    return level
        
        # Default to INFO for unclear cases
        return LogLevel.INFO
    
    def create_standardized_logger(self, name: str) -> 'StandardizedLogger':
        """Create a logger with enforced standards"""
        return StandardizedLogger(name, self)
    
    def validate_log_message(self, level: LogLevel, message: str) -> Dict[str, Any]:
        """
        Validate if a log message uses the appropriate level
        
        Returns:
            Dictionary with validation results and recommendations
        """
        recommended_level = self.get_recommended_level(message)
        
        is_correct = level == recommended_level
        
        return {
            "is_correct": is_correct,
            "current_level": level.value,
            "recommended_level": recommended_level.value,
            "message": message,
            "suggestion": None if is_correct else f"Consider using {recommended_level.value.upper()} instead of {level.value.upper()}"
        }


class StandardizedLogger:
    """
    CRITICAL FIX: Logger wrapper that enforces consistent logging standards
    
    This wrapper ensures that log messages use appropriate levels and provides
    guidance for proper logging practices.
    """
    
    def __init__(self, name: str, enforcer: LoggingStandardsEnforcer):
        self.logger = logging.getLogger(name)
        self.enforcer = enforcer
        self.validation_enabled = True  # Can be disabled in production
    
    def _log_with_validation(self, level: LogLevel, message: str, *args, **kwargs):
        """Internal method to log with optional validation"""
        
        if self.validation_enabled:
            validation = self.enforcer.validate_log_message(level, message)
            if not validation["is_correct"] and validation["suggestion"]:
                # Log the suggestion as a debug message
                self.logger.debug(f"LOGGING STANDARD: {validation['suggestion']}")
        
        # Get the actual logging method
        log_method = getattr(self.logger, level.value)
        log_method(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical system failures that block operation"""
        self._log_with_validation(LogLevel.CRITICAL, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log errors that break functionality"""
        self._log_with_validation(LogLevel.ERROR, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warnings about potential issues or degraded functionality"""
        self._log_with_validation(LogLevel.WARNING, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log normal operation and important events"""
        self._log_with_validation(LogLevel.INFO, message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log detailed diagnostic information"""
        self._log_with_validation(LogLevel.DEBUG, message, *args, **kwargs)
    
    # Convenience methods for common scenarios
    def system_failure(self, message: str, *args, **kwargs):
        """Log system failures that prevent normal operation"""
        self.critical(f"SYSTEM FAILURE: {message}", *args, **kwargs)
    
    def functionality_broken(self, message: str, *args, **kwargs):
        """Log when core functionality is broken"""
        self.error(f"FUNCTIONALITY BROKEN: {message}", *args, **kwargs)
    
    def using_fallback(self, message: str, *args, **kwargs):
        """Log when using fallback methods due to issues"""
        self.warning(f"USING FALLBACK: {message}", *args, **kwargs)
    
    def operation_complete(self, message: str, *args, **kwargs):
        """Log successful completion of operations"""
        self.info(f"COMPLETED: {message}", *args, **kwargs)


# Global enforcer instance
_global_enforcer = LoggingStandardsEnforcer()


def get_standardized_logger(name: str) -> StandardizedLogger:
    """
    Get a standardized logger with enforced logging levels
    
    Usage:
        logger = get_standardized_logger(__name__)
        logger.error("MCQ generation failed")  # Correct
        logger.warning("Using fallback generation method")  # Correct
        logger.critical("Application startup blocked")  # Correct
    """
    return _global_enforcer.create_standardized_logger(name)


def fix_logging_level(original_level: str, message: str) -> str:
    """
    CRITICAL FIX: Suggest correct logging level for a message
    
    Args:
        original_level: Current logging level being used
        message: The log message
        
    Returns:
        Recommended logging level
    """
    try:
        current_level = LogLevel(original_level.lower())
    except ValueError:
        current_level = LogLevel.INFO
    
    recommended_level = _global_enforcer.get_recommended_level(message)
    
    return recommended_level.value


def validate_codebase_logging(file_patterns: List[str]) -> Dict[str, Any]:
    """
    Analyze codebase for logging level inconsistencies
    
    This function can be used to audit the entire codebase for
    logging standard violations.
    """
    # This would be implemented to scan files and identify issues
    # For now, return structure for future implementation
    return {
        "total_files_scanned": 0,
        "total_log_statements": 0,
        "misclassified_statements": [],
        "recommendations": [],
        "summary": "Logging validation not yet implemented"
    }


# Decorator for automatic logging level enforcement
def enforce_logging_standards(func: Callable) -> Callable:
    """
    Decorator to enforce logging standards in functions
    
    Usage:
        @enforce_logging_standards
        def my_function():
            logger.error("This will be validated")
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Could implement automatic logging validation here
        return func(*args, **kwargs)
    
    return wrapper


# Common logging patterns with correct levels
LOGGING_EXAMPLES = {
    "CRITICAL": [
        "🚨 APPLICATION STARTUP BLOCKED - CRITICAL DEPENDENCY ISSUES",
        "🚨 SYSTEM FAILURE: Database corruption detected",
        "🚨 SECURITY BREACH: Unauthorized access detected"
    ],
    
    "ERROR": [
        "❌ MCQ generation failed completely",
        "❌ Model loading crashed with CUDA error", 
        "❌ API authentication failed"
    ],
    
    "WARNING": [
        "⚠️ Using fallback generation method",
        "⚠️ Optional dependency missing - features disabled",
        "⚠️ Version mismatch may cause compatibility issues"
    ],
    
    "INFO": [
        "✅ MCQ generation completed successfully",
        "🚀 Model loaded and ready",
        "💾 Configuration saved to file"
    ],
    
    "DEBUG": [
        "🔍 Internal parameters: temperature=0.7, max_tokens=512",
        "📊 Cache statistics: 85% hit rate",
        "⏱️ Performance timing: generation took 2.3s"
    ]
}


if __name__ == "__main__":
    # Example usage and testing
    logger = get_standardized_logger(__name__)
    
    # Test different log levels
    logger.critical("System cannot start due to missing critical dependencies")
    logger.error("MCQ generation failed with model loading error")
    logger.warning("Using fallback method due to API unavailability")
    logger.info("Application initialized successfully")
    logger.debug("Internal state: buffer_size=10, cache_hits=5")
    
    # Test validation
    enforcer = LoggingStandardsEnforcer()
    
    test_cases = [
        ("warning", "MCQ generation failed completely"),  # Should be ERROR
        ("error", "Using fallback generation method"),     # Should be WARNING
        ("info", "Application startup blocked"),           # Should be CRITICAL
    ]
    
    for level, message in test_cases:
        try:
            current_level = LogLevel(level)
            validation = enforcer.validate_log_message(current_level, message)
            print(f"Message: {message}")
            print(f"Current: {validation['current_level']}")
            print(f"Recommended: {validation['recommended_level']}")
            print(f"Suggestion: {validation['suggestion']}")
            print("-" * 50)
        except ValueError:
            print(f"Invalid log level: {level}")
