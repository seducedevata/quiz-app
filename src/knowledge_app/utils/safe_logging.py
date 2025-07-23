"""
Safe Logging Configuration

This module provides Unicode-safe logging configuration that prevents
logging system failures in environments with limited Unicode support.

Features:
- Automatic emoji replacement for compatibility
- Fallback ASCII formatting
- Thread-safe logging setup
- Environment detection for Unicode support
"""

import logging
import sys
import os
import threading
from typing import Dict, Optional
import unicodedata

# Thread-safe logging setup
_logging_lock = threading.RLock()
_logging_configured = False

# CRITICAL FIX: Emoji to ASCII mapping for environments with limited Unicode support
EMOJI_REPLACEMENTS = {
    # Status indicators
    '✅': '[OK]',
    '❌': '[ERROR]', 
    '⚠️': '[WARNING]',
    '🔥': '[FIRE]',
    '🚀': '[ROCKET]',
    '💥': '[BOOM]',
    
    # Process indicators
    '🔄': '[LOADING]',
    '⏳': '[WAITING]',
    '🎯': '[TARGET]',
    '🧠': '[BRAIN]',
    '🤖': '[AI]',
    '📝': '[NOTE]',
    
    # System indicators
    '🔧': '[CONFIG]',
    '🛡️': '[SECURITY]',
    '🧹': '[CLEANUP]',
    '💾': '[SAVE]',
    '📊': '[STATS]',
    '🔍': '[SEARCH]',
    
    # UI indicators
    '🎨': '[THEME]',
    '🖥️': '[COMPUTER]',
    '📚': '[BOOKS]',
    '🔑': '[KEY]',
    '⚡': '[FAST]',
    '✨': '[SPARKLE]',
    
    # Training indicators
    '🎓': '[EDUCATION]',
    '📈': '[PROGRESS]',
    '🏆': '[SUCCESS]',
    '🎮': '[GAME]',
    '🌊': '[STREAM]',
    
    # Emergency indicators
    '🆘': '[SOS]',
    '🚨': '[ALERT]',
    '⛔': '[STOP]',
    '🔒': '[LOCK]',
    
    # Additional common emojis
    '📄': '[DOC]',
    '📁': '[FOLDER]',
    '🗂️': '[FILES]',
    '💡': '[IDEA]',
    '🔗': '[LINK]',
    '📦': '[PACKAGE]',
    '🌐': '[WEB]',
    '🎪': '[CIRCUS]',
    '🎭': '[MASK]',
    '🎨': '[ART]',
}


class SafeFormatter(logging.Formatter):
    """
    CRITICAL FIX: Unicode-safe logging formatter that handles emoji characters
    and prevents logging system failures in environments with limited Unicode support.
    """
    
    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)
        self.unicode_support = self._detect_unicode_support()
        
    def _detect_unicode_support(self) -> bool:
        """Detect if the current environment supports Unicode output"""
        try:
            # Check if we can encode/decode Unicode properly
            test_emoji = "🚀✅❌"
            encoded = test_emoji.encode('utf-8')
            decoded = encoded.decode('utf-8')
            
            # Check if stdout supports Unicode
            if hasattr(sys.stdout, 'encoding'):
                encoding = sys.stdout.encoding or 'ascii'
                test_emoji.encode(encoding)
                return True
            
            return False
        except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
            return False
    
    def _sanitize_message(self, message: str) -> str:
        """
        CRITICAL FIX: Sanitize log messages by replacing Unicode characters
        that might cause issues in certain environments.
        """
        if not isinstance(message, str):
            message = str(message)
        
        if self.unicode_support:
            # Environment supports Unicode, but still normalize for safety
            try:
                # Normalize Unicode to prevent display issues
                message = unicodedata.normalize('NFC', message)
                return message
            except Exception:
                # Fall through to ASCII replacement
                pass
        
        # Replace emojis with ASCII equivalents
        sanitized = message
        for emoji, replacement in EMOJI_REPLACEMENTS.items():
            sanitized = sanitized.replace(emoji, replacement)
        
        # Remove any remaining non-ASCII characters that might cause issues
        try:
            # Try to encode as ASCII to find problematic characters
            sanitized.encode('ascii')
            return sanitized
        except UnicodeEncodeError:
            # Replace non-ASCII characters with safe alternatives
            safe_chars = []
            for char in sanitized:
                try:
                    char.encode('ascii')
                    safe_chars.append(char)
                except UnicodeEncodeError:
                    # Replace with ASCII equivalent or remove
                    if ord(char) < 128:
                        safe_chars.append(char)
                    else:
                        safe_chars.append('?')  # Placeholder for non-ASCII
            
            return ''.join(safe_chars)
    
    def format(self, record):
        """
        CRITICAL FIX: Format log record with Unicode safety
        """
        try:
            # Sanitize the message
            if hasattr(record, 'msg') and record.msg:
                record.msg = self._sanitize_message(str(record.msg))
            
            # Sanitize any arguments
            if hasattr(record, 'args') and record.args:
                safe_args = []
                for arg in record.args:
                    if isinstance(arg, str):
                        safe_args.append(self._sanitize_message(arg))
                    else:
                        safe_args.append(arg)
                record.args = tuple(safe_args)
            
            # Format using parent formatter
            formatted = super().format(record)
            
            # Final sanitization of the complete formatted message
            return self._sanitize_message(formatted)
            
        except Exception as e:
            # Emergency fallback - return a safe error message
            return f"[LOGGING_ERROR] Failed to format log message: {str(e)}"


def setup_safe_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    force_ascii: bool = False
) -> bool:
    """
    CRITICAL FIX: Set up thread-safe, Unicode-safe logging configuration
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
        force_ascii: Force ASCII-only output (default: auto-detect)
        
    Returns:
        bool: True if setup successful, False otherwise
    """
    global _logging_configured
    
    with _logging_lock:
        if _logging_configured:
            return True
        
        try:
            # Default format string
            if format_string is None:
                format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            
            # Create safe formatter
            formatter = SafeFormatter(format_string)
            
            # Force ASCII mode if requested
            if force_ascii:
                formatter.unicode_support = False
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(level)
            
            # Remove existing handlers to avoid duplicates
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            # Create console handler with safe formatter
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            
            # Add handler to root logger
            root_logger.addHandler(console_handler)
            
            # Test logging with Unicode characters
            test_logger = logging.getLogger('safe_logging_test')
            test_logger.info("Safe logging initialized - Unicode test: ✅🚀❌")
            
            _logging_configured = True
            return True
            
        except Exception as e:
            # Emergency fallback - use basic logging
            try:
                logging.basicConfig(
                    level=level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    force=True
                )
                print(f"[WARNING] Safe logging setup failed, using basic logging: {e}")
                _logging_configured = True
                return True
            except Exception as fallback_error:
                print(f"[CRITICAL] All logging setup failed: {fallback_error}")
                return False


def get_safe_logger(name: str) -> logging.Logger:
    """
    Get a logger with safe Unicode handling
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Ensure safe logging is set up
    if not _logging_configured:
        setup_safe_logging()
    
    return logging.getLogger(name)


def sanitize_log_message(message: str) -> str:
    """
    CRITICAL FIX: Sanitize a log message for safe output
    
    Args:
        message: Original message that may contain Unicode characters
        
    Returns:
        str: Sanitized message safe for all environments
    """
    formatter = SafeFormatter()
    return formatter._sanitize_message(message)


def test_unicode_support() -> Dict[str, bool]:
    """
    Test Unicode support in the current environment
    
    Returns:
        Dict with test results
    """
    results = {
        'stdout_encoding': False,
        'unicode_encode': False,
        'unicode_decode': False,
        'emoji_support': False,
        'overall_support': False
    }
    
    try:
        # Test stdout encoding
        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
            results['stdout_encoding'] = True
        
        # Test Unicode encoding/decoding
        test_string = "Test Unicode: 🚀✅❌⚠️"
        encoded = test_string.encode('utf-8')
        decoded = encoded.decode('utf-8')
        results['unicode_encode'] = True
        results['unicode_decode'] = True
        
        # Test emoji support specifically
        if hasattr(sys.stdout, 'encoding'):
            encoding = sys.stdout.encoding or 'ascii'
            "🚀".encode(encoding)
            results['emoji_support'] = True
        
        # Overall support
        results['overall_support'] = all([
            results['stdout_encoding'],
            results['unicode_encode'], 
            results['unicode_decode']
        ])
        
    except Exception as e:
        # Log the specific error for debugging
        results['error'] = str(e)
    
    return results


# Initialize safe logging when module is imported
if not _logging_configured:
    setup_safe_logging()
