"""
🛡️ SECURITY FIX #17: Comprehensive Input Sanitization System

This module provides enterprise-grade input validation and sanitization
to prevent injection attacks, XSS, path traversal, and other security vulnerabilities.
"""

import re
import html
import json
import logging
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class InputType(Enum):
    """Input type categories for context-specific validation"""
    TOPIC = "topic"
    FILENAME = "filename"
    API_KEY = "api_key"
    JSON_DATA = "json_data"
    SQL_QUERY = "sql_query"
    FILE_PATH = "file_path"
    URL = "url"
    EMAIL = "email"
    GENERAL_TEXT = "general_text"
    HTML_CONTENT = "html_content"
    QUIZ_ANSWER = "quiz_answer"
    TRAINING_CONFIG = "training_config"
    PDF_CONTENT = "pdf_content"  # Academic PDF content with relaxed sanitization

class SecurityLevel(Enum):
    """Security validation levels"""
    STRICT = "strict"      # Maximum security, minimal functionality
    BALANCED = "balanced"  # Good security with usability
    PERMISSIVE = "permissive"  # Basic security, maximum functionality

class ComprehensiveInputSanitizer:
    """
    🛡️ SECURITY FIX #17: Enterprise-grade input sanitization
    
    Features:
    - Context-aware validation
    - Multiple security levels
    - XSS prevention
    - SQL injection prevention
    - Path traversal prevention
    - Command injection prevention
    - Prompt injection prevention
    - File upload validation
    - JSON sanitization
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.BALANCED):
        self.security_level = security_level
        self._init_patterns()
        
        logger.info(f"🛡️ ComprehensiveInputSanitizer initialized with {security_level.value} security")
    
    def _init_patterns(self):
        """Initialize security patterns based on security level"""
        
        # XSS Prevention Patterns
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*=',
            r'onmouseover\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'<link[^>]*>',
            r'<meta[^>]*>',
            r'<style[^>]*>.*?</style>',
        ]
        
        # SQL Injection Patterns
        self.sql_injection_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)',
            r'(\b(UNION|OR|AND)\s+\d+\s*=\s*\d+)',
            r'(\b(OR|AND)\s+[\'"]?\w+[\'"]?\s*=\s*[\'"]?\w+[\'"]?)',
            r'(--|#|/\*|\*/)',
            r'(\bxp_cmdshell\b)',
            r'(\bsp_executesql\b)',
            r'(\bEXEC\s*\()',
        ]
        
        # Command Injection Patterns
        self.command_injection_patterns = [
            r'[;&|`$]',  # Simplified - removed () {} [] \ which are common in JSON
            r'\b(rm|del|format|fdisk|kill|shutdown|reboot)\b',
            r'\b(cat|type|more|less|head|tail)\b',
            r'\b(wget|curl|nc|netcat)\b',
            r'\b(python|perl|ruby|php|bash|sh|cmd|powershell)\b',
        ]

        # Academic Content Allowlists for PDF processing
        self.academic_math_patterns = [
            r'\|[^|]*\|',  # Absolute value notation |x|
            r'\$[^$]*\$',  # LaTeX inline math $equation$
            r'\\[a-zA-Z]+\{[^}]*\}',  # LaTeX commands \command{content}
            r'\b(sin|cos|tan|log|ln|exp|sqrt|sum|int|lim)\b',  # Math functions
            r'[a-zA-Z]\s*[=<>≤≥≠]\s*[a-zA-Z0-9]',  # Mathematical equations
        ]

        self.academic_code_patterns = [
            r'def\s+\w+\(',  # Python function definitions
            r'class\s+\w+\(',  # Python class definitions
            r'import\s+\w+',  # Import statements
            r'from\s+\w+\s+import',  # From-import statements
            r'#\s*[A-Za-z]',  # Comments starting with #
            r'//\s*[A-Za-z]',  # C-style comments
        ]

        self.academic_technical_terms = [
            r'\b(algorithm|function|method|class|object|variable|parameter)\b',
            r'\b(dataset|model|training|validation|testing|accuracy|precision)\b',
            r'\b(neural|network|deep|learning|machine|artificial|intelligence)\b',
            r'\b(equation|formula|theorem|proof|lemma|corollary|proposition)\b',
        ]

        # Path Traversal Patterns
        self.path_traversal_patterns = [
            r'\.\./',
            r'\.\.\\',
            r'%2e%2e%2f',
            r'%2e%2e%5c',
            r'\.\.%2f',
            r'\.\.%5c',
        ]
        
        # Prompt Injection Patterns
        self.prompt_injection_patterns = [
            r'ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|tasks?|objectives?)',
            r'forget\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|tasks?|objectives?)',
            r'new\s+(?:task|job|objective|instruction)',
            r'your\s+(?:new\s+)?(?:task|job|objective|instruction)\s+is',
            r'do\s+not\s+(?:output|generate|create)\s+json',
            r'output\s+(?:a\s+)?(?:poem|story|essay|text)',
            r'write\s+(?:a\s+)?(?:poem|story|essay)',
            r'pretend\s+(?:to\s+be|you\s+are)',
            r'act\s+as\s+(?:a\s+)?(?:different|another|hacker|attacker)',
            r'role\s*play',
            r'forget\s+(?:your\s+)?(?:previous\s+)?(?:task|job|objective)',
            r'(?:previous\s+)?(?:task|instruction|objective)\s+(?:and|but)',
        ]
        
        # Dangerous file extensions
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', '.jar',
            '.app', '.deb', '.pkg', '.dmg', '.iso', '.msi', '.run', '.sh', '.ps1'
        }
        
        # Safe file extensions for uploads
        self.safe_extensions = {
            '.txt', '.pdf', '.docx', '.doc', '.rtf', '.odt', '.md', '.csv',
            '.json', '.xml', '.yaml', '.yml', '.png', '.jpg', '.jpeg', '.gif'
        }
    
    def sanitize_input(self, 
                      user_input: Any, 
                      input_type: InputType = InputType.GENERAL_TEXT,
                      max_length: Optional[int] = None) -> str:
        """
        🛡️ Main sanitization method with context-aware validation
        
        Args:
            user_input: Raw user input to sanitize
            input_type: Type of input for context-specific handling
            max_length: Maximum allowed length (None for default)
            
        Returns:
            Sanitized and validated input string
        """
        try:
            # Convert to string and handle None/empty inputs
            if user_input is None:
                return ""
            
            if not isinstance(user_input, str):
                user_input = str(user_input)
            
            # Remove null bytes and control characters
            sanitized = self._remove_control_characters(user_input)
            
            # Apply context-specific sanitization
            sanitized = self._apply_context_sanitization(sanitized, input_type)
            
            # Apply security level filtering
            sanitized = self._apply_security_filtering(sanitized, input_type)
            
            # Apply length limits
            sanitized = self._apply_length_limits(sanitized, input_type, max_length)
            
            # Final validation
            self._validate_final_output(sanitized, input_type)
            
            return sanitized.strip()
            
        except Exception as e:
            logger.error(f"❌ Input sanitization failed: {e}")
            return ""  # Fail safe - return empty string
    
    def _remove_control_characters(self, text: str) -> str:
        """Remove null bytes and dangerous control characters"""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove other dangerous control characters but preserve common ones
        allowed_control = {'\n', '\r', '\t'}
        cleaned = ''.join(
            char for char in text 
            if ord(char) >= 32 or char in allowed_control
        )
        
        # Normalize line endings
        cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
        
        return cleaned
    
    def _apply_context_sanitization(self, text: str, input_type: InputType) -> str:
        """Apply context-specific sanitization rules"""
        
        if input_type == InputType.FILENAME:
            return self._sanitize_filename(text)
        elif input_type == InputType.FILE_PATH:
            return self._sanitize_file_path(text)
        elif input_type == InputType.API_KEY:
            return self._sanitize_api_key(text)
        elif input_type == InputType.URL:
            return self._sanitize_url(text)
        elif input_type == InputType.EMAIL:
            return self._sanitize_email(text)
        elif input_type == InputType.HTML_CONTENT:
            return self._sanitize_html(text)
        elif input_type == InputType.JSON_DATA:
            return self._sanitize_json_string(text)
        elif input_type == InputType.SQL_QUERY:
            return self._sanitize_sql(text)
        elif input_type == InputType.TOPIC:
            return self._sanitize_topic(text)
        elif input_type == InputType.QUIZ_ANSWER:
            return self._sanitize_quiz_answer(text)
        elif input_type == InputType.TRAINING_CONFIG:
            return self._sanitize_training_config(text)
        else:
            return self._sanitize_general_text(text)
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal and dangerous files"""
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
        
        # Check extension
        ext = Path(filename).suffix.lower()
        if ext in self.dangerous_extensions:
            logger.warning(f"⚠️ Dangerous file extension blocked: {ext}")
            return ""
        
        # Limit length
        if len(filename) > 255:
            name_part = filename[:200]
            ext_part = Path(filename).suffix
            filename = name_part + ext_part
        
        return filename
    
    def _sanitize_file_path(self, path: str) -> str:
        """Sanitize file path to prevent traversal attacks"""
        # Check for path traversal patterns
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                logger.warning(f"⚠️ Path traversal attempt blocked: {pattern}")
                return ""
        
        # Normalize path and ensure it's relative
        try:
            normalized = Path(path).as_posix()
            if normalized.startswith('/') or ':' in normalized:
                logger.warning(f"⚠️ Absolute path blocked: {path}")
                return ""
            return normalized
        except Exception:
            return ""
    
    def _sanitize_api_key(self, api_key: str) -> str:
        """Sanitize API key input"""
        # Remove whitespace and dangerous characters
        api_key = re.sub(r'[<>"\'\\\x00-\x1f\x7f-\x9f]', '', api_key.strip())
        
        # Basic format validation (alphanumeric, hyphens, underscores only)
        if not re.match(r'^[a-zA-Z0-9_-]+$', api_key):
            logger.warning("⚠️ Invalid API key format")
            return ""
        
        return api_key
    
    def _sanitize_url(self, url: str) -> str:
        """Sanitize URL input"""
        try:
            # Parse and validate URL
            parsed = urllib.parse.urlparse(url)
            
            # Only allow safe schemes
            safe_schemes = {'http', 'https', 'ftp', 'ftps'}
            if parsed.scheme.lower() not in safe_schemes:
                logger.warning(f"⚠️ Unsafe URL scheme: {parsed.scheme}")
                return ""
            
            # Reconstruct clean URL
            return urllib.parse.urlunparse(parsed)
        except Exception:
            return ""
    
    def _sanitize_email(self, email: str) -> str:
        """Sanitize email input"""
        # Basic email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return ""
        return email.lower().strip()
    
    def _sanitize_html(self, html_content: str) -> str:
        """Sanitize HTML content to prevent XSS"""
        # Remove dangerous HTML patterns
        for pattern in self.xss_patterns:
            html_content = re.sub(pattern, '', html_content, flags=re.IGNORECASE | re.DOTALL)
        
        # HTML escape remaining content
        return html.escape(html_content)
    
    def _sanitize_json_string(self, json_str: str) -> str:
        """Sanitize JSON string input"""
        try:
            # Parse and re-serialize to ensure valid JSON
            parsed = json.loads(json_str)
            # Sanitize the parsed data recursively
            sanitized_data = self._sanitize_json_data(parsed)
            return json.dumps(sanitized_data, ensure_ascii=True)
        except json.JSONDecodeError:
            logger.warning("⚠️ Invalid JSON format")
            return "{}"
    
    def _sanitize_sql(self, sql: str) -> str:
        """Sanitize SQL input (should generally be avoided)"""
        # Check for SQL injection patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                logger.warning(f"⚠️ SQL injection attempt blocked: {pattern}")
                return ""
        
        return sql
    
    def _sanitize_topic(self, topic: str) -> str:
        """Sanitize quiz topic input"""
        # Remove prompt injection patterns
        for pattern in self.prompt_injection_patterns:
            topic = re.sub(pattern, '[FILTERED]', topic, flags=re.IGNORECASE)
        
        # Remove excessive special characters
        topic = re.sub(r'[{}]+', '', topic)
        topic = re.sub(r'#{3,}', '##', topic)
        topic = re.sub(r'`{3,}', '``', topic)
        
        return topic
    
    def _sanitize_quiz_answer(self, answer: str) -> str:
        """Sanitize quiz answer input"""
        # Basic text sanitization
        answer = html.escape(answer)
        answer = re.sub(r'[<>"\'\\\x00-\x1f]', '', answer)
        return answer
    
    def _sanitize_training_config(self, config: str) -> str:
        """Sanitize training configuration input"""
        try:
            # Parse as JSON and validate structure
            config_data = json.loads(config)
            
            # Sanitize string values recursively
            sanitized_data = self._sanitize_json_data(config_data)
            
            return json.dumps(sanitized_data, ensure_ascii=True)
        except json.JSONDecodeError:
            logger.warning("⚠️ Invalid training config JSON")
            return "{}"
    
    def _sanitize_general_text(self, text: str) -> str:
        """General text sanitization"""
        # HTML escape
        text = html.escape(text)
        
        # Remove dangerous patterns based on security level
        if self.security_level == SecurityLevel.STRICT:
            # Remove all special characters except basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        elif self.security_level == SecurityLevel.BALANCED:
            # Remove dangerous characters but allow common symbols
            text = re.sub(r'[<>"\'\\\x00-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    def _sanitize_json_data(self, data: Any) -> Any:
        """Recursively sanitize JSON data structure"""
        if isinstance(data, dict):
            return {
                self._sanitize_json_string_value(key):
                self._sanitize_json_data(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_json_data(item) for item in data]
        elif isinstance(data, str):
            return self._sanitize_json_string_value(data)
        else:
            return data

    def _sanitize_json_string_value(self, text: str) -> str:
        """Sanitize string values within JSON without breaking JSON structure"""
        # Remove XSS patterns
        for pattern in self.xss_patterns:
            text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove SQL injection patterns - simplified for better matching
        sql_keywords = [
            r'\bDROP\s+TABLE\b',
            r'\bSELECT\b.*\bFROM\b',
            r'\bINSERT\s+INTO\b',
            r'\bUPDATE\b.*\bSET\b',
            r'\bDELETE\s+FROM\b',
            r'\bUNION\s+SELECT\b',
            r'\bEXEC\b',
            r'\bEXECUTE\b',
            r'--',
            r'/\*.*\*/',
        ]
        for pattern in sql_keywords:
            text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE)

        # Remove dangerous command patterns
        dangerous_commands = [
            r'\b(rm|del|format|fdisk|kill|shutdown|reboot)\b',
            r'\b(wget|curl|nc|netcat)\b',
        ]
        for pattern in dangerous_commands:
            text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE)

        return text

    def _is_academic_content(self, text: str) -> bool:
        """Check if text contains academic/research content that should be allowed"""
        # Check for mathematical expressions
        for pattern in self.academic_math_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        # Check for code examples
        for pattern in self.academic_code_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        # Check for technical terms
        for pattern in self.academic_technical_terms:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _apply_security_filtering(self, text: str, input_type: InputType) -> str:
        """Apply security-level specific filtering"""

        # Skip command injection filtering for JSON, training config, and PDF content
        if input_type not in [InputType.JSON_DATA, InputType.TRAINING_CONFIG, InputType.PDF_CONTENT]:
            # Check for command injection
            for pattern in self.command_injection_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.warning(f"⚠️ Command injection attempt blocked: {pattern}")
                    if self.security_level == SecurityLevel.STRICT:
                        return ""
                    else:
                        text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE)
        elif input_type == InputType.PDF_CONTENT:
            # For PDF content, apply relaxed filtering with academic content awareness
            for pattern in self.command_injection_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Check if this might be academic content before blocking
                    if not self._is_academic_content(text):
                        logger.warning(f"⚠️ Command injection attempt blocked in PDF: {pattern}")
                        if self.security_level == SecurityLevel.STRICT:
                            return ""
                        else:
                            text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE)
                    else:
                        logger.debug(f"📚 Academic content detected, allowing pattern: {pattern}")

        # Check for XSS patterns (always apply)
        for pattern in self.xss_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                logger.warning(f"⚠️ XSS attempt blocked: {pattern}")
                text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE | re.DOTALL)

        return text
    
    def _apply_length_limits(self, text: str, input_type: InputType, max_length: Optional[int]) -> str:
        """Apply appropriate length limits"""
        
        # Default length limits by input type
        default_limits = {
            InputType.TOPIC: 200,
            InputType.FILENAME: 255,
            InputType.API_KEY: 200,
            InputType.URL: 2048,
            InputType.EMAIL: 254,
            InputType.QUIZ_ANSWER: 500,
            InputType.GENERAL_TEXT: 10000,
            InputType.HTML_CONTENT: 50000,
            InputType.JSON_DATA: 100000,
            InputType.TRAINING_CONFIG: 50000,
        }
        
        limit = max_length or default_limits.get(input_type, 1000)
        
        if len(text) > limit:
            text = text[:limit]
            logger.warning(f"⚠️ Input truncated to {limit} characters")
        
        return text
    
    def _validate_final_output(self, text: str, input_type: InputType) -> None:
        """Final validation of sanitized output"""

        # Only check for the most critical patterns that should never survive
        critical_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
        ]

        # For certain input types, skip validation (like JSON which may contain special chars)
        if input_type in [InputType.JSON_DATA, InputType.TRAINING_CONFIG]:
            return

        for pattern in critical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.error(f"❌ Critical pattern survived sanitization: {pattern}")
                raise ValueError(f"Input failed final security validation")
    
    def validate_file_upload(self, filename: str, file_size: int, max_size: int = 10 * 1024 * 1024) -> Tuple[bool, str]:
        """
        Validate file upload for security
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Sanitize filename
            clean_filename = self.sanitize_input(filename, InputType.FILENAME)
            if not clean_filename:
                return False, "Invalid filename"
            
            # Check file size
            if file_size > max_size:
                return False, f"File too large (max {max_size // 1024 // 1024}MB)"
            
            # Check extension
            ext = Path(clean_filename).suffix.lower()
            if ext not in self.safe_extensions:
                return False, f"File type not allowed: {ext}"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"❌ File validation error: {e}")
            return False, "File validation failed"

# Global instance
_sanitizer = None

def get_input_sanitizer(security_level: SecurityLevel = SecurityLevel.BALANCED) -> ComprehensiveInputSanitizer:
    """Get or create global input sanitizer instance"""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = ComprehensiveInputSanitizer(security_level)
    return _sanitizer

def sanitize_input(user_input: Any, 
                  input_type: InputType = InputType.GENERAL_TEXT,
                  max_length: Optional[int] = None) -> str:
    """Convenience function for input sanitization"""
    sanitizer = get_input_sanitizer()
    return sanitizer.sanitize_input(user_input, input_type, max_length)
