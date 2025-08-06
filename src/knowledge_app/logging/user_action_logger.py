"""
Enhanced User Action Logger

This module provides comprehensive user interaction logging with advanced
features including session tracking, action patterns, and security monitoring.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading

# Import enhanced logging utilities
from ..utils.logging_config import get_logger, log_user_action, log_security_event
from ..utils.safe_logging import get_contextual_logger, get_performance_logger, timed_operation


@dataclass
class UserAction:
    """Structured user action data"""
    timestamp: datetime
    session_id: str
    user_id: str
    action_type: str
    action_name: str
    details: Dict[str, Any]
    duration: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class SessionTracker:
    """Track user sessions and detect patterns"""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.max_sessions = 1000  # Prevent memory overflow

    def start_session(self, session_id: str, user_id: str, metadata: Dict[str, Any] = None):
        """Start a new user session"""
        with self.lock:
            # Clean old sessions if needed
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_old_sessions()

            self.sessions[session_id] = {
                'user_id': user_id,
                'start_time': datetime.now(),
                'last_activity': datetime.now(),
                'action_count': 0,
                'actions': deque(maxlen=100),  # Keep last 100 actions
                'metadata': metadata or {},
                'flags': set()
            }

    def update_session(self, session_id: str, action: UserAction):
        """Update session with new action"""
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session['last_activity'] = action.timestamp
                session['action_count'] += 1
                session['actions'].append(action)

                # Detect suspicious patterns
                self._detect_suspicious_patterns(session_id, action)

    def end_session(self, session_id: str):
        """End a user session"""
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                duration = (datetime.now() - session['start_time']).total_seconds()

                # Log session summary
                logger = get_contextual_logger(__name__, "user")
                logger.info(f"Session ended: {session_id} - Duration: {duration:.1f}s, Actions: {session['action_count']}")

                del self.sessions[session_id]

    def _cleanup_old_sessions(self):
        """Remove sessions older than 24 hours"""
        cutoff = datetime.now() - timedelta(hours=24)
        old_sessions = [
            sid for sid, session in self.sessions.items()
            if session['last_activity'] < cutoff
        ]
        for sid in old_sessions:
            del self.sessions[sid]

    def _detect_suspicious_patterns(self, session_id: str, action: UserAction):
        """Detect suspicious user behavior patterns"""
        session = self.sessions[session_id]

        # Check for rapid-fire actions (potential bot)
        recent_actions = [a for a in session['actions'] if (action.timestamp - a.timestamp).total_seconds() < 60]
        if len(recent_actions) > 50:  # More than 50 actions per minute
            session['flags'].add('rapid_actions')
            log_security_event("RAPID_ACTIONS", {
                'session_id': session_id,
                'user_id': action.user_id,
                'action_count_per_minute': len(recent_actions)
            })

        # Check for repeated failed actions
        failed_actions = [a for a in session['actions'] if not a.success]
        if len(failed_actions) > 10:  # More than 10 failures
            session['flags'].add('multiple_failures')
            log_security_event("MULTIPLE_FAILURES", {
                'session_id': session_id,
                'user_id': action.user_id,
                'failure_count': len(failed_actions)
            })


class EnhancedUserActionLogger:
    """Enhanced user action logger with comprehensive tracking"""

    def __init__(self, log_dir: Path = None):
        self.logger = get_contextual_logger(__name__, "user")
        self.perf_logger = get_performance_logger(__name__, "user")
        self.session_tracker = SessionTracker()
        self.action_buffer: List[UserAction] = []
        self.buffer_lock = threading.RLock()
        self.max_buffer_size = 100

        # Action type categories for better organization
        self.action_categories = {
            'navigation': ['page_view', 'menu_click', 'tab_switch', 'scroll'],
            'quiz': ['quiz_start', 'quiz_complete', 'question_answer', 'hint_request'],
            'content': ['content_upload', 'content_delete', 'content_edit', 'content_view'],
            'settings': ['settings_change', 'preference_update', 'theme_change'],
            'auth': ['login', 'logout', 'register', 'password_change'],
            'system': ['error_occurred', 'performance_issue', 'crash', 'recovery']
        }

        # Initialize logging context
        self.logger.push_context("UserActionLogger")

    def start_session(self, session_id: str, user_id: str = None, metadata: Dict[str, Any] = None):
        """Start tracking a user session"""
        with timed_operation(__name__, "start_session", "user"):
            user_id = user_id or "anonymous"
            self.session_tracker.start_session(session_id, user_id, metadata)

            # Log session start
            log_user_action("session_start", user_id, {
                'session_id': session_id,
                'metadata': metadata
            })

            self.logger.info(f"✅ Started session tracking: {session_id} for user {user_id}")

    def log_action(self, action_type: str, action_name: str, session_id: str,
                   user_id: str = None, details: Dict[str, Any] = None,
                   duration: float = None, success: bool = True,
                   error: str = None, **kwargs):
        """Log a user action with comprehensive details"""

        # Create structured action
        action = UserAction(
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id or "anonymous",
            action_type=action_type,
            action_name=action_name,
            details=details or {},
            duration=duration,
            success=success,
            error=error,
            ip_address=kwargs.get('ip_address'),
            user_agent=kwargs.get('user_agent')
        )

        # Add to buffer
        with self.buffer_lock:
            self.action_buffer.append(action)
            if len(self.action_buffer) >= self.max_buffer_size:
                self._flush_buffer()

        # Update session tracking
        self.session_tracker.update_session(session_id, action)

        # Log based on action category
        category = self._get_action_category(action_type)
        self.logger.push_context(f"Action:{category}")

        # Format log message
        status_emoji = "✅" if success else "❌"
        duration_str = f" ({duration:.3f}s)" if duration else ""
        error_str = f" - Error: {error}" if error else ""

        self.logger.info(
            f"{status_emoji} {action_type}:{action_name}{duration_str} "
            f"- User: {action.user_id}, Session: {session_id[:8]}...{error_str}"
        )

        # Log detailed information for debugging
        self.logger.debug(f"Action details: {details}")

        self.logger.pop_context()

        # Log to centralized user action log
        log_user_action(f"{action_type}:{action_name}", user_id, {
            'session_id': session_id,
            'details': details,
            'duration': duration,
            'success': success,
            'error': error
        })

        # Performance warning for slow actions
        if duration and duration > 3.0:
            self.logger.warning(f"🐌 Slow user action: {action_name} took {duration:.3f}s")

    def end_session(self, session_id: str):
        """End session tracking"""
        with timed_operation(__name__, "end_session", "user"):
            self.session_tracker.end_session(session_id)
            self.logger.info(f"🏁 Ended session tracking: {session_id}")

    def _get_action_category(self, action_type: str) -> str:
        """Get category for action type"""
        for category, types in self.action_categories.items():
            if action_type in types:
                return category
        return "other"

    def _flush_buffer(self):
        """Flush action buffer to persistent storage"""
        if not self.action_buffer:
            return

        try:
            # Convert actions to JSON for potential persistence
            actions_data = [asdict(action) for action in self.action_buffer]

            # Log buffer flush
            self.logger.debug(f"💾 Flushed {len(self.action_buffer)} actions to buffer")

            # Clear buffer
            self.action_buffer.clear()

        except Exception as e:
            self.logger.error(f"❌ Failed to flush action buffer: {e}")

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of session activity"""
        session = self.session_tracker.sessions.get(session_id)
        if not session:
            return {}

        return {
            'session_id': session_id,
            'user_id': session['user_id'],
            'start_time': session['start_time'].isoformat(),
            'last_activity': session['last_activity'].isoformat(),
            'action_count': session['action_count'],
            'flags': list(session['flags']),
            'duration_minutes': (session['last_activity'] - session['start_time']).total_seconds() / 60
        }

    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about user actions"""
        stats = {
            'total_sessions': len(self.session_tracker.sessions),
            'action_categories': defaultdict(int),
            'recent_actions': len(self.action_buffer)
        }

        # Count actions by category
        for session in self.session_tracker.sessions.values():
            for action in session['actions']:
                category = self._get_action_category(action.action_type)
                stats['action_categories'][category] += 1

        return dict(stats)

    def log_quiz_interaction(self, session_id: str, quiz_id: str, action: str,
                           user_id: str = None, **kwargs):
        """Log quiz-specific interactions"""
        self.log_action(
            action_type="quiz",
            action_name=action,
            session_id=session_id,
            user_id=user_id,
            details={'quiz_id': quiz_id, **kwargs}
        )
    
    def log_content_interaction(self, session_id: str, content_id: str, action: str,
                              user_id: str = None, **kwargs):
        """Log content-specific interactions"""
        self.log_action(
            action_type="content",
            action_name=action,
            session_id=session_id,
            user_id=user_id,
            details={'content_id': content_id, **kwargs}
        )

    def log_navigation(self, session_id: str, page: str, user_id: str = None, **kwargs):
        """Log navigation events"""
        self.log_action(
            action_type="navigation",
            action_name="page_view",
            session_id=session_id,
            user_id=user_id,
            details={'page': page, **kwargs}
        )

    def log_error(self, session_id: str, error_type: str, error_message: str,
                  user_id: str = None, **kwargs):
        """Log user-encountered errors"""
        self.log_action(
            action_type="system",
            action_name="error_occurred",
            session_id=session_id,
            user_id=user_id,
            success=False,
            error=error_message,
            details={'error_type': error_type, **kwargs}
        )

    def shutdown(self):
        """Safely shutdown the logger"""
        self.logger.info("🔄 Shutting down user action logger...")

        # Flush remaining actions
        with self.buffer_lock:
            if self.action_buffer:
                self._flush_buffer()

        # End all active sessions
        for session_id in list(self.session_tracker.sessions.keys()):
            self.end_session(session_id)

        self.logger.pop_context()
        self.logger.info("✅ User action logger shutdown complete")


# Global instance
_user_logger: Optional[EnhancedUserActionLogger] = None
_logger_lock = threading.RLock()


def get_user_logger() -> EnhancedUserActionLogger:
    """Get the global user action logger instance"""
    global _user_logger

    with _logger_lock:
        if _user_logger is None:
            _user_logger = EnhancedUserActionLogger()
        return _user_logger


# Convenience functions for common logging patterns
def log_user_navigation(session_id: str, page: str, user_id: str = None, **kwargs):
    """Log user navigation"""
    get_user_logger().log_navigation(session_id, page, user_id, **kwargs)


def log_quiz_action(session_id: str, quiz_id: str, action: str, user_id: str = None, **kwargs):
    """Log quiz-related action"""
    get_user_logger().log_quiz_interaction(session_id, quiz_id, action, user_id, **kwargs)


def log_content_action(session_id: str, content_id: str, action: str, user_id: str = None, **kwargs):
    """Log content-related action"""
    get_user_logger().log_content_interaction(session_id, content_id, action, user_id, **kwargs)


def log_user_error(session_id: str, error_type: str, error_message: str,
                   user_id: str = None, **kwargs):
    """Log user-encountered error"""
    get_user_logger().log_error(session_id, error_type, error_message, user_id, **kwargs)


def start_user_session(session_id: str, user_id: str = None, **metadata):
    """Start tracking a user session"""
    get_user_logger().start_session(session_id, user_id, metadata)


def end_user_session(session_id: str):
    """End tracking a user session"""
    get_user_logger().end_session(session_id)
