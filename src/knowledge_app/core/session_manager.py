"""
Session management for Knowledge App
"""

import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions and progress tracking"""

    _instance = None

    def __new__(cls, config: Optional[Dict[str, Any]] = None):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._instance._initialized = False
        if config is not None:
            cls._instance._config = config
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if self._initialized:
            return

        if config is None:
            raise ValueError("Config is required for initialization")

        self._config = config
        self._initialized = True

        self.current_session = {
            "start_time": None,
            "end_time": None,
            "questions_answered": 0,
            "correct_answers": 0,
            "score": 0,
            "category": None,
            "difficulty": None,
        }

        self.sessions_dir = Path(
            config.get("sessions_dir", Path.home() / ".knowledge_app" / "sessions")
        )
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def start_session(
        self, category: Optional[str] = None, difficulty: Optional[str] = None
    ) -> None:
        """Start a new quiz session"""
        self.current_session = {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "questions_answered": 0,
            "correct_answers": 0,
            "score": 0,
            "category": category,
            "difficulty": difficulty,
        }
        logger.info(f"Started new session: {self.current_session}")

    def end_session(self) -> None:
        """End current session and save results"""
        self.current_session["end_time"] = datetime.now().isoformat()
        self.save_session()
        self.update_high_scores()
        logger.info(f"Ended session: {self.current_session}")

    def save_session(self) -> None:
        """Save session data to file"""
        session_file = self.sessions_dir / f"session_{int(time.time())}.json"
        try:
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(self.current_session, f, indent=4)
            logger.debug(f"Saved session to {session_file}")
        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def update_high_scores(self) -> None:
        """Update high scores"""
        try:
            high_scores = self._config.get("high_scores", [])
            high_scores.append(
                {
                    "score": self.current_session["score"],
                    "date": self.current_session["end_time"],
                    "correct_answers": self.current_session["correct_answers"],
                    "total_questions": self.current_session["questions_answered"],
                    "category": self.current_session["category"],
                    "difficulty": self.current_session["difficulty"],
                }
            )
            high_scores.sort(key=lambda x: x["score"], reverse=True)
            high_scores = high_scores[:10]  # Keep only top 10
            self._config["high_scores"] = high_scores
            logger.debug("Updated high scores")
        except Exception as e:
            logger.error(f"Error updating high scores: {e}")

    def get_high_scores(self) -> List[Dict[str, Any]]:
        """Get current high scores"""
        return self._config.get("high_scores", [])

    def update_progress(self, correct: bool, points: int = 1) -> None:
        """Update session progress"""
        self.current_session["questions_answered"] += 1
        if correct:
            self.current_session["correct_answers"] += 1
            self.current_session["score"] += points

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        stats = self.current_session.copy()
        if stats["questions_answered"] > 0:
            stats["accuracy"] = (stats["correct_answers"] / stats["questions_answered"]) * 100
        else:
            stats["accuracy"] = 0
        return stats

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all saved sessions"""
        sessions = []
        try:
            for file in self.sessions_dir.glob("*.json"):
                with open(file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                    sessions.append(session_data)
            return sorted(sessions, key=lambda x: x["start_time"], reverse=True)
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            return []

    def clear_session_history(self, days: Optional[int] = None) -> None:
        """
        Clear session history

        Args:
            days: Optional number of days to keep (None means clear all)
        """
        try:
            if days is None:
                # Clear all sessions
                for file in self.sessions_dir.glob("*.json"):
                    file.unlink()
            else:
                # Clear sessions older than specified days
                cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
                for file in self.sessions_dir.glob("*.json"):
                    if file.stat().st_mtime < cutoff:
                        file.unlink()
            logger.info(f"Cleared session history (days={days})")
        except Exception as e:
            logger.error(f"Error clearing session history: {e}")

    def force_save_current_session(self) -> bool:
        """🔧 FIX: Force save current session - RELIABLE alternative to __del__"""
        try:
            if self.current_session["start_time"] and not self.current_session["end_time"]:
                logger.warning("🚨 Force saving active session to prevent data loss")
                self.end_session()
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to force save session: {e}")
            return False

    def shutdown(self) -> None:
        """🔧 FIX: Explicit shutdown method for reliable session saving"""
        try:
            logger.info("🔧 SessionManager shutdown initiated")

            # Save any active session
            if self.force_save_current_session():
                logger.info("✅ Active session saved during shutdown")

            # Save any pending high scores
            try:
                if hasattr(self, '_config') and self._config:
                    # Force write config if it has high scores
                    high_scores = self._config.get("high_scores", [])
                    if high_scores:
                        logger.info(f"✅ Preserved {len(high_scores)} high scores during shutdown")
            except Exception as e:
                logger.error(f"❌ Failed to preserve high scores: {e}")

            logger.info("✅ SessionManager shutdown completed")

        except Exception as e:
            logger.error(f"❌ SessionManager shutdown failed: {e}")

    def __del__(self):
        """🔧 DEPRECATED: __del__ is unreliable - use shutdown() instead"""
        # 🔧 FIX: Log warning instead of relying on __del__ for critical operations
        logger.warning("⚠️ SessionManager.__del__ called - this is unreliable for data persistence")
        logger.warning("⚠️ Use shutdown() method explicitly for guaranteed session saving")

        # Still attempt save as last resort, but don't rely on it
        try:
            if hasattr(self, 'current_session') and self.current_session.get("start_time") and not self.current_session.get("end_time"):
                logger.warning("🚨 Attempting emergency session save in __del__ (unreliable)")
                self.end_session()
        except Exception as e:
            logger.error(f"❌ Emergency session save in __del__ failed: {e}")