"""
Audio management for Knowledge App - Pure QtWebEngine Compatible
STUB VERSION: Audio functionality disabled for pure QtWebEngine compatibility
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AudioManager:
    """Centralized audio management system - STUB for QtWebEngine compatibility"""

    _instance = None

    def __new__(cls, config: Optional[Dict[str, Any]] = None):
        if cls._instance is None:
            cls._instance = super(AudioManager, cls).__new__(cls)
            cls._instance._initialized = False
        if config is not None:
            cls._instance._config = config
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if self._initialized:
            return

        if config is None:
            config = {}  # Use empty config instead of requiring it

        self._config = config
        self._initialized = True

        # Sound effects - STUB
        self.sounds_dir = Path(config.get("sounds_dir", "assets/sounds"))
        self.sounds: Dict[str, str] = {}  # Just store filenames

        # Background music - STUB  
        self.music_dir = Path(config.get("music_dir", "assets/music"))
        self.volume = 50  # Just store value
        self.is_playing = False

        logger.info("🔇 AudioManager initialized in STUB mode (pure QtWebEngine compatibility)")

        # 🔧 BUG FIX 27: Provide clear user notification about audio status
        self._notify_audio_status()

    def load_sounds(self) -> None:
        """Load sound effects - STUB"""
        sound_files = {
            "correct": "correct.wav",
            "incorrect": "incorrect.wav", 
            "complete": "complete.wav",
            "tick": "tick.wav",
        }

        try:
            for name, file in sound_files.items():
                path = self.sounds_dir / file
                self.sounds[name] = str(path)  # Just store path
            logger.debug("Sound effects loaded (STUB mode)")
        except Exception as e:
            logger.debug(f"Sound loading failed (STUB mode): {e}")

    def play_sound(self, sound_name: str) -> None:
        """Play a sound effect - STUB"""
        if not self._config.get("sound_enabled", True):
            return
        logger.debug(f"Playing sound: {sound_name} (STUB mode)")

        # 🔧 BUG FIX 27: Notify user on first sound attempt
        if not hasattr(self, '_sound_notification_shown'):
            self._notify_audio_unavailable("sound effects")
            self._sound_notification_shown = True

    def load_music_playlist(self) -> None:
        """Load music files into playlist - STUB"""
        logger.debug("Music playlist loaded (STUB mode)")

    def play_music(self) -> None:
        """Start playing background music - STUB"""
        if not self._config.get("music_enabled", True):
            return
        self.is_playing = True
        logger.debug("Music started (STUB mode)")

    def pause_music(self) -> None:
        """Pause background music - STUB"""
        self.is_playing = False
        logger.debug("Music paused (STUB mode)")

    def stop_music(self) -> None:
        """Stop background music - STUB"""
        self.is_playing = False
        logger.debug("Music stopped (STUB mode)")

    def set_volume(self, volume: int) -> None:
        """Set audio volume - STUB"""
        self.volume = max(0, min(100, volume))  # Clamp between 0-100
        logger.debug(f"Volume set to {self.volume} (STUB mode)")

    def toggle_music(self) -> bool:
        """Toggle music playback - STUB"""
        self.is_playing = not self.is_playing
        logger.debug(f"Music toggled: {'playing' if self.is_playing else 'stopped'} (STUB mode)")
        return self.is_playing

    def cleanup(self) -> None:
        """Clean up audio resources - STUB"""
        self.is_playing = False
        self.sounds.clear()
        logger.debug("Audio resources cleaned up (STUB mode)")

    def _notify_audio_status(self):
        """🔧 BUG FIX 27: Notify user about audio status"""
        try:
            # Try to emit signal to UI if available
            from PyQt5.QtCore import QObject, pyqtSignal
            logger.info("ℹ️ Audio features are disabled for QtWebEngine compatibility")
        except Exception:
            pass  # UI not available

    def _notify_audio_unavailable(self, feature_type: str):
        """🔧 BUG FIX 27: Notify user when audio features are attempted but unavailable"""
        try:
            logger.info(f"ℹ️ {feature_type.title()} are not available in this version (QtWebEngine compatibility mode)")
            # Could emit signal to UI here if bridge is available
        except Exception:
            pass  # UI not available

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()