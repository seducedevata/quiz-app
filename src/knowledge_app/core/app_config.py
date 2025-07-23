"""
Application configuration module
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging
import warnings

logger = logging.getLogger(__name__)


class AppConfig:
    """Application configuration class"""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        🔧 FIX: Initialize with UnifiedConfigManager backend

        This prevents configuration conflicts by using a single source of truth.
        """
        # Issue deprecation warning
        warnings.warn(
            "AppConfig is deprecated. Use UnifiedConfigManager directly to avoid configuration conflicts.",
            DeprecationWarning,
            stacklevel=2
        )

        # Import here to avoid circular imports
        try:
            from .unified_config_manager import UnifiedConfigManager
            self._unified_manager = UnifiedConfigManager()
            logger.info("🔧 AppConfig redirecting to UnifiedConfigManager to prevent conflicts")
        except ImportError:
            logger.error("❌ UnifiedConfigManager not available - falling back to legacy mode")
            self._unified_manager = None
            self._init_legacy_mode()

        # Apply provided config if any
        if config_dict and self._unified_manager:
            self._apply_config_dict(config_dict)

    def _init_legacy_mode(self):
        """Initialize legacy mode if UnifiedConfigManager is not available"""
        self.base_dir = Path(__file__).parent.parent.parent.parent
        self.config_file = self.base_dir / "config" / "app_config.json"

        # Default configuration
        self._config = {
            "paths": {
                "base": str(self.base_dir),
                "data": str(self.base_dir / "data"),
                "image_cache": str(self.base_dir / "data" / "image_cache"),
                "models": str(self.base_dir / "data" / "models"),
                "user_data": str(self.base_dir / "data" / "user_data"),
                "uploaded_books": str(self.base_dir / "data" / "uploaded_books"),
                "logs": str(self.base_dir / "logs"),
                "cache": str(self.base_dir / "data" / "cache"),
                "processed_docs": str(self.base_dir / "data" / "processed_docs"),
                "lora_adapters": str(self.base_dir / "data" / "lora_adapters"),
                "sounds": str(self.base_dir / "assets" / "sounds"),
                "music": str(self.base_dir / "assets" / "music"),
            },
            "storage_config": {
                "image_cache_limit": 500 * 1024 * 1024,  # 500MB
                "model_cache_limit": 1024 * 1024 * 1024,  # 1GB
                "book_storage_limit": 100 * 1024 * 1024,  # 100MB
                "max_cache_size": 4 * 1024 * 1024 * 1024,  # 4GB
                "cleanup_threshold": 0.85,
            },
            "api_keys": {"cloud_inference": "", "tavily": ""},
            "app_settings": {
                "max_image_size": 1920,
                "image_quality": 85,
                "theme": "dark",
                "font_size": 12,
                "language": "en",
            },
            "storage_settings": {
                "cleanup_threshold_mb": 0.9,
                "cache_ttl": 3600,
                "max_book_size_mb": 100,  # Default 100MB limit for book files
                "max_image_size_mb": 5,  # Default 5MB limit for image files
            },
            "display_settings": {"enable_gpu": True, "enable_animations": True, "dark_mode": True},
        }

        # Update with provided config
        if config_dict:
            self.update_config(config_dict)

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with new values"""

        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d

        self._config = deep_update(self._config, config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)

    def get_value(self, key: str, default: Any = None) -> Any:
        """Alias for get_setting for backward compatibility"""
        return self.get_setting(key, default)

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation"""
        try:
            value = self._config
            for part in key.split("."):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation (alias for get_setting)"""
        return self.get_setting(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """
        🔧 FIX: Set configuration setting with dot notation support - redirected to UnifiedConfigManager
        """
        if self._unified_manager:
            self._unified_manager.set(key, value, save_immediately=True)
        else:
            # Legacy fallback
            keys = key.split(".")
            config = self._config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary syntax"""
        if self._unified_manager:
            return self._unified_manager.get(key)
        else:
            return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using dictionary syntax"""
        if self._unified_manager:
            self._unified_manager.set(key, value, save_immediately=True)
        else:
            self._config[key] = value

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to file"""
        if path is None:
            path = self.config_file

        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        """Load configuration from file"""
        if path is None:
            path = Path(__file__).parent.parent.parent.parent / "config" / "app_config.json"

        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                return cls(config_dict)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")

        return cls()