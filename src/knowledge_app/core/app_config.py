"""
🔧 DEPRECATED: Application configuration module

This module is deprecated and has been replaced by UnifiedConfigManager.
It now serves as a compatibility bridge to prevent breaking existing code.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging
import warnings

logger = logging.getLogger(__name__)


class AppConfig:
    """
    🔧 DEPRECATED: Application configuration class
    
    This class now redirects to UnifiedConfigManager to prevent configuration conflicts.
    All new code should use UnifiedConfigManager directly.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        🔧 COMPATIBILITY BRIDGE: Initialize with UnifiedConfigManager backend

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
            from .unified_config_manager import get_unified_config_manager
            self._unified_manager = get_unified_config_manager()
            logger.warning("🔧 AppConfig redirected to UnifiedConfigManager")
            
            # If config_dict provided, migrate it to unified manager
            if config_dict:
                for key, value in config_dict.items():
                    if self._unified_manager.get(key) is None:  # Don't overwrite existing
                        self._unified_manager.set(key, value)
                        
        except Exception as e:
            logger.error(f"❌ Failed to initialize UnifiedConfigManager bridge: {e}")
            self._unified_manager = None
            self._fallback_config = config_dict or {}
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        🔧 COMPATIBILITY: Get configuration value through unified manager
        """
        try:
            if self._unified_manager:
                return self._unified_manager.get(key, default)
            else:
                return self._fallback_config.get(key, default)
        except Exception as e:
            logger.error(f"❌ AppConfig.get failed for key '{key}': {e}")
            return default
            
    def set(self, key: str, value: Any) -> None:
        """
        🔧 COMPATIBILITY: Set configuration value through unified manager
        """
        try:
            if self._unified_manager:
                self._unified_manager.set(key, value)
            else:
                self._fallback_config[key] = value
                logger.warning(f"⚠️ AppConfig.set using fallback (not persistent): {key}")
        except Exception as e:
            logger.error(f"❌ AppConfig.set failed for key '{key}': {e}")
            
    def get_config(self) -> Dict[str, Any]:
        """
        🔧 COMPATIBILITY: Get all configuration as dict
        """
        try:
            if self._unified_manager:
                return self._unified_manager.get_all()
            else:
                return self._fallback_config.copy()
        except Exception as e:
            logger.error(f"❌ AppConfig.get_config failed: {e}")
            return {}
            
    def save(self) -> None:
        """
        🔧 COMPATIBILITY: Save configuration (handled automatically by unified manager)
        """
        try:
            if self._unified_manager:
                # UnifiedConfigManager handles persistence automatically
                logger.debug("Configuration automatically saved by UnifiedConfigManager")
            else:
                logger.warning("⚠️ AppConfig.save in fallback mode - changes not persistent")
        except Exception as e:
            logger.error(f"❌ AppConfig.save failed: {e}")


# 🔧 LEGACY COMPATIBILITY: Keep this for existing imports
def get_app_config(config_dict: Optional[Dict[str, Any]] = None) -> AppConfig:
    """
    🔧 COMPATIBILITY FUNCTION: Get AppConfig instance
    
    Issues deprecation warning and redirects to UnifiedConfigManager.
    """
    warnings.warn(
        "get_app_config is deprecated. Use get_unified_config_manager() directly.",
        DeprecationWarning,
        stacklevel=2
    )
    return AppConfig(config_dict)

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
            # Legacy fallback with type safety
            if not isinstance(key, str):
                logger.warning(f"⚠️ Config key expected string, got {type(key)}: {key}")
                key = str(key)
            
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
