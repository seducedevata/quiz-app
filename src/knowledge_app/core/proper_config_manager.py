"""
🔧 FIX: Proper Config Manager Bridge

This file provides a compatibility bridge for code that references the missing
proper_config_manager.py, redirecting to the UnifiedConfigManager.
"""

import warnings
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Issue deprecation warning
warnings.warn(
    "proper_config_manager is deprecated and was missing. Use unified_config_manager instead.",
    DeprecationWarning,
    stacklevel=2
)

def ProperConfigManager():
    """
    🔧 COMPATIBILITY BRIDGE: Redirects to UnifiedConfigManager
    
    This ensures existing code continues to work while migrating to unified config.
    """
    try:
        from .unified_config_manager import get_unified_config_manager
        logger.warning("🔧 ProperConfigManager redirected to UnifiedConfigManager")
        return get_unified_config_manager()
    except ImportError:
        logger.error("❌ UnifiedConfigManager not available - returning empty config")
        return {}

def get_config():
    """Get configuration - redirected to unified config manager"""
    return ProperConfigManager()

def get_config_manager():
    """Get config manager - redirected to unified config manager"""
    try:
        from .unified_config_manager import get_unified_config_manager
        return get_unified_config_manager()
    except ImportError:
        logger.error("❌ UnifiedConfigManager not available")
        return None

# For backwards compatibility
class ConfigManager:
    """Legacy config manager compatibility bridge"""
    def __init__(self):
        warnings.warn(
            "ConfigManager is deprecated. Use UnifiedConfigManager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        try:
            from .unified_config_manager import get_unified_config_manager
            self._manager = get_unified_config_manager()
        except ImportError:
            self._manager = None
            
    def get(self, key: str, default: Any = None) -> Any:
        if self._manager:
            return self._manager.get(key, default)
        return default
        
    def set(self, key: str, value: Any) -> None:
        if self._manager:
            self._manager.set(key, value)
            
    def save(self) -> None:
        if self._manager:
            self._manager.save()

# Export for compatibility
__all__ = ['ProperConfigManager', 'get_config', 'get_config_manager', 'ConfigManager']
