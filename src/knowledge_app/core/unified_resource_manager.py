"""
🛡️ CRITICAL ARCHITECTURE FIX #19: Remove Zombie Resource Manager

This file is kept ONLY as a deprecation stub to prevent import errors.
All functionality has been moved to ConsolidatedResourceManager.

The UnifiedResourceManager was supposed to be deprecated but was left as a
"zombie" that forwards calls, which is dangerous because developers could
accidentally use the old interface and bypass the intended unified system.
"""

import logging
import warnings

logger = logging.getLogger(__name__)


class UnifiedResourceManager:
    """
    🚫 DEPRECATED ZOMBIE RESOURCE MANAGER - DO NOT USE
    
    This class is deprecated and will raise errors if used.
    Use ConsolidatedResourceManager instead.
    """
    
    def __init__(self):
        """Raise error to prevent usage of deprecated resource manager"""
        raise DeprecationWarning(
            "🚫 UnifiedResourceManager is DEPRECATED and removed (Bug #19 fix).\n"
            "This zombie class was kept alive too long and caused resource conflicts.\n"
            "Use ConsolidatedResourceManager instead:\n"
            "\n"
            "  from .memory_consolidation import ConsolidatedResourceManager\n"
            "  manager = ConsolidatedResourceManager()\n"
            "\n"
            "See: https://github.com/seducedevata/quiz-app/issues/19"
        )
    
    def __new__(cls):
        """Prevent instantiation completely"""
        raise DeprecationWarning(
            "🚫 UnifiedResourceManager has been completely removed (Bug #19 fix).\n"
            "Use ConsolidatedResourceManager instead."
        )
    
    @staticmethod
    def register_resource(*args, **kwargs):
        """All static methods also deprecated"""
        raise DeprecationWarning(
            "🚫 UnifiedResourceManager.register_resource() is deprecated.\n"
            "Use ConsolidatedResourceManager.register_resource() instead."
        )
    
    @staticmethod
    def cleanup_all(*args, **kwargs):
        """All static methods also deprecated"""
        raise DeprecationWarning(
            "🚫 UnifiedResourceManager.cleanup_all() is deprecated.\n"
            "Use ConsolidatedResourceManager.cleanup_all() instead."
        )


# Also prevent module-level access
def __getattr__(name):
    """Prevent any access to deprecated functionality"""
    if name in ['UnifiedResourceManager', 'get_unified_resource_manager']:
        raise DeprecationWarning(
            f"🚫 {name} is deprecated and removed (Bug #19 fix).\n"
            "Use ConsolidatedResourceManager from memory_consolidation module instead."
        )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
