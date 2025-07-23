"""
🔧 FIX: Deprecation Helper for Old DI Containers

This provides deprecation warnings to guide developers away from the old
fragmented DI containers towards the unified solution.
"""

import warnings
import logging
from typing import Type, Any

logger = logging.getLogger(__name__)

def warn_deprecated_container(container_name: str, replacement: str = "UnifiedDIContainer"):
    """Issue deprecation warning for old DI containers"""
    message = (
        f"⚠️ {container_name} is DEPRECATED and will be removed. "
        f"Use {replacement} from unified_di_container instead. "
        f"See architectural bug fixes for migration guide."
    )
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    logger.warning(f"🔧 DEPRECATION: {message}")

def get_unified_container_replacement():
    """Get the unified container as replacement for deprecated containers"""
    from .unified_di_container import get_unified_container, configure_unified_container
    container = get_unified_container()
    configure_unified_container()  # Ensure it's configured
    return container
