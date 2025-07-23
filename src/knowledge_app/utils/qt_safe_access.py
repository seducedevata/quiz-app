"""
Qt Safe Access Utilities

This module provides utilities for safely accessing Qt objects that may have been
deleted by Qt's garbage collector, preventing RuntimeError crashes.
"""

import logging
from typing import Any, Optional, Callable, TypeVar
from PyQt5.QtCore import QObject
# Removed QtWidgets dependency for pure QtWebEngine app

logger = logging.getLogger(__name__)

T = TypeVar("T")


class QtObjectState:
    """Represents the state of a Qt object"""

    VALID = "valid"
    DELETED = "deleted"
    UNKNOWN = "unknown"


def is_qt_object_valid(obj: Any) -> bool:
    """
    Safely check if a Qt object is still valid (not deleted by Qt).

    Args:
        obj: The Qt object to check

    Returns:
        bool: True if the object is valid, False if deleted or invalid
    """
    if obj is None:
        return False

    try:
        # For QObject-derived classes, try to access a basic property
        if isinstance(obj, QObject):
            # Try to access the object name - this will fail if object is deleted
            _ = obj.objectName()
            return True

        # For QGraphicsEffect objects, try to access a basic property
        if isinstance(obj, QGraphicsEffect):
            # Try to access enabled state - this will fail if object is deleted
            _ = obj.isEnabled()
            return True

        # For other Qt objects, try to access a common property
        if hasattr(obj, "isVisible"):
            _ = obj.isVisible()
            return True

        # If we can't determine validity, assume it's valid
        return True

    except RuntimeError as e:
        # This is the typical error when Qt objects are deleted
        if "wrapped C/C++ object" in str(e) or "has been deleted" in str(e):
            return False
        # Re-raise other RuntimeErrors
        raise
    except Exception:
        # For any other exception, assume the object is invalid
        return False


def safe_qt_call(obj: Any, method_name: str, *args, **kwargs) -> tuple[bool, Any]:
    """
    Safely call a method on a Qt object.

    Args:
        obj: The Qt object
        method_name: Name of the method to call
        *args: Arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method

    Returns:
        tuple: (success: bool, result: Any)
    """
    try:
        if not is_qt_object_valid(obj):
            return False, None

        method = getattr(obj, method_name, None)
        if method is None:
            logger.warning(f"Method {method_name} not found on object {type(obj)}")
            return False, None

        result = method(*args, **kwargs)
        return True, result

    except RuntimeError as e:
        if "wrapped C/C++ object" in str(e) or "has been deleted" in str(e):
            logger.debug(f"Qt object deleted during method call: {method_name}")
            return False, None
        raise
    except Exception as e:
        logger.warning(f"Error calling {method_name}: {e}")
        return False, None


def safe_qt_property_access(obj: Any, property_name: str, default_value: Any = None) -> Any:
    """
    Safely access a property on a Qt object.

    Args:
        obj: The Qt object
        property_name: Name of the property to access
        default_value: Value to return if access fails

    Returns:
        The property value or default_value if access fails
    """
    try:
        if not is_qt_object_valid(obj):
            return default_value

        return getattr(obj, property_name, default_value)

    except RuntimeError as e:
        if "wrapped C/C++ object" in str(e) or "has been deleted" in str(e):
            logger.debug(f"Qt object deleted during property access: {property_name}")
            return default_value
        raise
    except Exception as e:
        logger.warning(f"Error accessing property {property_name}: {e}")
        return default_value


def safe_qt_operation(
    operation: Callable[[], T],
    default: Optional[T] = None,
    error_message: str = "Qt operation failed",
) -> Optional[T]:
    """
    Safely execute a Qt operation with automatic error handling.

    Args:
        operation: The operation to execute
        default: Value to return if operation fails
        error_message: Error message to log on failure

    Returns:
        The operation result or default value
    """
    try:
        return operation()
    except RuntimeError as e:
        if "wrapped C/C++ object" in str(e) or "has been deleted" in str(e):
            logger.debug(f"{error_message}: Qt object deleted")
            return default
        logger.error(f"{error_message}: {e}")
        return default
    except Exception as e:
        logger.warning(f"{error_message}: {e}")
        return default


class SafeQtObjectManager:
    """
    Manager for safely handling Qt objects that may be deleted.
    """

    def __init__(self):
        self._objects = {}
        self._creation_callbacks = {}

    def register_object(
        self, name: str, obj: Any, creation_callback: Optional[Callable[[], Any]] = None
    ):
        """
        Register a Qt object for safe management.

        Args:
            name: Unique name for the object
            obj: The Qt object to manage
            creation_callback: Function to recreate the object if needed
        """
        self._objects[name] = obj
        if creation_callback:
            self._creation_callbacks[name] = creation_callback

    def get_object(self, name: str, auto_recreate: bool = True) -> Optional[Any]:
        """
        Get a registered Qt object, optionally recreating it if deleted.

        Args:
            name: Name of the object
            auto_recreate: Whether to automatically recreate deleted objects

        Returns:
            The Qt object or None if not available
        """
        if name not in self._objects:
            return None

        obj = self._objects[name]

        if not is_qt_object_valid(obj):
            if auto_recreate and name in self._creation_callbacks:
                try:
                    new_obj = self._creation_callbacks[name]()
                    self._objects[name] = new_obj
                    logger.debug(f"Recreated Qt object: {name}")
                    return new_obj
                except Exception as e:
                    logger.error(f"Failed to recreate Qt object {name}: {e}")
                    return None
            else:
                logger.debug(f"Qt object {name} is no longer valid")
                return None

        return obj

    def is_object_valid(self, name: str) -> bool:
        """Check if a registered object is still valid."""
        if name not in self._objects:
            return False
        return is_qt_object_valid(self._objects[name])

    def cleanup(self):
        """Clean up all registered objects."""
        self._objects.clear()
        self._creation_callbacks.clear()