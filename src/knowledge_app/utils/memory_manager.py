"""
Memory Manager for Knowledge App

This module provides memory management utilities for the application,
including garbage collection and memory monitoring.
"""

import gc
import logging
import psutil
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """Memory management utilities"""

    def __init__(self):
        self.process = psutil.Process()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        try:
            # System memory
            system_memory = psutil.virtual_memory()

            # Process memory
            process_memory = self.process.memory_info()

            # GPU memory if available
            gpu_memory = {}
            if torch.cuda.is_available():
                try:
                    gpu_memory = {
                        "allocated": torch.cuda.memory_allocated(),
                        "cached": torch.cuda.memory_reserved(),
                        "total": torch.cuda.get_device_properties(0).total_memory,
                    }
                except Exception as e:
                    logger.debug(f"Error getting GPU memory info: {e}")

            return {
                "system": {
                    "total": system_memory.total,
                    "available": system_memory.available,
                    "percent": system_memory.percent,
                    "used": system_memory.used,
                },
                "process": {
                    "rss": process_memory.rss,
                    "vms": process_memory.vms,
                    "percent": self.process.memory_percent(),
                },
                "gpu": gpu_memory,
            }

        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {}

    def collect_garbage(self) -> int:
        """Force garbage collection and return number of objects collected"""
        try:
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force garbage collection
            collected = gc.collect()

            logger.debug(f"Garbage collection freed {collected} objects")
            return collected

        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
            return 0

    def get_memory_usage_mb(self) -> float:
        """Get current process memory usage in MB"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def get_gpu_memory_usage_mb(self) -> Dict[str, float]:
        """Get GPU memory usage in MB"""
        if not torch.cuda.is_available():
            return {}

        try:
            return {
                "allocated": torch.cuda.memory_allocated() / 1024 / 1024,
                "cached": torch.cuda.memory_reserved() / 1024 / 1024,
                "total": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
            }
        except Exception as e:
            logger.debug(f"Error getting GPU memory usage: {e}")
            return {}

    def is_memory_available(self, required_mb: float) -> bool:
        """Check if required memory is available"""
        try:
            available_mb = psutil.virtual_memory().available / 1024 / 1024
            return available_mb >= required_mb
        except Exception:
            return True  # Assume available if can't check

    def log_memory_status(self):
        """Log current memory status"""
        try:
            info = self.get_memory_info()

            system = info.get("system", {})
            process = info.get("process", {})
            gpu = info.get("gpu", {})

            logger.info(f"Memory Status:")
            logger.info(
                f"  System: {system.get('percent', 0):.1f}% used "
                f"({system.get('used', 0) / 1024**3:.1f}GB / "
                f"{system.get('total', 0) / 1024**3:.1f}GB)"
            )

            logger.info(
                f"  Process: {process.get('percent', 0):.1f}% "
                f"({process.get('rss', 0) / 1024**2:.1f}MB)"
            )

            if gpu:
                gpu_percent = (gpu.get("allocated", 0) / gpu.get("total", 1)) * 100
                logger.info(
                    f"  GPU: {gpu_percent:.1f}% used "
                    f"({gpu.get('allocated', 0) / 1024**3:.1f}GB / "
                    f"{gpu.get('total', 0) / 1024**3:.1f}GB)"
                )

        except Exception as e:
            logger.debug(f"Error logging memory status: {e}")


    def sanitize_string(self, text: str) -> str:
        """Remove null bytes and other problematic characters from string to prevent JavaScript errors"""
        if not text or not isinstance(text, str):
            return ""
        # Remove null bytes, control characters, and other problematic characters
        sanitized = text
        # Remove null bytes (both \x00 and \0 representations)
        sanitized = sanitized.replace('\x00', '').replace('\0', '')
        # Remove other control characters that can cause issues
        control_chars = ''.join(chr(i) for i in range(32) if i not in [9, 10, 13])  # Allow tab, newline, carriage return
        for char in control_chars:
            sanitized = sanitized.replace(char, '')
        # Handle Unicode surrogate pairs that can cause issues in JavaScript
        sanitized = ''.join(c for c in sanitized if not (0xD800 <= ord(c) <= 0xDFFF))
        return sanitized.strip()
    
    def sanitize_object(self, obj: Any) -> Any:
        """Recursively sanitize an object by removing problematic characters from all strings
        
        Handles various data structures including nested dictionaries, lists, tuples, and sets.
        Also handles edge cases like None values and custom objects.
        """
        if obj is None:
            return None
        elif isinstance(obj, str):
            return self.sanitize_string(obj)
        elif isinstance(obj, (list, tuple)):
            # Handle both lists and tuples while preserving the original type
            return type(obj)(self.sanitize_object(item) for item in obj)
        elif isinstance(obj, dict):
            # Sanitize both keys and values in dictionaries
            # For keys that aren't strings, convert to string first, then sanitize
            return {self.sanitize_string(str(key)) if isinstance(key, str) else key: 
                    self.sanitize_object(value) for key, value in obj.items()}
        elif isinstance(obj, set):
            # Handle sets
            return {self.sanitize_object(item) for item in obj}
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by sanitizing their attributes
            for attr, value in obj.__dict__.items():
                setattr(obj, attr, self.sanitize_object(value))
            return obj
        # Return other types (int, float, bool, etc.) unchanged
        return obj


# Global memory manager instance
_memory_manager = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager