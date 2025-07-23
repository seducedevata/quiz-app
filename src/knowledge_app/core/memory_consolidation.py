"""
[HOT] MEMORY CONSOLIDATION - Unified Resource & Memory Management

This module consolidates ResourceManager and MemoryManager to eliminate
duplication and conflicts. It provides a single source of truth for all
resource and memory management operations.

CRITICAL FIX: Addresses the issue where both ResourceManager and MemoryManager
were handling memory monitoring independently, leading to conflicting actions.
"""

import logging
import threading
import weakref
import gc
import time
import psutil
from typing import Dict, Any, Optional, List, Set, Callable, Union
from enum import Enum
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

logger = logging.getLogger(__name__)

# GPU management imports (lazy loaded)
torch = None
cuda_available = False

try:
    import torch as torch_module
    torch = torch_module
    cuda_available = torch.cuda.is_available()
except ImportError:
    torch = None
    cuda_available = False

logger.info("[CONFIG] Memory consolidation module created - use ConsolidatedResourceManager instead of separate ResourceManager/MemoryManager")

class ResourceType(Enum):
    """Types of resources managed"""
    MODEL = "model"
    TOKENIZER = "tokenizer"
    SESSION = "session"
    CACHE = "cache"
    THREAD = "thread"
    FILE_HANDLE = "file_handle"
    NETWORK_CONNECTION = "network_connection"
    GPU_MEMORY = "gpu_memory"
    UI_COMPONENT = "ui_component"
    WORKER = "worker"


class ConsolidatedResourceManager(QObject):
    """
    [HOT] CONSOLIDATED RESOURCE MANAGER
    
    Single unified system for all resource and memory management.
    Eliminates conflicts between ResourceManager and MemoryManager.
    """
    
    # Signals for monitoring
    memory_usage_updated = pyqtSignal(dict)
    resource_leak_detected = pyqtSignal(str)
    cleanup_completed = pyqtSignal(int)  # number of resources cleaned
    
    _instance: Optional["ConsolidatedResourceManager"] = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        super().__init__()
        
        with self._lock:
            if self._initialized:
                return
                
            # Resource tracking
            self._resources: Dict[str, Dict[str, Any]] = {}
            self._weak_refs: Dict[str, weakref.ref] = {}
            self._cleanup_callbacks: Dict[str, Callable] = {}
            self._resource_lock = threading.RLock()
            
            # Memory management
            self._memory_threshold_percent = 85  # 85% memory threshold
            self._gpu_memory_threshold_percent = 90  # 90% GPU memory threshold
            self._last_gc_time = time.time()
            self._gc_interval = 30.0  # Run GC every 30 seconds
            
            # Monitoring timer
            self._memory_timer = None
            self._monitoring_active = False
            
            # Statistics
            self._stats = {
                "total_allocated": 0,
                "total_cleaned": 0,
                "gc_runs": 0,
                "memory_leaks_detected": 0,
                "peak_memory_usage": 0,
                "peak_gpu_memory_usage": 0
            }
            
            # System info
            self._process = psutil.Process()
            
            self._initialized = True
            logger.info("🛡️ ConsolidatedResourceManager initialized - unified resource management active")
    
    def start_monitoring(self):
        """Start memory and resource monitoring"""
        if self._monitoring_active:
            return
            
        self._monitoring_active = True
        
        # Use QTimer for thread-safe monitoring
        self._memory_timer = QTimer()
        self._memory_timer.timeout.connect(self._monitor_resources)
        self._memory_timer.start(5000)  # Monitor every 5 seconds
        
        logger.info("[SEARCH] Consolidated resource monitoring started")

    def add_cleanup_callback(self, callback: Callable):
        """[CONFIG] UNIFIED: Add a cleanup callback"""
        with self._resource_lock:
            if callback not in self._cleanup_callbacks:
                self._cleanup_callbacks[f"callback_{len(self._cleanup_callbacks)}"] = callback
                logger.debug(f"[OK] Cleanup callback registered: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")

    def unregister_resource(self, resource_id: str):
        """[CONFIG] UNIFIED: Unregister a resource"""
        with self._resource_lock:
            if resource_id in self._resources:
                resource_info = self._resources[resource_id]
                resource_info["active"] = False
                del self._resources[resource_id]

                # Remove weak reference if exists
                if resource_id in self._weak_refs:
                    del self._weak_refs[resource_id]

                # Remove cleanup callback if exists
                if resource_id in self._cleanup_callbacks:
                    del self._cleanup_callbacks[resource_id]

                self._stats["total_cleaned"] += 1
                logger.debug(f"🗑️ Resource unregistered: {resource_id}")

    def emergency_cleanup(self):
        """[CONFIG] UNIFIED: Emergency cleanup of all resources"""
        logger.warning("[EMERGENCY] EMERGENCY CLEANUP initiated by ConsolidatedResourceManager!")

        cleaned_count = 0
        try:
            # Call all cleanup callbacks
            with self._resource_lock:
                for callback_id, callback in list(self._cleanup_callbacks.items()):
                    try:
                        callback()
                        cleaned_count += 1
                    except Exception as e:
                        logger.error(f"[ERROR] Cleanup callback {callback_id} failed: {e}")

                # Clear all tracked resources
                self._resources.clear()
                self._weak_refs.clear()
                self._cleanup_callbacks.clear()

            # Force garbage collection
            for _ in range(3):
                gc.collect()

            # GPU memory cleanup if available
            if torch and cuda_available:
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(f"[WARNING] GPU cleanup failed: {e}")

            self._stats["total_cleaned"] += cleaned_count
            logger.info(f"[OK] Emergency cleanup completed: {cleaned_count} resources cleaned")
            self.cleanup_completed.emit(cleaned_count)

        except Exception as e:
            logger.error(f"[ERROR] Emergency cleanup failed: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if self._memory_timer:
            self._memory_timer.stop()
            self._memory_timer = None
        self._monitoring_active = False
        logger.info("[SEARCH] Consolidated resource monitoring stopped")
    
    def register_resource(self, resource: Any, resource_type: ResourceType, 
                         resource_id: Optional[str] = None,
                         cleanup_callback: Optional[Callable] = None) -> str:
        """Register a resource for management"""
        with self._resource_lock:
            if resource_id is None:
                resource_id = f"{resource_type.value}_{int(time.time() * 1000)}_{id(resource)}"
            
            # Store resource info
            self._resources[resource_id] = {
                "type": resource_type,
                "allocated_time": time.time(),
                "last_used": time.time(),
                "memory_usage": self._estimate_memory_usage(resource),
                "active": True
            }
            
            # Create weak reference for automatic cleanup
            def cleanup_ref(ref):
                self._handle_resource_deleted(resource_id)
            
            self._weak_refs[resource_id] = weakref.ref(resource, cleanup_ref)
            
            # Store cleanup callback
            if cleanup_callback:
                self._cleanup_callbacks[resource_id] = cleanup_callback
            
            self._stats["total_allocated"] += 1
            
            logger.debug(f"[DOC] Registered resource: {resource_id}")
            return resource_id
    
    def unregister_resource(self, resource_id: str):
        """Manually unregister a resource"""
        with self._resource_lock:
            if resource_id in self._resources:
                # Call cleanup callback if exists
                if resource_id in self._cleanup_callbacks:
                    try:
                        self._cleanup_callbacks[resource_id]()
                    except Exception as e:
                        logger.warning(f"[WARNING] Error in cleanup callback for {resource_id}: {e}")
                
                # Remove from tracking
                self._resources.pop(resource_id, None)
                self._weak_refs.pop(resource_id, None)
                self._cleanup_callbacks.pop(resource_id, None)
                
                self._stats["total_cleaned"] += 1
                logger.debug(f"🗑️ Unregistered resource: {resource_id}")
    
    def _handle_resource_deleted(self, resource_id: str):
        """Handle automatic resource deletion via weak reference"""
        with self._resource_lock:
            if resource_id in self._resources:
                self._resources[resource_id]["active"] = False
                logger.debug(f"🗑️ Resource automatically cleaned: {resource_id}")
                
                # Clean up after delay to avoid issues
                def delayed_cleanup():
                    self.unregister_resource(resource_id)
                
                # Use QTimer for thread-safe delayed cleanup
                timer = QTimer()
                timer.singleShot(1000, delayed_cleanup)
    
    def _estimate_memory_usage(self, resource: Any) -> int:
        """Estimate memory usage of a resource in bytes"""
        try:
            import sys
            return sys.getsizeof(resource)
        except Exception:
            return 0
    
    def _monitor_resources(self):
        """Monitor resources and trigger cleanup if needed"""
        try:
            # Check system memory
            memory_info = self._get_memory_usage()
            system_memory_percent = memory_info.get("system", {}).get("percent", 0)
            
            # Update peak usage
            if system_memory_percent > self._stats["peak_memory_usage"]:
                self._stats["peak_memory_usage"] = system_memory_percent
            
            # Check GPU memory if available
            gpu_memory_percent = 0
            if cuda_available:
                try:
                    gpu_memory_percent = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
                    if gpu_memory_percent > self._stats["peak_gpu_memory_usage"]:
                        self._stats["peak_gpu_memory_usage"] = gpu_memory_percent
                except Exception:
                    pass
            
            # Emit monitoring signal
            self.memory_usage_updated.emit({
                "system_memory_percent": system_memory_percent,
                "gpu_memory_percent": gpu_memory_percent,
                "active_resources": len([r for r in self._resources.values() if r.get("active", True)]),
                "stats": self._stats.copy()
            })
            
            # Trigger cleanup if thresholds exceeded
            cleanup_triggered = False
            if system_memory_percent > self._memory_threshold_percent:
                logger.warning(f"[WARNING] System memory usage {system_memory_percent:.1f}% exceeds threshold {self._memory_threshold_percent}%")
                self._trigger_memory_cleanup()
                cleanup_triggered = True
            
            if gpu_memory_percent > self._gpu_memory_threshold_percent:
                logger.warning(f"[WARNING] GPU memory usage {gpu_memory_percent:.1f}% exceeds threshold {self._gpu_memory_threshold_percent}%")
                self._trigger_gpu_cleanup()
                cleanup_triggered = True
            
            # Periodic garbage collection
            if time.time() - self._last_gc_time > self._gc_interval:
                self._trigger_garbage_collection()
                cleanup_triggered = True
            
            # Detect resource leaks
            self._detect_resource_leaks()
            
        except Exception as e:
            logger.error(f"[ERROR] Error in resource monitoring: {e}")
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            process_memory = self._process.memory_info()
            
            result = {
                "system": {
                    "total": system_memory.total,
                    "available": system_memory.available,
                    "percent": system_memory.percent,
                    "used": system_memory.used
                },
                "process": {
                    "rss": process_memory.rss,
                    "vms": process_memory.vms,
                    "percent": self._process.memory_percent()
                }
            }
            
            # GPU memory if available
            if cuda_available:
                try:
                    result["gpu"] = {
                        "allocated": torch.cuda.memory_allocated(),
                        "cached": torch.cuda.memory_reserved(),
                        "max_allocated": torch.cuda.max_memory_allocated()
                    }
                except Exception:
                    pass
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting memory usage: {e}")
            return {}
    
    def _trigger_memory_cleanup(self):
        """Trigger memory cleanup procedures"""
        logger.info("[CLEAN] Triggering memory cleanup...")
        
        # Clean up inactive resources
        cleanup_count = 0
        with self._resource_lock:
            inactive_resources = [
                rid for rid, info in self._resources.items() 
                if not info.get("active", True)
            ]
            
            for resource_id in inactive_resources:
                self.unregister_resource(resource_id)
                cleanup_count += 1
        
        # Force garbage collection
        self._trigger_garbage_collection()
        
        logger.info(f"[CLEAN] Memory cleanup completed: {cleanup_count} resources cleaned")
        self.cleanup_completed.emit(cleanup_count)
    
    def _trigger_gpu_cleanup(self):
        """Trigger GPU memory cleanup"""
        if cuda_available:
            try:
                torch.cuda.empty_cache()
                logger.info("[GAME] GPU memory cache cleared")
            except Exception as e:
                logger.error(f"[ERROR] Error clearing GPU cache: {e}")
    
    def _trigger_garbage_collection(self):
        """Trigger garbage collection"""
        try:
            collected = gc.collect()
            self._last_gc_time = time.time()
            self._stats["gc_runs"] += 1
            logger.debug(f"🗑️ Garbage collection: {collected} objects collected")
        except Exception as e:
            logger.error(f"[ERROR] Error in garbage collection: {e}")
    
    def _detect_resource_leaks(self):
        """Detect potential resource leaks"""
        try:
            current_time = time.time()
            leak_threshold = 300  # 5 minutes
            
            with self._resource_lock:
                potential_leaks = []
                for resource_id, info in self._resources.items():
                    if info.get("active", True):
                        age = current_time - info["allocated_time"]
                        if age > leak_threshold:
                            potential_leaks.append((resource_id, age))
                
                if potential_leaks:
                    self._stats["memory_leaks_detected"] += len(potential_leaks)
                    for resource_id, age in potential_leaks:
                        logger.warning(f"[EMERGENCY] Potential resource leak detected: {resource_id} (age: {age:.1f}s)")
                        self.resource_leak_detected.emit(f"{resource_id} (age: {age:.1f}s)")
                        
        except Exception as e:
            logger.error(f"[ERROR] Error detecting resource leaks: {e}")
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource management statistics"""
        with self._resource_lock:
            active_resources = len([r for r in self._resources.values() if r.get("active", True)])
            
            return {
                **self._stats,
                "active_resources": active_resources,
                "total_tracked": len(self._resources),
                "memory_info": self._get_memory_usage()
            }
    
    def cleanup_all(self):
        """Clean up all resources and stop monitoring"""
        logger.info("[CLEAN] Starting complete resource cleanup...")
        
        # Stop monitoring first
        self.stop_monitoring()
        
        # Clean up all resources
        cleanup_count = 0
        with self._resource_lock:
            resource_ids = list(self._resources.keys())
            for resource_id in resource_ids:
                self.unregister_resource(resource_id)
                cleanup_count += 1
        
        # Final cleanup
        self._trigger_garbage_collection()
        if cuda_available:
            self._trigger_gpu_cleanup()
        
        logger.info(f"[CLEAN] Complete cleanup finished: {cleanup_count} resources cleaned")
        self.cleanup_completed.emit(cleanup_count)


# Global instance for easy access
def get_consolidated_resource_manager() -> ConsolidatedResourceManager:
    """Get the global consolidated resource manager instance"""
    return ConsolidatedResourceManager()


# CRITICAL FIX: Deprecate old managers
class ResourceManagerDeprecationWarning:
    """Warning class for deprecated ResourceManager usage"""
    
    def __init__(self):
        import warnings
        warnings.warn(
            "ResourceManager is deprecated. Use ConsolidatedResourceManager instead.",
            DeprecationWarning,
            stacklevel=3
        )


class MemoryManagerDeprecationWarning:
    """Warning class for deprecated MemoryManager usage"""
    
    def __init__(self):
        import warnings
        warnings.warn(
            "MemoryManager is deprecated. Use ConsolidatedResourceManager instead.",
            DeprecationWarning,
            stacklevel=3
        )