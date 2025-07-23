"""
Resource Manager for the Knowledge App.
Handles resource allocation, cleanup, and monitoring.
"""

import os
import gc
import logging
import psutil
import threading
import weakref
import torch
import multiprocessing
from typing import Dict, Any, Set, Optional, List, Tuple, Callable
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be managed"""

    UI = auto()
    MODEL = auto()
    WORKER = auto()
    FILE = auto()
    MEMORY = auto()
    GPU = auto()
    NETWORK = auto()
    DATABASE = auto()
    SYSTEM = auto()


class ResourcePriority(Enum):
    """Priority levels for resources"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


@dataclass
class ResourceStats:
    """Statistics for a resource"""

    creation_time: datetime
    last_access_time: datetime
    access_count: int = 0
    memory_usage: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    is_active: bool = True


class ResourceManager:
    """Manages application resources and cleanup"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        """Initialize resource manager (singleton)"""
        if not hasattr(self, "_initialized"):
            # Resource tracking
            self._resources: Dict[str, Any] = {}
            self._resource_types: Dict[str, ResourceType] = {}
            self._resource_priorities: Dict[str, ResourcePriority] = {}
            self._cleanup_handlers: Dict[str, Callable[[], None]] = {}
            self._resource_stats: Dict[str, ResourceStats] = {}
            self._resource_refs: Dict[str, weakref.ref] = {}
            self._last_report_time = datetime.now()
            self._report_interval = timedelta(minutes=5)

            # System resources
            self._process = psutil.Process()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._gpu_memory_allocated = 0
            self._cpu_count = multiprocessing.cpu_count()

            # Thread pool for async operations
            self._thread_pool = ThreadPoolExecutor(max_workers=4)

            # Memory monitoring
            self._memory_monitor = None
            self._stop_monitoring = threading.Event()
            self._memory_threshold = 0.85  # 85% memory threshold
            self._gpu_memory_threshold = 0.90  # 90% GPU memory threshold
            self._last_cleanup = datetime.now()
            self._cleanup_interval = timedelta(minutes=5)

            # Initialize
            self._setup_cpu_resources()
            self._initialized = True

            # Start monitoring
            self.start_memory_monitoring()

    def _setup_cpu_resources(self):
        """Set up CPU-related resources"""
        try:
            # Set process priority
            if os.name == "nt":  # Windows
                self._process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:  # Unix
                self._process.nice(10)  # Lower priority

            # Set thread count based on CPU cores
            torch.set_num_threads(max(1, self._cpu_count - 1))

        except Exception as e:
            logger.error(f"Error setting up CPU resources: {e}")

    def start_memory_monitoring(self):
        """Start memory monitoring thread"""
        if self._memory_monitor is None or not self._memory_monitor.is_alive():
            self._stop_monitoring.clear()
            self._memory_monitor = threading.Thread(
                target=self._monitor_memory, name="ResourceMonitor", daemon=True
            )
            self._memory_monitor.start()

    def stop_memory_monitoring(self):
        """Stop memory monitoring thread"""
        if self._memory_monitor is not None:
            self._stop_monitoring.set()
            if self._memory_monitor.is_alive():
                self._memory_monitor.join(timeout=2.0)
            self._memory_monitor = None

    def _monitor_memory(self):
        """Memory monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                self._check_resource_usage()
                threading.Event().wait(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                threading.Event().wait(5)

    def _check_resource_usage(self):
        """Check system resource usage"""
        try:
            current_time = datetime.now()
            if current_time - self._last_cleanup < self._cleanup_interval:
                return

            needs_cleanup = False

            # Check system memory
            memory = psutil.virtual_memory()
            if memory.percent > self._memory_threshold * 100:
                logger.warning(f"High memory usage: {memory.percent}%")
                needs_cleanup = True

            # Check GPU memory
            if self._device == "cuda":
                for i in range(torch.cuda.device_count()):
                    try:
                        allocated = torch.cuda.memory_allocated(i)
                        total = torch.cuda.get_device_properties(i).total_memory
                        usage = allocated / total

                        if usage > self._gpu_memory_threshold:
                            logger.warning(f"High GPU {i} memory usage: {usage*100:.1f}%")
                            needs_cleanup = True
                    except Exception as e:
                        logger.error(f"Error checking GPU {i} memory: {e}")

            if needs_cleanup:
                self.cleanup_resources()
                self._last_cleanup = current_time

        except Exception as e:
            logger.error(f"Error checking resource usage: {e}")

    def register_resource(
        self,
        name: str,
        resource: Any,
        resource_type: ResourceType,
        cleanup_handler: Optional[Callable[[], None]] = None,
        priority: ResourcePriority = ResourcePriority.MEDIUM,
    ) -> None:
        """
        Register a resource for management

        Args:
            name: Unique resource identifier
            resource: The resource object to manage
            resource_type: Type of resource
            cleanup_handler: Optional cleanup function
            priority: Resource priority level
        """
        if name in self._resources:
            logger.warning(f"Resource {name} already registered, updating")

        self._resources[name] = resource
        self._resource_types[name] = resource_type
        self._resource_priorities[name] = priority

        if cleanup_handler:
            self._cleanup_handlers[name] = cleanup_handler

        # Create or update stats
        now = datetime.now()
        if name not in self._resource_stats:
            self._resource_stats[name] = ResourceStats(creation_time=now, last_access_time=now)

        # Create weak reference if possible
        try:
            self._resource_refs[name] = weakref.ref(resource)
        except TypeError:
            logger.debug(f"Could not create weak reference for {name}")

        self._update_resource_stats(name)
        logger.debug(f"Registered resource: {name} ({resource_type.name})")

    def get_resource(self, name: str) -> Optional[Any]:
        """
        Get a registered resource

        Args:
            name: Resource identifier

        Returns:
            The resource object if found, None otherwise
        """
        if name in self._resources:
            self._update_resource_stats(name)
            return self._resources[name]
        return None

    def unregister_resource(self, name: str) -> None:
        """
        Unregister a resource

        Args:
            name: Resource identifier
        """
        if name in self._resources:
            if name in self._cleanup_handlers:
                try:
                    self._cleanup_handlers[name]()
                except Exception as e:
                    logger.error(f"Error cleaning up resource {name}: {e}")

            self._resources.pop(name, None)
            self._resource_types.pop(name, None)
            self._resource_priorities.pop(name, None)
            self._cleanup_handlers.pop(name, None)
            self._resource_stats.pop(name, None)
            self._resource_refs.pop(name, None)

            logger.debug(f"Unregistered resource: {name}")

    def _update_resource_stats(self, name: str) -> None:
        """Update statistics for a resource"""
        if name not in self._resource_stats:
            return

        stats = self._resource_stats[name]
        stats.last_access_time = datetime.now()
        stats.access_count += 1

        # Try to get memory usage
        resource = self._resources.get(name)
        if resource:
            try:
                stats.memory_usage = self._process.memory_info().rss / 1024 / 1024  # MB
            except Exception:
                pass

            # Try to get GPU memory usage
            try:
                if torch.cuda.is_available() and hasattr(resource, "cuda"):
                    device = resource.cuda.current_device()
                    stats.gpu_memory_usage = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
            except Exception:
                pass

    def generate_resource_report(self) -> None:
        """Generate a report of resource usage"""
        now = datetime.now()
        if now - self._last_report_time < self._report_interval:
            return

        self._last_report_time = now

        logger.info("=== Resource Usage Report ===")

        # Group resources by type
        by_type: Dict[ResourceType, List[str]] = {}
        for name, res_type in self._resource_types.items():
            if res_type not in by_type:
                by_type[res_type] = []
            by_type[res_type].append(name)

        # Report by type
        for res_type, names in by_type.items():
            logger.info(f"\n{res_type.name} Resources:")
            for name in names:
                stats = self._resource_stats.get(name)
                if stats:
                    age = now - stats.creation_time
                    last_access = now - stats.last_access_time

                    # Check if resource is still alive
                    if name in self._resource_refs:
                        ref = self._resource_refs[name]
                        stats.is_active = ref() is not None

                    status = "ACTIVE" if stats.is_active else "INACTIVE"
                    priority = self._resource_priorities[name].name

                    logger.info(
                        f"  {name}:\n"
                        f"    Status: {status}\n"
                        f"    Priority: {priority}\n"
                        f"    Age: {age.total_seconds():.1f}s\n"
                        f"    Last Access: {last_access.total_seconds():.1f}s ago\n"
                        f"    Access Count: {stats.access_count}"
                    )

                    if stats.memory_usage is not None:
                        logger.info(f"    Memory Usage: {stats.memory_usage:.1f}MB")
                    if stats.gpu_memory_usage is not None:
                        logger.info(f"    GPU Memory Usage: {stats.gpu_memory_usage:.1f}MB")

        logger.info("===========================")

    def cleanup_resources(
        self,
        resource_type: Optional[ResourceType] = None,
        min_priority: ResourcePriority = ResourcePriority.LOW,
    ) -> None:
        """
        Clean up registered resources

        Args:
            resource_type: Optional type to clean up
            min_priority: Minimum priority level to clean up
        """
        with self._lock:
            # Get resources to clean up
            resources_to_clean = []
            for name, resource_ref in list(self._resources.items()):
                if resource_type and self._resource_types.get(name) != resource_type:
                    continue

                resource_priority = self._resource_priorities.get(name, ResourcePriority.LOW)
                if resource_priority.value < min_priority.value:
                    continue

                # Handle both direct resource and weak reference
                if hasattr(resource_ref, "__call__"):
                    resource = resource_ref()
                else:
                    resource = resource_ref

                if resource is not None:
                    resources_to_clean.append((name, resource))
                else:
                    self.unregister_resource(name)

            # Sort by priority (highest first)
            resources_to_clean.sort(
                key=lambda x: self._resource_priorities.get(x[0], ResourcePriority.LOW).value,
                reverse=True,
            )

            # Clean up resources
            for name, resource in resources_to_clean:
                try:
                    if name in self._cleanup_handlers:
                        self._cleanup_handlers[name]()
                    elif hasattr(resource, "cleanup"):
                        resource.cleanup()
                    elif hasattr(resource, "close"):
                        resource.close()

                    # Update statistics
                    self._update_resource_stats(name)

                except Exception as e:
                    logger.error(f"Error cleaning up resource {name}: {e}")
                    continue

            # Force garbage collection
            gc.collect()

            # Clear GPU cache if needed
            if self._device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error clearing GPU cache: {e}")

    def get_resource_stats(self) -> Dict[str, ResourceStats]:
        """Get resource statistics"""
        return self._resource_stats

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        try:
            stats = {
                "system": dict(psutil.virtual_memory()._asdict()),
                "process": {
                    "memory_percent": self._process.memory_percent(),
                    "memory_info": dict(self._process.memory_info()._asdict()),
                },
            }

            if self._device == "cuda":
                gpu_stats = {}
                for i in range(torch.cuda.device_count()):
                    try:
                        allocated = torch.cuda.memory_allocated(i)
                        reserved = torch.cuda.memory_reserved(i)
                        total = torch.cuda.get_device_properties(i).total_memory
                        gpu_stats[f"gpu_{i}"] = {
                            "allocated": allocated,
                            "reserved": reserved,
                            "total": total,
                            "utilization": allocated / total,
                        }
                    except Exception as e:
                        logger.error(f"Error getting GPU {i} stats: {e}")
                stats["gpu"] = gpu_stats

            return stats
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {}

    def monitor_memory(self, threshold: float = 0.9) -> bool:
        """
        Monitor memory usage and trigger cleanup if needed

        Args:
            threshold: Memory usage threshold (0-1) to trigger cleanup

        Returns:
            bool: True if cleanup was triggered
        """
        try:
            memory_info = self.get_memory_usage()
            if memory_info.get("system", {}).get("percent", 0) > threshold * 100:
                logger.warning("Memory usage exceeded threshold, triggering cleanup")
                self.cleanup_resources(min_priority=ResourcePriority.MEDIUM)
                return True
        except Exception as e:
            logger.error(f"Error monitoring memory: {e}")

        return False

    @property
    def device(self) -> str:
        """Get current compute device"""
        return self._device

    def cleanup(self) -> None:
        """Perform complete cleanup"""
        try:
            # Stop monitoring
            self.stop_memory_monitoring()

            # Clean up all resources
            self.cleanup_resources()

            # Shut down thread pool
            self._thread_pool.shutdown(wait=False)

            # Clear all tracking
            self._resources.clear()
            self._cleanup_handlers.clear()
            self._resource_types.clear()
            self._resource_priorities.clear()
            self._resource_stats.clear()
            self._resource_refs.clear()

            # Force final garbage collection
            gc.collect()

            if self._device == "cuda":
                torch.cuda.empty_cache()

            # Generate final report
            self.generate_resource_report()

            logger.info("Resource cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()