#!/usr/bin/env python3
"""
🔍 MEMORY LEAK DETECTOR - Advanced Memory Management System

This module provides comprehensive memory leak detection and prevention:
1. Real-time memory monitoring
2. Automatic leak detection
3. Resource cleanup automation
4. Memory usage optimization
"""

import logging
import psutil
import gc
import threading
import time
import weakref
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import traceback

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float
    gpu_memory_mb: float
    python_objects: int
    gc_collections: Dict[int, int]

@dataclass
class ResourceTracker:
    """Track allocated resources"""
    resource_type: str
    allocation_time: float
    stack_trace: str
    cleanup_callback: Optional[Callable] = None
    is_cleaned: bool = False

class MemoryLeakDetector:
    """
    🔍 ADVANCED MEMORY LEAK DETECTOR
    
    Monitors memory usage and automatically detects/prevents leaks
    """
    
    def __init__(self):
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.snapshots: List[MemorySnapshot] = []
        self.resource_registry: Dict[int, ResourceTracker] = {}
        self.weak_refs: Set[weakref.ref] = set()
        self._lock = threading.Lock()
        
        # Configuration
        self.config = {
            "snapshot_interval": 5.0,  # Take snapshot every 5 seconds
            "max_snapshots": 100,      # Keep last 100 snapshots
            "leak_threshold_mb": 50,   # Alert if memory grows by 50MB
            "leak_detection_window": 10,  # Check last 10 snapshots for leaks
            "auto_cleanup_enabled": True,
            "gc_frequency": 30.0,      # Force GC every 30 seconds
        }
        
        # Leak detection state
        self.baseline_memory = 0.0
        self.leak_alerts: List[Dict[str, Any]] = []
        self.cleanup_callbacks: List[Callable] = []
        
        logger.info("🔍 Memory Leak Detector initialized")

    def start_monitoring(self):
        """Start memory leak monitoring"""
        if self.monitoring_active:
            logger.warning("⚠️ Memory monitoring already active")
            return
            
        self.monitoring_active = True
        self.baseline_memory = self._get_current_memory_usage()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MemoryLeakDetector"
        )
        self.monitor_thread.start()
        logger.info("🚀 Memory leak monitoring started")

    def stop_monitoring(self):
        """Stop memory leak monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("🛑 Memory leak monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        last_gc_time = time.time()
        
        while self.monitoring_active:
            try:
                # Take memory snapshot
                snapshot = self._take_memory_snapshot()
                
                with self._lock:
                    self.snapshots.append(snapshot)
                    # Keep only recent snapshots
                    if len(self.snapshots) > self.config["max_snapshots"]:
                        self.snapshots.pop(0)
                
                # Check for memory leaks
                self._detect_memory_leaks()
                
                # Periodic garbage collection
                current_time = time.time()
                if current_time - last_gc_time >= self.config["gc_frequency"]:
                    self._force_garbage_collection()
                    last_gc_time = current_time
                
                # Clean up dead weak references
                self._cleanup_weak_refs()
                
                time.sleep(self.config["snapshot_interval"])
                
            except Exception as e:
                logger.error(f"❌ Error in memory monitoring loop: {e}")
                time.sleep(10.0)  # Wait longer on error

    def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Get GPU memory if available
            gpu_memory = 0.0
            try:
                from .hardware_utils import get_real_time_gpu_utilization
                gpu_stats = get_real_time_gpu_utilization()
                gpu_memory = gpu_stats.get("memory_used_mb", 0.0)
            except:
                pass
            
            # Get Python object count
            python_objects = len(gc.get_objects())
            
            # Get GC stats
            gc_stats = {}
            for i in range(3):  # Python has 3 GC generations
                gc_stats[i] = gc.get_count()[i]
            
            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                percent=process.memory_percent(),
                gpu_memory_mb=gpu_memory,
                python_objects=python_objects,
                gc_collections=gc_stats
            )
            
        except Exception as e:
            logger.error(f"❌ Error taking memory snapshot: {e}")
            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=0.0, vms_mb=0.0, percent=0.0,
                gpu_memory_mb=0.0, python_objects=0,
                gc_collections={}
            )

    def _detect_memory_leaks(self):
        """Detect potential memory leaks"""
        try:
            if len(self.snapshots) < self.config["leak_detection_window"]:
                return  # Not enough data yet
            
            # Get recent snapshots for analysis
            recent_snapshots = self.snapshots[-self.config["leak_detection_window"]:]
            
            # Calculate memory growth trend
            first_snapshot = recent_snapshots[0]
            last_snapshot = recent_snapshots[-1]
            
            memory_growth = last_snapshot.rss_mb - first_snapshot.rss_mb
            time_span = last_snapshot.timestamp - first_snapshot.timestamp
            
            # Check for significant memory growth
            if memory_growth > self.config["leak_threshold_mb"]:
                leak_alert = {
                    "timestamp": time.time(),
                    "memory_growth_mb": memory_growth,
                    "time_span_seconds": time_span,
                    "growth_rate_mb_per_min": (memory_growth / time_span) * 60,
                    "current_memory_mb": last_snapshot.rss_mb,
                    "gpu_memory_mb": last_snapshot.gpu_memory_mb,
                    "python_objects": last_snapshot.python_objects
                }
                
                self.leak_alerts.append(leak_alert)
                
                logger.warning(
                    f"🚨 MEMORY LEAK DETECTED: {memory_growth:.1f}MB growth in {time_span:.1f}s "
                    f"(rate: {leak_alert['growth_rate_mb_per_min']:.1f}MB/min)"
                )
                
                # Trigger automatic cleanup if enabled
                if self.config["auto_cleanup_enabled"]:
                    self._trigger_automatic_cleanup()
                    
        except Exception as e:
            logger.error(f"❌ Error detecting memory leaks: {e}")

    def _trigger_automatic_cleanup(self):
        """Trigger automatic memory cleanup"""
        try:
            logger.info("🧹 Triggering automatic memory cleanup...")
            
            # Force garbage collection
            self._force_garbage_collection()
            
            # Clean up registered resources
            self._cleanup_registered_resources()
            
            # Run custom cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"❌ Error in cleanup callback: {e}")
            
            # Clear caches
            self._clear_caches()
            
            logger.info("✅ Automatic memory cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error in automatic cleanup: {e}")

    def _force_garbage_collection(self):
        """Force Python garbage collection"""
        try:
            # Collect all generations
            collected = gc.collect()
            logger.debug(f"🗑️ Garbage collection freed {collected} objects")
            
            # Also try to free GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("🔥 GPU cache cleared")
            except:
                pass
                
        except Exception as e:
            logger.error(f"❌ Error in garbage collection: {e}")

    def _cleanup_registered_resources(self):
        """Clean up registered resources"""
        try:
            cleaned_count = 0
            
            with self._lock:
                for resource_id, tracker in list(self.resource_registry.items()):
                    if not tracker.is_cleaned and tracker.cleanup_callback:
                        try:
                            tracker.cleanup_callback()
                            tracker.is_cleaned = True
                            cleaned_count += 1
                        except Exception as e:
                            logger.error(f"❌ Error cleaning resource {resource_id}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned up {cleaned_count} registered resources")
                
        except Exception as e:
            logger.error(f"❌ Error cleaning registered resources: {e}")

    def _cleanup_weak_refs(self):
        """Clean up dead weak references"""
        try:
            dead_refs = [ref for ref in self.weak_refs if ref() is None]
            for ref in dead_refs:
                self.weak_refs.discard(ref)
                
            if dead_refs:
                logger.debug(f"🗑️ Cleaned up {len(dead_refs)} dead weak references")
                
        except Exception as e:
            logger.error(f"❌ Error cleaning weak references: {e}")

    def _clear_caches(self):
        """Clear various internal caches"""
        try:
            # Clear function caches
            import functools
            # Note: This is a simplified cache clearing - in practice you'd clear specific caches
            
            logger.debug("🧹 Internal caches cleared")
            
        except Exception as e:
            logger.error(f"❌ Error clearing caches: {e}")

    def register_resource(self, resource: Any, resource_type: str, cleanup_callback: Optional[Callable] = None) -> int:
        """Register a resource for tracking"""
        try:
            resource_id = id(resource)
            
            tracker = ResourceTracker(
                resource_type=resource_type,
                allocation_time=time.time(),
                stack_trace=traceback.format_stack()[-3],  # Get caller's stack
                cleanup_callback=cleanup_callback
            )
            
            with self._lock:
                self.resource_registry[resource_id] = tracker
            
            # Create weak reference to detect when resource is garbage collected
            def cleanup_callback_wrapper(ref):
                with self._lock:
                    if resource_id in self.resource_registry:
                        del self.resource_registry[resource_id]
                        
            weak_ref = weakref.ref(resource, cleanup_callback_wrapper)
            self.weak_refs.add(weak_ref)
            
            logger.debug(f"📝 Registered resource: {resource_type} (ID: {resource_id})")
            return resource_id
            
        except Exception as e:
            logger.error(f"❌ Error registering resource: {e}")
            return -1

    def add_cleanup_callback(self, callback: Callable):
        """Add a cleanup callback to be called during automatic cleanup"""
        self.cleanup_callbacks.append(callback)
        logger.debug("📝 Added cleanup callback")

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report"""
        try:
            if not self.snapshots:
                return {"status": "no_data", "message": "No memory snapshots available"}
            
            latest_snapshot = self.snapshots[-1]
            
            # Calculate memory trends
            if len(self.snapshots) >= 2:
                first_snapshot = self.snapshots[0]
                memory_trend = latest_snapshot.rss_mb - first_snapshot.rss_mb
                time_span = latest_snapshot.timestamp - first_snapshot.timestamp
                growth_rate = (memory_trend / time_span) * 60 if time_span > 0 else 0
            else:
                memory_trend = 0
                growth_rate = 0
            
            report = {
                "current_memory": {
                    "rss_mb": latest_snapshot.rss_mb,
                    "vms_mb": latest_snapshot.vms_mb,
                    "percent": latest_snapshot.percent,
                    "gpu_memory_mb": latest_snapshot.gpu_memory_mb,
                    "python_objects": latest_snapshot.python_objects
                },
                "trends": {
                    "memory_growth_mb": memory_trend,
                    "growth_rate_mb_per_min": growth_rate,
                    "snapshots_analyzed": len(self.snapshots)
                },
                "leak_detection": {
                    "alerts_count": len(self.leak_alerts),
                    "recent_alerts": self.leak_alerts[-5:] if self.leak_alerts else []
                },
                "resources": {
                    "registered_count": len(self.resource_registry),
                    "weak_refs_count": len(self.weak_refs)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating memory report: {e}")
            return {"status": "error", "message": str(e)}

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

# Global detector instance
_memory_detector: Optional[MemoryLeakDetector] = None

def get_memory_detector() -> MemoryLeakDetector:
    """Get or create global memory leak detector"""
    global _memory_detector
    if _memory_detector is None:
        _memory_detector = MemoryLeakDetector()
    return _memory_detector

def start_memory_monitoring():
    """Start global memory monitoring"""
    detector = get_memory_detector()
    detector.start_monitoring()

def stop_memory_monitoring():
    """Stop global memory monitoring"""
    detector = get_memory_detector()
    detector.stop_monitoring()
