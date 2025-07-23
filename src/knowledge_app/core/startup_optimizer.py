"""
Startup Performance Optimizer

This module provides comprehensive startup optimization for the Knowledge App,
including lazy loading, dependency management, and performance monitoring.

Features:
- Lazy import management
- Startup performance monitoring
- Progressive initialization
- Memory usage optimization
- Dependency health checking
"""

import logging
import time
import sys
import warnings
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class StartupOptimizer:
    """
    Comprehensive startup optimizer that manages lazy loading,
    performance monitoring, and progressive initialization.
    """

    def __init__(self):
        self.start_time = time.time()
        self.initialization_phases = {}
        self.lazy_imports = {}
        self.performance_metrics = {}
        self.warning_suppressions_applied = False

        # Apply critical optimizations immediately
        self._apply_immediate_optimizations()

    def _apply_immediate_optimizations(self):
        """Apply critical optimizations that must happen immediately"""
        try:
            # Suppress startup warnings that clutter output
            self._suppress_startup_warnings()

            # Optimize Python import behavior
            self._optimize_import_behavior()

            logger.debug("✅ Immediate startup optimizations applied")

        except Exception as e:
            logger.warning(f"Failed to apply immediate optimizations: {e}")

    def _suppress_startup_warnings(self):
        """Suppress warnings that clutter startup output"""
        if self.warning_suppressions_applied:
            return

        try:
            # Suppress pydantic warnings
            warnings.filterwarnings(
                "ignore", message=".*GetCoreSchemaHandler.*", category=UserWarning
            )
            warnings.filterwarnings(
                "ignore", message=".*DataFrame/numpy support limited.*", category=UserWarning
            )

            # Suppress deprecated package warnings
            warnings.filterwarnings(
                "ignore", message=".*pkg_resources.*deprecated.*", category=DeprecationWarning
            )
            warnings.filterwarnings(
                "ignore", message=".*quantulum3.*deprecated.*", category=UserWarning
            )

            # Suppress ML library warnings during startup
            warnings.filterwarnings("ignore", message=".*torch.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*transformers.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*CUDA.*", category=UserWarning)

            # Suppress FAISS warnings
            warnings.filterwarnings(
                "ignore", message=".*Failed to load GPU Faiss.*", category=UserWarning
            )
            warnings.filterwarnings("ignore", message=".*FAISS.*", category=UserWarning)

            self.warning_suppressions_applied = True
            logger.debug("✅ Startup warning suppressions applied")

        except Exception as e:
            logger.warning(f"Failed to suppress startup warnings: {e}")

    def _optimize_import_behavior(self):
        """Optimize Python import behavior for faster startup"""
        try:
            # Reduce import overhead by optimizing sys.path
            # Remove duplicate paths
            seen = set()
            sys.path[:] = [x for x in sys.path if not (x in seen or seen.add(x))]

            logger.debug("✅ Import behavior optimized")

        except Exception as e:
            logger.warning(f"Failed to optimize import behavior: {e}")

    def start_phase(self, phase_name: str) -> None:
        """Start timing an initialization phase"""
        self.initialization_phases[phase_name] = {
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "status": "running",
        }
        logger.debug(f"🔄 Starting phase: {phase_name}")

    def end_phase(self, phase_name: str, status: str = "completed") -> float:
        """End timing an initialization phase"""
        if phase_name not in self.initialization_phases:
            logger.warning(f"Phase {phase_name} was not started")
            return 0.0

        phase = self.initialization_phases[phase_name]
        phase["end_time"] = time.time()
        phase["duration"] = phase["end_time"] - phase["start_time"]
        phase["status"] = status

        logger.debug(f"✅ Phase {phase_name} {status} in {phase['duration']:.3f}s")
        return phase["duration"]

    def register_lazy_import(self, module_name: str, import_func: Callable) -> None:
        """Register a lazy import function"""
        self.lazy_imports[module_name] = {
            "import_func": import_func,
            "imported": False,
            "import_time": None,
            "module": None,
        }
        logger.debug(f"📦 Registered lazy import: {module_name}")

    def get_lazy_import(self, module_name: str) -> Any:
        """Get a lazily imported module"""
        if module_name not in self.lazy_imports:
            raise ValueError(f"Lazy import {module_name} not registered")

        lazy_import = self.lazy_imports[module_name]

        if not lazy_import["imported"]:
            start_time = time.time()
            try:
                lazy_import["module"] = lazy_import["import_func"]()
                lazy_import["imported"] = True
                lazy_import["import_time"] = time.time() - start_time
                logger.debug(f"📦 Lazy imported {module_name} in {lazy_import['import_time']:.3f}s")
            except Exception as e:
                logger.error(f"Failed to lazy import {module_name}: {e}")
                raise

        return lazy_import["module"]

    def defer_heavy_initialization(self, component_name: str, init_func: Callable) -> None:
        """Defer heavy initialization to background thread"""

        def background_init():
            try:
                start_time = time.time()
                init_func()
                duration = time.time() - start_time
                logger.info(
                    f"✅ Background initialization of {component_name} completed in {duration:.3f}s"
                )
            except Exception as e:
                logger.error(f"❌ Background initialization of {component_name} failed: {e}")

        thread = threading.Thread(
            target=background_init, name=f"Init-{component_name}", daemon=True
        )
        thread.start()
        logger.debug(f"🔄 Deferred {component_name} initialization to background")

    def get_startup_summary(self) -> Dict[str, Any]:
        """Get a summary of startup performance"""
        total_time = time.time() - self.start_time

        summary = {
            "total_startup_time": total_time,
            "phases": self.initialization_phases.copy(),
            "lazy_imports": {
                name: {"imported": info["imported"], "import_time": info["import_time"]}
                for name, info in self.lazy_imports.items()
            },
            "performance_metrics": self.performance_metrics.copy(),
        }

        return summary

    def log_startup_summary(self) -> None:
        """Log a summary of startup performance"""
        summary = self.get_startup_summary()
        total_time = summary["total_startup_time"]

        logger.info(f"🚀 Startup completed in {total_time:.3f}s")

        # Log phase timings
        for phase_name, phase_info in summary["phases"].items():
            if phase_info["duration"]:
                logger.info(
                    f"  📋 {phase_name}: {phase_info['duration']:.3f}s ({phase_info['status']})"
                )

        # Log lazy import statistics
        imported_count = sum(1 for info in summary["lazy_imports"].values() if info["imported"])
        total_lazy = len(summary["lazy_imports"])
        logger.info(f"  📦 Lazy imports: {imported_count}/{total_lazy} loaded")


# Global startup optimizer instance
_global_optimizer: Optional[StartupOptimizer] = None


def get_startup_optimizer() -> StartupOptimizer:
    """Get the global startup optimizer instance"""
    global _global_optimizer

    if _global_optimizer is None:
        _global_optimizer = StartupOptimizer()

    return _global_optimizer


def optimize_startup():
    """Apply startup optimizations"""
    optimizer = get_startup_optimizer()
    return optimizer


def start_phase(phase_name: str) -> None:
    """Start timing an initialization phase"""
    optimizer = get_startup_optimizer()
    optimizer.start_phase(phase_name)


def end_phase(phase_name: str, status: str = "completed") -> float:
    """End timing an initialization phase"""
    optimizer = get_startup_optimizer()
    return optimizer.end_phase(phase_name, status)


def register_lazy_import(module_name: str, import_func: Callable) -> None:
    """Register a lazy import function"""
    optimizer = get_startup_optimizer()
    optimizer.register_lazy_import(module_name, import_func)


def get_lazy_import(module_name: str) -> Any:
    """Get a lazily imported module"""
    optimizer = get_startup_optimizer()
    return optimizer.get_lazy_import(module_name)


def defer_heavy_initialization(component_name: str, init_func: Callable) -> None:
    """Defer heavy initialization to background thread"""
    optimizer = get_startup_optimizer()
    optimizer.defer_heavy_initialization(component_name, init_func)


def log_startup_summary() -> None:
    """Log a summary of startup performance"""
    optimizer = get_startup_optimizer()
    optimizer.log_startup_summary()