"""
Ultra-Aggressive Lazy Loading System

This module implements an ultra-aggressive lazy loading system that defers
ALL heavy imports and initializations until absolutely necessary, targeting
sub-3 second startup times and minimal memory footprint.
"""

import sys
import time
import threading
import logging
import weakref
from typing import Dict, Any, Optional, Callable, Type, Union
from functools import wraps
import importlib
from pathlib import Path

logger = logging.getLogger(__name__)


class LazyImport:
    """
    Lazy import wrapper that defers module loading until first access
    """

    def __init__(self, module_name: str, attribute: Optional[str] = None):
        self.module_name = module_name
        self.attribute = attribute
        self._module = None
        self._loaded = False
        self._loading = False
        self._load_time = None

    def __getattr__(self, name):
        """Load module on first attribute access"""
        if not self._loaded:
            self._load_module()

        if self.attribute:
            return getattr(getattr(self._module, self.attribute), name)
        else:
            return getattr(self._module, name)

    def __call__(self, *args, **kwargs):
        """Support calling the imported object directly"""
        if not self._loaded:
            self._load_module()

        if self.attribute:
            obj = getattr(self._module, self.attribute)
        else:
            obj = self._module

        return obj(*args, **kwargs)

    def _load_module(self):
        """Actually load the module"""
        if self._loading:
            # Prevent circular loading
            return

        self._loading = True
        start_time = time.time()

        try:
            logger.debug(f"🔄 Lazy loading: {self.module_name}")
            self._module = importlib.import_module(self.module_name)
            self._loaded = True
            self._load_time = time.time() - start_time
            logger.debug(f"✅ Loaded {self.module_name} in {self._load_time:.3f}s")

        except ImportError as e:
            logger.error(f"❌ Failed to lazy load {self.module_name}: {e}")
            raise
        finally:
            self._loading = False

    @property
    def is_loaded(self) -> bool:
        """Check if module is loaded"""
        return self._loaded

    @property
    def load_time(self) -> Optional[float]:
        """Get load time in seconds"""
        return self._load_time


class UltraLazyLoader:
    """
    Ultra-aggressive lazy loading system that minimizes startup overhead
    """

    def __init__(self):
        self.lazy_imports: Dict[str, LazyImport] = {}
        self.deferred_initializations: Dict[str, Callable] = {}
        self.initialization_order: Dict[str, int] = {}
        self.initialized_components: set = set()
        self.loading_stats: Dict[str, float] = {}

        # Critical startup phase - defer everything possible
        self.startup_phase = True
        self.startup_complete_time = None

        logger.info("🚀 Ultra-aggressive lazy loader initialized")

    def register_lazy_import(
        self, name: str, module_name: str, attribute: Optional[str] = None
    ) -> LazyImport:
        """Register a module for lazy loading"""
        lazy_import = LazyImport(module_name, attribute)
        self.lazy_imports[name] = lazy_import
        logger.debug(f"📝 Registered lazy import: {name} -> {module_name}")
        return lazy_import

    def register_deferred_init(self, name: str, init_func: Callable, priority: int = 50) -> None:
        """Register a component for deferred initialization"""
        self.deferred_initializations[name] = init_func
        self.initialization_order[name] = priority
        logger.debug(f"📝 Registered deferred init: {name} (priority: {priority})")

    def get_lazy_import(self, name: str) -> Optional[LazyImport]:
        """Get a lazy import by name"""
        return self.lazy_imports.get(name)

    def initialize_component(self, name: str, force: bool = False) -> bool:
        """Initialize a specific component"""
        if name in self.initialized_components and not force:
            return True

        if name not in self.deferred_initializations:
            logger.warning(f"Component '{name}' not registered for deferred initialization")
            return False

        start_time = time.time()

        try:
            logger.info(f"🔄 Initializing component: {name}")
            init_func = self.deferred_initializations[name]
            result = init_func()

            self.initialized_components.add(name)
            load_time = time.time() - start_time
            self.loading_stats[name] = load_time

            logger.info(f"✅ Component '{name}' initialized in {load_time:.3f}s")
            return result if result is not None else True

        except Exception as e:
            logger.error(f"❌ Failed to initialize component '{name}': {e}")
            return False

    def initialize_by_priority(self, max_priority: Optional[int] = None) -> Dict[str, bool]:
        """Initialize components by priority order"""
        results = {}

        # Sort by priority (higher priority = earlier initialization)
        sorted_components = sorted(
            self.deferred_initializations.items(),
            key=lambda x: self.initialization_order.get(x[0], 50),
            reverse=True,
        )

        for name, init_func in sorted_components:
            priority = self.initialization_order.get(name, 50)

            if max_priority is not None and priority > max_priority:
                continue

            if name not in self.initialized_components:
                results[name] = self.initialize_component(name)

        return results

    def complete_startup_phase(self) -> None:
        """Mark startup phase as complete and begin background initialization"""
        if not self.startup_phase:
            return

        self.startup_phase = False
        self.startup_complete_time = time.time()

        logger.info("🎯 Startup phase completed - beginning background initialization")

        # Start background initialization thread
        threading.Thread(
            target=self._background_initialization, name="BackgroundInitializer", daemon=True
        ).start()

    def _background_initialization(self) -> None:
        """Initialize remaining components in background"""
        logger.info("🔄 Starting background component initialization")

        # Initialize remaining components with priority < 80
        results = self.initialize_by_priority(max_priority=79)

        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)

        logger.info(
            f"✅ Background initialization completed: {success_count}/{total_count} components"
        )

    def get_loading_stats(self) -> Dict[str, Any]:
        """Get comprehensive loading statistics"""
        total_load_time = sum(self.loading_stats.values())

        # Get lazy import stats
        lazy_stats = {}
        for name, lazy_import in self.lazy_imports.items():
            lazy_stats[name] = {"loaded": lazy_import.is_loaded, "load_time": lazy_import.load_time}

        return {
            "startup_complete_time": self.startup_complete_time,
            "total_component_load_time": total_load_time,
            "initialized_components": len(self.initialized_components),
            "total_registered_components": len(self.deferred_initializations),
            "component_stats": self.loading_stats.copy(),
            "lazy_import_stats": lazy_stats,
            "startup_phase": self.startup_phase,
        }

    def force_load_all(self) -> Dict[str, bool]:
        """Force load all registered components (for testing/debugging)"""
        logger.warning("🚨 Force loading ALL components - this defeats lazy loading!")

        # Load all lazy imports
        for name, lazy_import in self.lazy_imports.items():
            if not lazy_import.is_loaded:
                try:
                    lazy_import._load_module()
                except Exception as e:
                    logger.error(f"Failed to force load {name}: {e}")

        # Initialize all components
        return self.initialize_by_priority()


# Global ultra lazy loader instance
_global_loader: Optional[UltraLazyLoader] = None


def get_ultra_lazy_loader() -> UltraLazyLoader:
    """Get the global ultra lazy loader"""
    global _global_loader
    if _global_loader is None:
        _global_loader = UltraLazyLoader()
    return _global_loader


def lazy_import(
    module_name: str, attribute: Optional[str] = None, name: Optional[str] = None
) -> LazyImport:
    """Create a lazy import"""
    loader = get_ultra_lazy_loader()
    import_name = name or module_name.split(".")[-1]
    return loader.register_lazy_import(import_name, module_name, attribute)


def deferred_init(name: str, priority: int = 50):
    """Decorator for deferred initialization"""

    def decorator(func: Callable) -> Callable:
        loader = get_ultra_lazy_loader()
        loader.register_deferred_init(name, func, priority)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return loader.initialize_component(name)

        return wrapper

    return decorator


def complete_startup():
    """Mark startup as complete and begin background loading"""
    loader = get_ultra_lazy_loader()
    loader.complete_startup_phase()


def get_loading_stats() -> Dict[str, Any]:
    """Get loading statistics"""
    loader = get_ultra_lazy_loader()
    return loader.get_loading_stats()


# Pre-register common heavy imports for ultra-lazy loading
def setup_common_lazy_imports():
    """Setup common heavy imports for lazy loading"""
    loader = get_ultra_lazy_loader()

    # ML/AI Libraries (heaviest imports)
    loader.register_lazy_import("torch", "torch")
    loader.register_lazy_import("transformers", "transformers")
    loader.register_lazy_import("datasets", "datasets")
    loader.register_lazy_import("numpy", "numpy")
    loader.register_lazy_import("pandas", "pandas")
    loader.register_lazy_import("sklearn", "sklearn")

    # Computer Vision
    loader.register_lazy_import("cv2", "cv2")
    loader.register_lazy_import("PIL", "PIL")

    # NLP Libraries
    loader.register_lazy_import("spacy", "spacy")
    loader.register_lazy_import("nltk", "nltk")

    # Scientific Computing
    loader.register_lazy_import("scipy", "scipy")
    loader.register_lazy_import("matplotlib", "matplotlib")

    # Vector Databases
    loader.register_lazy_import("faiss", "faiss")
    loader.register_lazy_import("chromadb", "chromadb")

    logger.info("📚 Common heavy imports registered for ultra-lazy loading")


# Initialize common lazy imports immediately
setup_common_lazy_imports()