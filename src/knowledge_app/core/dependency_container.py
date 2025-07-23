"""
🔧 CRITICAL FIX for Bug 38: Dependency Injection Container

This module provides clear ownership and lifecycle management for all critical components,
eliminating the tangled mess of dependencies and global singletons.
"""

import logging
import threading
import weakref
from typing import Dict, Any, Optional, Type, Callable, Set
from enum import Enum

logger = logging.getLogger(__name__)

class ComponentLifecycle(Enum):
    """Component lifecycle states"""
    SINGLETON = "singleton"      # Single instance for entire application
    TRANSIENT = "transient"      # New instance every time
    SCOPED = "scoped"           # Single instance per scope (e.g., per quiz session)

class ComponentState(Enum):
    """Component states"""
    NOT_CREATED = "not_created"
    CREATING = "creating"
    READY = "ready"
    FAILED = "failed"
    DISPOSED = "disposed"

class ComponentRegistration:
    """Registration information for a component"""
    def __init__(self, component_type: Type, factory: Callable, 
                 lifecycle: ComponentLifecycle, dependencies: Set[str] = None):
        self.component_type = component_type
        self.factory = factory
        self.lifecycle = lifecycle
        self.dependencies = dependencies or set()
        self.state = ComponentState.NOT_CREATED
        self.instance = None
        self.error = None

class DependencyContainer:
    """
    🔧 CRITICAL FIX for Bug 38: Clear ownership and dependency management
    
    Provides explicit dependency injection and lifecycle management,
    eliminating the tangled web of global singletons and unclear ownership.
    """
    
    _instance: Optional["DependencyContainer"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._registrations: Dict[str, ComponentRegistration] = {}
        self._instances: Dict[str, Any] = {}
        self._scoped_instances: Dict[str, Dict[str, Any]] = {}  # scope_id -> component_name -> instance
        self._creation_lock = threading.RLock()
        self._current_scope = None
        
        logger.info("🔧 DependencyContainer initialized - clear ownership model active")
        
        # Register core components
        self._register_core_components()
    
    def _register_core_components(self):
        """Register core application components with clear ownership"""
        
        # Unified Inference Manager - Singleton
        self.register(
            name="unified_inference_manager",
            factory=self._create_unified_inference_manager,
            lifecycle=ComponentLifecycle.SINGLETON,
            dependencies=set()
        )
        
        # Global Model Singleton - Singleton
        self.register(
            name="global_model_singleton", 
            factory=self._create_global_model_singleton,
            lifecycle=ComponentLifecycle.SINGLETON,
            dependencies=set()
        )
        
        # MCQ Manager - Scoped (per quiz session)
        self.register(
            name="mcq_manager",
            factory=self._create_mcq_manager,
            lifecycle=ComponentLifecycle.SCOPED,
            dependencies={"unified_inference_manager"}
        )
        
        # Training Manager - Scoped
        self.register(
            name="training_manager",
            factory=self._create_training_manager,
            lifecycle=ComponentLifecycle.SCOPED,
            dependencies={"unified_inference_manager"}
        )
        
        # Unified Fallback Manager - Singleton
        self.register(
            name="unified_fallback_manager",
            factory=self._create_unified_fallback_manager,
            lifecycle=ComponentLifecycle.SINGLETON,
            dependencies=set()
        )
    
    def register(self, name: str, factory: Callable, lifecycle: ComponentLifecycle, 
                dependencies: Set[str] = None, component_type: Type = None):
        """Register a component with the container"""
        with self._creation_lock:
            if name in self._registrations:
                logger.warning(f"⚠️ Component '{name}' already registered, replacing")
            
            self._registrations[name] = ComponentRegistration(
                component_type=component_type,
                factory=factory,
                lifecycle=lifecycle,
                dependencies=dependencies or set()
            )
            
            logger.info(f"📝 Registered component '{name}' with {lifecycle.value} lifecycle")
    
    def get(self, name: str, scope_id: str = None) -> Any:
        """Get a component instance with proper dependency resolution"""
        with self._creation_lock:
            if name not in self._registrations:
                raise ValueError(f"Component '{name}' not registered")
            
            registration = self._registrations[name]
            
            # Handle different lifecycles
            if registration.lifecycle == ComponentLifecycle.SINGLETON:
                return self._get_singleton(name, registration)
            elif registration.lifecycle == ComponentLifecycle.SCOPED:
                return self._get_scoped(name, registration, scope_id or self._current_scope or "default")
            else:  # TRANSIENT
                return self._create_instance(name, registration)
    
    def _get_singleton(self, name: str, registration: ComponentRegistration) -> Any:
        """
        🛡️ CRITICAL FIX: Get or create singleton instance with circular dependency detection
        """
        if name in self._instances:
            return self._instances[name]

        if registration.state == ComponentState.CREATING:
            raise RuntimeError(f"Circular dependency detected for singleton '{name}' - already being created")

        if registration.state == ComponentState.FAILED:
            raise RuntimeError(f"Singleton '{name}' failed to create: {registration.error}")

        # 🛡️ CRITICAL FIX: Set CREATING state BEFORE calling _create_instance to detect cycles
        registration.state = ComponentState.CREATING

        try:
            instance = self._create_instance(name, registration)
            self._instances[name] = instance

            # Mark as ready after successful creation
            registration.state = ComponentState.READY
            return instance

        except Exception as e:
            # Reset state on failure
            registration.state = ComponentState.FAILED
            registration.error = str(e)
            logger.error(f"❌ Singleton creation failed for '{name}': {e}")
            raise
    
    def _get_scoped(self, name: str, registration: ComponentRegistration, scope_id: str) -> Any:
        """Get or create scoped instance"""
        if scope_id not in self._scoped_instances:
            self._scoped_instances[scope_id] = {}
        
        scope_instances = self._scoped_instances[scope_id]
        
        if name in scope_instances:
            return scope_instances[name]
        
        instance = self._create_instance(name, registration)
        scope_instances[name] = instance
        return instance
    
    def _create_instance(self, name: str, registration: ComponentRegistration) -> Any:
        """
        🛡️ CRITICAL FIX: Create component instance with dependency resolution

        Note: For singletons, state should already be set to CREATING by caller.
        For transient instances, we set it here.
        """
        try:
            # 🛡️ CRITICAL FIX: Only set CREATING state if not already set (for transient instances)
            if registration.state == ComponentState.NOT_CREATED:
                registration.state = ComponentState.CREATING
            
            # Resolve dependencies first
            dependencies = {}
            for dep_name in registration.dependencies:
                dependencies[dep_name] = self.get(dep_name)
            
            # Create instance
            if dependencies:
                instance = registration.factory(**dependencies)
            else:
                instance = registration.factory()
            
            registration.state = ComponentState.READY
            registration.instance = weakref.ref(instance) if hasattr(instance, '__weakref__') else instance
            
            logger.info(f"✅ Created component '{name}' with dependencies: {list(registration.dependencies)}")
            return instance
            
        except Exception as e:
            registration.state = ComponentState.FAILED
            registration.error = str(e)
            logger.error(f"❌ Failed to create component '{name}': {e}")
            raise
    
    def start_scope(self, scope_id: str):
        """Start a new dependency scope"""
        self._current_scope = scope_id
        logger.info(f"🔄 Started dependency scope: {scope_id}")
    
    def end_scope(self, scope_id: str = None):
        """End a dependency scope and cleanup scoped instances"""
        target_scope = scope_id or self._current_scope
        
        if target_scope and target_scope in self._scoped_instances:
            scope_instances = self._scoped_instances[target_scope]
            
            # Cleanup scoped instances
            for name, instance in scope_instances.items():
                try:
                    if hasattr(instance, 'cleanup'):
                        instance.cleanup()
                    elif hasattr(instance, 'dispose'):
                        instance.dispose()
                except Exception as e:
                    logger.warning(f"⚠️ Error cleaning up scoped instance '{name}': {e}")
            
            del self._scoped_instances[target_scope]
            logger.info(f"🧹 Cleaned up dependency scope: {target_scope}")
        
        if self._current_scope == target_scope:
            self._current_scope = None
    
    def dispose(self):
        """Dispose of all managed instances"""
        with self._creation_lock:
            # Cleanup all scoped instances
            for scope_id in list(self._scoped_instances.keys()):
                self.end_scope(scope_id)
            
            # Cleanup singletons
            for name, instance in self._instances.items():
                try:
                    if hasattr(instance, 'cleanup'):
                        instance.cleanup()
                    elif hasattr(instance, 'dispose'):
                        instance.dispose()
                except Exception as e:
                    logger.warning(f"⚠️ Error disposing singleton '{name}': {e}")
            
            self._instances.clear()
            logger.info("🧹 DependencyContainer disposed of all instances")
    
    # Factory methods for core components
    def _create_unified_inference_manager(self):
        """Factory for UnifiedInferenceManager"""
        from .unified_inference_manager import get_unified_inference_manager
        return get_unified_inference_manager()
    
    def _create_global_model_singleton(self):
        """Factory for GlobalModelSingleton"""
        from .global_model_singleton import GlobalModelSingleton
        return GlobalModelSingleton()
    
    def _create_mcq_manager(self, unified_inference_manager):
        """
        🚀 BUG FIX: Factory for MCQManager using UnifiedConfigManager
        This fixes Bug 6 by using the single source of truth for configuration
        """
        from .mcq_manager import MCQManager

        # 🚀 BUG FIX: Use UnifiedConfigManager instead of hardcoded config
        try:
            from .unified_config_manager import UnifiedConfigManager
            config_manager = UnifiedConfigManager()

            # Get MCQ-specific configuration from unified config
            config = {
                'use_cache': config_manager.get('mcq.use_cache', True),
                'timeout': config_manager.get('mcq.timeout', 60.0),  # Respect user's timeout setting
                'max_retries': config_manager.get('mcq.max_retries', 3),
                'fallback_enabled': config_manager.get('mcq.fallback_enabled', True)
            }

            logger.info(f"🔧 MCQManager config from UnifiedConfigManager: timeout={config['timeout']}s")

        except ImportError:
            logger.warning("⚠️ UnifiedConfigManager not available - using fallback defaults")
            # Fallback to reasonable defaults (not the old hardcoded ones)
            config = {
                'use_cache': True,
                'timeout': 60.0,  # More reasonable default than 15s
                'max_retries': 3,
                'fallback_enabled': True
            }

        return MCQManager(config)
    
    def _create_training_manager(self, unified_inference_manager):
        """Factory for TrainingManager with proper dependencies"""
        # Import and create training manager when needed
        # from .training_manager import TrainingManager
        # return TrainingManager(unified_inference_manager)
        return None  # Placeholder for now
    
    def _create_unified_fallback_manager(self):
        """Factory for UnifiedFallbackManager"""
        from .unified_fallback_manager import get_unified_fallback_manager
        return get_unified_fallback_manager()

# Global container instance
_dependency_container = None

def get_dependency_container() -> DependencyContainer:
    """
    🔧 DEPRECATED: Get the global dependency container instance
    
    WARNING: This container is deprecated. Use UnifiedDIContainer instead.
    """
    from .deprecated_di_warning import warn_deprecated_container
    warn_deprecated_container("DependencyContainer") 
    
    global _dependency_container
    if _dependency_container is None:
        _dependency_container = DependencyContainer()
    return _dependency_container
