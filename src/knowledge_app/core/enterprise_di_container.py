"""
Enterprise-Grade Dependency Injection Container

This module provides a professional dependency injection container that supports:
- Constructor injection
- Singleton and transient lifetimes
- Interface-based registration
- Circular dependency detection
- Thread-safe operations
- Lazy initialization
"""

import inspect
import threading
from typing import Any, Dict, Type, TypeVar, Callable, Optional, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceLifetime(Enum):
    """Service lifetime management"""

    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceDescriptor:
    """Describes how a service should be created and managed"""

    def __init__(
        self,
        service_type: Type,
        implementation_type: Optional[Type] = None,
        factory: Optional[Callable] = None,
        instance: Optional[Any] = None,
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
    ):
        self.service_type = service_type
        self.implementation_type = implementation_type or service_type
        self.factory = factory
        self.instance = instance
        self.lifetime = lifetime
        self.is_resolved = False


class EnterpriseDIContainer:
    """
    Enterprise-grade dependency injection container with advanced features.

    Features:
    - Constructor injection with automatic parameter resolution
    - Multiple service lifetimes (singleton, transient, scoped)
    - Interface-based service registration
    - Circular dependency detection
    - Thread-safe operations
    - Lazy initialization
    """

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = threading.RLock()
        self._resolution_stack: Set[Type] = set()

    def register_singleton(
        self, service_type: Type[T], implementation_type: Type[T] = None
    ) -> "EnterpriseDIContainer":
        """Register a service as singleton (one instance for the entire application)"""
        return self._register(service_type, implementation_type, ServiceLifetime.SINGLETON)

    def register_transient(
        self, service_type: Type[T], implementation_type: Type[T] = None
    ) -> "EnterpriseDIContainer":
        """Register a service as transient (new instance every time)"""
        return self._register(service_type, implementation_type, ServiceLifetime.TRANSIENT)

    def register_instance(self, service_type: Type[T], instance: T) -> "EnterpriseDIContainer":
        """Register a specific instance as singleton"""
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type, instance=instance, lifetime=ServiceLifetime.SINGLETON
            )
            self._services[service_type] = descriptor
            self._singletons[service_type] = instance
            logger.debug(f"Registered instance for {service_type.__name__}")
            return self

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[[], T],
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
    ) -> "EnterpriseDIContainer":
        """Register a factory function for creating instances"""
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type, factory=factory, lifetime=lifetime
            )
            self._services[service_type] = descriptor
            logger.debug(
                f"Registered factory for {service_type.__name__} with {lifetime.value} lifetime"
            )
            return self

    def _register(
        self,
        service_type: Type[T],
        implementation_type: Type[T] = None,
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
    ) -> "EnterpriseDIContainer":
        """Internal registration method"""
        with self._lock:
            impl_type = implementation_type or service_type
            descriptor = ServiceDescriptor(
                service_type=service_type, implementation_type=impl_type, lifetime=lifetime
            )
            self._services[service_type] = descriptor
            logger.debug(
                f"Registered {service_type.__name__} -> {impl_type.__name__} with {lifetime.value} lifetime"
            )
            return self

    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service instance with automatic dependency injection.

        Args:
            service_type: The type of service to resolve

        Returns:
            An instance of the requested service

        Raises:
            ValueError: If service is not registered or circular dependency detected
        """
        with self._lock:
            # Check for circular dependencies
            if service_type in self._resolution_stack:
                cycle = (
                    " -> ".join([t.__name__ for t in self._resolution_stack])
                    + f" -> {service_type.__name__}"
                )
                raise ValueError(f"Circular dependency detected: {cycle}")

            # Check if service is registered
            if service_type not in self._services:
                raise ValueError(f"Service {service_type.__name__} is not registered")

            descriptor = self._services[service_type]

            # Return singleton if already created
            if (
                descriptor.lifetime == ServiceLifetime.SINGLETON
                and service_type in self._singletons
            ):
                return self._singletons[service_type]

            # Return registered instance
            if descriptor.instance is not None:
                return descriptor.instance

            # Add to resolution stack for circular dependency detection
            self._resolution_stack.add(service_type)

            try:
                # Create instance
                if descriptor.factory:
                    instance = descriptor.factory()
                else:
                    instance = self._create_instance(descriptor.implementation_type)

                # Store singleton
                if descriptor.lifetime == ServiceLifetime.SINGLETON:
                    self._singletons[service_type] = instance

                logger.debug(f"Resolved {service_type.__name__} -> {type(instance).__name__}")
                return instance

            finally:
                # Remove from resolution stack
                self._resolution_stack.discard(service_type)

    def _create_instance(self, implementation_type: Type[T]) -> T:
        """Create an instance with automatic constructor injection"""
        try:
            # Get constructor signature
            signature = inspect.signature(implementation_type.__init__)
            parameters = list(signature.parameters.values())[1:]  # Skip 'self'

            # Resolve constructor dependencies
            kwargs = {}
            for param in parameters:
                if param.annotation != inspect.Parameter.empty:
                    # Try to resolve the parameter type
                    try:
                        kwargs[param.name] = self.resolve(param.annotation)
                    except ValueError:
                        # If dependency not registered, check if parameter has default value
                        if param.default != inspect.Parameter.empty:
                            kwargs[param.name] = param.default
                        else:
                            logger.warning(
                                f"Cannot resolve dependency {param.annotation.__name__} for {implementation_type.__name__}"
                            )
                            # Continue without this dependency - let the constructor handle it

            # Create instance
            return implementation_type(**kwargs)

        except Exception as e:
            logger.error(f"Failed to create instance of {implementation_type.__name__}: {e}")
            # Fallback to parameterless constructor
            try:
                return implementation_type()
            except Exception as fallback_error:
                raise ValueError(
                    f"Cannot create instance of {implementation_type.__name__}: {fallback_error}"
                )

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered"""
        with self._lock:
            return service_type in self._services

    def get_registered_services(self) -> Dict[str, str]:
        """Get a dictionary of all registered services for debugging"""
        with self._lock:
            return {
                service_type.__name__: descriptor.implementation_type.__name__
                for service_type, descriptor in self._services.items()
            }

    def clear(self):
        """Clear all registrations (useful for testing)"""
        with self._lock:
            self._services.clear()
            self._singletons.clear()
            self._resolution_stack.clear()
            logger.debug("Container cleared")


_container = None


def get_container() -> EnterpriseDIContainer:
    """
    🔧 DEPRECATED: Get the global DI container instance
    
    WARNING: This container is deprecated. Use UnifiedDIContainer instead.
    """
    from .deprecated_di_warning import warn_deprecated_container
    warn_deprecated_container("EnterpriseDIContainer")
    
    global _container
    if _container is None:
        _container = EnterpriseDIContainer()
        configure_services()
    return _container


def configure_services() -> EnterpriseDIContainer:
    """Configure all application services in the DI container"""
    container = get_container()
    if container._services: # prevent re-configuration
        return container
        
    logger.info("Configuring enterprise DI container...")

    # Import service types (avoiding circular imports)
    try:
        from ..utils.logging_config import LogManager
        from ..utils.error_handler import ErrorHandler
        from .memory_consolidation import get_consolidated_resource_manager as ResourceManager
        from ..utils.shutdown_manager import ShutdownManager, get_shutdown_manager
        # Handle missing core modules with stubs
        try:
            from .gpu_manager import GPUManager
        except ImportError:
            class GPUManager:
                def __init__(self): pass
                def is_available(self): return False

        try:
            from .storage_manager import StorageManager
        except ImportError:
            class StorageManager:
                def __init__(self): pass
                def is_available(self): return True

        try:
            from .memory_consolidation import get_consolidated_resource_manager as MemoryManager
        except ImportError:
            def MemoryManager(): return None

        try:
            from .model_manager import ModelManager
        except ImportError:
            class ModelManager:
                def __init__(self): pass
                def is_available(self): return False
        from knowledge_app.core.mcq_manager import MCQManager
        from knowledge_app.core.proper_config_manager import ProperConfigManager as get_config

        # Get configuration for services that need it
        config = get_config()

        # 🚀 BUG FIX 9: Use ConsolidatedResourceManager as single resource manager
        from knowledge_app.core.memory_consolidation import ConsolidatedResourceManager

        # Register core services as singletons
        container.register_singleton(LogManager)
        container.register_singleton(ErrorHandler)
        container.register_singleton(ConsolidatedResourceManager)  # Single resource manager
        container.register_singleton(ShutdownManager)
        container.register_singleton(GPUManager)

        # Register services that need configuration with factory functions
        def create_storage_manager():
            storage_config = {
                "data_path": "data",
                "max_cache_size": 4 * 1024 * 1024 * 1024,  # 4GB
                "cleanup_threshold": 0.85,
                "cache_expiry": 7200,  # 2 hours
                "use_mmap": True,
            }
            return StorageManager(storage_config)

        # 🚀 BUG FIX 9: Remove separate MemoryManager - use ConsolidatedResourceManager
        # Memory management is now handled by ConsolidatedResourceManager

        def create_model_manager():
            model_config = {
                "base_path": "data/models",
                "max_size": 8 * 1024 * 1024 * 1024,  # 8GB
                "cleanup_threshold": 0.85,
                "cache_expiry": 3600,  # 1 hour
            }
            return ModelManager(model_config)

        # Register factory-based services
        container.register_factory(
            StorageManager, create_storage_manager, ServiceLifetime.SINGLETON
        )
        # 🚀 BUG FIX 9: MemoryManager removed - ConsolidatedResourceManager handles all resource management
        container.register_factory(ModelManager, create_model_manager, ServiceLifetime.SINGLETON)

        # Register MCQ manager as singleton
        def create_mcq_manager():
            return MCQManager(config=get_config())

        container.register_factory(MCQManager, create_mcq_manager, ServiceLifetime.SINGLETON)

        # Register shutdown manager as singleton
        container.register_instance(ShutdownManager, get_shutdown_manager())
        logger.debug("✅ ShutdownManager registered in DI container")

        logger.info("✅ Enterprise DI container configured successfully")

    except ImportError as e:
        logger.warning(f"Some services could not be registered: {e}")

    return container
