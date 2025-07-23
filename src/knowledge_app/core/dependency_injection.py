"""
Dependency Injection Container

This module provides a simple dependency injection container for managing
application services and their dependencies.
"""

from typing import Dict, Type, Any, Callable, TypeVar, cast, Optional, Union, Set
import threading
import inspect
import functools

T = TypeVar("T")


class ServiceNotFoundException(Exception):
    """Exception raised when a requested service is not found in the container"""

    pass


class DependencyContainer:
    """
    🔧 DEPRECATED: A dependency injection container that manages service registrations and instances.

    WARNING: This container is deprecated. Use UnifiedDIContainer instead.

    This container supports:
    - Registering service types with their implementations
    - Singleton or transient service lifetimes
    - Factory function registration
    - Automatic dependency resolution
    - Thread-safe singleton instantiation
    - Named service registration
    """

    def __init__(self):
        """Initialize a new dependency container"""
        from .deprecated_di_warning import warn_deprecated_container
        warn_deprecated_container("DependencyContainer (dependency_injection.py)")
        
        self._registrations: Dict[Type, Dict[str, Dict]] = {}
        self._singletons: Dict[Type, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        service_type: Type[T],
        implementation: Type[T] = None,
        singleton: bool = False,
        instance: T = None,
        name: str = None,
    ) -> None:
        """
        Register a service with the container.

        Args:
            service_type: The interface or abstract type to register
            implementation: The concrete implementation class (None if providing an instance)
            singleton: Whether this service should be a singleton
            instance: An existing instance to use (implies singleton=True)
            name: Optional name for the registration (for multiple implementations of same interface)
        """
        with self._lock:
            # Initialize registration dictionary for this type if needed
            if service_type not in self._registrations:
                self._registrations[service_type] = {}
                self._singletons[service_type] = {}

            reg_name = name or "default"

            if instance is not None:
                # If an instance is provided, always register as singleton
                self._singletons[service_type][reg_name] = instance
                self._registrations[service_type][reg_name] = {
                    "implementation": type(instance),
                    "singleton": True,
                    "factory": None,
                    "instance_type": type(instance),  # Store the exact type for comparison
                }
            else:
                # Register the type
                self._registrations[service_type][reg_name] = {
                    "implementation": implementation or service_type,
                    "singleton": singleton,
                    "factory": None,
                    "instance_type": implementation,  # Store the exact implementation type
                }

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[[], T],
        singleton: bool = False,
        name: str = None,
    ) -> None:
        """
        Register a factory function for creating service instances.

        Args:
            service_type: The type to register
            factory: Function that creates instances of the service
            singleton: Whether this service should be a singleton
            name: Optional name for the registration
        """
        with self._lock:
            # Initialize registration dictionary for this type if needed
            if service_type not in self._registrations:
                self._registrations[service_type] = {}
                self._singletons[service_type] = {}

            reg_name = name or "default"

            self._registrations[service_type][reg_name] = {
                "implementation": None,
                "singleton": singleton,
                "factory": factory,
                "instance_type": None,  # Will be determined when instance is created
            }

    def get(self, service_type: Type[T], name: str = None) -> T:
        """
        Get an instance of the requested service.

        Args:
            service_type: The type of service to retrieve
            name: Optional name of the registration to retrieve

        Returns:
            An instance of the requested service

        Raises:
            ServiceNotFoundException: If the service is not registered
        """
        with self._lock:
            reg_name = name or "default"

            # Check if type is registered
            if not self.has(service_type, name):
                raise ServiceNotFoundException(
                    f"Service {service_type.__name__}{f' with name {name}' if name else ''} not registered"
                )

            registration = self._registrations[service_type][reg_name]

            # Check if it's a singleton and we have an instance
            if registration["singleton"] and reg_name in self._singletons[service_type]:
                return cast(T, self._singletons[service_type][reg_name])

            # Create a new instance
            instance = self._create_instance(service_type, registration)

            # Update the instance_type if it was created by a factory
            if registration["factory"] is not None:
                registration["instance_type"] = type(instance)

            # Store the instance if it's a singleton
            if registration["singleton"]:
                self._singletons[service_type][reg_name] = instance

            return instance

    def has(self, service_type: Type, name: str = None) -> bool:
        """
        Check if a service type is registered.

        Args:
            service_type: The service type to check
            name: Optional name of the registration

        Returns:
            bool: True if registered, False otherwise
        """
        reg_name = name or "default"
        return service_type in self._registrations and reg_name in self._registrations[service_type]

    def _create_instance(self, service_type: Type[T], registration: Dict) -> T:
        """
        Create a new instance of the service.

        Args:
            service_type: The type of service to create
            registration: The registration info for the service

        Returns:
            A new instance of the service
        """
        if registration["factory"] is not None:
            # Use factory function if provided
            return cast(T, registration["factory"]())

        implementation = registration["implementation"]

        # Get constructor parameters
        try:
            signature = inspect.signature(implementation.__init__)
        except (ValueError, TypeError):
            # If __init__ is not accessible or is a built-in class
            return cast(T, implementation())

        params = {}

        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue

            # For *args and **kwargs parameters, just pass empty values
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue

            # Try to resolve parameter from container
            param_type = param.annotation
            if param_type is not inspect.Parameter.empty and param_type in self._registrations:
                params[param_name] = self.get(param_type)
            elif param.default is not inspect.Parameter.empty:
                # Use default value if available
                params[param_name] = param.default
            else:
                # For parameters we can't resolve, pass None (could be handled by the implementation)
                params[param_name] = None

        # Create the instance with resolved parameters
        return cast(T, implementation(**params))

    def clear(self) -> None:
        """Clear all registrations and singleton instances"""
        with self._lock:
            self._registrations.clear()
            self._singletons.clear()
