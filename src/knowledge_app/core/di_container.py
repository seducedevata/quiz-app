"""
Dependency Injection Container for Knowledge App
"""

from typing import Dict, Any, Type, TypeVar, Optional
from .interfaces import IQuestionService, IModelManager, ITrainingService, IQuestionRepository
from .question_service import QuestionService
from .model_manager_service import ModelManagerService
# from training.core.training_service import TrainingService  # TODO: Create training service
try:
    from training.core.training_service import TrainingService
except ImportError:
    # Stub implementation for missing training service
    class TrainingService:
        def __init__(self):
            pass
        def is_available(self):
            return False
from .question_repository import QuestionRepository
from .memory_consolidation import get_consolidated_resource_manager as MemoryManager
from .memory_consolidation import get_consolidated_resource_manager as ResourceManager

T = TypeVar("T")


class DIContainer:
    """
    🔧 DEPRECATED: Dependency Injection Container for managing application services
    
    WARNING: This container is deprecated. Use UnifiedDIContainer instead.
    """

    def __init__(self):
        from .deprecated_di_warning import warn_deprecated_container
        warn_deprecated_container("DIContainer")
        
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}

    def register(self, interface: Type[T], implementation: Type[T]) -> None:
        """
        Register a service implementation

        Args:
            interface: The interface type
            implementation: The implementation type
        """
        self._services[interface] = implementation

    def register_singleton(self, interface: Type[T], instance: T) -> None:
        """
        Register a singleton instance

        Args:
            interface: The interface type
            instance: The singleton instance
        """
        self._singletons[interface] = instance

    def resolve(self, interface: Type[T]) -> Optional[T]:
        """
        Resolve a service implementation

        Args:
            interface: The interface to resolve

        Returns:
            The resolved instance or None if not found
        """
        # Check singletons first
        if interface in self._singletons:
            return self._singletons[interface]

        # Check registered services
        if interface in self._services:
            implementation = self._services[interface]
            instance = implementation()
            return instance

        return None

    @classmethod
    def configure_default(cls) -> "DIContainer":
        """
        Configure the default container with standard services

        Returns:
            Configured DIContainer instance
        """
        container = cls()

        # Register core services
        container.register(IQuestionService, QuestionService)
        container.register(IModelManager, ModelManagerService)
        container.register(ITrainingService, TrainingService)
        container.register(IQuestionRepository, QuestionRepository)

        # Register singletons
        container.register_singleton(MemoryManager, MemoryManager())
        container.register_singleton(ResourceManager, ResourceManager())

        return container
