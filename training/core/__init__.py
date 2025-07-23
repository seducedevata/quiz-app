"""
Core Training Components

Contains the main training logic and orchestration components.
"""

# Core training components - Only import what actually exists
# TODO: Create actual implementations for these training components

# For now, use the implementations from src/knowledge_app/core/
try:
    from src.knowledge_app.core.training_orchestrator import TrainingOrchestrator
    from src.knowledge_app.core.training_manager import TrainingManager
    from src.knowledge_app.core.golden_path_trainer import GoldenPathTrainer
    from src.knowledge_app.core.quiz_service import TrainingService
except ImportError:
    # Fallback stubs if imports fail
    class TrainingOrchestrator:
        def __init__(self, *args, **kwargs): pass
    class TrainingManager:
        def __init__(self, *args, **kwargs): pass
    class GoldenPathTrainer:
        def __init__(self, *args, **kwargs): pass
    class TrainingService:
        def __init__(self, *args, **kwargs): pass

# Placeholder classes for missing components
class TrainingController:
    def __init__(self, *args, **kwargs): pass
class TrainingWorker:
    def __init__(self, *args, **kwargs): pass
class TrainingThread:
    def __init__(self, *args, **kwargs): pass
class TrainingCallbacks:
    def __init__(self, *args, **kwargs): pass
class TrainingMetrics:
    def __init__(self, *args, **kwargs): pass
class TrainingEstimator:
    def __init__(self, *args, **kwargs): pass
class TrainingDataProcessor:
    def __init__(self, *args, **kwargs): pass
class TrainingManagement:
    def __init__(self, *args, **kwargs): pass
class AutoTrainingManager:
    def __init__(self, *args, **kwargs): pass

__all__ = [
    "TrainingOrchestrator",
    "TrainingManager",
    "TrainingController", 
    "TrainingService",
    "TrainingWorker",
    "TrainingThread",
    "TrainingCallbacks",
    "TrainingMetrics",
    "TrainingEstimator",
    "TrainingDataProcessor",
    "TrainingManagement",
    "AutoTrainingManager",
    "GoldenPathTrainer"
] 