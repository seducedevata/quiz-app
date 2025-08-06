"""
Base MCQ Generator class for the Knowledge App
Provides a unified interface for different MCQ generation backends
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MCQGenerator(ABC):
    """Abstract base class for MCQ generators"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the MCQ generator"""
        pass

    @abstractmethod
    def generate_mcq(
        self, topic: str, context: str = "", num_questions: int = 1
    ) -> List[Dict[str, Any]]:
        """Generate MCQ questions for a given topic and context"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the generator is available and ready to use"""
        pass

    def cleanup(self):
        """Clean up resources"""
        pass


class DefaultMCQGenerator(MCQGenerator):
    """Default implementation - NO SAMPLE QUESTIONS! Forces real AI generation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # NO MORE SAMPLE QUESTIONS!

    def initialize(self) -> bool:
        """Initialize the default generator"""
        self.is_initialized = True
        logger.info("Default MCQ Generator initialized - will force real AI generation")
        return True

    def generate_mcq(
        self, topic: str, context: str = "", num_questions: int = 1
    ) -> List[Dict[str, Any]]:
        """NO HARDCODED QUESTIONS! This should never be used - forces proper AI generation."""
        logger.error(f"❌ DEFAULT GENERATOR CALLED - This means AI generation failed!")
        logger.error(f"❌ Topic: {topic}, Context: {context}")
        logger.error("🚨 NO HARDCODED CONTENT AVAILABLE - Configure proper AI models")
        raise Exception(f"Default generator called for '{topic}' - no hardcoded content available")

    def is_available(self) -> bool:
        """Default generator should not be used"""
        return False


# Note: MCQ generation is now handled by UnifiedInferenceManager
# This file only contains the base interface and default fallback
