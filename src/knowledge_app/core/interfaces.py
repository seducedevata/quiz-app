"""
Core interfaces for the Knowledge App

This module contains the interfaces that define the core abstractions
used throughout the application. Classes can implement these interfaces
to provide concrete implementations of these abstractions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class DataStorageInterface(ABC):
    """Interface for data storage operations"""

    @abstractmethod
    def save_data(self, collection: str, key: str, data: Any) -> bool:
        """Save data to storage

        Args:
            collection: The collection/category to save under
            key: The key to save the data with
            data: The data to save

        Returns:
            bool: True if save was successful
        """
        pass

    @abstractmethod
    def load_data(self, collection: str, key: str) -> Any:
        """Load data from storage

        Args:
            collection: The collection/category to load from
            key: The key to load data for

        Returns:
            Any: The loaded data, or None if not found
        """
        pass

    @abstractmethod
    def delete_data(self, collection: str, key: str) -> bool:
        """Delete data from storage

        Args:
            collection: The collection/category to delete from
            key: The key to delete

        Returns:
            bool: True if deletion was successful
        """
        pass

    @abstractmethod
    def list_keys(self, collection: str) -> List[str]:
        """List all keys in a collection

        Args:
            collection: The collection/category to list keys from

        Returns:
            List[str]: List of keys in the collection
        """
        pass


class ConfigurationInterface(ABC):
    """Interface for configuration management"""

    @abstractmethod
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value

        Args:
            key: The configuration key
            default: The default value to return if key not found

        Returns:
            Any: The configuration value
        """
        pass

    @abstractmethod
    def set_value(self, key: str, value: Any) -> None:
        """Set a configuration value

        Args:
            key: The configuration key
            value: The value to set
        """
        pass

    @abstractmethod
    def save(self) -> bool:
        """Save the configuration

        Returns:
            bool: True if save was successful
        """
        pass

    @abstractmethod
    def load(self) -> bool:
        """Load the configuration

        Returns:
            bool: True if load was successful
        """
        pass


class Model(ABC):
    """Interface for ML models"""

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make a prediction using the model

        Args:
            input_data: The input data for prediction

        Returns:
            Any: The prediction result
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the model

        Returns:
            str: The model name
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the model's serializable state

        Returns:
            Dict[str, Any]: The model state as a dictionary
        """
        pass


from dataclasses import dataclass
from datetime import datetime


@dataclass
class Question:
    """Represents a quiz question"""

    id: str
    text: str
    answer: str
    category: str
    difficulty: int
    created_at: datetime
    metadata: Dict[str, Any]
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None


class Quiz:
    """Represents a quiz with questions"""

    def __init__(self, name: str, questions: List[Question], description: Optional[str] = None):
        self.name = name
        self.questions = questions
        self.description = description


class IModelManager(ABC):
    """Interface for model management operations"""

    @abstractmethod
    def load_model(self, model_name: str) -> Optional[Model]:
        """Load a model by name

        Args:
            model_name: The name of the model to load

        Returns:
            Optional[Model]: The loaded model or None if not found
        """
        pass

    @abstractmethod
    def list_available_models(self) -> List[str]:
        """List all available models

        Returns:
            List[str]: List of model names
        """
        pass


class ModelManagerProtocol(IModelManager):
    """Extended interface for model management with additional capabilities"""

    @abstractmethod
    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get metadata about a model

        Args:
            model_name: The name of the model

        Returns:
            Dict[str, Any]: Model metadata
        """
        pass


class IQuestionService(ABC):
    """Interface for question generation service"""

    @abstractmethod
    def generate_question(self, category: str, difficulty: int) -> Question:
        """Generate a new question

        Args:
            category: Question category
            difficulty: Question difficulty level (1-5)

        Returns:
            Generated question
        """
        pass

    @abstractmethod
    def validate_question(self, question: Question) -> bool:
        """Validate a question

        Args:
            question: Question to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def grade_answer(self, question: Question, user_answer: str) -> float:
        """Grade a user's answer

        Args:
            question: The question being answered
            user_answer: User's answer

        Returns:
            Score between 0 and 1
        """
        pass

    @abstractmethod
    def get_question_stats(self, question: Question) -> Dict[str, Any]:
        """Get statistics for a question

        Args:
            question: Question to analyze

        Returns:
            Dictionary of statistics
        """
        pass