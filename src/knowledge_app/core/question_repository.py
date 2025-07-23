"""
Question repository implementation
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from .interfaces import IQuestionRepository, Question

logger = logging.getLogger(__name__)


class QuestionRepository(IQuestionRepository):
    """File-based question repository implementation"""

    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.path.join(os.path.dirname(__file__), "..", "data")
        self.questions_dir = os.path.join(self.base_path, "questions")
        os.makedirs(self.questions_dir, exist_ok=True)

    def add_question(self, question: Question) -> bool:
        """
        Add a question to the repository

        Args:
            question: Question to add

        Returns:
            bool: True if successful
        """
        try:
            # Generate filename from timestamp and question ID
            filename = f"{int(datetime.now().timestamp())}_{question.id}.json"
            filepath = os.path.join(self.questions_dir, filename)

            # Save question to file
            with open(filepath, "w") as f:
                json.dump(question.to_dict(), f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error adding question: {e}")
            return False

    def get_question(self, question_id: str) -> Optional[Question]:
        """
        Get a question by ID

        Args:
            question_id: Question ID

        Returns:
            Question if found, None otherwise
        """
        try:
            # Find file containing question
            for filename in os.listdir(self.questions_dir):
                if filename.endswith(f"_{question_id}.json"):
                    filepath = os.path.join(self.questions_dir, filename)
                    with open(filepath) as f:
                        data = json.load(f)
                        return Question.from_dict(data)

            return None

        except Exception as e:
            logger.error(f"Error getting question: {e}")
            return None

    def get_questions(self, limit: int = None, offset: int = 0) -> List[Question]:
        """
        Get multiple questions

        Args:
            limit: Maximum number of questions to return
            offset: Number of questions to skip

        Returns:
            List of questions
        """
        try:
            questions = []

            # Get all question files
            files = sorted(os.listdir(self.questions_dir))

            # Apply offset and limit
            if offset:
                files = files[offset:]
            if limit:
                files = files[:limit]

            # Load questions
            for filename in files:
                if filename.endswith(".json"):
                    filepath = os.path.join(self.questions_dir, filename)
                    with open(filepath) as f:
                        data = json.load(f)
                        questions.append(Question.from_dict(data))

            return questions

        except Exception as e:
            logger.error(f"Error getting questions: {e}")
            return []

    def update_question(self, question: Question) -> bool:
        """
        Update a question

        Args:
            question: Question to update

        Returns:
            bool: True if successful
        """
        try:
            # Find existing question file
            for filename in os.listdir(self.questions_dir):
                if filename.endswith(f"_{question.id}.json"):
                    filepath = os.path.join(self.questions_dir, filename)

                    # Update question
                    with open(filepath, "w") as f:
                        json.dump(question.to_dict(), f, indent=2)

                    return True

            return False

        except Exception as e:
            logger.error(f"Error updating question: {e}")
            return False

    def delete_question(self, question_id: str) -> bool:
        """
        Delete a question

        Args:
            question_id: ID of question to delete

        Returns:
            bool: True if successful
        """
        try:
            # Find and delete question file
            for filename in os.listdir(self.questions_dir):
                if filename.endswith(f"_{question_id}.json"):
                    filepath = os.path.join(self.questions_dir, filename)
                    os.remove(filepath)
                    return True

            return False

        except Exception as e:
            logger.error(f"Error deleting question: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up repository resources"""
        pass  # No cleanup needed for file-based repository