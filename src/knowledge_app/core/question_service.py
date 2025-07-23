"""
Question service implementation
"""

# CRITICAL MEMORY FIX: Import only lightweight modules during startup
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance


# CRITICAL MEMORY FIX: Heavy ML imports will be done lazily when question service is first used
def _lazy_import_transformers():
    """Lazy import transformers pipeline to reduce startup memory"""
    try:
        from transformers import pipeline

        return pipeline
    except ImportError:
        logger.warning("Transformers not available, using fallback")
        return None


from .interfaces import IQuestionService, Question, IModelManager
from .model_factory import create_model_manager

logger = logging.getLogger(__name__)


class QuestionService(IQuestionService):
    """Implementation of the question service"""

    def __init__(self, model_manager: Optional[IModelManager] = None):
        """
        Initialize the question service

        Args:
            model_manager: Optional model manager instance
        """
        self._model_manager = model_manager or create_model_manager()
        self._qa_pipeline = None
        self._initialize_nlp()

    def _initialize_nlp(self) -> None:
        """Initialize NLP components"""
        try:
            pipeline = _lazy_import_transformers()
            if pipeline:
                self._qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    device="cuda" if self._model_manager.is_model_loaded() else "cpu",
                )
            else:
                self._qa_pipeline = None
        except Exception as e:
            logger.error(f"Failed to initialize QA pipeline: {e}")
            self._qa_pipeline = None

    def generate_question(self, category: str, difficulty: int) -> Question:
        """
        Generate a new question

        Args:
            category: Question category
            difficulty: Question difficulty level (1-5)

        Returns:
            Generated question

        Raises:
            ValueError: If category or difficulty is invalid
        """
        if not category or not isinstance(category, str):
            raise ValueError("Invalid category")
        if not isinstance(difficulty, int) or difficulty < 1 or difficulty > 5:
            raise ValueError("Difficulty must be between 1 and 5")

        try:
            # Generate question using the model
            prompt = self._create_question_prompt(category, difficulty)
            response = self._model_manager.generate(prompt)

            # Parse response into question and answer
            question_text, answer = self._parse_model_response(response)

            # Create question object
            return Question(
                id=str(uuid.uuid4()),
                text=question_text,
                answer=answer,
                category=category,
                difficulty=difficulty,
                created_at=datetime.now(),
                metadata={
                    "source": "model_generated",
                    "model": self._model_manager.__class__.__name__,
                    "prompt_tokens": len(word_tokenize(prompt)),
                    "response_tokens": len(word_tokenize(response)),
                },
            )

        except Exception as e:
            logger.error(f"Failed to generate question: {e}")
            raise

    def validate_question(self, question: Question) -> bool:
        """
        Validate a question

        Args:
            question: Question to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation
            if not question.text or not question.answer:
                return False

            # Check question format
            if not question.text.strip().endswith("?"):
                return False

            # Check minimum lengths
            if len(word_tokenize(question.text)) < 3:
                return False
            if len(word_tokenize(question.answer)) < 1:
                return False

            # Validate using QA pipeline
            if self._qa_pipeline:
                result = self._qa_pipeline(question=question.text, context=question.answer)
                if result["score"] < 0.5:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating question: {e}")
            return False

    def grade_answer(self, question: Question, user_answer: str) -> float:
        """
        Grade a user's answer

        Args:
            question: The question being answered
            user_answer: User's answer

        Returns:
            Score between 0 and 1
        """
        try:
            if not user_answer:
                return 0.0

            # Normalize answers
            correct = question.answer.lower().strip()
            user = user_answer.lower().strip()

            # Exact match
            if correct == user:
                return 1.0

            # Token comparison
            correct_tokens = set(word_tokenize(correct))
            user_tokens = set(word_tokenize(user))

            # Calculate token overlap
            overlap = len(correct_tokens.intersection(user_tokens))
            total = len(correct_tokens.union(user_tokens))

            if total == 0:
                return 0.0

            # Calculate base score from token overlap
            base_score = overlap / total

            # Apply edit distance penalty
            distance = edit_distance(correct, user)
            max_distance = max(len(correct), len(user))
            distance_score = 1 - (distance / max_distance if max_distance > 0 else 1)

            # Combine scores
            final_score = (base_score * 0.7) + (distance_score * 0.3)

            return min(max(final_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error grading answer: {e}")
            return 0.0

    def get_question_stats(self, question: Question) -> Dict[str, Any]:
        """
        Get statistics for a question

        Args:
            question: Question to analyze

        Returns:
            Dictionary of statistics
        """
        try:
            question_tokens = word_tokenize(question.text)
            answer_tokens = word_tokenize(question.answer)

            return {
                "question_length": len(question.text),
                "answer_length": len(question.answer),
                "question_tokens": len(question_tokens),
                "answer_tokens": len(answer_tokens),
                "created_at": question.created_at.isoformat(),
                "category": question.category,
                "difficulty": question.difficulty,
                "metadata": question.metadata,
            }

        except Exception as e:
            logger.error(f"Error getting question stats: {e}")
            return {}

    def _create_question_prompt(self, category: str, difficulty: int) -> str:
        """Create a prompt for question generation"""
        difficulty_desc = ["very easy", "easy", "moderate", "hard", "very hard"][difficulty - 1]
        return f"Generate a {difficulty_desc} question about {category}. Format: Q: [question] A: [answer]"

    def _parse_model_response(self, response: str) -> tuple[str, str]:
        """Parse model response into question and answer"""
        try:
            # Split into question and answer parts
            parts = response.split("A:")
            if len(parts) != 2:
                raise ValueError("Invalid response format")

            question = parts[0].replace("Q:", "").strip()
            answer = parts[1].strip()

            # Ensure question ends with question mark
            if not question.endswith("?"):
                question += "?"

            return question, answer

        except Exception as e:
            logger.error(f"Error parsing model response: {e}")
            raise ValueError("Failed to parse model response")