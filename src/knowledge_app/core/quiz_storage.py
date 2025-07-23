import sqlite3
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from ..config import AppConfig
import os
from pathlib import Path
from .interfaces import Quiz, Question
import uuid

logger = logging.getLogger(__name__)


class QuizStorage:
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        if not self.storage_dir.exists():
            self.storage_dir.mkdir(parents=True)

        self.db_connection = None
        # Initialize the database
        self._init_db()

    def _init_db(self):
        """Initialize the database with required tables"""
        try:
            self.db_connection = sqlite3.connect(self.storage_dir / "quiz_database.sqlite")
            cursor = self.db_connection.cursor()

            # Create quizzes table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS quizzes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deleted_at TIMESTAMP DEFAULT NULL
                )
            """
            )

            # Create questions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    quiz_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    options TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deleted_at TIMESTAMP DEFAULT NULL,
                    metadata TEXT,
                    FOREIGN KEY (quiz_id) REFERENCES quizzes(id)
                )
            """
            )

            self.db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            if self.db_connection:
                self.db_connection.close()
                self.db_connection = None
            raise

    def save_quiz(self, quiz) -> bool:
        """Save a quiz to the storage

        Args:
            quiz: The quiz to save

        Returns:
            bool: True if save was successful
        """
        try:
            quiz_id = getattr(quiz, "id", str(uuid.uuid4()))
            quiz_file = self.storage_dir / f"{quiz_id}.json"

            # Convert quiz to serializable dict
            questions_data = []
            try:
                for q in quiz.questions:
                    questions_data.append(
                        {
                            "question": getattr(q, "question", getattr(q, "text", "")),
                            "answer": getattr(q, "answer", ""),
                            "options": getattr(q, "options", []),
                            "metadata": getattr(q, "metadata", {}),
                        }
                    )
            except AttributeError as e:
                logger.error(f"Failed to save quiz: {e}")
                return False

            quiz_data = {
                "id": quiz_id,
                "title": getattr(quiz, "title", getattr(quiz, "name", "Untitled Quiz")),
                "questions": questions_data,
            }

            with open(quiz_file, "w") as f:
                json.dump(quiz_data, f)

            return True

        except Exception as e:
            logger.error(f"Failed to save quiz: {e}")
            return False

    def store_quiz(self, quiz_data: Dict) -> str:
        """Store a new quiz in the database"""
        try:
            quiz_id = str(uuid.uuid4())
            cursor = self.db_connection.cursor()

            # Insert quiz
            cursor.execute(
                """
                INSERT INTO quizzes (id, title, created_at)
                VALUES (?, ?, ?)
            """,
                (quiz_id, quiz_data.get("title", "Untitled Quiz"), datetime.now().isoformat()),
            )

            # Insert questions
            for question in quiz_data.get("questions", []):
                options_json = json.dumps(question.get("options", {}))
                metadata_json = json.dumps(question.get("metadata", {}))

                cursor.execute(
                    """
                    INSERT INTO questions (
                        quiz_id, question, answer, options,
                        created_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        quiz_id,
                        question.get("question", ""),
                        question.get("answer", ""),
                        options_json,
                        datetime.now().isoformat(),
                        metadata_json,
                    ),
                )

            self.db_connection.commit()
            return quiz_id

        except Exception as e:
            logger.error(f"Failed to store quiz: {e}")
            raise

    def load_quiz(self, quiz_id: str) -> Optional[Quiz]:
        """Load a quiz from storage

        Args:
            quiz_id: ID of the quiz to load

        Returns:
            Optional[Quiz]: The loaded quiz or None if not found
        """
        try:
            quiz_file = self.storage_dir / f"{quiz_id}.json"

            if not quiz_file.exists():
                logger.warning(f"Quiz file not found: {quiz_id}")
                return None

            with open(quiz_file, "r") as f:
                quiz_data = json.load(f)

            # Create Question objects
            questions = []
            for q_data in quiz_data.get("questions", []):
                question = Question(
                    text=q_data.get("question", ""),
                    options=q_data.get("options", []),
                    correct_answer=(
                        q_data.get("options", []).index(q_data.get("answer", ""))
                        if q_data.get("answer") in q_data.get("options", [])
                        else 0
                    ),
                    explanation=q_data.get("metadata", {}).get("explanation", ""),
                )
                questions.append(question)

            # Create Quiz object
            quiz = Quiz(
                name=quiz_data.get("title", "Untitled Quiz"),
                questions=questions,
                description=quiz_data.get("description", ""),
            )

            # Add the id attribute to match what tests expect
            quiz.id = quiz_id
            quiz.title = quiz_data.get("title", "Untitled Quiz")

            return quiz

        except Exception as e:
            logger.error(f"Failed to load quiz: {e}")
            return None

    def delete_quiz(self, quiz_id: str) -> bool:
        """Delete a quiz from storage

        Args:
            quiz_id: ID of the quiz to delete

        Returns:
            bool: True if deletion was successful
        """
        try:
            quiz_file = self.storage_dir / f"{quiz_id}.json"

            if not quiz_file.exists():
                logger.warning(f"Quiz file not found for deletion: {quiz_id}")
                return False

            os.remove(quiz_file)
            return True

        except Exception as e:
            logger.error(f"Failed to delete quiz: {e}")
            return False

    def list_quizzes(self) -> List[Quiz]:
        """List all quizzes in storage

        Returns:
            List[Quiz]: List of quiz objects
        """
        try:
            quizzes = []
            for quiz_file in self.storage_dir.glob("*.json"):
                if quiz_file.name == "quiz_database.sqlite":
                    continue

                try:
                    with open(quiz_file, "r") as f:
                        quiz_data = json.load(f)

                    # Create a simple Quiz object with just the metadata
                    quiz = type(
                        "QuizInfo",
                        (),
                        {
                            "id": quiz_data.get("id", quiz_file.stem),
                            "title": quiz_data.get("title", "Untitled Quiz"),
                            "question_count": len(quiz_data.get("questions", [])),
                        },
                    )

                    quizzes.append(quiz)
                except Exception as e:
                    logger.error(f"Error reading quiz file {quiz_file}: {e}")

            return quizzes

        except Exception as e:
            logger.error(f"Failed to list quizzes: {e}")
            return []

    def cleanup(self):
        """Clean up resources"""
        if self.db_connection:
            try:
                self.db_connection.close()
                self.db_connection = None
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")