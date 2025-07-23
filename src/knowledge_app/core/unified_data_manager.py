"""
🛡️ CRITICAL ARCHITECTURE FIX #14: Unified Data Manager
This solves the "Schizophrenic Database" problem by providing a single,
consistent interface for all quiz and question data operations.

The system previously had:
- quiz_storage.py: Simple JSON file-based storage  
- question_history_storage.py: Complex SQLite database storage

This creates confusion, data fragmentation, and inconsistent access patterns.
The unified data manager provides:
- Single source of truth for all quiz data
- Consistent data formats and validation
- Unified backup and migration strategies  
- Clear separation of concerns between storage backends
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import uuid

logger = logging.getLogger(__name__)


@dataclass
class CanonicalQuestion:
    """
    🛡️ CRITICAL DATA INTEGRITY FIX #15: Canonical question format
    
    This eliminates the need for _normalize_to_canonical function by
    enforcing a single, consistent data format from the start.
    """
    id: str
    question_text: str
    options: List[str]  # Always a list, never a dict
    correct_answer_index: int  # Always an index, never a letter or text
    explanation: str
    difficulty: str
    question_type: str
    topic: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        """Validate data integrity on creation"""
        if not self.question_text or not self.question_text.strip():
            raise ValueError("Question text cannot be empty")
        
        if not isinstance(self.options, list) or len(self.options) < 2:
            raise ValueError("Options must be a list with at least 2 items")
        
        if not isinstance(self.correct_answer_index, int):
            raise ValueError("Correct answer must be an integer index")
        
        if self.correct_answer_index < 0 or self.correct_answer_index >= len(self.options):
            raise ValueError(f"Correct answer index {self.correct_answer_index} out of range for {len(self.options)} options")
        
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    @classmethod
    def from_legacy_format(cls, legacy_data: Dict[str, Any], topic: str = "", metadata: Optional[Dict[str, Any]] = None) -> 'CanonicalQuestion':
        """
        Convert legacy MCQ formats to canonical format
        This replaces the fragile _normalize_to_canonical function with type-safe conversion
        """
        try:
            # Extract question text
            question_text = legacy_data.get('question', '').strip()
            if not question_text:
                raise ValueError("No question text found in legacy data")
            
            # Handle options - convert dict or list to consistent list format
            options_raw = legacy_data.get('options', [])
            if isinstance(options_raw, dict):
                # Convert {"A": "option1", "B": "option2"} to ["option1", "option2"]
                sorted_keys = sorted(options_raw.keys())
                options = [options_raw[key] for key in sorted_keys]
            elif isinstance(options_raw, list):
                options = list(options_raw)
            else:
                raise ValueError(f"Invalid options format: {type(options_raw)}")
            
            # Handle correct answer - convert letter/text to index
            correct_raw = legacy_data.get('correct_answer') or legacy_data.get('correct', '')
            if isinstance(correct_raw, str):
                if correct_raw.upper() in ['A', 'B', 'C', 'D', 'E']:
                    # Letter format (A, B, C, D)
                    correct_answer_index = ord(correct_raw.upper()) - ord('A')
                elif correct_raw.isdigit():
                    # String number format
                    correct_answer_index = int(correct_raw)
                elif isinstance(options_raw, dict) and correct_raw in options_raw:
                    # Answer is a key in the options dict
                    correct_answer_index = list(options_raw.keys()).index(correct_raw)
                else:
                    # Try to find the answer text in the options list
                    try:
                        correct_answer_index = options.index(correct_raw)
                    except ValueError:
                        raise ValueError(f"Cannot match correct answer '{correct_raw}' to options")
            elif isinstance(correct_raw, int):
                correct_answer_index = correct_raw
            else:
                raise ValueError(f"Invalid correct answer format: {correct_raw} ({type(correct_raw)})")
            
            # Extract other fields with defaults
            explanation = legacy_data.get('explanation', 'No explanation provided')
            difficulty = legacy_data.get('difficulty', 'medium')
            question_type = legacy_data.get('question_type', 'mixed')
            
            # Generate ID if not present
            question_id = legacy_data.get('id', str(uuid.uuid4()))
            
            return cls(
                id=question_id,
                question_text=question_text,
                options=options,
                correct_answer_index=correct_answer_index,
                explanation=explanation,
                difficulty=difficulty,
                question_type=question_type,
                topic=topic,
                metadata=metadata or {}
            )
            
        except Exception as e:
            logger.error(f"Failed to convert legacy question data: {e}")
            logger.error(f"Legacy data: {legacy_data}")
            raise ValueError(f"Cannot convert legacy question format: {e}")


class UnifiedDataManager:
    """
    🛡️ CRITICAL ARCHITECTURE FIX #14: Single source of truth for all quiz data
    
    This eliminates the confusion between quiz_storage.py and question_history_storage.py
    by providing one unified interface that manages both storage backends transparently.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread safety for concurrent access
        self._lock = threading.RLock()
        
        # Initialize both storage backends
        self.db_path = self.data_dir / "unified_quiz_database.sqlite"
        self.json_backup_dir = self.data_dir / "quiz_backups"
        self.json_backup_dir.mkdir(exist_ok=True)
        
        self._init_database()
        
        logger.info("🛡️ UnifiedDataManager initialized with single source of truth")
    
    def _init_database(self):
        """Initialize the unified database schema"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create unified questions table with canonical schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS canonical_questions (
                    id TEXT PRIMARY KEY,
                    question_text TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    correct_answer_index INTEGER NOT NULL,
                    explanation TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    
                    -- Validation constraints
                    CHECK (question_text != ''),
                    CHECK (correct_answer_index >= 0),
                    CHECK (difficulty IN ('easy', 'medium', 'hard', 'expert')),
                    CHECK (question_type IN ('conceptual', 'numerical', 'mixed'))
                )
            """)
            
            # Create quiz sessions table for tracking user performance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quiz_sessions (
                    session_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    total_questions INTEGER NOT NULL DEFAULT 0,
                    correct_answers INTEGER NOT NULL DEFAULT 0
                )
            """)
            
            # Create question attempts table for detailed analytics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS question_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    user_answer_index INTEGER,
                    is_correct BOOLEAN NOT NULL,
                    time_taken_seconds REAL,
                    attempted_at TEXT NOT NULL,
                    
                    FOREIGN KEY (question_id) REFERENCES canonical_questions (id),
                    FOREIGN KEY (session_id) REFERENCES quiz_sessions (session_id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_topic ON canonical_questions (topic)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_difficulty ON canonical_questions (difficulty)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_type ON canonical_questions (question_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_session ON question_attempts (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_question ON question_attempts (question_id)")
            
            conn.commit()
            logger.info("🛡️ Unified database schema initialized")
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper error handling and cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def store_question(self, question: CanonicalQuestion) -> bool:
        """
        Store a question in the unified database with automatic backup
        
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            with self._lock:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Store in primary database
                    cursor.execute("""
                        INSERT OR REPLACE INTO canonical_questions 
                        (id, question_text, options_json, correct_answer_index, explanation, 
                         difficulty, question_type, topic, metadata_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        question.id,
                        question.question_text,
                        json.dumps(question.options),
                        question.correct_answer_index,
                        question.explanation,
                        question.difficulty,
                        question.question_type,
                        question.topic,
                        json.dumps(question.metadata or {}),
                        question.created_at
                    ))
                    
                    conn.commit()
                    
                    # Create JSON backup
                    self._backup_question_to_json(question)
                    
                    logger.debug(f"📝 Stored question {question.id} successfully")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to store question {question.id}: {e}")
            return False
    
    def get_questions(self, topic: Optional[str] = None, difficulty: Optional[str] = None, 
                     question_type: Optional[str] = None, limit: Optional[int] = None) -> List[CanonicalQuestion]:
        """
        Retrieve questions with optional filtering
        
        Args:
            topic: Filter by topic (optional)
            difficulty: Filter by difficulty (optional)  
            question_type: Filter by question type (optional)
            limit: Maximum number of questions to return (optional)
            
        Returns:
            List[CanonicalQuestion]: List of matching questions
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Build query dynamically based on filters
                where_clauses = []
                params = []
                
                if topic:
                    where_clauses.append("topic = ?")
                    params.append(topic)
                
                if difficulty:
                    where_clauses.append("difficulty = ?")
                    params.append(difficulty)
                
                if question_type:
                    where_clauses.append("question_type = ?")
                    params.append(question_type)
                
                where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                limit_sql = f"LIMIT {limit}" if limit else ""
                
                query = f"""
                    SELECT * FROM canonical_questions 
                    {where_sql}
                    ORDER BY created_at DESC
                    {limit_sql}
                """
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to CanonicalQuestion objects
                questions = []
                for row in rows:
                    questions.append(CanonicalQuestion(
                        id=row['id'],
                        question_text=row['question_text'],
                        options=json.loads(row['options_json']),
                        correct_answer_index=row['correct_answer_index'],
                        explanation=row['explanation'],
                        difficulty=row['difficulty'],
                        question_type=row['question_type'],
                        topic=row['topic'],
                        metadata=json.loads(row['metadata_json'] or '{}'),
                        created_at=row['created_at']
                    ))
                
                logger.debug(f"📚 Retrieved {len(questions)} questions")
                return questions
                
        except Exception as e:
            logger.error(f"Failed to retrieve questions: {e}")
            return []
    
    def _backup_question_to_json(self, question: CanonicalQuestion):
        """Create JSON backup for redundancy and migration"""
        try:
            backup_file = self.json_backup_dir / f"question_{question.id}.json"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(question), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to create JSON backup for question {question.id}: {e}")
    
    def migrate_legacy_data(self):
        """
        🛡️ CRITICAL MIGRATION: Convert legacy quiz_storage.py and question_history_storage.py data
        to the unified canonical format
        """
        logger.info("🔄 Starting legacy data migration...")
        
        # Migrate from old quiz_storage JSON files
        self._migrate_json_files()
        
        # Migrate from old question_history_storage database
        self._migrate_legacy_database()
        
        logger.info("✅ Legacy data migration completed")
    
    def _migrate_json_files(self):
        """Migrate individual JSON quiz files"""
        try:
            # Look for old quiz files
            old_quiz_dir = Path("data/quizzes")
            if old_quiz_dir.exists():
                for json_file in old_quiz_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            legacy_quiz = json.load(f)
                        
                        # Convert to canonical format
                        if 'questions' in legacy_quiz:
                            for q_data in legacy_quiz['questions']:
                                try:
                                    canonical_q = CanonicalQuestion.from_legacy_format(
                                        q_data, 
                                        topic=legacy_quiz.get('topic', 'Unknown'),
                                        metadata={'source': 'migrated_json', 'original_file': str(json_file)}
                                    )
                                    self.store_question(canonical_q)
                                except Exception as e:
                                    logger.warning(f"Failed to migrate question from {json_file}: {e}")
                        
                        logger.info(f"📦 Migrated quiz file: {json_file}")
                        
                    except Exception as e:
                        logger.error(f"Failed to migrate quiz file {json_file}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to migrate JSON files: {e}")
    
    def _migrate_legacy_database(self):
        """Migrate from old question_history_storage database"""
        try:
            # Look for old database
            old_db_path = Path("data/question_history.db")
            if old_db_path.exists():
                with sqlite3.connect(old_db_path) as old_conn:
                    old_conn.row_factory = sqlite3.Row
                    cursor = old_conn.cursor()
                    
                    # Get all questions from old format
                    cursor.execute("SELECT * FROM questions")
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        try:
                            # Convert old format to canonical
                            legacy_data = {
                                'question': row.get('question_text', ''),
                                'options': json.loads(row.get('options_json', '[]')),
                                'correct_answer': row.get('correct_answer', ''),
                                'explanation': row.get('explanation', ''),
                                'difficulty': row.get('difficulty', 'medium'),
                                'question_type': row.get('question_type', 'mixed')
                            }
                            
                            canonical_q = CanonicalQuestion.from_legacy_format(
                                legacy_data,
                                topic=row.get('topic', 'Unknown'),
                                metadata={'source': 'migrated_db', 'original_id': row.get('id')}
                            )
                            
                            self.store_question(canonical_q)
                            
                        except Exception as e:
                            logger.warning(f"Failed to migrate database question {row.get('id')}: {e}")
                
                logger.info(f"📦 Migrated legacy database: {old_db_path}")
                
        except Exception as e:
            logger.error(f"Failed to migrate legacy database: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get unified statistics across all data"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get question counts by category
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_questions,
                        COUNT(DISTINCT topic) as unique_topics,
                        COUNT(DISTINCT difficulty) as difficulty_levels,
                        COUNT(DISTINCT question_type) as question_types
                    FROM canonical_questions
                """)
                
                stats = dict(cursor.fetchone())
                
                # Get breakdown by topic
                cursor.execute("""
                    SELECT topic, COUNT(*) as count 
                    FROM canonical_questions 
                    GROUP BY topic 
                    ORDER BY count DESC
                """)
                
                stats['by_topic'] = dict(cursor.fetchall())
                
                # Get breakdown by difficulty
                cursor.execute("""
                    SELECT difficulty, COUNT(*) as count 
                    FROM canonical_questions 
                    GROUP BY difficulty
                """)
                
                stats['by_difficulty'] = dict(cursor.fetchall())
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


# Global instance for easy access
_unified_data_manager = None


def get_unified_data_manager() -> UnifiedDataManager:
    """Get global unified data manager instance"""
    global _unified_data_manager
    if _unified_data_manager is None:
        _unified_data_manager = UnifiedDataManager()
    return _unified_data_manager
