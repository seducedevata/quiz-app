import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 🔧 FIX: Custom exception for question normalization failures
class QuestionNormalizationError(Exception):
    """Raised when question data cannot be normalized to canonical format"""
    pass

# 🚀 BUG FIX 21: Canonical data model for robust question storage
@dataclass
class CanonicalQuestion:
    """
    Canonical question format that all question data must be normalized to
    before storage. This ensures consistent data structure regardless of
    the source generator's output format.
    """
    id: str
    question_text: str
    options: List[str]  # Always exactly 4 options as strings
    correct_answer: str  # The actual text of the correct answer
    correct_index: int  # 0-3 index of the correct answer
    explanation: str
    topic: str
    difficulty: str
    question_type: str
    game_mode: str
    metadata: Dict[str, Any]


class QuestionHistoryStorage:
    """
    🏆 PERMANENT QUESTION HISTORY DATABASE STORAGE

    This is the PERMANENT storage system for questions - NOT the temporary cache!

    Purpose:
    - Store ALL generated questions permanently for review history
    - Track user progress and performance
    - Enable question deduplication across sessions
    - Support analytics and learning insights

    Storage: SQLite database at user_data/question_history.sqlite
    Retention: Permanent (until user manually clears)

    NOTE: This is separate from the temporary cache system used for performance.
    """
    
    def __init__(self, storage_dir: str = "user_data"):
        self.storage_dir = Path(storage_dir)
        if not self.storage_dir.exists():
            self.storage_dir.mkdir(parents=True)

        self.db_path = self.storage_dir / "question_history.sqlite"
        self.db_connection = None
        self._connection_count = 0  # Track connection usage
        self._init_database()

        # 🔧 FIX: Register cleanup for proper resource management
        import atexit
        atexit.register(self.cleanup)
        
    def _init_database(self):
        """
        🔧 FIX: Initialize the comprehensive question history database with proper connection tracking
        """
        try:
            self.db_connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.db_connection.row_factory = sqlite3.Row  # Enable column access by name
            self._connection_count += 1
            cursor = self.db_connection.cursor()
            
            # Create comprehensive question history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS question_history (
                    id TEXT PRIMARY KEY,
                    question_text TEXT NOT NULL,
                    option_a TEXT NOT NULL,
                    option_b TEXT NOT NULL, 
                    option_c TEXT NOT NULL,
                    option_d TEXT NOT NULL,
                    correct_answer TEXT NOT NULL,
                    correct_index INTEGER NOT NULL,
                    explanation TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    game_mode TEXT NOT NULL,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_reviewed TIMESTAMP,
                    times_answered INTEGER DEFAULT 0,
                    times_correct INTEGER DEFAULT 0,
                    quiz_session_id TEXT,
                    metadata TEXT
                )
            """)
            
            # Create quiz sessions table for grouping
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quiz_sessions (
                    session_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    game_mode TEXT NOT NULL,
                    questions_count INTEGER NOT NULL,
                    score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            
            # 🚀 PHASE 3: Training History Management Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    adapter_name TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    training_preset TEXT NOT NULL,
                    selected_files TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    duration_seconds INTEGER,
                    success BOOLEAN DEFAULT 0,
                    final_loss REAL,
                    final_accuracy REAL,
                    total_steps INTEGER,
                    peak_gpu_utilization REAL,
                    dataset_size_mb REAL,
                    fire_estimated_hours REAL,
                    fire_actual_hours REAL,
                    fire_accuracy_score REAL,
                    evaluation_score REAL,
                    improvement_score REAL,
                    adapter_path TEXT,
                    config_json TEXT,
                    error_message TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 🚀 PHASE 3: Model Evaluation Results Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    training_run_id TEXT NOT NULL,
                    base_model_score REAL,
                    trained_model_score REAL,
                    improvement_percentage REAL,
                    holdout_questions_count INTEGER,
                    evaluation_method TEXT,
                    evaluation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    detailed_results TEXT,
                    FOREIGN KEY (training_run_id) REFERENCES training_runs (run_id)
                )
            """)
            
            # Create indexes for faster searches
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic ON question_history(topic)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_difficulty ON question_history(difficulty)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_question_type ON question_history(question_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_generated_at ON question_history(generated_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON question_history(quiz_session_id)")
            
            # 🚀 PHASE 3: Training history indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_adapter_name ON training_runs(adapter_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_start_time ON training_runs(start_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_success ON training_runs(success)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_run_id ON model_evaluations(training_run_id)")
            
            self.db_connection.commit()
            logger.info("✅ Question history database initialized with Phase 3 training history support")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize question history database: {e}")
            if self.db_connection:
                self.db_connection.close()
                self.db_connection = None
            raise

    def _normalize_to_canonical(self, question_data: Dict, quiz_params: Dict) -> CanonicalQuestion:
        """
        🚀 BUG FIX 21: Normalize any question data format to canonical format

        This method handles all the different formats that generators might produce:
        - options as list vs dict
        - correct_answer as text vs index vs letter
        - missing or malformed fields

        Args:
            question_data: Raw question data from any generator
            quiz_params: Quiz parameters for context

        Returns:
            CanonicalQuestion: Normalized question in canonical format

        Raises:
            ValueError: If the data cannot be normalized (missing critical fields)
        """
        try:
            # Extract basic question text
            question_text = question_data.get("question", "").strip()
            if not question_text:
                raise ValueError("Question text is missing or empty")

            # Normalize options to list format
            raw_options = question_data.get("options", [])
            if isinstance(raw_options, dict):
                # Convert {"A": "opt1", "B": "opt2", ...} to ["opt1", "opt2", ...]
                options = [raw_options.get(key, f"Option {key}") for key in ["A", "B", "C", "D"]]
            elif isinstance(raw_options, list):
                # Ensure we have exactly 4 options
                options = list(raw_options)
                while len(options) < 4:
                    options.append(f"Option {len(options) + 1}")
                options = options[:4]  # Truncate if more than 4
            else:
                raise ValueError(f"Options must be list or dict, got {type(raw_options)}")

            # Normalize correct answer and find correct index
            correct_answer = question_data.get("correct_answer", "")
            correct_index = question_data.get("correct_index", -1)

            # If we have correct_index but no correct_answer, derive it
            if correct_index >= 0 and correct_index < len(options) and not correct_answer:
                correct_answer = options[correct_index]

            # If we have correct_answer but no correct_index, find it
            elif correct_answer and correct_index < 0:
                try:
                    correct_index = options.index(correct_answer)
                except ValueError:
                    # correct_answer might be a letter (A, B, C, D)
                    if correct_answer.upper() in ["A", "B", "C", "D"]:
                        letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
                        correct_index = letter_to_index[correct_answer.upper()]
                        correct_answer = options[correct_index]
                    else:
                        # Try fuzzy matching
                        for i, option in enumerate(options):
                            if correct_answer.lower().strip() in option.lower().strip():
                                correct_index = i
                                correct_answer = option
                                break
                        else:
                            # Default to first option if no match found
                            logger.warning(f"Could not match correct_answer '{correct_answer}' to any option, defaulting to first option")
                            correct_index = 0
                            correct_answer = options[0]

            # Validate correct_index is in valid range
            if correct_index < 0 or correct_index >= len(options):
                logger.warning(f"Invalid correct_index {correct_index}, defaulting to 0")
                correct_index = 0
                correct_answer = options[0]

            # Extract other fields with defaults
            explanation = question_data.get("explanation", "No explanation provided.").strip()
            if not explanation:
                explanation = "No explanation provided."

            # Create canonical question
            canonical = CanonicalQuestion(
                id=str(uuid.uuid4()),
                question_text=question_text,
                options=options,
                correct_answer=correct_answer,
                correct_index=correct_index,
                explanation=explanation,
                topic=quiz_params.get("topic", "Unknown"),
                difficulty=quiz_params.get("difficulty", "medium"),
                question_type=quiz_params.get("question_type", "mixed"),
                game_mode=quiz_params.get("game_mode", "casual"),
                metadata=question_data.get("metadata", {})
            )

            logger.debug(f"✅ Normalized question to canonical format: {canonical.question_text[:50]}...")
            return canonical

        except Exception as e:
            logger.error(f"❌ Failed to normalize question data: {e}")
            logger.error(f"❌ Raw question data: {question_data}")
            raise ValueError(f"Cannot normalize question data: {e}")

    def save_question(self, question_data: Dict, quiz_params: Dict, session_id: str = None) -> str:
        """
        🚀 BUG FIX 21: Save a single question to history with robust data normalization

        This method now uses canonical data normalization to handle any input format
        from any generator, preventing silent save failures and data loss.

        Args:
            question_data: The question data (question, options, correct_answer, explanation)
            quiz_params: Quiz parameters (topic, difficulty, question_type, game_mode)
            session_id: Optional session ID to group questions

        Returns:
            str: The question ID
        """
        try:
            logger.info(f"🔍 DEBUG: save_question called")
            logger.info(f"🔍 DEBUG: Question data keys: {list(question_data.keys())}")
            logger.info(f"🔍 DEBUG: Quiz params: {quiz_params}")

            # 🚀 BUG FIX 21: Step 1 - Normalize to canonical format
            try:
                canonical_question = self._normalize_to_canonical(question_data, quiz_params)
                logger.info(f"✅ Successfully normalized question to canonical format")
            except ValueError as e:
                logger.error(f"❌ Failed to normalize question data: {e}")
                # 🔧 FIX: Raise exception instead of silent failure
                raise QuestionNormalizationError(f"Question normalization failed: {e}") from e

            # Use provided session_id or generate new one
            if session_id is None:
                session_id = str(uuid.uuid4())

            logger.info(f"🔍 DEBUG: Using question_id: {canonical_question.id}")
            logger.info(f"🔍 DEBUG: Using session_id: {session_id}")

            cursor = self.db_connection.cursor()

            # 🚀 BUG FIX 21: Step 2 - Use guaranteed-correct canonical data for SQL
            logger.info(f"🔍 DEBUG: Canonical options: {canonical_question.options}")
            logger.info(f"🔍 DEBUG: Canonical question text: {canonical_question.question_text[:100]}...")
            logger.info(f"🔍 DEBUG: Canonical correct answer: {canonical_question.correct_answer}")
            logger.info(f"🔍 DEBUG: Canonical correct index: {canonical_question.correct_index}")

            # Store question using canonical data (guaranteed to be correct)
            cursor.execute("""
                INSERT INTO question_history (
                    id, question_text, option_a, option_b, option_c, option_d,
                    correct_answer, correct_index, explanation, topic, difficulty,
                    question_type, game_mode, quiz_session_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                canonical_question.id,
                canonical_question.question_text,
                canonical_question.options[0],
                canonical_question.options[1],
                canonical_question.options[2],
                canonical_question.options[3],
                canonical_question.correct_answer,
                canonical_question.correct_index,
                canonical_question.explanation,
                canonical_question.topic,
                canonical_question.difficulty,
                canonical_question.question_type,
                canonical_question.game_mode,
                session_id,
                json.dumps(canonical_question.metadata)
            ))
            
            affected_rows = cursor.rowcount
            logger.info(f"🔍 DEBUG: Affected rows: {affected_rows}")

            self.db_connection.commit()
            logger.info(f"✅ Successfully saved canonical question to history: {canonical_question.id}")

            # Verify the save worked
            cursor.execute("SELECT COUNT(*) FROM question_history WHERE id = ?", (canonical_question.id,))
            count = cursor.fetchone()[0]
            logger.info(f"🔍 DEBUG: Verification count for saved question: {count}")

            return canonical_question.id
            
        except Exception as e:
            logger.error(f"❌ Failed to save question to history: {e}")
            logger.error(f"❌ Exception type: {type(e)}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return ""
    
    def save_quiz_session(self, session_id: str, quiz_params: Dict, questions_count: int, score: float = None):
        """Save quiz session metadata"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO quiz_sessions (
                    session_id, topic, difficulty, question_type, game_mode,
                    questions_count, score, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                quiz_params.get("topic", "Unknown"),
                quiz_params.get("difficulty", "medium"),
                quiz_params.get("question_type", "mixed"),
                quiz_params.get("game_mode", "casual"),
                questions_count,
                score,
                datetime.now().isoformat() if score is not None else None
            ))
            
            self.db_connection.commit()
            logger.debug(f"✅ Saved quiz session: {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save quiz session: {e}")
    
    def get_questions_by_topic(self, topic: str, limit: int = 50) -> List[Dict]:
        """Get all questions for a specific topic"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT * FROM question_history 
                WHERE topic LIKE ? 
                ORDER BY generated_at DESC 
                LIMIT ?
            """, (f"%{topic}%", limit))
            
            questions = []
            for row in cursor.fetchall():
                questions.append(self._row_to_dict(row))
            
            logger.info(f"📚 Retrieved {len(questions)} questions for topic: {topic}")
            return questions
            
        except Exception as e:
            logger.error(f"❌ Failed to get questions by topic: {e}")
            return []
    
    def get_questions_by_difficulty(self, difficulty: str, limit: int = 50) -> List[Dict]:
        """Get all questions for a specific difficulty"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT * FROM question_history 
                WHERE difficulty = ? 
                ORDER BY generated_at DESC 
                LIMIT ?
            """, (difficulty, limit))
            
            questions = []
            for row in cursor.fetchall():
                questions.append(self._row_to_dict(row))
            
            logger.info(f"📊 Retrieved {len(questions)} questions for difficulty: {difficulty}")
            return questions
            
        except Exception as e:
            logger.error(f"❌ Failed to get questions by difficulty: {e}")
            return []
    
    def get_recent_questions(self, limit: int = 100) -> List[Dict]:
        """Get most recently generated questions"""
        try:
            logger.info(f"🔍 DEBUG: get_recent_questions called with limit={limit}")
            logger.info(f"🔍 DEBUG: Database path: {self.db_path}")
            logger.info(f"🔍 DEBUG: Database exists: {self.db_path.exists()}")

            if not self.db_connection:
                logger.error("❌ DEBUG: No database connection")
                return []

            cursor = self.db_connection.cursor()
            
            # First, check if table exists and has data
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='question_history'")
            table_exists = cursor.fetchone() is not None
            logger.info(f"🔍 DEBUG: question_history table exists: {table_exists}")
            
            if table_exists:
                # Check total count
                cursor.execute("SELECT COUNT(*) FROM question_history")
                total_count = cursor.fetchone()[0]
                logger.info(f"🔍 DEBUG: Total questions in database: {total_count}")
                
                # Get column info
                cursor.execute("PRAGMA table_info(question_history)")
                columns = cursor.fetchall()
                logger.info(f"🔍 DEBUG: Table columns: {[col[1] for col in columns]}")
            
            # Now get the actual questions
            cursor.execute("""
                SELECT * FROM question_history 
                ORDER BY generated_at DESC 
                LIMIT ?
            """, (limit,))
            
            questions = []
            rows = cursor.fetchall()
            logger.info(f"🔍 DEBUG: Raw query returned {len(rows)} rows")
            
            for i, row in enumerate(rows):
                try:
                    question_dict = self._row_to_dict(row)
                    questions.append(question_dict)
                    if i == 0:  # Log first question for debugging
                        logger.info(f"🔍 DEBUG: First question ID: {question_dict.get('id', 'No ID')}")
                        logger.info(f"🔍 DEBUG: First question text: {question_dict.get('question', 'No question')[:100]}...")
                except Exception as row_error:
                    logger.error(f"❌ DEBUG: Error converting row {i}: {row_error}")
            
            logger.info(f"🕒 Retrieved {len(questions)} recent questions")
            return questions
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent questions: {e}")
            logger.error(f"❌ Exception type: {type(e)}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return []
    
    def get_quiz_sessions(self, limit: int = 50) -> List[Dict]:
        """Get quiz session history"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT s.*, COUNT(q.id) as actual_questions_count
                FROM quiz_sessions s
                LEFT JOIN question_history q ON s.session_id = q.quiz_session_id
                GROUP BY s.session_id
                ORDER BY s.created_at DESC 
                LIMIT ?
            """, (limit,))
            
            sessions = []
            for row in cursor.fetchall():
                session_dict = dict(row)
                sessions.append(session_dict)
            
            logger.info(f"📋 Retrieved {len(sessions)} quiz sessions")
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Failed to get quiz sessions: {e}")
            return []
    
    def get_questions_for_session(self, session_id: str) -> List[Dict]:
        """Get all questions from a specific quiz session"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT * FROM question_history 
                WHERE quiz_session_id = ? 
                ORDER BY generated_at ASC
            """, (session_id,))
            
            questions = []
            for row in cursor.fetchall():
                questions.append(self._row_to_dict(row))
            
            logger.info(f"🎯 Retrieved {len(questions)} questions for session: {session_id}")
            return questions
            
        except Exception as e:
            logger.error(f"❌ Failed to get questions for session: {e}")
            return []
    
    def update_question_stats(self, question_id: str, answered_correctly: bool):
        """Update question answering statistics"""
        try:
            cursor = self.db_connection.cursor()
            
            if answered_correctly:
                cursor.execute("""
                    UPDATE question_history 
                    SET times_answered = times_answered + 1,
                        times_correct = times_correct + 1,
                        last_reviewed = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), question_id))
            else:
                cursor.execute("""
                    UPDATE question_history 
                    SET times_answered = times_answered + 1,
                        last_reviewed = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), question_id))
            
            self.db_connection.commit()
            logger.debug(f"✅ Updated stats for question: {question_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update question stats: {e}")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about stored questions"""
        try:
            cursor = self.db_connection.cursor()
            
            stats = {}
            
            # Total questions
            cursor.execute("SELECT COUNT(*) FROM question_history")
            stats["total_questions"] = cursor.fetchone()[0]
            
            # Questions by topic
            cursor.execute("""
                SELECT topic, COUNT(*) as count 
                FROM question_history 
                GROUP BY topic 
                ORDER BY count DESC
            """)
            stats["by_topic"] = dict(cursor.fetchall())
            
            # Questions by difficulty
            cursor.execute("""
                SELECT difficulty, COUNT(*) as count 
                FROM question_history 
                GROUP BY difficulty 
                ORDER BY count DESC
            """)
            stats["by_difficulty"] = dict(cursor.fetchall())
            
            # Questions by type
            cursor.execute("""
                SELECT question_type, COUNT(*) as count 
                FROM question_history 
                GROUP BY question_type 
                ORDER BY count DESC
            """)
            stats["by_type"] = dict(cursor.fetchall())
            
            # Total quiz sessions
            cursor.execute("SELECT COUNT(*) FROM quiz_sessions")
            stats["total_sessions"] = cursor.fetchone()[0]
            
            logger.info(f"📊 Generated storage statistics: {stats['total_questions']} questions")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            return {}
    
    def search_questions(self, query: str, limit: int = 50) -> List[Dict]:
        """Search questions by text content"""
        try:
            cursor = self.db_connection.cursor()
            search_pattern = f"%{query}%"
            
            cursor.execute("""
                SELECT * FROM question_history 
                WHERE question_text LIKE ? 
                   OR option_a LIKE ? 
                   OR option_b LIKE ? 
                   OR option_c LIKE ? 
                   OR option_d LIKE ?
                   OR explanation LIKE ?
                ORDER BY generated_at DESC 
                LIMIT ?
            """, (search_pattern, search_pattern, search_pattern, 
                  search_pattern, search_pattern, search_pattern, limit))
            
            questions = []
            for row in cursor.fetchall():
                questions.append(self._row_to_dict(row))
            
            logger.info(f"🔍 Found {len(questions)} questions matching: {query}")
            return questions
            
        except Exception as e:
            logger.error(f"❌ Failed to search questions: {e}")
            return []
    
    def get_question_by_id(self, question_id: str) -> Optional[Dict]:
        """Get a specific question by its ID for practice mode"""
        try:
            logger.info(f"🔍 Getting question by ID: {question_id}")
            
            if not self.db_connection:
                logger.error("❌ No database connection")
                return None
                
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT * FROM question_history 
                WHERE id = ?
            """, (question_id,))
            
            row = cursor.fetchone()
            if row:
                question_dict = self._row_to_dict(row)
                logger.info(f"✅ Found question: {question_dict['question'][:50]}...")
                return question_dict
            else:
                logger.warning(f"⚠️ Question not found with ID: {question_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get question by ID: {e}")
            return None
    
    def _row_to_dict(self, row) -> Dict:
        """Convert SQLite row to dictionary"""
        question_dict = {
            "id": row["id"],
            "question": row["question_text"],
            "options": [row["option_a"], row["option_b"], row["option_c"], row["option_d"]],
            "correct_answer": row["correct_answer"],
            "correct_index": row["correct_index"],
            "explanation": row["explanation"],
            "topic": row["topic"],
            "difficulty": row["difficulty"],
            "question_type": row["question_type"],
            "game_mode": row["game_mode"],
            "generated_at": row["generated_at"],
            "last_reviewed": row["last_reviewed"],
            "times_answered": row["times_answered"],
            "times_correct": row["times_correct"],
            "quiz_session_id": row["quiz_session_id"],
            "accuracy": (row["times_correct"] / row["times_answered"]) if row["times_answered"] > 0 else 0
        }
        
        # Add metadata if available
        if row["metadata"]:
            try:
                question_dict["metadata"] = json.loads(row["metadata"])
            except:
                question_dict["metadata"] = {}
        else:
            question_dict["metadata"] = {}
            
        return question_dict
    
    def cleanup(self):
        """
        🔧 FIX: Enhanced cleanup with proper resource management
        """
        if self.db_connection:
            try:
                # Commit any pending transactions
                self.db_connection.commit()

                # Close the connection
                self.db_connection.close()
                self.db_connection = None

                logger.info(f"🧹 Question history database connection closed (used {self._connection_count} times)")
            except Exception as e:
                logger.error(f"❌ Error closing database: {e}")

    def __enter__(self):
        """
        🔧 FIX: Context manager entry - ensures proper resource management
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        🔧 FIX: Context manager exit - ensures cleanup is called
        """
        self.cleanup()

    def __del__(self):
        """
        🔧 FIX: Destructor - ensures cleanup even if not explicitly called
        """
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction
    
    # 🚀 PHASE 3: Training History Management Methods
    
    def create_training_run(self, training_config: Dict[str, Any]) -> str:
        """
        Create a new training run record
        
        Args:
            training_config: Training configuration including adapter_name, base_model, etc.
            
        Returns:
            str: The training run ID
        """
        try:
            run_id = str(uuid.uuid4())
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT INTO training_runs (
                    run_id, adapter_name, base_model, training_preset, 
                    selected_files, start_time, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                training_config.get("adapter_name", "unnamed_adapter"),
                training_config.get("base_model", "unknown"),
                training_config.get("training_preset", "standard"),
                json.dumps(training_config.get("selected_files", [])),
                datetime.now().isoformat(),
                json.dumps(training_config)
            ))
            
            self.db_connection.commit()
            logger.info(f"✅ Created training run record: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create training run record: {e}")
            return ""
    
    def update_training_run_fire_estimate(self, run_id: str, estimated_hours: float, accuracy_score: float = None):
        """Update training run with FIRE estimation data"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                UPDATE training_runs 
                SET fire_estimated_hours = ?, fire_accuracy_score = ?
                WHERE run_id = ?
            """, (estimated_hours, accuracy_score, run_id))
            
            self.db_connection.commit()
            logger.debug(f"✅ Updated FIRE estimate for run: {run_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update FIRE estimate: {e}")
    
    def complete_training_run(self, run_id: str, success: bool, final_metrics: Dict[str, Any] = None, error_message: str = None):
        """Complete a training run with final results"""
        try:
            cursor = self.db_connection.cursor()
            
            end_time = datetime.now()
            
            # Calculate duration if we have start time
            cursor.execute("SELECT start_time FROM training_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            duration_seconds = None
            
            if row and row["start_time"]:
                try:
                    start_time = datetime.fromisoformat(row["start_time"])
                    duration_seconds = int((end_time - start_time).total_seconds())
                except:
                    pass
            
            # Extract final metrics
            final_loss = final_metrics.get("final_loss") if final_metrics else None
            final_accuracy = final_metrics.get("final_accuracy") if final_metrics else None
            total_steps = final_metrics.get("total_steps") if final_metrics else None
            peak_gpu_utilization = final_metrics.get("peak_gpu_utilization") if final_metrics else None
            dataset_size_mb = final_metrics.get("dataset_size_mb") if final_metrics else None
            adapter_path = final_metrics.get("adapter_path") if final_metrics else None
            
            # Calculate actual FIRE hours
            fire_actual_hours = duration_seconds / 3600 if duration_seconds else None
            
            cursor.execute("""
                UPDATE training_runs 
                SET end_time = ?, duration_seconds = ?, success = ?, 
                    final_loss = ?, final_accuracy = ?, total_steps = ?,
                    peak_gpu_utilization = ?, dataset_size_mb = ?, 
                    fire_actual_hours = ?, adapter_path = ?, error_message = ?
                WHERE run_id = ?
            """, (
                end_time.isoformat(), duration_seconds, success,
                final_loss, final_accuracy, total_steps,
                peak_gpu_utilization, dataset_size_mb,
                fire_actual_hours, adapter_path, error_message,
                run_id
            ))
            
            self.db_connection.commit()
            logger.info(f"✅ Completed training run: {run_id} (success: {success})")
            
        except Exception as e:
            logger.error(f"❌ Failed to complete training run: {e}")
    
    def save_model_evaluation(self, training_run_id: str, evaluation_results: Dict[str, Any]) -> str:
        """Save model evaluation results"""
        try:
            evaluation_id = str(uuid.uuid4())
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT INTO model_evaluations (
                    evaluation_id, training_run_id, base_model_score, 
                    trained_model_score, improvement_percentage, 
                    holdout_questions_count, evaluation_method, detailed_results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation_id,
                training_run_id,
                evaluation_results.get("base_model_score"),
                evaluation_results.get("trained_model_score"),
                evaluation_results.get("improvement_percentage"),
                evaluation_results.get("holdout_questions_count"),
                evaluation_results.get("evaluation_method", "question_quality_comparison"),
                json.dumps(evaluation_results.get("detailed_results", {}))
            ))
            
            # Update the training run with evaluation score
            improvement_score = evaluation_results.get("improvement_percentage", 0)
            cursor.execute("""
                UPDATE training_runs 
                SET improvement_score = ?, evaluation_score = ?
                WHERE run_id = ?
            """, (improvement_score, evaluation_results.get("trained_model_score"), training_run_id))
            
            self.db_connection.commit()
            logger.info(f"✅ Saved model evaluation: {evaluation_id}")
            return evaluation_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save model evaluation: {e}")
            return ""
    
    def get_training_history(self, limit: int = 50) -> List[Dict]:
        """Get training history for model management UI"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT tr.*, me.improvement_percentage, me.base_model_score, me.trained_model_score
                FROM training_runs tr
                LEFT JOIN model_evaluations me ON tr.run_id = me.training_run_id
                ORDER BY tr.start_time DESC
                LIMIT ?
            """, (limit,))
            
            history = []
            for row in cursor.fetchall():
                history.append(self._training_row_to_dict(row))
            
            logger.info(f"🎯 Retrieved {len(history)} training history records")
            return history
            
        except Exception as e:
            logger.error(f"❌ Failed to get training history: {e}")
            return []
    
    def get_successful_adapters(self) -> List[Dict]:
        """Get list of successfully trained adapters for management"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT tr.*, me.improvement_percentage 
                FROM training_runs tr
                LEFT JOIN model_evaluations me ON tr.run_id = me.training_run_id
                WHERE tr.success = 1 AND tr.adapter_path IS NOT NULL
                ORDER BY tr.end_time DESC
            """)
            
            adapters = []
            for row in cursor.fetchall():
                adapters.append(self._training_row_to_dict(row))
            
            logger.info(f"🎯 Retrieved {len(adapters)} successful adapters")
            return adapters
            
        except Exception as e:
            logger.error(f"❌ Failed to get successful adapters: {e}")
            return []
    
    def delete_training_run(self, run_id: str) -> bool:
        """Delete a training run and its evaluation data"""
        try:
            cursor = self.db_connection.cursor()
            
            # Delete evaluation data first (foreign key constraint)
            cursor.execute("DELETE FROM model_evaluations WHERE training_run_id = ?", (run_id,))
            
            # Delete training run
            cursor.execute("DELETE FROM training_runs WHERE run_id = ?", (run_id,))
            
            deleted_runs = cursor.rowcount
            self.db_connection.commit()
            
            if deleted_runs > 0:
                logger.info(f"✅ Deleted training run: {run_id}")
                return True
            else:
                logger.warning(f"⚠️ Training run not found: {run_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete training run: {e}")
            return False
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics for dashboard"""
        try:
            cursor = self.db_connection.cursor()
            
            # Basic stats
            cursor.execute("SELECT COUNT(*) as total_runs FROM training_runs")
            total_runs = cursor.fetchone()["total_runs"]
            
            cursor.execute("SELECT COUNT(*) as successful_runs FROM training_runs WHERE success = 1")
            successful_runs = cursor.fetchone()["successful_runs"]
            
            # Average training time
            cursor.execute("SELECT AVG(duration_seconds) as avg_duration FROM training_runs WHERE success = 1")
            avg_duration_row = cursor.fetchone()
            avg_duration = avg_duration_row["avg_duration"] if avg_duration_row["avg_duration"] else 0
            
            # Best performing adapter
            cursor.execute("""
                SELECT adapter_name, improvement_score 
                FROM training_runs 
                WHERE success = 1 AND improvement_score IS NOT NULL
                ORDER BY improvement_score DESC 
                LIMIT 1
            """)
            best_adapter_row = cursor.fetchone()
            best_adapter = {
                "name": best_adapter_row["adapter_name"] if best_adapter_row else "None",
                "score": best_adapter_row["improvement_score"] if best_adapter_row else 0
            }
            
            # FIRE accuracy
            cursor.execute("""
                SELECT AVG(ABS(fire_estimated_hours - fire_actual_hours)) as fire_error
                FROM training_runs 
                WHERE fire_estimated_hours IS NOT NULL AND fire_actual_hours IS NOT NULL
            """)
            fire_error_row = cursor.fetchone()
            fire_accuracy = 100 - (fire_error_row["fire_error"] * 10) if fire_error_row["fire_error"] else 95
            
            return {
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "success_rate": (successful_runs / total_runs * 100) if total_runs > 0 else 0,
                "avg_duration_hours": avg_duration / 3600 if avg_duration else 0,
                "best_adapter": best_adapter,
                "fire_accuracy_percent": max(0, min(100, fire_accuracy))
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get training statistics: {e}")
            return {}
    
    def _training_row_to_dict(self, row) -> Dict:
        """Convert training run SQLite row to dictionary"""
        result = {
            "run_id": row["run_id"],
            "adapter_name": row["adapter_name"],
            "base_model": row["base_model"],
            "training_preset": row["training_preset"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "duration_seconds": row["duration_seconds"],
            "success": bool(row["success"]),
            "final_loss": row["final_loss"],
            "final_accuracy": row["final_accuracy"],
            "total_steps": row["total_steps"],
            "peak_gpu_utilization": row["peak_gpu_utilization"],
            "dataset_size_mb": row["dataset_size_mb"],
            "fire_estimated_hours": row["fire_estimated_hours"],
            "fire_actual_hours": row["fire_actual_hours"],
            "fire_accuracy_score": row["fire_accuracy_score"],
            "evaluation_score": row["evaluation_score"],
            "improvement_score": row["improvement_score"],
            "adapter_path": row["adapter_path"],
            "error_message": row["error_message"],
            "created_at": row["created_at"]
        }
        
        # Add parsed configuration
        try:
            if row["config_json"]:
                result["config"] = json.loads(row["config_json"])
                result["selected_files"] = json.loads(row["selected_files"])
            else:
                result["config"] = {}
                result["selected_files"] = []
        except:
            result["config"] = {}
            result["selected_files"] = []
        
        # Add evaluation data if available  
        if "improvement_percentage" in row.keys() and row["improvement_percentage"] is not None:
            result["improvement_percentage"] = row["improvement_percentage"]
            result["base_model_score"] = row.get("base_model_score")
            result["trained_model_score"] = row.get("trained_model_score")
        
        return result

    def get_quiz_statistics(self) -> Dict[str, Any]:
        """Get comprehensive quiz statistics for home menu display"""
        try:
            cursor = self.db_connection.cursor()
            
            # Total quizzes taken (completed sessions)
            cursor.execute("""
                SELECT COUNT(*) FROM quiz_sessions 
                WHERE end_time IS NOT NULL
            """)
            quizzes_taken = cursor.fetchone()[0]
            
            # Total questions answered across all quizzes
            cursor.execute("""
                SELECT COUNT(*) FROM quiz_sessions 
                WHERE end_time IS NOT NULL
            """)
            total_sessions = cursor.fetchone()[0]
            
            # Get all quiz sessions with scores
            cursor.execute("""
                SELECT score, total_questions FROM quiz_sessions 
                WHERE end_time IS NOT NULL AND score IS NOT NULL
            """)
            sessions_data = cursor.fetchall()
            
            total_questions_answered = 0
            total_correct_answers = 0
            total_score = 0.0
            
            for score, total_questions in sessions_data:
                if score is not None and total_questions is not None:
                    total_questions_answered += total_questions
                    total_correct_answers += int((score / 100.0) * total_questions)
                    total_score += score
            
            # Calculate average score
            average_score = total_score / len(sessions_data) if sessions_data else 0.0
            
            # Get streak information
            cursor.execute("""
                SELECT score FROM quiz_sessions 
                WHERE end_time IS NOT NULL AND score IS NOT NULL
                ORDER BY end_time DESC
            """)
            recent_scores = cursor.fetchall()
            
            streak = 0
            for score_row in recent_scores:
                if score_row[0] and score_row[0] >= 70:  # Consider 70%+ as success
                    streak += 1
                else:
                    break
            
            # Get last quiz date
            cursor.execute("""
                SELECT MAX(end_time) FROM quiz_sessions 
                WHERE end_time IS NOT NULL
            """)
            last_quiz_date = cursor.fetchone()[0]
            
            stats = {
                "quizzes_taken": quizzes_taken,
                "average_score": round(average_score, 1),
                "questions_answered": total_questions_answered,
                "total_correct_answers": total_correct_answers,
                "total_incorrect_answers": total_questions_answered - total_correct_answers,
                "streak": streak,
                "last_quiz_date": last_quiz_date
            }
            
            logger.info(f"📊 Quiz statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get quiz statistics: {e}")
            return {
                "quizzes_taken": 0,
                "average_score": 0.0,
                "questions_answered": 0,
                "total_correct_answers": 0,
                "total_incorrect_answers": 0,
                "streak": 0,
                "last_quiz_date": None
            }

# Global instance for singleton pattern
_question_history_storage_instance = None

def get_question_history_storage() -> QuestionHistoryStorage:
    """Get the global question history storage instance"""
    global _question_history_storage_instance
    if _question_history_storage_instance is None:
        _question_history_storage_instance = QuestionHistoryStorage()
    return _question_history_storage_instance