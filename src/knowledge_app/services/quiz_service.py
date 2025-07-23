"""
🔧 Quiz Service

This service handles all quiz-related business logic, extracted from the PythonBridge
to follow the Single Responsibility Principle.

CRITICAL FIX: Eliminates the "God Object" anti-pattern by separating concerns:
- Quiz state management
- Question generation and buffering
- Answer submission and scoring
- Quiz navigation
"""

import logging
import threading
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class QuizService:
    """
    🔧 FIX: Dedicated service for quiz management
    
    This service handles all quiz-related operations that were previously
    mixed into the PythonBridge "God Object".
    """
    
    def __init__(self, mcq_manager=None):
        self.mcq_manager = mcq_manager
        
        # Quiz state
        self.current_quiz: Optional[Dict[str, Any]] = None
        self.quiz_questions: List[Dict[str, Any]] = []
        self.current_question_index = 0
        
        # Enhanced navigation state management
        self.highest_question_answered = -1
        self.navigation_history: List[str] = []
        
        # Question buffering
        self.question_buffer: List[Dict[str, Any]] = []
        self.question_history: List[Dict[str, Any]] = []
        self.buffer_size = 5
        self.min_buffer_threshold = 2
        
        # Session tracking
        self.current_session_id: Optional[str] = None
        self.session_lock = threading.RLock()
        
        # Thread safety
        self.quiz_lock = threading.RLock()
        
        logger.info("🔧 QuizService initialized")
    
    def start_quiz(self, quiz_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a new quiz with the given parameters
        
        Args:
            quiz_params: Quiz configuration parameters
            
        Returns:
            Dict with quiz start result
        """
        try:
            with self.quiz_lock:
                # Generate new session ID
                import time
                new_session_id = f"quiz_{int(time.time() * 1000)}_{id(self)}"
                
                with self.session_lock:
                    self.current_session_id = new_session_id
                
                # Initialize quiz state
                self.current_quiz = {
                    'topic': quiz_params.get('topic', 'General Knowledge'),
                    'difficulty': quiz_params.get('difficulty', 'medium'),
                    'num_questions': quiz_params.get('num_questions', 5),
                    'mode': quiz_params.get('mode', 'auto'),
                    'submode': quiz_params.get('submode', 'mixed'),
                    'score': 0,
                    'total_answered': 0,
                    'start_time': time.time(),
                    'session_id': new_session_id
                }
                
                # Reset navigation state
                self.current_question_index = 0
                self.highest_question_answered = -1
                self.navigation_history = []
                self.quiz_questions = []
                self.question_buffer = []
                
                logger.info(f"🚀 Quiz started: {self.current_quiz['topic']} ({new_session_id})")
                
                return {
                    'success': True,
                    'session_id': new_session_id,
                    'quiz_config': self.current_quiz
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to start quiz: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def submit_answer(self, answer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit an answer for the current question
        
        Args:
            answer_data: Answer submission data
            
        Returns:
            Dict with submission result
        """
        try:
            with self.quiz_lock:
                if not self.current_quiz:
                    return {'success': False, 'error': 'No active quiz'}
                
                # Check for double submission (idempotency)
                if (self.current_question_index < len(self.question_history) and 
                    self.question_history[self.current_question_index].get("answered")):
                    logger.warning("⚠️ Question already answered - preventing double submission")
                    return {'success': False, 'error': 'Question already answered'}
                
                # Process answer
                is_correct = self._check_answer_correctness(answer_data)
                
                if is_correct:
                    self.current_quiz['score'] += 1
                
                self.current_quiz['total_answered'] += 1
                
                # Update navigation state
                self.highest_question_answered = max(self.highest_question_answered, self.current_question_index)
                self.navigation_history.append(f"answered_q{self.current_question_index}")
                
                # Mark question as answered
                if self.current_question_index < len(self.question_history):
                    self.question_history[self.current_question_index]["answered"] = True
                
                logger.info(f"✅ Answer submitted: {'CORRECT' if is_correct else 'INCORRECT'}")
                logger.info(f"📊 Score: {self.current_quiz['score']}/{self.current_quiz['total_answered']}")
                
                return {
                    'success': True,
                    'correct': is_correct,
                    'score': self.current_quiz['score'],
                    'total_answered': self.current_quiz['total_answered']
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to submit answer: {e}")
            return {'success': False, 'error': str(e)}
    
    def navigate_to_question(self, direction: str) -> Dict[str, Any]:
        """
        Navigate to a different question
        
        Args:
            direction: 'next', 'previous', 'next_new'
            
        Returns:
            Dict with navigation result
        """
        try:
            with self.quiz_lock:
                if not self.current_quiz:
                    return {'success': False, 'error': 'No active quiz'}
                
                old_index = self.current_question_index
                
                if direction == 'next':
                    if self.current_question_index < len(self.question_history) - 1:
                        self.current_question_index += 1
                        self.navigation_history.append(f"next_to_q{self.current_question_index}")
                elif direction == 'previous':
                    if self.current_question_index > 0:
                        self.current_question_index -= 1
                        self.navigation_history.append(f"prev_to_q{self.current_question_index}")
                elif direction == 'next_new':
                    # Navigate to next unanswered question
                    next_new_index = self.highest_question_answered + 1
                    if next_new_index < len(self.quiz_questions):
                        self.current_question_index = next_new_index
                        self.navigation_history.append(f"next_new_to_q{next_new_index}")
                    else:
                        return {'success': False, 'error': 'No more new questions available'}
                
                if old_index != self.current_question_index:
                    logger.info(f"🧭 Navigated from Q{old_index + 1} to Q{self.current_question_index + 1}")
                    return {
                        'success': True,
                        'new_index': self.current_question_index,
                        'question_data': self._get_current_question_data()
                    }
                else:
                    return {'success': False, 'error': 'Cannot navigate in that direction'}
                
        except Exception as e:
            logger.error(f"❌ Navigation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_quiz_status(self) -> Dict[str, Any]:
        """Get current quiz status"""
        try:
            with self.quiz_lock:
                if not self.current_quiz:
                    return {'active': False}
                
                return {
                    'active': True,
                    'session_id': self.current_session_id,
                    'topic': self.current_quiz['topic'],
                    'difficulty': self.current_quiz['difficulty'],
                    'current_question': self.current_question_index + 1,
                    'total_questions': len(self.quiz_questions),
                    'score': self.current_quiz['score'],
                    'total_answered': self.current_quiz['total_answered'],
                    'highest_answered': self.highest_question_answered + 1,
                    'buffer_size': len(self.question_buffer)
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get quiz status: {e}")
            return {'active': False, 'error': str(e)}
    
    def end_quiz(self) -> Dict[str, Any]:
        """End the current quiz"""
        try:
            with self.quiz_lock:
                if not self.current_quiz:
                    return {'success': False, 'error': 'No active quiz'}
                
                # Calculate final results
                final_score = self.current_quiz['score']
                total_questions = self.current_quiz['total_answered']
                percentage = (final_score / total_questions * 100) if total_questions > 0 else 0
                
                quiz_summary = {
                    'topic': self.current_quiz['topic'],
                    'difficulty': self.current_quiz['difficulty'],
                    'final_score': final_score,
                    'total_questions': total_questions,
                    'percentage': percentage,
                    'session_id': self.current_session_id,
                    'duration': time.time() - self.current_quiz['start_time']
                }
                
                # Clear quiz state
                self.current_quiz = None
                self.current_session_id = None
                self.quiz_questions = []
                self.current_question_index = 0
                self.highest_question_answered = -1
                self.navigation_history = []
                
                logger.info(f"🏁 Quiz ended: {final_score}/{total_questions} ({percentage:.1f}%)")
                
                return {
                    'success': True,
                    'summary': quiz_summary
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to end quiz: {e}")
            return {'success': False, 'error': str(e)}
    
    def _check_answer_correctness(self, answer_data: Dict[str, Any]) -> bool:
        """Check if the submitted answer is correct"""
        try:
            # Implementation depends on answer format
            # This is a simplified version
            submitted_answer = answer_data.get('selected_option', '')
            correct_answer = answer_data.get('correct_answer', '')
            
            return submitted_answer.strip().lower() == correct_answer.strip().lower()
            
        except Exception as e:
            logger.error(f"❌ Error checking answer correctness: {e}")
            return False
    
    def _get_current_question_data(self) -> Optional[Dict[str, Any]]:
        """Get data for the current question"""
        try:
            if (0 <= self.current_question_index < len(self.question_history)):
                question_data = self.question_history[self.current_question_index].copy()
                question_data["question_number"] = self.current_question_index + 1
                question_data["total_questions"] = len(self.question_history)
                return question_data
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting current question data: {e}")
            return None
