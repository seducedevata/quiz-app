"""
🔧 CRITICAL FIX for Bug 37: Unified Fallback Manager

This module provides a single, coherent strategy for handling AI generation failures
across the entire application, eliminating conflicting fallback methods.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Types of generation failures"""
    NETWORK_ERROR = "network_error"
    MODEL_UNAVAILABLE = "model_unavailable"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    RATE_LIMITED = "rate_limited"
    AUTHENTICATION_ERROR = "auth_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    UNKNOWN_ERROR = "unknown_error"

class FallbackStrategy(Enum):
    """🔧 FIX: Enhanced fallback strategies for comprehensive error handling"""
    RETRY_WITH_SIMPLER_PROMPT = "retry_simple"
    SWITCH_TO_DIFFERENT_MODEL = "switch_model"
    SWITCH_TO_ONLINE_MODE = "switch_online"
    SWITCH_TO_OFFLINE_MODE = "switch_offline"
    USE_PLACEHOLDER_CONTENT = "use_placeholder"
    FAIL_GRACEFULLY = "fail_graceful"

class FallbackResult:
    """Standardized fallback result"""
    def __init__(self, success: bool, content: Optional[Dict[str, Any]] = None, 
                 message: str = "", strategy_used: Optional[FallbackStrategy] = None):
        self.success = success
        self.content = content
        self.message = message
        self.strategy_used = strategy_used
        self.timestamp = time.time()

class UnifiedFallbackManager:
    """
    🔧 CRITICAL FIX for Bug 37: Single, coherent fallback strategy
    
    Eliminates conflicting fallback methods by providing a unified approach
    to handling AI generation failures across the entire application.
    """
    
    _instance: Optional["UnifiedFallbackManager"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._fallback_attempts = {}  # Track attempts per topic
        self._max_attempts = 3
        
        logger.info("🔧 UnifiedFallbackManager initialized - single fallback strategy active")
    
    def handle_generation_failure(self, topic: str, difficulty: str, 
                                 error: Exception, context: Dict[str, Any] = None) -> FallbackResult:
        """
        🔧 UNIFIED: Handle AI generation failure with consistent strategy
        
        This is the single entry point for all fallback handling in the application.
        """
        try:
            logger.warning(f"🔄 Handling generation failure for '{topic}' ({difficulty}): {error}")
            
            # Track attempts
            attempt_key = f"{topic}_{difficulty}"
            attempts = self._fallback_attempts.get(attempt_key, 0) + 1
            self._fallback_attempts[attempt_key] = attempts
            
            # Choose strategy based on attempt count and error type
            if attempts == 1:
                # First failure: Try simpler approach
                return self._retry_with_simpler_prompt(topic, difficulty, error, context)
            elif attempts == 2:
                # Second failure: Graceful degradation with educational content
                return self._graceful_degradation(topic, difficulty, error, context)
            else:
                # Final failure: Emergency educational content
                return self._emergency_educational_content(topic, difficulty, error, context)
                
        except Exception as fallback_error:
            logger.error(f"❌ Fallback manager itself failed: {fallback_error}")
            return self._absolute_emergency_fallback(topic, difficulty)
    
    def _retry_with_simpler_prompt(self, topic: str, difficulty: str, 
                                  error: Exception, context: Dict[str, Any]) -> FallbackResult:
        """Strategy 1: Suggest retry with simpler prompt"""
        logger.info(f"🔄 Strategy 1: Suggesting simpler prompt retry for '{topic}'")
        
        return FallbackResult(
            success=False,
            message=f"AI generation failed for '{topic}'. Retrying with simpler approach...",
            strategy_used=FallbackStrategy.RETRY_WITH_SIMPLER
        )
    
    def _graceful_degradation(self, topic: str, difficulty: str, 
                             error: Exception, context: Dict[str, Any]) -> FallbackResult:
        """Strategy 2: Provide educational fallback content"""
        logger.info(f"🔄 Strategy 2: Graceful degradation for '{topic}'")
        
        # Create educational fallback question
        safe_topic = self._sanitize_topic(topic)
        
        fallback_question = {
            "question": f"Which of the following best describes an important aspect of {safe_topic}?",
            "options": {
                "A": f"It involves fundamental principles and theoretical frameworks",
                "B": f"It requires systematic study and analytical thinking", 
                "C": f"It has practical applications in various fields",
                "D": f"All of the above are correct"
            },
            "correct": "D",
            "explanation": f"The study of {safe_topic} encompasses fundamental principles, systematic analysis, and practical applications, making all the options correct.",
            "metadata": {
                "generation_method": "graceful_degradation",
                "ai_generation_failed": True,
                "fallback_strategy": "educational_content",
                "original_error": str(error)[:100],
                "note": "This is educational fallback content generated when AI models were unavailable"
            }
        }
        
        return FallbackResult(
            success=True,
            content=fallback_question,
            message=f"AI generation failed. Providing educational fallback content for '{topic}'.",
            strategy_used=FallbackStrategy.GRACEFUL_DEGRADATION
        )
    
    def _emergency_educational_content(self, topic: str, difficulty: str, 
                                     error: Exception, context: Dict[str, Any]) -> FallbackResult:
        """Strategy 3: Emergency educational content as last resort"""
        logger.warning(f"🚨 Strategy 3: Emergency educational content for '{topic}'")
        
        safe_topic = self._sanitize_topic(topic)
        
        emergency_question = {
            "question": f"What is a key characteristic of studying {safe_topic}?",
            "options": {
                "A": f"It requires understanding of core concepts",
                "B": f"It involves critical thinking and analysis", 
                "C": f"It builds foundational knowledge",
                "D": f"All of the above"
            },
            "correct": "D",
            "explanation": f"Studying {safe_topic} involves understanding core concepts, critical thinking, and building foundational knowledge.",
            "metadata": {
                "generation_method": "emergency_educational",
                "ai_generation_failed": True,
                "fallback_strategy": "emergency_content",
                "attempts": self._fallback_attempts.get(f"{topic}_{difficulty}", 0),
                "original_error": str(error)[:100],
                "note": "This is emergency educational content - AI models were unavailable after multiple attempts"
            }
        }
        
        return FallbackResult(
            success=True,
            content=emergency_question,
            message=f"Multiple AI generation failures. Providing emergency educational content for '{topic}'.",
            strategy_used=FallbackStrategy.EMERGENCY_CONTENT
        )
    
    def _absolute_emergency_fallback(self, topic: str, difficulty: str) -> FallbackResult:
        """Absolute last resort when even fallback manager fails"""
        logger.error(f"🚨 ABSOLUTE EMERGENCY: Fallback manager failed for '{topic}'")
        
        return FallbackResult(
            success=False,
            message=f"Critical system failure. Unable to generate content for '{topic}'. Please try again later.",
            strategy_used=FallbackStrategy.FAIL_WITH_MESSAGE
        )
    
    def _sanitize_topic(self, topic: str) -> str:
        """Sanitize topic for safe use in educational content"""
        if not topic or not isinstance(topic, str):
            return "general knowledge"
        
        # Clean and make educational
        clean_topic = topic.strip().lower()
        
        # Educational mappings
        educational_mappings = {
            'atoms': 'atomic structure and chemistry',
            'quantum': 'quantum mechanics and physics',
            'python': 'Python programming and computer science',
            'math': 'mathematics and mathematical concepts',
            'history': 'historical events and analysis',
            'biology': 'biological systems and life sciences',
            'physics': 'physical laws and phenomena',
            'chemistry': 'chemical reactions and molecular science'
        }
        
        # Check for direct mappings
        for key, value in educational_mappings.items():
            if key in clean_topic:
                return value
        
        # Default educational framing
        return f"the academic study of {clean_topic}" if clean_topic else "general knowledge"
    
    def reset_attempts(self, topic: str = None, difficulty: str = None):
        """Reset fallback attempts for a topic or all topics"""
        if topic and difficulty:
            attempt_key = f"{topic}_{difficulty}"
            if attempt_key in self._fallback_attempts:
                del self._fallback_attempts[attempt_key]
                logger.info(f"🔄 Reset fallback attempts for '{topic}' ({difficulty})")
        else:
            self._fallback_attempts.clear()
            logger.info("🔄 Reset all fallback attempts")

# Global instance
_unified_fallback_manager = None

def get_unified_fallback_manager() -> UnifiedFallbackManager:
    """Get the global unified fallback manager instance"""
    global _unified_fallback_manager
    if _unified_fallback_manager is None:
        _unified_fallback_manager = UnifiedFallbackManager()
    return _unified_fallback_manager
