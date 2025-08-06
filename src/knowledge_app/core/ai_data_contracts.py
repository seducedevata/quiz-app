"""
🛡️ AI Data Contracts - Strict validation for AI-generated data

This module provides Pydantic models for enforcing strict data contracts
on all AI-generated content to ensure consistency, security, and reliability.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
import re
import json


class QuestionType(str, Enum):
    """Valid question types"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    NUMERICAL = "numerical"
    SHORT_ANSWER = "short_answer"
    FILL_IN_BLANK = "fill_in_blank"


class DifficultyLevel(str, Enum):
    """Valid difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class QuestionContract(BaseModel):
    """🛡️ Strict validation for AI-generated questions"""
    
    question: str = Field(..., min_length=10, max_length=2000)
    question_type: QuestionType
    difficulty: DifficultyLevel
    options: List[str] = Field(..., min_items=2, max_items=5)
    correct_answer: Union[int, str, bool] = Field(...)
    explanation: str = Field(..., min_length=20, max_length=2000)
    topic: str = Field(..., min_length=2, max_length=100)
    estimated_time: int = Field(..., ge=30, le=3600)  # seconds
    
    @validator('question')
    def validate_question(cls, v):
        """Sanitize and validate question text"""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>'"\x00-\x1f\x7f-\x9f]', '', v.strip())
        
        if len(sanitized) < 10:
            raise ValueError("Question too short")
        
        # Check for prompt injection attempts
        dangerous_patterns = [
            r'ignore\s+previous',
            r'forget\s+instructions',
            r'system\s+prompt',
            r'admin\s+override',
            r'debug\s+mode',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                raise ValueError(f"Potential prompt injection detected: {pattern}")
        
        return sanitized
    
    @validator('options')
    def validate_options(cls, v, values):
        """Validate options based on question type"""
        if 'question_type' not in values:
            return v
        
        q_type = values['question_type']
        
        if q_type == QuestionType.TRUE_FALSE:
            if len(v) != 2:
                raise ValueError("True/False questions must have exactly 2 options")
            if not any(opt.lower() in ['true', 'false', 'yes', 'no'] for opt in v):
                raise ValueError("True/False questions must have boolean options")
        
        elif q_type == QuestionType.MULTIPLE_CHOICE:
            if len(v) < 2 or len(v) > 5:
                raise ValueError("Multiple choice questions must have 2-5 options")
        
        # Sanitize all options
        sanitized = []
        for opt in v:
            if not opt or not opt.strip():
                raise ValueError("Option cannot be empty")
            sanitized.append(re.sub(r'[<>'"\x00-\x1f\x7f-\x9f]', '', opt.strip()))
        
        return sanitized
    
    @validator('correct_answer')
    def validate_correct_answer(cls, v, values):
        """Validate correct answer based on question type"""
        if 'options' not in values or 'question_type' not in values:
            return v
        
        options = values['options']
        q_type = values['question_type']
        
        if q_type == QuestionType.TRUE_FALSE:
            if not isinstance(v, bool):
                raise ValueError("True/False questions must have boolean correct answer")
        
        elif q_type == QuestionType.MULTIPLE_CHOICE:
            if not isinstance(v, int) or v < 0 or v >= len(options):
                raise ValueError(f"Correct answer index must be between 0 and {len(options)-1}")
        
        elif q_type == QuestionType.NUMERICAL:
            if not isinstance(v, (int, float)):
                raise ValueError("Numerical questions must have numeric correct answer")
        
        return v
    
    @validator('explanation')
    def validate_explanation(cls, v):
        """Sanitize and validate explanation"""
        if not v or not v.strip():
            raise ValueError("Explanation cannot be empty")
        
        sanitized = re.sub(r'[<>'"\x00-\x1f\x7f-\x9f]', '', v.strip())
        
        if len(sanitized) < 20:
            raise ValueError("Explanation too short")
        
        return sanitized


class QuizGenerationResponse(BaseModel):
    """🛡️ Strict validation for AI quiz generation responses"""
    
    questions: List[QuestionContract] = Field(..., min_items=1, max_items=50)
    topic: str = Field(..., min_length=2, max_length=100)
    total_questions: int = Field(..., ge=1, le=50)
    difficulty_distribution: Dict[str, int] = Field(...)
    estimated_total_time: int = Field(..., ge=60, le=3600)
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('questions')
    def validate_questions(cls, v):
        """Validate question list"""
        if not v:
            raise ValueError("Questions list cannot be empty")
        
        # Ensure unique questions
        seen_questions = set()
        for q in v:
            if q.question in seen_questions:
                raise ValueError("Duplicate question detected")
            seen_questions.add(q.question)
        
        return v
    
    @validator('difficulty_distribution')
    def validate_difficulty_distribution(cls, v):
        """Validate difficulty distribution"""
        valid_difficulties = {'easy', 'medium', 'hard', 'expert'}
        
        if not isinstance(v, dict):
            raise ValueError("Difficulty distribution must be a dictionary")
        
        for difficulty, count in v.items():
            if difficulty not in valid_difficulties:
                raise ValueError(f"Invalid difficulty: {difficulty}")
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"Count must be non-negative integer for {difficulty}")
        
        return v


class AIResponseValidator(BaseModel):
    """🛡️ Centralized validator for all AI responses"""
    
    response_type: str = Field(..., regex=r'^(quiz|training|evaluation|summary)$')
    content: Dict[str, Any] = Field(...)
    timestamp: str = Field(...)
    model_version: str = Field(..., min_length=1, max_length=50)
    validation_hash: str = Field(..., min_length=32, max_length=64)
    
    @validator('content')
    def validate_content(cls, v):
        """Deep validation of response content"""
        if not isinstance(v, dict):
            raise ValueError("Content must be a dictionary")
        
        # Ensure required fields are present
        if 'questions' in v:
            # Validate questions using QuizGenerationResponse
            try:
                quiz_response = QuizGenerationResponse(**v)
                return quiz_response.dict()
            except Exception as e:
                raise ValueError(f"Invalid quiz format: {e}")
        
        return v
    
    @validator('validation_hash')
    def validate_hash(cls, v):
        """Validate response integrity hash"""
        if not re.match(r'^[a-fA-F0-9]{32,64}$', v):
            raise ValueError("Validation hash must be a valid hexadecimal string")
        return v


class PromptInjectionDetector(BaseModel):
    """🛡️ Detect and prevent prompt injection attacks"""
    
    input_text: str = Field(..., min_length=1, max_length=10000)
    detected_patterns: List[str] = Field(default_factory=list)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    is_safe: bool = Field(...)
    
    @validator('input_text')
    def scan_for_injection(cls, v):
        """Scan input for prompt injection attempts"""
        injection_patterns = [
            r'ignore\s+all\s+previous\s+instructions',
            r'forget\s+everything\s+before',
            r'system\s+override',
            r'admin\s+access',
            r'debug\s+mode\s+enabled',
            r'bypass\s+security',
            r'expose\s+api\s+keys',
            r'print\s+all\s+variables',
            r'execute\s+system\s+command',
            r'drop\s+database',
            r'delete\s+all\s+files',
        ]
        
        detected = []
        for pattern in injection_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                detected.append(pattern)
        
        return v


class DataContractEnforcer:
    """🛡️ Enforces data contracts across the application"""
    
    def __init__(self):
        self.validation_rules = {
            'question': QuestionContract,
            'quiz_response': QuizGenerationResponse,
            'ai_response': AIResponseValidator,
            'prompt_safety': PromptInjectionDetector,
        }
    
    def validate_ai_response(self, response: Dict[str, Any], response_type: str) -> Dict[str, Any]:
        """Validate AI response using appropriate contract"""
        if response_type not in self.validation_rules:
            raise ValueError(f"Unknown response type: {response_type}")
        
        validator_class = self.validation_rules[response_type]
        validated = validator_class(**response)
        return validated.dict()
    
    def check_prompt_safety(self, text: str) -> PromptInjectionDetector:
        """Check text for prompt injection attempts"""
        detector = PromptInjectionDetector(
            input_text=text,
            detected_patterns=[],
            risk_score=0.0,
            is_safe=True
        )
        
        # Scan for injection patterns
        injection_patterns = [
            r'ignore\s+all\s+previous\s+instructions',
            r'forget\s+everything\s+before',
            r'system\s+override',
            r'admin\s+access',
            r'debug\s+mode\s+enabled',
            r'bypass\s+security',
            r'expose\s+api\s+keys',
            r'print\s+all\s+variables',
            r'execute\s+system\s+command',
            r'drop\s+database',
            r'delete\s+all\s+files',
        ]
        
        detected = []
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(pattern)
        
        detector.detected_patterns = detected
        detector.risk_score = min(len(detected) * 0.2, 1.0)
        detector.is_safe = detector.risk_score < 0.5
        
        return detector
    
    def generate_validation_hash(self, data: Dict[str, Any]) -> str:
        """Generate validation hash for data integrity"""
        import hashlib
        import json
        
        # Sort keys for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()


# Global instance for easy access
data_contract_enforcer = DataContractEnforcer()
