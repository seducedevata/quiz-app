"""
🛡️ API VALIDATION MODELS - Enterprise-Grade Input Validation

This module defines Pydantic models for all data contracts between the JavaScript 
frontend and Python backend, eliminating "implicit trust" security vulnerabilities.

SECURITY PRINCIPLE: Never trust the frontend to send perfectly formed data.
Every payload from JavaScript must be validated before processing.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Optional, Union, Any
from enum import Enum
import re


class DifficultyLevel(str, Enum):
    """Valid difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium" 
    HARD = "hard"
    EXPERT = "expert"


class QuestionType(str, Enum):
    """Valid question types"""
    NUMERICAL = "numerical"
    CONCEPTUAL = "conceptual"
    MIXED = "mixed"


class GameMode(str, Enum):
    """Valid game modes"""
    CASUAL = "casual"
    TIMED = "timed"
    CHALLENGE = "challenge"
    EXPERT = "expert"


class ApiProvider(str, Enum):
    """Valid API providers"""
    OPENROUTER = "openrouter"
    GROQ = "groq"
    OPENAI = "openai"
    OLLAMA = "ollama"


class StartQuizPayload(BaseModel):
    """🛡️ SECURITY: Validate quiz start parameters"""
    
    topic: str = Field(..., min_length=2, max_length=200, description="Quiz topic")
    num_questions: int = Field(default=5, ge=1, le=50, description="Number of questions (1-50)")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    question_type: QuestionType = Field(default=QuestionType.MIXED)
    game_mode: GameMode = Field(default=GameMode.CASUAL)
    context: Optional[str] = Field(default="", max_length=10000, description="Optional RAG context")
    adapter_name: Optional[str] = Field(default=None, max_length=100, description="LoRA adapter name")
    
    @validator('topic')
    def validate_topic(cls, v):
        """Sanitize and validate topic input"""
        if not v or not v.strip():
            raise ValueError("Topic cannot be empty")
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\'\\\x00-\x1f\x7f-\x9f]', '', v.strip())
        
        if len(sanitized) < 2:
            raise ValueError("Topic too short after sanitization")
            
        return sanitized
    
    @validator('context')
    def validate_context(cls, v):
        """Validate context if provided"""
        if v is None:
            return ""
        
        # Remove potentially dangerous characters from context
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', str(v))
        return sanitized[:10000]  # Enforce max length


class UserSettingsPayload(BaseModel):
    """🛡️ SECURITY: Validate user settings including API keys"""
    
    api_keys: Dict[str, str] = Field(default_factory=dict)
    providers_enabled: Dict[str, bool] = Field(default_factory=dict)
    generation_mode: Optional[str] = Field(default="auto", regex=r'^(auto|online|offline)$')
    ui_theme: Optional[str] = Field(default="dark", regex=r'^(dark|light)$')
    question_timeout: Optional[int] = Field(default=60, ge=10, le=300)
    
    @validator('api_keys')
    def validate_api_keys(cls, v):
        """Validate API keys structure and content"""
        if not isinstance(v, dict):
            raise ValueError("api_keys must be a dictionary")
        
        validated_keys = {}
        valid_providers = {e.value for e in ApiProvider}
        
        for provider, key in v.items():
            if provider not in valid_providers:
                continue  # Skip unknown providers
                
            if not isinstance(key, str):
                continue  # Skip non-string keys
                
            # Basic API key format validation
            clean_key = str(key).strip()
            if len(clean_key) < 10:  # Minimum reasonable API key length
                continue
                
            # Remove any potentially dangerous characters
            sanitized_key = re.sub(r'[\x00-\x1f\x7f-\x9f<>"\']', '', clean_key)
            if len(sanitized_key) >= 10:
                validated_keys[provider] = sanitized_key
        
        return validated_keys
    
    @validator('providers_enabled')
    def validate_providers_enabled(cls, v):
        """Validate providers enabled structure"""
        if not isinstance(v, dict):
            return {}
        
        validated_providers = {}
        valid_providers = {e.value for e in ApiProvider}
        
        for provider, enabled in v.items():
            if provider in valid_providers and isinstance(enabled, bool):
                validated_providers[provider] = enabled
        
        return validated_providers


class SubmitAnswerPayload(BaseModel):
    """🛡️ SECURITY: Validate answer submission"""
    
    answer_index: int = Field(..., ge=0, le=3, description="Answer index (0-3)")
    question_id: Optional[str] = Field(default=None, max_length=100)
    time_taken: Optional[float] = Field(default=None, ge=0, le=3600)  # Max 1 hour
    
    @validator('question_id')
    def validate_question_id(cls, v):
        """Sanitize question ID"""
        if v is None:
            return None
        
        # Only allow alphanumeric, dashes, underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', str(v))
        return sanitized[:100] if sanitized else None


class ApiKeyTestPayload(BaseModel):
    """🛡️ SECURITY: Validate API key testing parameters"""
    
    provider: ApiProvider = Field(...)
    api_key: str = Field(..., min_length=10, max_length=500)
    
    @validator('api_key')
    def validate_api_key(cls, v):
        """Sanitize API key for testing"""
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        
        # Remove dangerous characters but preserve key structure
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f<>"\']', '', v.strip())
        
        if len(sanitized) < 10:
            raise ValueError("API key too short after sanitization")
            
        return sanitized


class QuestionHistoryQueryPayload(BaseModel):
    """🛡️ SECURITY: Validate question history queries"""
    
    page: int = Field(default=0, ge=0, le=1000)
    page_size: int = Field(default=20, ge=1, le=100)
    topic_filter: Optional[str] = Field(default=None, max_length=200)
    difficulty_filter: Optional[DifficultyLevel] = Field(default=None)
    
    @validator('topic_filter')
    def validate_topic_filter(cls, v):
        """Sanitize topic filter"""
        if v is None:
            return None
        
        sanitized = re.sub(r'[<>"\'\\\x00-\x1f\x7f-\x9f]', '', str(v).strip())
        return sanitized if len(sanitized) >= 2 else None


class SearchQuestionsPayload(BaseModel):
    """🛡️ SECURITY: Validate question search parameters"""
    
    query: str = Field(..., min_length=2, max_length=500)
    limit: int = Field(default=50, ge=1, le=100)
    
    @validator('query')
    def validate_search_query(cls, v):
        """Sanitize search query"""
        if not v or not v.strip():
            raise ValueError("Search query cannot be empty")
        
        # Allow alphanumeric, spaces, and common punctuation for search
        sanitized = re.sub(r'[<>"\'\\\x00-\x1f\x7f-\x9f]', '', v.strip())
        
        if len(sanitized) < 2:
            raise ValueError("Search query too short after sanitization")
            
        return sanitized


class TrainingConfigPayload(BaseModel):
    """🛡️ SECURITY: Validate training configuration parameters"""
    
    adapter_name: str = Field(..., min_length=3, max_length=50)
    base_model: str = Field(..., min_length=3, max_length=100)
    selected_files: List[str] = Field(..., min_items=1, max_items=20)
    training_preset: str = Field(default="default", max_length=50)
    learning_rate: Optional[float] = Field(default=None, gt=0, le=1)
    epochs: Optional[int] = Field(default=None, ge=1, le=100)
    
    @validator('adapter_name')
    def validate_adapter_name(cls, v):
        """Validate adapter name format"""
        # Only allow alphanumeric, dashes, underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', v.strip())
        
        if len(sanitized) < 3:
            raise ValueError("Adapter name too short after sanitization")
            
        return sanitized
    
    @validator('selected_files')
    def validate_selected_files(cls, v):
        """Validate file paths"""
        if not isinstance(v, list):
            raise ValueError("selected_files must be a list")
        
        validated_files = []
        for file_path in v:
            if not isinstance(file_path, str):
                continue
                
            # Basic path sanitization
            sanitized = re.sub(r'[<>"\'\x00-\x1f\x7f-\x9f]', '', str(file_path).strip())
            
            # Must be reasonable file path length
            if 3 <= len(sanitized) <= 500:
                validated_files.append(sanitized)
        
        if not validated_files:
            raise ValueError("No valid files after sanitization")
            
        return validated_files


class GenericPayload(BaseModel):
    """🛡️ SECURITY: Fallback validation for generic payloads"""
    
    data: Union[Dict[str, Any], List[Any], str, int, float, bool] = Field(...)
    
    @validator('data')
    def validate_generic_data(cls, v):
        """Basic sanitization for generic data"""
        if isinstance(v, str):
            # Sanitize string data
            return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', v[:10000])
        elif isinstance(v, (int, float, bool)):
            return v
        elif isinstance(v, (dict, list)):
            # For complex structures, implement recursive sanitization if needed
            return v
        else:
            raise ValueError(f"Unsupported data type: {type(v)}")


def validate_bridge_payload(payload: Any, expected_model: BaseModel) -> BaseModel:
    """
    🛡️ MASTER VALIDATION FUNCTION
    
    Central function to validate any payload from JavaScript frontend.
    Returns validated and sanitized data or raises ValidationError.
    
    Args:
        payload: Raw payload from QVariant/JSON string
        expected_model: Pydantic model class to validate against
        
    Returns:
        Validated model instance
        
    Raises:
        ValidationError: If payload doesn't match expected schema
        ValueError: If payload is fundamentally invalid
    """
    from pydantic import ValidationError
    import json
    
    try:
        # Handle different input types from Qt bridge
        if hasattr(payload, 'toVariant'):
            # QVariant input
            data = payload.toVariant()
        elif isinstance(payload, str):
            # JSON string input
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}")
        else:
            # Direct dict/primitive input
            data = payload
        
        # Validate using the expected model
        return expected_model(**data)
        
    except ValidationError as e:
        # Re-raise with context
        raise ValidationError(f"Payload validation failed for {expected_model.__name__}: {e}")
    except Exception as e:
        raise ValueError(f"Payload processing failed: {e}") 