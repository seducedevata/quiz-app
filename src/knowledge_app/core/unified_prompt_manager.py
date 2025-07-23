"""
🔧 UNIFIED PROMPT MANAGER - Single Source of Truth for All MCQ Prompts

This module centralizes all prompt engineering logic to eliminate redundancy
and ensure consistency across all MCQ generators.
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class PromptType(Enum):
    """Types of prompts supported by the unified system"""
    EXPERT_BATCH = "expert_batch"
    EXPERT_OPTIMIZED = "expert_optimized"
    STANDARD = "standard"
    NUMERICAL = "numerical"
    CONCEPTUAL = "conceptual"
    ONLINE_API = "online_api"
    OFFLINE_LOCAL = "offline_local"

class DifficultyLevel(Enum):
    """Difficulty levels with specific requirements"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

class UnifiedPromptManager:
    """
    🔧 CENTRALIZED PROMPT ENGINEERING - Single source of truth
    
    Eliminates the massive prompt redundancy across 4+ files by providing
    a single, authoritative prompt generation system.
    """
    
    def __init__(self):
        self.difficulty_configs = {
            DifficultyLevel.EASY: {
                "description": "basic recall of specific facts and fundamental definitions",
                "target_audience": "undergraduate students",
                "cognitive_requirements": "simple recognition and recall",
                "complexity": "straightforward application of basic concepts",
                "min_question_length": 80,
                "examples": "What is the definition of..., Which of the following is..."
            },
            DifficultyLevel.MEDIUM: {
                "description": "analytical thinking and concept application",
                "target_audience": "advanced undergraduate students",
                "cognitive_requirements": "analysis and application of concepts",
                "complexity": "multi-step reasoning with specific scenarios",
                "min_question_length": 100,
                "examples": "How would you apply..., What happens when..."
            },
            DifficultyLevel.HARD: {
                "description": "complex synthesis and multi-step reasoning",
                "target_audience": "graduate students",
                "cognitive_requirements": "synthesis of multiple advanced concepts",
                "complexity": "expert-level analysis of specific cases and mechanisms",
                "min_question_length": 120,
                "examples": "Analyze the relationship between..., Evaluate the impact of..."
            },
            DifficultyLevel.EXPERT: {
                "description": "cutting-edge research-level analysis",
                "target_audience": "PhD students and researchers",
                "cognitive_requirements": "advanced theoretical understanding and research application",
                "complexity": "PhD dissertation-level complexity with recent research concepts",
                "min_question_length": 150,
                "examples": "Critically evaluate the theoretical framework..., Design an experiment to test..."
            }
        }
        
        logger.info("🔧 UnifiedPromptManager initialized - centralized prompt engineering active")
    
    def create_prompt(self, prompt_type: PromptType, topic: str, context: str = "", 
                     difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
                     question_type: str = "mixed", **kwargs) -> str:
        """
        🎯 CREATE UNIFIED PROMPT - Single method for all prompt types
        
        Args:
            prompt_type: Type of prompt to generate
            topic: Question topic
            context: Additional context
            difficulty: Difficulty level
            question_type: Type of question (numerical, conceptual, mixed)
            **kwargs: Additional parameters
            
        Returns:
            Optimized prompt string
        """
        
        # Get difficulty configuration
        diff_config = self.difficulty_configs[difficulty]
        
        # Create base prompt components
        base_components = self._create_base_components(topic, context, diff_config, question_type)
        
        # Generate prompt based on type
        if prompt_type == PromptType.EXPERT_BATCH:
            return self._create_expert_batch_prompt(base_components, **kwargs)
        elif prompt_type == PromptType.EXPERT_OPTIMIZED:
            return self._create_expert_optimized_prompt(base_components, **kwargs)
        elif prompt_type == PromptType.NUMERICAL:
            return self._create_numerical_prompt(base_components, **kwargs)
        elif prompt_type == PromptType.CONCEPTUAL:
            return self._create_conceptual_prompt(base_components, **kwargs)
        elif prompt_type == PromptType.ONLINE_API:
            return self._create_online_api_prompt(base_components, **kwargs)
        elif prompt_type == PromptType.OFFLINE_LOCAL:
            return self._create_offline_local_prompt(base_components, **kwargs)
        else:
            return self._create_standard_prompt(base_components, **kwargs)
    
    def _create_base_components(self, topic: str, context: str, diff_config: Dict, question_type: str) -> Dict[str, Any]:
        """Create reusable prompt components"""
        
        # Question type specific enforcement
        type_enforcement = ""
        if question_type.lower() == "numerical":
            type_enforcement = """🔢 NUMERICAL REQUIREMENT:
- Question MUST involve calculations, numbers, formulas, or quantitative analysis
- Include specific numerical values with appropriate units
- All answer options MUST be numerical values
- Question should require mathematical problem-solving
- FORBIDDEN: Conceptual questions without calculations"""
        elif question_type.lower() == "conceptual":
            type_enforcement = """🧠 CONCEPTUAL REQUIREMENT:
- Focus on understanding, principles, and theory
- Test deep theoretical knowledge
- Avoid pure calculation questions
- Emphasize mechanisms and relationships"""
        
        # Context instruction
        context_instruction = ""
        if context and context.strip():
            context_instruction = f"""
**Context for Question Generation:**
{context.strip()}

Use this context to inform your question, but do NOT simply copy sentences from it.
"""
        
        return {
            "topic": topic,
            "context": context,
            "context_instruction": context_instruction,
            "difficulty_config": diff_config,
            "type_enforcement": type_enforcement,
            "question_type": question_type
        }
    
    def _create_expert_batch_prompt(self, components: Dict[str, Any], **kwargs) -> str:
        """Create expert batch generation prompt"""
        num_questions = kwargs.get('num_questions', 1)
        
        return f"""Generate {num_questions} PhD-level multiple choice questions about {components['topic']}.

{components['context_instruction']}

{components['type_enforcement']}

EXPERT REQUIREMENTS:
- {components['difficulty_config']['description']}
- Target audience: {components['difficulty_config']['target_audience']}
- Minimum question length: {components['difficulty_config']['min_question_length']} characters
- Include advanced terminology and recent research concepts (2020-2024)
- Test synthesis of multiple advanced concepts

OUTPUT FORMAT - Generate ALL {num_questions} questions as JSON array:
[
  {{
    "question": "Advanced question text ending with ?",
    "options": {{"A": "option", "B": "option", "C": "option", "D": "option"}},
    "correct": "A",
    "explanation": "Detailed explanation with advanced concepts"
  }}
]

Generate the complete JSON array now:"""
    
    def _create_expert_optimized_prompt(self, components: Dict[str, Any], **kwargs) -> str:
        """Create optimized expert single-question prompt"""
        
        return f"""Generate a PhD-level {components['question_type']} question about {components['topic']}.

{components['context_instruction']}

{components['type_enforcement']}

EXPERT REQUIREMENTS:
- Advanced, graduate-level complexity
- Specific, detailed scenarios
- No basic "what is" questions
- Include recent research concepts (2020-2024)
- Minimum {components['difficulty_config']['min_question_length']} characters

OUTPUT FORMAT (JSON):
{{
  "question": "Advanced question text here ending with ?",
  "options": {{"A": "option", "B": "option", "C": "option", "D": "option"}},
  "correct": "A",
  "explanation": "Detailed explanation with advanced concepts"
}}

Generate the JSON now:"""
    
    def _create_numerical_prompt(self, components: Dict[str, Any], **kwargs) -> str:
        """Create numerical question prompt"""
        
        return f"""Generate a {components['difficulty_config']['description']} numerical question about {components['topic']}.

{components['context_instruction']}

🔢 NUMERICAL REQUIREMENTS:
- MUST involve calculations, formulas, or quantitative analysis
- Include specific numerical values with units
- All options must be numerical answers
- Show mathematical problem-solving

DIFFICULTY: {components['difficulty_config']['target_audience']}
COGNITIVE LEVEL: {components['difficulty_config']['cognitive_requirements']}

OUTPUT FORMAT (JSON):
{{
  "question": "Numerical question with calculations ending with ?",
  "options": {{"A": "numerical value", "B": "numerical value", "C": "numerical value", "D": "numerical value"}},
  "correct": "A",
  "explanation": "Step-by-step calculation explanation"
}}

Generate the JSON now:"""
    
    def _create_conceptual_prompt(self, components: Dict[str, Any], **kwargs) -> str:
        """Create conceptual question prompt"""
        
        return f"""Generate a {components['difficulty_config']['description']} conceptual question about {components['topic']}.

{components['context_instruction']}

🧠 CONCEPTUAL REQUIREMENTS:
- Focus on understanding principles and theory
- Test deep theoretical knowledge
- Emphasize mechanisms and relationships
- Avoid pure calculation questions

DIFFICULTY: {components['difficulty_config']['target_audience']}
COGNITIVE LEVEL: {components['difficulty_config']['cognitive_requirements']}

OUTPUT FORMAT (JSON):
{{
  "question": "Conceptual question about understanding ending with ?",
  "options": {{"A": "concept option", "B": "concept option", "C": "concept option", "D": "concept option"}},
  "correct": "A",
  "explanation": "Theoretical explanation of concepts"
}}

Generate the JSON now:"""
    
    def _create_online_api_prompt(self, components: Dict[str, Any], **kwargs) -> str:
        """Create prompt optimized for online API calls"""
        
        return f"""You are an expert MCQ generator. Generate only valid JSON.

Create a {components['difficulty_config']['description']} question about {components['topic']}.

{components['context_instruction']}

{components['type_enforcement']}

Requirements:
- Target: {components['difficulty_config']['target_audience']}
- Complexity: {components['difficulty_config']['complexity']}
- Length: minimum {components['difficulty_config']['min_question_length']} characters

Return only this JSON structure:
{{
  "question": "Question text ending with ?",
  "options": {{"A": "option", "B": "option", "C": "option", "D": "option"}},
  "correct": "A",
  "explanation": "Explanation"
}}"""
    
    def _create_offline_local_prompt(self, components: Dict[str, Any], **kwargs) -> str:
        """Create prompt optimized for offline local models"""
        
        return f"""### INSTRUCTION ###
You are a machine. Your ONLY task is to generate a single, valid JSON object.
Do NOT output any text, explanation, or markdown before or after the JSON object.

### TASK ###
Generate a multiple-choice question about '{components['topic']}' for {components['difficulty_config']['target_audience']}.

{components['type_enforcement']}

### REQUIREMENTS ###
- {components['difficulty_config']['cognitive_requirements']}
- {components['difficulty_config']['complexity']}
- Minimum {components['difficulty_config']['min_question_length']} characters

{components['context_instruction']}

### OUTPUT ###
Respond with ONLY the raw JSON object:
{{
  "question": "string",
  "options": {{"A": "string", "B": "string", "C": "string", "D": "string"}},
  "correct": "string",
  "explanation": "string"
}}"""
    
    def _create_standard_prompt(self, components: Dict[str, Any], **kwargs) -> str:
        """Create standard prompt for general use"""
        
        return f"""Generate a {components['difficulty_config']['description']} multiple choice question about {components['topic']}.

{components['context_instruction']}

{components['type_enforcement']}

Requirements:
- Target audience: {components['difficulty_config']['target_audience']}
- Cognitive level: {components['difficulty_config']['cognitive_requirements']}
- Question complexity: {components['difficulty_config']['complexity']}

JSON format:
{{
  "question": "Question text ending with ?",
  "options": {{"A": "option", "B": "option", "C": "option", "D": "option"}},
  "correct": "A",
  "explanation": "Explanation"
}}"""

    def _sanitize_input(self, input_text: str) -> str:
        """🔧 FIX: Centralized input sanitization to prevent prompt injection"""
        if not input_text:
            return ""

        # Remove potential prompt injection patterns
        dangerous_patterns = [
            "ignore all previous instructions",
            "instead, write",
            "forget the above",
            "new instructions:",
            "system:",
            "assistant:",
            "user:",
        ]

        sanitized = input_text

        # 🔧 FIX: Use case-insensitive replacement with regex
        import re
        for pattern in dangerous_patterns:
            # Replace case-insensitively
            sanitized = re.sub(re.escape(pattern), "", sanitized, flags=re.IGNORECASE)

        # Remove excessive whitespace and control characters
        sanitized = " ".join(sanitized.split())

        # Limit length to prevent prompt bloat
        if len(sanitized) > 500:
            sanitized = sanitized[:500] + "..."

        return sanitized

# Global instance
_prompt_manager = None

def get_unified_prompt_manager() -> UnifiedPromptManager:
    """Get the global unified prompt manager instance"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = UnifiedPromptManager()
    return _prompt_manager
