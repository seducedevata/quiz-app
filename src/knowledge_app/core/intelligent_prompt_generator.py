#!/usr/bin/env python3
"""
Intelligent Prompt Generator - Creates smart prompts from resolved topics
"""

import logging
from typing import Dict, List, Optional
from .intelligent_topic_resolver import get_intelligent_topic_resolver

logger = logging.getLogger(__name__)

class IntelligentPromptGenerator:
    """
    🧠 INTELLIGENT PROMPT GENERATOR
    Creates sophisticated prompts that generate meaningful questions from ANY input
    """
    
    def __init__(self):
        self.topic_resolver = get_intelligent_topic_resolver()
        
        # Subject-specific prompt templates
        self.subject_templates = {
            "computer science": {
                "conceptual": """You are an expert computer science educator. Create a multiple choice question about {topic}.

Context: {context}

Focus on fundamental concepts, algorithms, or practical applications. Make the question test understanding rather than memorization.

Generate a JSON response with this exact format:
{{
  "question": "Clear, specific question about {topic}",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct": "Option A",
  "explanation": "Detailed explanation of why the answer is correct"
}}""",
                
                "practical": """You are a senior software engineer creating a practical question about {topic}.

Context: {context}

Create a question that tests practical knowledge or problem-solving skills related to {topic}.

Generate a JSON response with this exact format:
{{
  "question": "Practical question about implementing or using {topic}",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct": "Option A", 
  "explanation": "Explanation focusing on practical applications"
}}"""
            },
            
            "mathematics": {
                "conceptual": """You are a mathematics professor creating an educational question about {topic}.

Context: {context}

Create a question that tests mathematical understanding and reasoning related to {topic}.

Generate a JSON response with this exact format:
{{
  "question": "Mathematical question about {topic}",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct": "Option A",
  "explanation": "Mathematical explanation with reasoning"
}}""",
                
                "problem_solving": """You are creating a mathematical problem-solving question about {topic}.

Context: {context}

Create a question that requires mathematical thinking and problem-solving skills.

Generate a JSON response with this exact format:
{{
  "question": "Problem-solving question involving {topic}",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct": "Option A",
  "explanation": "Step-by-step solution explanation"
}}"""
            },
            
            "science": {
                "conceptual": [
                    """You are a science educator creating a question about {topic}.

Context: {context}

Create a question that tests scientific understanding and concepts related to {topic}.

Generate a JSON response with this exact format:
{{
  "question": "Scientific question about {topic}",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct": "Option A",
  "explanation": "Scientific explanation with reasoning"
}}""",
                    """You are a research scientist developing an educational question about {topic}.

Context: {context}

Design a question that explores the principles and mechanisms of {topic}.

Generate a JSON response with this exact format:
{{
  "question": "Research-based question about {topic}",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct": "Option A",
  "explanation": "Detailed scientific reasoning"
}}""",
                    """You are creating an advanced science question about {topic}.

Context: {context}

Develop a question that challenges understanding of {topic} from a theoretical perspective.

Generate a JSON response with this exact format:
{{
  "question": "Advanced theoretical question about {topic}",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct": "Option A",
  "explanation": "Comprehensive scientific explanation"
}}"""
                ],
                
                "application": """You are creating a real-world application question about {topic}.

Context: {context}

Create a question about how {topic} applies to real-world situations or phenomena.

Generate a JSON response with this exact format:
{{
  "question": "Real-world application question about {topic}",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct": "Option A",
  "explanation": "Explanation connecting theory to practice"
}}"""
            },
            
            "general": {
                "analytical": """You are an expert educator creating an analytical thinking question about {topic}.

Context: {context}

Create a question that promotes critical thinking and analysis related to {topic}.

Generate a JSON response with this exact format:
{{
  "question": "Analytical question about {topic}",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct": "Option A",
  "explanation": "Explanation promoting deeper understanding"
}}""",
                
                "comprehensive": """You are creating a comprehensive question about {topic}.

Context: {context}

Create a well-rounded question that tests understanding of {topic} from multiple angles.

Generate a JSON response with this exact format:
{{
  "question": "Comprehensive question about {topic}",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct": "Option A",
  "explanation": "Thorough explanation covering key aspects"
}}"""
            }
        }
    
    def generate_intelligent_prompt(self, raw_input: str, difficulty: str = "medium", 
                                  question_type: str = "mixed") -> Dict[str, any]:
        """
        🧠 Generate intelligent prompt from ANY user input
        
        Args:
            raw_input: Any user input (even random text like "dfs")
            difficulty: Question difficulty level
            question_type: Type of question to generate
            
        Returns:
            Dict with prompt and metadata
        """
        # Step 1: Resolve the topic intelligently
        resolution = self.topic_resolver.resolve_topic(raw_input)
        
        # Step 2: Select appropriate prompt template
        prompt_template = self._select_prompt_template(resolution, question_type)
        
        # Step 3: Enhance context based on resolution method
        enhanced_context = self._enhance_context(resolution, difficulty)
        
        # Step 4: Generate the final prompt using INQUISITOR'S MANDATE
        final_prompt = self._create_inquisitor_mandate_prompt(
            topic=resolution["resolved_topic"],
            context=enhanced_context,
            difficulty=difficulty,
            original_input=resolution["original_input"]
        )
        
        return {
            "prompt": final_prompt,
            "resolution": resolution,
            "enhanced_context": enhanced_context,
            "confidence": resolution["confidence"],
            "metadata": {
                "original_input": raw_input,
                "resolved_topic": resolution["resolved_topic"],
                "subject_area": resolution["subject_area"],
                "resolution_method": resolution["resolution_method"],
                "difficulty": difficulty,
                "question_type": question_type
            }
        }
    
    def _select_prompt_template(self, resolution: Dict, question_type: str) -> str:
        """Select the best prompt template based on resolution - with randomness for variety"""
        import random

        subject_area = resolution["subject_area"]

        # Map subject areas to template categories
        if subject_area in ["computer science", "programming", "networking"]:
            template_category = "computer science"
        elif subject_area in ["mathematics", "algebra", "geometry", "calculus"]:
            template_category = "mathematics"
        elif subject_area in ["physics", "chemistry", "biology", "astronomy"]:
            template_category = "science"
        else:
            template_category = "general"

        # Select specific template type
        templates = self.subject_templates[template_category]

        def get_template(template_key):
            """Get template, handling both single templates and lists"""
            template = templates.get(template_key, templates[list(templates.keys())[0]])
            if isinstance(template, list):
                # 🔥 RANDOM SELECTION for variety
                return random.choice(template)
            return template

        if question_type == "conceptual":
            return get_template("conceptual")
        elif question_type == "practical":
            return get_template("practical") or get_template("application")
        elif question_type == "numerical":
            # 🔢 FORCE NUMERICAL CALCULATION QUESTIONS
            return {
                "name": "numerical_calculation",
                "description": "Generate calculation-based questions requiring mathematical problem-solving",
                "prompt_template": """🔢 CRITICAL: Generate a CALCULATION-BASED question that REQUIRES mathematical problem-solving.

MANDATORY REQUIREMENTS:
- Question MUST involve specific numbers, calculations, formulas, or quantitative analysis
- Include numerical values (masses, energies, wavelengths, frequencies, etc.)
- All answer options MUST be numerical values with appropriate units
- Question should require step-by-step mathematical solution
- Focus on calculations like: energy levels, electron transitions, atomic masses, binding energies, wavelengths
- Use formulas like E=hf, E=-13.6/n², λ=hc/E, etc.

FORBIDDEN: Conceptual questions, theory explanations, qualitative comparisons, electron configuration patterns

EXAMPLES OF GOOD NUMERICAL QUESTIONS:
- "Calculate the energy of an electron in the n=3 shell of hydrogen (ground state = -13.6 eV)"
- "What is the wavelength of light emitted when an electron transitions from n=4 to n=2 in hydrogen?"
- "Calculate the binding energy of the outermost electron in sodium (ionization energy = 5.14 eV)"

Generate a numerical calculation question about {topic} at {difficulty} level."""
            }
        else:  # mixed or default
            # Choose based on resolution confidence
            if resolution["confidence"] > 0.8:
                return get_template("conceptual")
            else:
                return get_template("analytical") or get_template("comprehensive")
    
    def _enhance_context(self, resolution: Dict, difficulty: str) -> str:
        """Enhance context based on resolution method and difficulty"""
        base_context = resolution.get("context", "") or ""

        # Ensure we have a valid base context
        if not base_context:
            base_context = f"Educational content about {resolution.get('resolved_topic', 'the given topic')}"

        # Add resolution-specific context
        if resolution["resolution_method"] == "acronym_database":
            enhanced = f"{base_context}. "
            if resolution.get("alternative_meanings"):
                enhanced += f"Note that '{resolution['original_input']}' can also refer to: {', '.join(resolution['alternative_meanings'])}. "
            enhanced += f"Focus on the primary meaning: {resolution['resolved_topic']}."

        elif resolution["resolution_method"] == "fuzzy_matching":
            enhanced = f"{base_context}. This interpretation accounts for possible typos or variations in spelling."

        elif resolution["resolution_method"] == "intelligent_interpretation":
            enhanced = f"{base_context}. This educational interpretation transforms the input into a meaningful learning opportunity."

        elif resolution["resolution_method"] == "adaptive_creation":
            enhanced = f"{base_context}. This demonstrates how any input can become a foundation for learning and critical thinking."

        else:
            enhanced = base_context

        # Ensure enhanced is not None
        if not enhanced:
            enhanced = f"Educational content about {resolution.get('resolved_topic', 'the given topic')}"

        # Add difficulty-specific context
        if difficulty == "easy":
            enhanced += " Focus on fundamental concepts and basic understanding."
        elif difficulty == "medium":
            enhanced += " Include moderate complexity and practical applications."
        elif difficulty == "hard":
            enhanced += " Incorporate advanced concepts and complex reasoning."
        elif difficulty == "expert":
            enhanced += " Require deep expertise and sophisticated analysis."

        return enhanced
    
    def _add_difficulty_enhancements(self, prompt: str, difficulty: str) -> str:
        """Add difficulty-specific enhancements to the prompt"""
        if difficulty == "easy":
            enhancement = "\n\nDifficulty: EASY - Use simple language, basic concepts, and straightforward reasoning."
        elif difficulty == "medium":
            enhancement = "\n\nDifficulty: MEDIUM - Use moderate complexity, practical examples, and clear reasoning."
        elif difficulty == "hard":
            enhancement = "\n\nDifficulty: HARD - Use advanced concepts, complex scenarios, and sophisticated reasoning."
        elif difficulty == "expert":
            enhancement = "\n\nDifficulty: EXPERT - Use cutting-edge concepts, research-level complexity, and expert-level analysis."
        else:
            enhancement = ""
        
        return prompt + enhancement

    def _create_inquisitor_mandate_prompt(self, topic: str, context: str, difficulty: str, original_input: str) -> str:
        """
        🎯 Create INQUISITOR'S MANDATE prompt for maximum quality
        Uses your advanced prompting system instead of basic templates
        """
        from .inquisitor_prompt import _create_inquisitor_prompt

        # Use the advanced Inquisitor's Mandate system
        return _create_inquisitor_prompt(context, topic, difficulty)

# Global instance
_prompt_generator = None

def get_intelligent_prompt_generator() -> IntelligentPromptGenerator:
    """Get the global prompt generator instance"""
    global _prompt_generator
    if _prompt_generator is None:
        _prompt_generator = IntelligentPromptGenerator()
    return _prompt_generator
