"""
🔧 SIMPLIFIED PROMPT SYSTEM - Fix for Over-Engineered Prompts

This module provides clean, effective prompts that avoid the over-engineering issues
found in the current system (excessive [EMERGENCY], [FORBIDDEN], [TARGET] markers).
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def create_simplified_numerical_prompt(topic: str, difficulty: str = "medium") -> str:
    """
    🔧 FIX: Create clean, effective numerical question prompt
    
    Replaces over-engineered prompts with clear, concise instructions.
    """
    examples = """
Example 1:
Question: "Calculate the energy of a photon with wavelength 500 nm."
Options: A) 2.48 eV  B) 3.12 eV  C) 1.86 eV  D) 4.25 eV
Correct: A
Explanation: "Using E = hc/λ = (4.14×10⁻¹⁵ eV·s)(3×10⁸ m/s)/(500×10⁻⁹ m) = 2.48 eV"

Example 2:
Question: "Determine the mass of an object with momentum 15 kg·m/s and velocity 3 m/s."
Options: A) 5 kg  B) 8 kg  C) 12 kg  D) 45 kg
Correct: A
Explanation: "Using p = mv, mass = p/v = 15 kg·m/s ÷ 3 m/s = 5 kg"
"""

    return f"""Create 1 numerical calculation question about {topic}.

Requirements:
- Start with: "Calculate", "Determine", "Find", or "Compute"
- Include specific numbers and units in the question
- All 4 options must be numerical values with units
- Provide a brief calculation in the explanation

{examples}

Generate your question about {topic} now. Return as JSON:
{{
    "question": "Calculate/Determine/Find/Compute [question about {topic}]?",
    "options": {{"A": "number unit", "B": "number unit", "C": "number unit", "D": "number unit"}},
    "correct": "A",
    "explanation": "Brief calculation showing the answer"
}}"""


def create_simplified_conceptual_prompt(topic: str, difficulty: str = "medium") -> str:
    """
    🔧 FIX: Create clean, effective conceptual question prompt
    
    Replaces over-engineered prompts with clear, concise instructions.
    """
    examples = """
Example 1:
Question: "Why does the photoelectric effect demonstrate the particle nature of light?"
Options: 
A) Light exhibits wave interference patterns
B) Photon energy depends only on frequency, not intensity
C) Light can be diffracted around obstacles  
D) Light travels at constant speed in vacuum
Correct: B
Explanation: "The photoelectric effect shows that increasing light intensity doesn't increase electron kinetic energy, only frequency does, proving light consists of discrete photon particles."

Example 2:
Question: "How does quantum tunneling allow particles to pass through energy barriers?"
Options:
A) Particles gain enough thermal energy to overcome the barrier
B) Wave function probability extends beyond the barrier
C) Particles are accelerated by electric fields
D) The barrier height decreases over time
Correct: B
Explanation: "In quantum mechanics, the wave function has non-zero probability beyond barriers, allowing particles to 'tunnel' through classically forbidden regions."
"""

    return f"""Create 1 conceptual understanding question about {topic}.

Requirements:
- Start with: "Why", "How", "What explains", or "Describe"
- Focus on principles, theories, and understanding
- All 4 options must be descriptive explanations
- No calculations or specific numerical values

{examples}

Generate your question about {topic} now. Return as JSON:
{{
    "question": "Why/How/What/Describe [question about {topic}]?",
    "options": {{"A": "descriptive explanation", "B": "descriptive explanation", "C": "descriptive explanation", "D": "descriptive explanation"}},
    "correct": "A",
    "explanation": "Clear explanation of the concept or principle"
}}"""


def create_simplified_mixed_prompt(topic: str, difficulty: str = "medium") -> str:
    """
    🔧 FIX: Create clean, effective mixed question prompt
    
    Can generate either numerical or conceptual questions.
    """
    return f"""Create 1 multiple choice question about {topic}.

Choose ONE approach:
1. NUMERICAL: Calculation-based with numerical options and units
2. CONCEPTUAL: Understanding-based with descriptive options

Requirements:
- Question must end with "?"
- All options must be substantial (20+ characters)
- Provide clear explanation
- Focus specifically on {topic}

Return as JSON:
{{
    "question": "Your question about {topic}?",
    "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}},
    "correct": "A",
    "explanation": "Clear explanation"
}}"""


def get_simplified_prompt(topic: str, question_type: str, difficulty: str = "medium") -> str:
    """
    🔧 MAIN FIX: Get simplified prompt instead of over-engineered version
    
    Args:
        topic: Question topic
        question_type: 'numerical', 'conceptual', or 'mixed'
        difficulty: Question difficulty level
        
    Returns:
        str: Clean, effective prompt
    """
    if question_type.lower() == "numerical":
        return create_simplified_numerical_prompt(topic, difficulty)
    elif question_type.lower() == "conceptual":
        return create_simplified_conceptual_prompt(topic, difficulty)
    else:
        return create_simplified_mixed_prompt(topic, difficulty)


def enhance_prompt_for_retry(original_prompt: str, failure_reason: str, attempt: int) -> str:
    """
    🔧 FIX: Simple retry enhancement without over-engineering
    
    Args:
        original_prompt: The original prompt
        failure_reason: Why the previous attempt failed
        attempt: Retry attempt number
        
    Returns:
        str: Enhanced prompt for retry
    """
    if attempt == 1:
        enhancement = f"""
⚠️ RETRY NEEDED - Previous attempt failed validation
Failure reason: {failure_reason}

Please ensure your question follows the requirements exactly as specified.
"""
    else:
        enhancement = f"""
🔄 FINAL RETRY - Multiple validation failures
Previous attempts failed because: {failure_reason}

Critical requirements:
- Follow the format exactly as shown in examples
- Ensure content matches the question type requirements
- Double-check that all options are appropriate for the question type
"""

    return f"{enhancement}\n{original_prompt}"
