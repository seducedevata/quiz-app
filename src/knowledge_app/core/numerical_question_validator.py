"""
Centralized Numerical Question Validator
🚀 BUG FIX: Provides consistent numerical question validation across all generators
"""

import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class NumericalQuestionValidator:
    """
    Centralized validator for numerical questions to ensure consistency across all generators
    """
    
    def __init__(self):
        # Conceptual patterns that should NOT appear in numerical questions
        self.conceptual_patterns = [
            'explain', 'describe', 'why', 'how', 'what is the concept',
            'what is the principle', 'what is the theory', 'what is the definition',
            'which statement', 'which principle', 'which theory', 'which concept',
            'analyze', 'compare', 'contrast', 'discuss', 'evaluate'
        ]
        
        # Required numerical starters for numerical questions
        self.numerical_starters = [
            'calculate', 'determine', 'find', 'compute',
            'what is the value', 'what is the magnitude', 'what is the energy',
            'what is the wavelength', 'what is the frequency', 'what is the mass',
            'what is the charge', 'what is the number', 'how many', 'at what'
        ]
        
        # Common units that should appear in numerical questions
        self.common_units = [
            'ev', 'mev', 'kev', 'gev', 'j', 'joule', 'nm', 'pm', 'fm', 'cm', 'm', 'mm', 'km',
            'hz', 'khz', 'mhz', 'ghz', 'thz', 's', 'ms', 'ns', 'ps', 'fs', 'kg', 'g', 'mg',
            'c', 'k', 'kelvin', 'celsius', 'tesla', 't', 'gauss', 'amp', 'ampere', 'volt', 'v',
            'ohm', 'watt', 'w', 'pascal', 'pa', 'bar', 'atm', 'mol', 'rad', 'deg', 'degree'
        ]
    
    def validate_numerical_question(self, question_data: Dict[str, Any], question_type: str) -> bool:
        """
        Validate that a question is truly numerical if question_type is 'numerical'
        
        Args:
            question_data: The generated question data
            question_type: The requested question type
            
        Returns:
            bool: True if valid, False if invalid
        """
        try:
            if question_type.lower() != "numerical":
                return True  # No validation needed for non-numerical questions
            
            question = question_data.get('question', '').lower()
            options = question_data.get('options', {})
            
            if not question or not options:
                logger.warning("🚫 Numerical question validation FAILED: Missing question or options")
                return False
            
            # Check if question contains any conceptual patterns
            for pattern in self.conceptual_patterns:
                if pattern in question:
                    logger.warning(f"🚫 Numerical question validation FAILED: Contains conceptual pattern '{pattern}'")
                    logger.warning(f"❌ Question: '{question[:100]}...'")
                    return False
            
            # Check for numerical calculation starters (required for numerical questions)
            has_numerical_starter = any(starter in question for starter in self.numerical_starters)
            if not has_numerical_starter:
                logger.warning(f"🚫 Numerical question validation FAILED: No numerical calculation starter found")
                logger.warning(f"❌ Question: '{question[:100]}...'")
                return False
            
            # Check that all options are numerical (contain numbers and units)
            numerical_options = 0
            option_values = list(options.values()) if isinstance(options, dict) else options
            
            for option in option_values:
                option_str = str(option).lower()
                # Check if option contains numbers and possibly units
                has_number = any(char.isdigit() for char in option_str)
                
                if has_number:
                    numerical_options += 1
            
            # Require at least 3/4 options to be numerical
            if numerical_options < 3:
                logger.warning(f"🚫 Numerical question validation FAILED: Only {numerical_options}/4 options are numerical")
                logger.warning(f"❌ Options: {option_values}")
                return False
            
            logger.info(f"✅ Numerical question validation PASSED: Pure calculation question detected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating numerical question: {e}")
            return True  # Default to accepting the question if validation fails
    
    def create_retry_prompt_enhancement(self, original_prompt: str, question_type: str, failed_attempt: int) -> str:
        """
        Create an enhanced prompt for retry attempts when validation fails
        
        Args:
            original_prompt: The original prompt that failed
            question_type: The question type that failed validation
            failed_attempt: The number of failed attempts (1, 2, etc.)
            
        Returns:
            str: Enhanced prompt with stronger enforcement
        """
        if question_type.lower() != "numerical":
            return original_prompt
        
        # Progressive enforcement based on attempt number
        if failed_attempt == 1:
            enforcement = """
🚨 CRITICAL RETRY - PREVIOUS ATTEMPT FAILED NUMERICAL VALIDATION 🚨
The previous question was rejected for being too conceptual.
MANDATORY REQUIREMENTS:
- Question MUST start with "Calculate", "Determine", "Find", or "Compute"
- Question MUST include specific numerical values and units
- ALL options MUST be numerical values with units (e.g., "2.5 eV", "450 nm")
- NO conceptual words: explain, describe, why, how, analyze, compare
"""
        else:
            enforcement = """
🔥 FINAL RETRY - MULTIPLE FAILURES DETECTED 🔥
ZERO TOLERANCE ENFORCEMENT:
- Question MUST be a pure calculation problem
- Question MUST require mathematical computation to solve
- Include specific numbers, formulas, and units in the question
- ALL four options MUST be different numerical values with proper units
- Example good question: "Calculate the energy of a photon with wavelength 500 nm"
- Example good options: "A) 2.48 eV", "B) 3.12 eV", "C) 1.86 eV", "D) 4.25 eV"
"""
        
        return f"{original_prompt}\n{enforcement}"


# Global instance for easy access
_numerical_validator = None

def get_numerical_validator() -> NumericalQuestionValidator:
    """Get the global numerical question validator instance"""
    global _numerical_validator
    if _numerical_validator is None:
        _numerical_validator = NumericalQuestionValidator()
    return _numerical_validator
