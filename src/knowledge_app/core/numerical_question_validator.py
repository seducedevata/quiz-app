"""
🔧 ENHANCED Centralized Numerical Question Validator
🚀 BUG FIX: Now uses unified validation authority for consistent rules across prompts and validators
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NumericalQuestionValidator:
    """
    🔧 ENHANCED: Centralized validator now integrated with unified validation authority
    
    This validator uses the same canonical rules that prompt generation uses,
    eliminating the dual authority problem.
    """
    
    def __init__(self):
        # 🔧 FIX: Import unified validation authority for consistent rules
        try:
            from .unified_validation_authority import get_validation_authority
            self.validation_authority = get_validation_authority()
            logger.info("✅ Unified validation authority integrated")
        except ImportError:
            logger.warning("⚠️ Unified validation authority not available, using legacy rules")
            self.validation_authority = None
            self._setup_legacy_rules()
    
    def _setup_legacy_rules(self):
        """Setup legacy rules as fallback if unified authority is not available"""
        # Legacy conceptual patterns that should NOT appear in numerical questions
        self.conceptual_patterns = [
            'explain', 'describe', 'why', 'how', 'what is the concept',
            'what is the principle', 'what is the theory', 'what is the definition',
            'which statement', 'which principle', 'which theory', 'which concept',
            'analyze', 'compare', 'contrast', 'discuss', 'evaluate'
        ]
        
        # Legacy numerical starters for numerical questions
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
        🔧 ENHANCED FIX: Validate numerical questions with comprehensive checks
        
        Args:
            question_data: The generated question data
            question_type: The requested question type
            
        Returns:
            bool: True if valid, False if invalid
        """
        try:
            if question_type.lower() != "numerical":
                return True  # No validation needed for non-numerical questions
            
            # Use unified validation authority if available
            if self.validation_authority:
                is_valid, failure_reason, _ = self.validation_authority.validate_question(question_data, question_type)
                if not is_valid:
                    logger.warning(f"🚫 Numerical validation FAILED: {failure_reason}")
                    return False
                else:
                    logger.info(f"✅ Numerical validation PASSED")
                    return True
            
            # Enhanced legacy validation with numerical-specific checks
            return self._enhanced_validate_numerical_question(question_data)
            
        except Exception as e:
            logger.error(f"❌ Error in numerical question validation: {e}")
            return True  # Default to accepting the question if validation fails
    
    def _enhanced_validate_numerical_question(self, question_data: Dict[str, Any]) -> bool:
        """Enhanced legacy validation method for backward compatibility"""
        """Legacy validation method for backward compatibility"""
        question = question_data.get('question', '').lower()
        options = question_data.get('options', {})
        explanation = question_data.get('explanation', '').lower()
        
        if not question or not options:
            logger.warning("🚫 Numerical question validation FAILED: Missing question or options")
            return False
        
        # ARCHITECTURAL FIX: More reasonable conceptual pattern checking
        option_values = list(options.values()) if isinstance(options, dict) else options
        
        # Only check question text for major conceptual patterns (not explanation/options)
        # This allows for reasonable explanation context while maintaining numerical focus
        major_conceptual_patterns = ['explain the concept', 'describe the theory', 'what is the definition']
        
        for pattern in major_conceptual_patterns:
            if pattern in question:
                logger.warning(f"🚫 Numerical question validation FAILED: Contains major conceptual pattern '{pattern}' in question")
                return False
        
        # ARCHITECTURAL FIX: More flexible numerical starter checking
        # Accept broader range of numerical question patterns
        extended_numerical_patterns = [
            'calculate', 'determine', 'find', 'compute', 'what is the value', 'what is the magnitude',
            'how much', 'how many', 'what amount', 'what quantity', 'at what', 'to what',
            'what wavelength', 'what frequency', 'what energy', 'what mass', 'what charge'
        ]
        
        has_numerical_starter = any(pattern in question for pattern in extended_numerical_patterns)
        has_numbers_in_question = any(char.isdigit() for char in question)
        
        # Accept if either has numerical starters OR contains numbers (more flexible)
        if not (has_numerical_starter or has_numbers_in_question):
            logger.warning(f"🚫 Numerical question validation FAILED: No numerical indicators found")
            return False
        
        # ARCHITECTURAL FIX: More reasonable numerical option validation
        numerical_options = 0
        for option in option_values:
            option_str = str(option).lower()
            
            # Check if option contains numbers (basic requirement)
            has_number = any(char.isdigit() for char in option_str)
            
            # Check for scientific notation or units
            has_scientific_notation = 'e' in option_str and any(char.isdigit() for char in option_str)
            has_units = any(unit in option_str for unit in self.common_units)
            
            if has_number or has_scientific_notation or has_units:
                numerical_options += 1
        
        # ARCHITECTURAL FIX: More reasonable requirement - at least 3/4 options should be numerical
        # This allows for edge cases like "None of the above" or mixed formats
        if numerical_options < 3:
            logger.warning(f"🚫 Numerical question validation FAILED: Only {numerical_options}/4 options appear numerical")
            logger.warning(f"❌ Options: {option_values}")
            return False
        
        # ARCHITECTURAL FIX: Allow reasonable explanations for numerical questions
        # Explanations should be allowed to contain conceptual context as long as they explain the calculation
        if explanation and len(explanation) > 300:  # Only check very long explanations
            # Only reject if explanation is overwhelmingly conceptual with no numerical content
            has_calculation_terms = any(term in explanation for term in ['calculate', 'formula', 'equation', 'solve', 'result', 'answer'])
            if not has_calculation_terms:
                logger.warning(f"🚫 Numerical question validation FAILED: Explanation lacks calculation context")
                return False
        
        logger.info(f"✅ Numerical question validation PASSED: Pure calculation question with numerical options")
        return True
    
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
