"""
� UNIFIED VALIDATION AUTHORITY - Evolved for Trust-Based Prompt Architecture

This module provides clean, intelligent validation that works harmoniously with
the new "One Prompt to Rule Them All" system. It validates AI output quality
without imposing the constraint-heavy mindset of the legacy system.

EVOLUTION:
- FROM: Constraint-based rules with FORBIDDEN/BANNED lists
- TO: Quality-focused validation that trusts AI intelligence
- INTEGRATION: Seamless cooperation with UnifiedPromptBuilder

The validation authority now focuses on:
✅ Structural integrity (proper JSON, required fields)
✅ Academic appropriateness (difficulty-matched complexity)
✅ Content quality (substantial options, clear explanations)
✅ Format consistency (not rigid constraints)
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class UnifiedValidationAuthority:
    """
    � EVOLVED VALIDATION AUTHORITY - Quality-Focused, Trust-Based Validation
    
    This class provides intelligent validation that complements the unified prompt system:
    1. Validates structural integrity and format compliance
    2. Assesses academic appropriateness and content quality  
    3. Provides constructive feedback for improvement
    4. Maintains consistency without rigid constraints
    
    Works harmoniously with UnifiedPromptBuilder for seamless question generation.
    """
    
    def __init__(self):
        # Quality standards for different difficulty levels
        self.difficulty_standards = {
            'easy': {
                'min_option_length': 80,
                'min_explanation_length': 100,
                'complexity_keywords': ['basic', 'fundamental', 'simple', 'introductory'],
                'academic_level': 'high school'
            },
            'medium': {
                'min_option_length': 100,
                'min_explanation_length': 150,
                'complexity_keywords': ['analyze', 'apply', 'connect', 'moderate'],
                'academic_level': 'undergraduate'
            },
            'hard': {
                'min_option_length': 120,
                'min_explanation_length': 200,
                'complexity_keywords': ['advanced', 'sophisticated', 'complex', 'graduate'],
                'academic_level': 'graduate level'
            },
            'expert': {
                'min_option_length': 150,
                'min_explanation_length': 250,
                'complexity_keywords': ['cutting-edge', 'research-level', 'specialized', 'interdisciplinary'],
                'academic_level': 'PhD/research level'
            }
        }
        
        # Content quality indicators (positive approach)
        self.quality_indicators = {
            'numerical': {
                'good_starters': ['calculate', 'determine', 'find', 'compute', 'solve', 'evaluate'],
                'content_markers': ['numbers', 'units', 'formulas', 'equations', 'values'],
                'explanation_elements': ['calculation', 'derivation', 'formula', 'method']
            },
            'conceptual': {
                'good_starters': ['explain', 'why', 'how', 'describe', 'analyze', 'what'],
                'content_markers': ['principle', 'theory', 'concept', 'mechanism', 'process'],
                'explanation_elements': ['understanding', 'reasoning', 'logic', 'analysis']
            }
        }
    
    def validate_question_quality(self, question_data: Dict[str, Any], 
                                 difficulty: str = "medium", 
                                 question_type: str = "mixed") -> Tuple[bool, str, Dict[str, Any]]:
        """
        🎯 INTELLIGENT QUALITY VALIDATION - Focus on excellence, not constraints
        
        Validates question quality using positive indicators rather than restrictive rules.
        Works cooperatively with the UnifiedPromptBuilder system.
        
        Args:
            question_data: Generated question data
            difficulty: Expected difficulty level
            question_type: Type of question (numerical, conceptual, mixed)
            
        Returns:
            Tuple[bool, str, Dict]: (is_valid, feedback_message, quality_metrics)
        """
        try:
            # 1. Structural integrity check
            structure_valid, structure_msg = self._validate_structure(question_data)
            if not structure_valid:
                return False, structure_msg, {"structural_issues": True}
            
            # 2. Difficulty appropriateness check
            difficulty_valid, difficulty_msg, metrics = self._validate_difficulty_appropriateness(
                question_data, difficulty
            )
            
            # 3. Content quality assessment
            quality_score = self._assess_content_quality(question_data, question_type)
            metrics["quality_score"] = quality_score
            
            # 4. Overall validation decision
            overall_valid = structure_valid and difficulty_valid and quality_score >= 0.7
            
            if overall_valid:
                return True, f"✅ Excellent question quality (score: {quality_score:.2f})", metrics
            else:
                feedback = self._generate_constructive_feedback(structure_msg, difficulty_msg, quality_score)
                return False, feedback, metrics
                
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return False, f"Validation failed due to technical error: {e}", {"error": True}
    
    def _validate_structure(self, question_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate basic structural requirements"""
        required_fields = ['question', 'options', 'correct', 'explanation']
        
        for field in required_fields:
            if field not in question_data:
                return False, f"Missing required field: {field}"
        
        # Question should end with ?
        if not question_data['question'].strip().endswith('?'):
            return False, "Question should end with a question mark"
        
        # Should have 4 options
        options = question_data['options']
        if isinstance(options, dict):
            if len(options) != 4 or not all(k in options for k in ['A', 'B', 'C', 'D']):
                return False, "Must have exactly 4 options labeled A, B, C, D"
        elif isinstance(options, list):
            if len(options) != 4:
                return False, "Must have exactly 4 options"
        else:
            return False, "Options must be either dict with A/B/C/D keys or list of 4 items"
        
        # Valid correct answer
        if question_data['correct'] not in ['A', 'B', 'C', 'D']:
            return False, "Correct answer must be A, B, C, or D"
        
        return True, "Structure is valid"
    
    def _validate_difficulty_appropriateness(self, question_data: Dict[str, Any], 
                                           difficulty: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate that content matches expected difficulty level"""
        standards = self.difficulty_standards.get(difficulty, self.difficulty_standards['medium'])
        metrics = {}
        
        # Check option lengths
        options = question_data['options']
        option_values = list(options.values()) if isinstance(options, dict) else options
        
        avg_option_length = sum(len(str(opt)) for opt in option_values) / len(option_values)
        metrics["avg_option_length"] = avg_option_length
        
        min_required = standards['min_option_length']
        if avg_option_length < min_required:
            return False, f"Options too brief for {difficulty} level (avg: {avg_option_length:.0f}, need: {min_required}+)", metrics
        
        # Check explanation length
        explanation_length = len(question_data['explanation'])
        metrics["explanation_length"] = explanation_length
        
        min_explanation = standards['min_explanation_length']
        if explanation_length < min_explanation:
            return False, f"Explanation too brief for {difficulty} level ({explanation_length} chars, need: {min_explanation}+)", metrics
        
        # Check for complexity indicators
        all_text = f"{question_data['question']} {' '.join(str(opt) for opt in option_values)} {question_data['explanation']}".lower()
        
        complexity_indicators = sum(1 for keyword in standards['complexity_keywords'] if keyword in all_text)
        metrics["complexity_indicators"] = complexity_indicators
        
        return True, f"Content appropriate for {difficulty} level", metrics
    
    def _assess_content_quality(self, question_data: Dict[str, Any], question_type: str) -> float:
        """Assess overall content quality using positive indicators"""
        quality_score = 0.0
        max_score = 5.0
        
        question_text = question_data['question'].lower()
        options = question_data['options']
        option_values = list(options.values()) if isinstance(options, dict) else options
        explanation = question_data['explanation'].lower()
        
        # 1. Question starter quality (1 point)
        if question_type in self.quality_indicators:
            good_starters = self.quality_indicators[question_type]['good_starters']
            if any(starter in question_text for starter in good_starters):
                quality_score += 1.0
        else:
            quality_score += 0.5  # Neutral for mixed type
        
        # 2. Content depth indicators (1 point)
        if question_type in self.quality_indicators:
            content_markers = self.quality_indicators[question_type]['content_markers']
            all_content = f"{question_text} {' '.join(str(opt).lower() for opt in option_values)}"
            marker_count = sum(1 for marker in content_markers if marker in all_content)
            quality_score += min(1.0, marker_count / 3.0)  # Up to 1 point for content markers
        
        # 3. Option quality (1.5 points)
        option_quality = 0.0
        for opt in option_values:
            opt_str = str(opt).strip()
            if len(opt_str) > 50:  # Substantial content
                option_quality += 0.25
            if len(opt_str) > 100:  # Very detailed
                option_quality += 0.125
        quality_score += min(1.5, option_quality)
        
        # 4. Explanation quality (1 point)
        if question_type in self.quality_indicators:
            explanation_elements = self.quality_indicators[question_type]['explanation_elements']
            element_count = sum(1 for element in explanation_elements if element in explanation)
            quality_score += min(1.0, element_count / 2.0)
        else:
            quality_score += 0.5 if len(explanation) > 100 else 0.3
        
        # 5. Overall coherence (0.5 points)
        if len(explanation) > 150 and not any(word in explanation for word in ['error', 'fail', 'invalid']):
            quality_score += 0.5
        
        return min(1.0, quality_score / max_score)
    
    def _generate_constructive_feedback(self, structure_msg: str, difficulty_msg: str, quality_score: float) -> str:
        """Generate constructive feedback for improvement"""
        feedback_parts = []
        
        if "Structure is valid" not in structure_msg:
            feedback_parts.append(f"🔧 Structure: {structure_msg}")
        
        if quality_score < 0.7:
            feedback_parts.append(f"📈 Quality: Score {quality_score:.2f}/1.0 - needs more depth and detail")
        
        if feedback_parts:
            return "Areas for improvement:\n" + "\n".join(feedback_parts)
        else:
            return difficulty_msg
    
    def get_improvement_suggestions(self, question_data: Dict[str, Any], 
                                  difficulty: str, question_type: str) -> List[str]:
        """
        🎯 CONSTRUCTIVE GUIDANCE - Suggest specific improvements
        
        Provides actionable suggestions rather than just identifying problems.
        """
        suggestions = []
        
        options = question_data.get('options', {})
        option_values = list(options.values()) if isinstance(options, dict) else options
        
        # Length-based suggestions
        standards = self.difficulty_standards.get(difficulty, self.difficulty_standards['medium'])
        avg_length = sum(len(str(opt)) for opt in option_values) / len(option_values)
        
        if avg_length < standards['min_option_length']:
            suggestions.append(f"💡 Expand options to {standards['min_option_length']}+ characters for {difficulty} level")
        
        explanation_length = len(question_data.get('explanation', ''))
        if explanation_length < standards['min_explanation_length']:
            suggestions.append(f"💡 Provide more detailed explanation ({standards['min_explanation_length']}+ characters)")
        
        # Content-specific suggestions
        if question_type == 'numerical':
            all_text = str(question_data).lower()
            if not any(indicator in all_text for indicator in ['calculate', 'determine', 'find']):
                suggestions.append("💡 Use calculation-focused language (calculate, determine, find)")
            if not any(char.isdigit() for char in all_text):
                suggestions.append("💡 Include specific numerical values and units")
        
        elif question_type == 'conceptual':
            if not any(word in str(question_data).lower() for word in ['explain', 'why', 'how']):
                suggestions.append("💡 Focus on conceptual understanding (explain why/how)")
        
        return suggestions

    def create_enhancement_for_retry(self, failure_reason: str, attempt: int, 
                                   difficulty: str, question_type: str) -> str:
        """
        🔄 CONSTRUCTIVE RETRY GUIDANCE - Help AI improve, don't just constrain
        
        Creates enhancement instructions that guide the AI toward better output
        rather than just listing what went wrong.
        """
        urgency = "🎯 IMPROVEMENT NEEDED" if attempt == 1 else "🚨 FINAL OPPORTUNITY"
        
        # Get specific guidance based on failure
        specific_guidance = self._get_targeted_guidance(failure_reason, difficulty, question_type)
        
        enhancement = f"""
{urgency} - Let's create an exceptional question together

Previous feedback: {failure_reason}

{specific_guidance}

Your expertise is valued - let's channel it into creating the perfect {difficulty}-level question that demonstrates the academic rigor expected at this level.
"""
        return enhancement
    
    def _get_targeted_guidance(self, failure_reason: str, difficulty: str, question_type: str) -> str:
        """Generate specific, positive guidance based on the failure reason"""
        guidance_parts = []
        
        # Difficulty-specific guidance
        standards = self.difficulty_standards.get(difficulty, self.difficulty_standards['medium'])
        guidance_parts.append(f"🎓 Target {standards['academic_level']} complexity")
        guidance_parts.append(f"📏 Aim for {standards['min_option_length']}+ characters per option")
        
        # Content-specific guidance  
        if question_type in self.quality_indicators:
            indicators = self.quality_indicators[question_type]
            guidance_parts.append(f"🎯 Consider starting with: {', '.join(indicators['good_starters'][:3])}")
            guidance_parts.append(f"💡 Include elements like: {', '.join(indicators['content_markers'][:3])}")
        
        # Specific issue guidance
        if "too brief" in failure_reason.lower():
            guidance_parts.append("📝 Expand with more detail, examples, and comprehensive explanations")
        
        if "structure" in failure_reason.lower():
            guidance_parts.append("🏗️ Ensure proper JSON format with question, options (A-D), correct answer, and explanation")
        
        return "\n".join(guidance_parts)


# 🎯 CLEAN INTEGRATION FUNCTIONS

_validation_authority_instance: Optional[UnifiedValidationAuthority] = None

def get_validation_authority() -> UnifiedValidationAuthority:
    """Get the global unified validation authority instance"""
    global _validation_authority_instance
    if _validation_authority_instance is None:
        _validation_authority_instance = UnifiedValidationAuthority()
        logger.debug("✅ Initialized UnifiedValidationAuthority")
    return _validation_authority_instance

def validate_generated_question(question_data: Dict[str, Any], 
                               difficulty: str = "medium", 
                               question_type: str = "mixed") -> Tuple[bool, str, Dict[str, Any]]:
    """
    🎯 MAIN VALIDATION FUNCTION - Clean interface for question validation
    
    Args:
        question_data: Generated question to validate
        difficulty: Expected difficulty level
        question_type: Type of question
        
    Returns:
        Tuple[bool, str, Dict]: (is_valid, feedback, quality_metrics)
    """
    authority = get_validation_authority()
    return authority.validate_question_quality(question_data, difficulty, question_type)

def get_enhancement_for_retry(failure_reason: str, attempt: int, 
                            difficulty: str, question_type: str) -> str:
    """
    🔄 RETRY ENHANCEMENT - Get constructive guidance for question improvement
    """
    authority = get_validation_authority()
    return authority.create_enhancement_for_retry(failure_reason, attempt, difficulty, question_type)
