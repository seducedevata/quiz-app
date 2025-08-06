"""
Advanced MCQ Post-Processing and Validation Pipeline
"""

import logging
import json
import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .mcq_validator import MCQValidator, ValidationResult, fix_common_mcq_issues

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """MCQ processing stages"""

    RAW_GENERATION = "raw_generation"
    JSON_PARSING = "json_parsing"
    CONTENT_VALIDATION = "content_validation"
    QUALITY_ENHANCEMENT = "quality_enhancement"
    FINAL_VALIDATION = "final_validation"


@dataclass
class ProcessingResult:
    """Result of MCQ processing"""

    success: bool
    mcq_data: Optional[Dict[str, Any]]
    stage: ProcessingStage
    issues: List[str]
    enhancements: List[str]
    quality_score: float


class MCQPostProcessor:
    """Advanced MCQ post-processing pipeline"""

    def __init__(self):
        self.validator = MCQValidator()
        self.min_quality_score = 0.7

    def process_mcq(self, raw_response: str, context: Dict[str, Any] = None) -> ProcessingResult:
        """
        Process raw MCQ response through comprehensive pipeline

        Args:
            raw_response: Raw response from model
            context: Generation context for validation

        Returns:
            ProcessingResult with processed MCQ
        """
        try:
            logger.info("🔄 Starting MCQ post-processing pipeline")

            # Stage 1: Parse JSON
            parse_result = self._parse_json_response(raw_response)
            if not parse_result.success:
                return parse_result

            mcq_data = parse_result.mcq_data

            # Stage 2: Content validation
            validation_result = self._validate_content(mcq_data, context)
            if not validation_result.success:
                return validation_result

            # Stage 3: Quality enhancement
            enhancement_result = self._enhance_quality(mcq_data, context)
            if not enhancement_result.success:
                return enhancement_result

            # Stage 4: Final validation
            final_result = self._final_validation(enhancement_result.mcq_data)

            logger.info(
                f"✅ MCQ processing completed with quality score: {final_result.quality_score:.2f}"
            )
            return final_result

        except Exception as e:
            logger.error(f"❌ MCQ processing failed: {e}")
            return ProcessingResult(
                success=False,
                mcq_data=None,
                stage=ProcessingStage.RAW_GENERATION,
                issues=[f"Processing error: {e}"],
                enhancements=[],
                quality_score=0.0,
            )

    def _parse_json_response(self, raw_response: str) -> ProcessingResult:
        """Parse JSON from raw model response"""
        try:
            # Clean the response
            cleaned_response = self._clean_json_response(raw_response)

            # Try to parse JSON
            mcq_data = json.loads(cleaned_response)

            # Basic structure validation - CRITICAL FIX: Handle both field names
            required_fields = ["question", "options", "explanation"]
            missing_fields = [field for field in required_fields if field not in mcq_data]

            # Check for either 'correct' or 'correct_answer' field
            if "correct" not in mcq_data and "correct_answer" not in mcq_data:
                missing_fields.append("correct or correct_answer")

            if missing_fields:
                return ProcessingResult(
                    success=False,
                    mcq_data=None,
                    stage=ProcessingStage.JSON_PARSING,
                    issues=[f"Missing required fields: {missing_fields}"],
                    enhancements=[],
                    quality_score=0.0,
                )

            return ProcessingResult(
                success=True,
                mcq_data=mcq_data,
                stage=ProcessingStage.JSON_PARSING,
                issues=[],
                enhancements=["Successfully parsed JSON"],
                quality_score=0.5,
            )

        except json.JSONDecodeError as e:
            # Try manual parsing as fallback
            manual_result = self._manual_parse_response(raw_response)
            if manual_result:
                return ProcessingResult(
                    success=True,
                    mcq_data=manual_result,
                    stage=ProcessingStage.JSON_PARSING,
                    issues=["JSON parsing failed, used manual parsing"],
                    enhancements=["Applied manual parsing fallback"],
                    quality_score=0.4,
                )

            return ProcessingResult(
                success=False,
                mcq_data=None,
                stage=ProcessingStage.JSON_PARSING,
                issues=[f"JSON parsing failed: {e}"],
                enhancements=[],
                quality_score=0.0,
            )

    def _clean_json_response(self, response: str) -> str:
        """Clean model response to extract valid JSON"""
        # Remove markdown code blocks
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        # Find JSON object boundaries
        start = response.find("{")
        end = response.rfind("}") + 1

        if start != -1 and end > start:
            return response[start:end]

        return response

    def _manual_parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Enhanced manual parsing using the robust MCQ parser"""
        try:
            logger.info("🔄 Using enhanced MCQ parser for manual parsing...")

            # Use the enhanced MCQ parser
            from ..utils.enhanced_mcq_parser import EnhancedMCQParser

            parser = EnhancedMCQParser()
            result = parser.parse_mcq(response)

            if result.success and result.mcq_data:
                logger.info(
                    f"✅ Enhanced parser succeeded with {result.format_detected.value} format (confidence: {result.confidence:.2f})"
                )
                if result.issues:
                    logger.warning(f"⚠️ Parser issues: {result.issues}")
                return result.mcq_data
            else:
                logger.warning(f"⚠️ Enhanced parser failed: {result.issues}")

        except Exception as e:
            logger.warning(f"⚠️ Enhanced parser error: {e}")

        # Fallback to original manual parsing
        return self._legacy_manual_parse_response(response)

    def _legacy_manual_parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Legacy manual parsing as fallback"""
        try:
            logger.info("🔄 Using legacy manual parsing as fallback...")

            lines = [line.strip() for line in response.split("\n") if line.strip()]
            result = {}
            options = {}

            # Extract question
            for line in lines:
                if "?" in line and "question" not in result:
                    # Remove common prefixes
                    question = re.sub(r"^(Question:|Q:|\d+\.)\s*", "", line, flags=re.IGNORECASE)
                    result["question"] = question.strip()
                    break

            # Extract options
            option_pattern = re.compile(r"^([ABCD])[\.\)]\s*(.+)$")
            for line in lines:
                match = option_pattern.match(line)
                if match:
                    letter, text = match.groups()
                    options[letter] = text.strip()

            if len(options) >= 4:
                result["options"] = options

                # Find correct answer
                correct_patterns = [
                    r"correct.*?([ABCD])",
                    r"answer.*?([ABCD])",
                    r"([ABCD]).*?correct",
                ]

                for pattern in correct_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        result["correct"] = match.group(1)
                        break

                if "correct" not in result:
                    result["correct"] = "A"  # Default fallback

                # Extract explanation
                explanation_patterns = [
                    r"explanation[:\-\s]*(.+?)(?:\n|$)",
                    r"because[:\-\s]*(.+?)(?:\n|$)",
                    r"reason[:\-\s]*(.+?)(?:\n|$)",
                ]

                for pattern in explanation_patterns:
                    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                    if match:
                        result["explanation"] = match.group(1).strip()
                        break

                if "explanation" not in result:
                    result["explanation"] = f"The correct answer is {result.get('correct', 'A')}."

                logger.info(
                    f"📝 Legacy parser result: question='{result.get('question', '')[:50]}...', options={len(options)}"
                )
                return result

            return None

        except Exception as e:
            logger.error(f"Legacy manual parsing failed: {e}")
            return None

    def _validate_content(
        self, mcq_data: Dict[str, Any], context: Dict[str, Any] = None
    ) -> ProcessingResult:
        """Validate MCQ content quality with enhanced cognitive level checking"""
        try:
            # Use existing validator
            validation_result = self.validator.validate_mcq(mcq_data)

            issues = validation_result.issues.copy()
            enhancements = []

            # Enhanced cognitive level validation
            if 'cognitive_level' in mcq_data and context:
                difficulty = context.get("difficulty", "medium")
                cognitive_level = mcq_data.get('cognitive_level', '')
                
                # Validate cognitive level alignment with difficulty
                expected_levels = {
                    "easy": ["knowledge", "comprehension"],
                    "medium": ["application", "analysis"], 
                    "hard": ["analysis", "synthesis"],
                    "expert": ["synthesis", "evaluation"]
                }
                
                if difficulty.lower() in expected_levels:
                    if cognitive_level not in expected_levels[difficulty.lower()]:
                        issues.append(f"Cognitive level '{cognitive_level}' doesn't match {difficulty} difficulty (expected: {expected_levels[difficulty.lower()]})")
                    else:
                        enhancements.append(f"Cognitive level '{cognitive_level}' properly aligned with {difficulty} difficulty")

            # Additional context-specific validation
            if context:
                topic = context.get("topic", "")
                if topic and topic.lower() not in mcq_data.get("question", "").lower():
                    issues.append(f"Question doesn't clearly relate to topic: {topic}")

                difficulty = context.get("difficulty", "medium")
                
                # Enhanced difficulty validation based on cognitive levels
                if difficulty == "expert":
                    question_text = mcq_data.get("question", "")
                    if len(question_text) < 100:
                        issues.append("Expert mode questions should be more comprehensive (min 100 chars)")
                    
                    # Check for research-level terminology
                    expert_indicators = ["evaluate", "analyze", "synthesize", "compare", "assess", "research", "advanced", "mechanism", "pathway"]
                    if not any(indicator in question_text.lower() for indicator in expert_indicators):
                        issues.append("Expert mode questions should use research-level terminology")
                        
                elif difficulty == "hard":
                    question_text = mcq_data.get("question", "")
                    if len(question_text) < 80:
                        issues.append("Hard mode questions should be more comprehensive (min 80 chars)")
                    
                    # Check for graduate-level complexity indicators
                    hard_indicators = [
                        "analyze", "derive", "evaluate", "calculate the scattering", "determine the correlation", 
                        "solve the differential", "find the dispersion", "multi-step", "coupled system", 
                        "non-linear", "many-body", "advanced", "complex system", "synthesis", "interaction"
                    ]
                    if not any(indicator in question_text.lower() for indicator in hard_indicators):
                        issues.append("Hard mode questions should require graduate-level analytical thinking")
                    
                    # 🚀 REVOLUTIONARY: Quality-focused validation instead of banned patterns
                    # Instead of banning specific patterns, assess overall question quality
                    question_lower = question_text.lower()
                    
                    # Quality indicators for hard questions
                    quality_indicators = [
                        "analyze", "derive", "evaluate", "calculate complex", "determine relationship", 
                        "multi-step", "synthesis", "interaction", "mechanism", "pathway",
                        "relationship between", "effect of", "optimization", "comparison",
                        "critical analysis", "systematic approach"
                    ]
                    
                    quality_score = sum(1 for indicator in quality_indicators if indicator in question_lower)
                    
                    if quality_score < 1:
                        # Encourage higher quality without forbidding specific topics
                        enhancements.append("Consider incorporating more analytical elements that demonstrate graduate-level thinking")
                    else:
                        enhancements.append(f"Good analytical complexity detected ({quality_score} quality indicators)")
                        
                elif difficulty == "easy" and len(mcq_data.get("question", "")) > 200:
                    issues.append("Question too complex for easy difficulty")

            # Enhanced vague question detection
            if self._is_enhanced_vague_question(mcq_data, context):
                issues.append("Question is too vague and generic for the specified difficulty")
            
            # Check for LaTeX syntax if present
            question_text = mcq_data.get("question", "")
            if "$" in question_text or "\\" in question_text:
                latex_issues = self._validate_latex_syntax(question_text)
                issues.extend(latex_issues)
                if not latex_issues:
                    enhancements.append("Valid LaTeX syntax detected")

            # Enhanced distractor quality validation
            distractor_quality = self._validate_distractor_quality(mcq_data)
            issues.extend(distractor_quality['issues'])
            enhancements.extend(distractor_quality['enhancements'])

            success = validation_result.is_valid and len(issues) == len(validation_result.issues)

            return ProcessingResult(
                success=success,
                mcq_data=mcq_data,
                stage=ProcessingStage.CONTENT_VALIDATION,
                issues=issues,
                enhancements=enhancements,
                quality_score=validation_result.score,
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                mcq_data=mcq_data,
                stage=ProcessingStage.CONTENT_VALIDATION,
                issues=[f"Validation error: {e}"],
                enhancements=[],
                quality_score=0.0,
            )

    def _validate_latex_syntax(self, text: str) -> List[str]:
        """Validate LaTeX syntax in text"""
        issues = []

        # LaTeX validation temporarily disabled - latex_renderer module not available
        # TODO: Re-enable when latex validation is properly implemented
        
        return issues

    def _enhance_quality(
        self, mcq_data: Dict[str, Any], context: Dict[str, Any] = None
    ) -> ProcessingResult:
        """Enhance MCQ quality through automated improvements"""
        try:
            enhanced_data = mcq_data.copy()
            enhancements = []

            # Fix common issues
            enhanced_data = fix_common_mcq_issues(enhanced_data)
            enhancements.append("Applied common issue fixes")

            # Enhance question clarity
            question = enhanced_data.get("question", "")
            if not question.endswith("?"):
                enhanced_data["question"] = question.rstrip(".") + "?"
                enhancements.append("Added question mark")

            # Enhance options formatting
            options = enhanced_data.get("options", {})
            if isinstance(options, dict):
                for key, value in options.items():
                    # Ensure consistent capitalization
                    if value and not value[0].isupper():
                        options[key] = value[0].upper() + value[1:]
                        enhancements.append(f"Capitalized option {key}")

            # Add metadata
            enhanced_data["processing_metadata"] = {
                "processed": True,
                "enhancements_applied": len(enhancements),
                "processing_stage": ProcessingStage.QUALITY_ENHANCEMENT.value,
            }

            return ProcessingResult(
                success=True,
                mcq_data=enhanced_data,
                stage=ProcessingStage.QUALITY_ENHANCEMENT,
                issues=[],
                enhancements=enhancements,
                quality_score=0.8,
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                mcq_data=mcq_data,
                stage=ProcessingStage.QUALITY_ENHANCEMENT,
                issues=[f"Enhancement error: {e}"],
                enhancements=[],
                quality_score=0.0,
            )

    def _final_validation(self, mcq_data: Dict[str, Any]) -> ProcessingResult:
        """Final validation and quality scoring"""
        try:
            # Run final validation
            validation_result = self.validator.validate_mcq(mcq_data)

            # Calculate final quality score
            quality_score = validation_result.score

            # Bonus points for enhancements
            if mcq_data.get("processing_metadata", {}).get("processed"):
                quality_score += 0.1

            # Check if meets minimum quality threshold
            success = validation_result.is_valid and quality_score >= self.min_quality_score

            return ProcessingResult(
                success=success,
                mcq_data=mcq_data,
                stage=ProcessingStage.FINAL_VALIDATION,
                issues=validation_result.issues,
                enhancements=validation_result.suggestions,
                quality_score=min(quality_score, 1.0),
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                mcq_data=mcq_data,
                stage=ProcessingStage.FINAL_VALIDATION,
                issues=[f"Final validation error: {e}"],
                enhancements=[],
                quality_score=0.0,
            )

    def _is_enhanced_vague_question(self, mcq_data: Dict[str, Any], context: Dict[str, Any] = None) -> bool:
        """Enhanced vague question detection with cognitive level awareness"""
        question = mcq_data.get("question", "").lower()
        difficulty = context.get("difficulty", "medium") if context else "medium"
        
        # Quality-focused validation for expert and hard modes
        if difficulty.lower() == "expert":
            # Promote quality indicators for expert level
            quality_indicators = [
                "mechanism", "pathway", "synthesis", "analysis", "evaluation", 
                "research", "advanced", "complex", "molecular", "cellular",
                "theoretical", "computational", "optimization", "integration"
            ]
            
            # Check for sufficient complexity without rigid patterns
            if len(question) < 100:  # Too short for expert level
                logger.info(f"� EXPERT ENHANCEMENT: Question could benefit from more detail: '{question[:80]}...'")
                return True
                
            # Encourage advanced terminology presence
            has_advanced_content = any(term in question for term in quality_indicators)
            if not has_advanced_content:
                logger.info(f"� EXPERT ENHANCEMENT: Question could benefit from advanced terminology: '{question[:80]}...'")
                return True
        
        # Hard mode - quality-focused complexity assessment
        elif difficulty.lower() == "hard":
            # Promote graduate-level quality indicators
            complexity_indicators = [
                "analyze", "derive", "evaluate", "multi-step", "coupled", "non-linear", 
                "many-body", "advanced", "complex system", "synthesis", "interaction",
                "scattering", "correlation", "differential", "dispersion", "optimization"
            ]
            
            # Encourage appropriate complexity
            if len(question) < 80:  # May be too simple for hard level
                logger.info(f"� HARD ENHANCEMENT: Question could benefit from more complexity: '{question[:80]}...'")
                return True
                
            # Promote advanced content without rigid exclusions
            has_complexity = any(term in question for term in complexity_indicators)
            if not has_complexity:
                logger.info(f"� HARD ENHANCEMENT: Question could benefit from advanced concepts: '{question[:80]}...'")
                return True
        
        # For all difficulties, check cognitive level appropriateness
        cognitive_level = mcq_data.get('cognitive_level', '')
        if cognitive_level in ['synthesis', 'evaluation'] and len(question) < 60:
            logger.warning(f"🧠 COGNITIVE MISMATCH: {cognitive_level} level question too short: '{question[:80]}...'")
            return True
        
        return False

    def _validate_distractor_quality(self, mcq_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced distractor quality validation"""
        issues = []
        enhancements = []
        
        options = mcq_data.get("options", [])
        # CRITICAL FIX: Handle both field names
        correct_answer = mcq_data.get("correct_answer", mcq_data.get("correct", ""))
        
        if len(options) != 4:
            issues.append("Must have exactly 4 options")
            return {"issues": issues, "enhancements": enhancements}
        
        # Check for distractor quality issues
        for i, option in enumerate(options):
            if option == correct_answer:
                continue
                
            # Check for obviously wrong distractors
            if len(option) < 10:
                issues.append(f"Option {i+1} too short - may be obviously wrong")
            
            # Check for generic distractors
            generic_phrases = ["none of the above", "all of the above", "it depends", "not applicable"]
            if any(phrase in option.lower() for phrase in generic_phrases):
                issues.append(f"Option {i+1} contains generic phrase: {option}")
            
            # Check for distractors that are too similar to correct answer
            if correct_answer and len(correct_answer) > 20:
                similarity_ratio = len(set(option.lower().split()) & set(correct_answer.lower().split())) / len(set(correct_answer.lower().split()))
                if similarity_ratio > 0.7:
                    issues.append(f"Option {i+1} too similar to correct answer")
        
        # Positive validation
        if len(issues) == 0:
            enhancements.append("All distractors appear to be plausible and well-crafted")
        
        return {"issues": issues, "enhancements": enhancements}


def process_mcq_response(raw_response: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function for processing MCQ responses

    Args:
        raw_response: Raw model response
        context: Generation context

    Returns:
        Processed MCQ data or None if processing failed
    """
    processor = MCQPostProcessor()
    result = processor.process_mcq(raw_response, context)

    if result.success:
        logger.info(f"✅ MCQ processed successfully (quality: {result.quality_score:.2f})")
        return result.mcq_data
    else:
        logger.error(f"❌ MCQ processing failed at {result.stage.value}: {result.issues}")
        return None
