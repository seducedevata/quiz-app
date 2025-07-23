"""
Advanced MCQ Validation System - The Quality Sieve

This module implements comprehensive validation for MCQ quality,
ensuring only high-quality questions reach the user.
"""

import json
import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of MCQ validation"""

    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: List[str]
    suggestions: List[str]


class MCQValidator:
    """
    The Quality Sieve - Comprehensive MCQ validation system

    This validator ensures that generated MCQs meet high quality standards
    before being presented to users.
    """

    def __init__(self):
        self.min_question_length = 10
        self.max_question_length = 300
        self.min_option_length = 3
        self.max_option_length = 200
        self.min_explanation_length = 20

    def validate_mcq(self, mcq_data: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive validation of MCQ data

        Args:
            mcq_data: Dictionary containing MCQ data

        Returns:
            ValidationResult with validation status and details
        """
        issues = []
        suggestions = []
        score = 1.0

        try:
            # 1. Structure validation
            structure_score, structure_issues = self._validate_structure(mcq_data)
            issues.extend(structure_issues)
            score *= structure_score

            if structure_score < 0.5:
                return ValidationResult(False, score, issues, suggestions)

            # 2. Content quality validation
            content_score, content_issues, content_suggestions = self._validate_content_quality(
                mcq_data
            )
            issues.extend(content_issues)
            suggestions.extend(content_suggestions)
            score *= content_score

            # 3. Option quality validation
            option_score, option_issues, option_suggestions = self._validate_options(mcq_data)
            issues.extend(option_issues)
            suggestions.extend(option_suggestions)
            score *= option_score

            # 4. Answer validation
            answer_score, answer_issues = self._validate_answer(mcq_data)
            issues.extend(answer_issues)
            score *= answer_score

            # 5. Explanation validation
            explanation_score, explanation_issues = self._validate_explanation(mcq_data)
            issues.extend(explanation_issues)
            score *= explanation_score

            # Determine if valid (score threshold)
            is_valid = score >= 0.7 and len(issues) == 0

            logger.info(f"MCQ validation completed: score={score:.2f}, valid={is_valid}")

            return ValidationResult(is_valid, score, issues, suggestions)

        except Exception as e:
            logger.error(f"Error during MCQ validation: {e}")
            return ValidationResult(False, 0.0, [f"Validation error: {e}"], [])

    def _validate_structure(self, mcq_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Validate basic JSON structure"""
        issues = []
        score = 1.0

        # Required keys - CRITICAL FIX: Accept both 'correct' and 'correct_answer'
        required_keys = ["question", "options", "explanation"]
        for key in required_keys:
            if key not in mcq_data:
                issues.append(f"Missing required key: {key}")
                score *= 0.5

        # Check for either 'correct' or 'correct_answer' field
        if "correct" not in mcq_data and "correct_answer" not in mcq_data:
            issues.append("Missing required key: correct or correct_answer")
            score *= 0.5

        # Options structure
        if "options" in mcq_data:
            options = mcq_data["options"]
            if not isinstance(options, dict):
                issues.append("Options must be a dictionary")
                score *= 0.3
            else:
                expected_keys = ["A", "B", "C", "D"]
                for key in expected_keys:
                    if key not in options:
                        issues.append(f"Missing option: {key}")
                        score *= 0.8

        # Correct answer validation
        if "correct" in mcq_data:
            correct = mcq_data["correct"]
            if correct not in ["A", "B", "C", "D"]:
                issues.append(f"Invalid correct answer format: {correct}")
                score *= 0.5

        return score, issues

    def _validate_content_quality(
        self, mcq_data: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """Validate content quality"""
        issues = []
        suggestions = []
        score = 1.0

        question = mcq_data.get("question", "")

        # Length validation
        if len(question) < self.min_question_length:
            issues.append(
                f"Question too short: {len(question)} chars (min: {self.min_question_length})"
            )
            score *= 0.6
        elif len(question) > self.max_question_length:
            issues.append(
                f"Question too long: {len(question)} chars (max: {self.max_question_length})"
            )
            score *= 0.8

        # Question mark validation
        if not question.strip().endswith("?"):
            suggestions.append("Question should end with a question mark")
            score *= 0.95

        # Avoid simple recall questions
        simple_patterns = [r"^What is\s+\w+\?$", r"^Define\s+\w+", r"^List\s+", r"^Name\s+"]

        for pattern in simple_patterns:
            if re.match(pattern, question, re.IGNORECASE):
                suggestions.append(
                    "Consider creating a question that requires analysis rather than simple recall"
                )
                score *= 0.9
                break

        # Check for vague questions
        vague_words = ["something", "anything", "things", "stuff", "it"]
        if any(word in question.lower() for word in vague_words):
            suggestions.append("Question contains vague terms - consider being more specific")
            score *= 0.9

        return score, issues, suggestions

    def _validate_options(self, mcq_data: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Validate option quality"""
        issues = []
        suggestions = []
        score = 1.0

        options = mcq_data.get("options", {})

        if not options:
            return 0.0, ["No options provided"], []

        option_texts = list(options.values())

        # Length validation
        for i, option in enumerate(option_texts):
            if len(option) < self.min_option_length:
                issues.append(f"Option {list(options.keys())[i]} too short: {len(option)} chars")
                score *= 0.8
            elif len(option) > self.max_option_length:
                issues.append(f"Option {list(options.keys())[i]} too long: {len(option)} chars")
                score *= 0.9

        # Check for duplicate options
        if len(set(option_texts)) != len(option_texts):
            issues.append("Duplicate options detected")
            score *= 0.5

        # Check for obviously wrong options
        obvious_wrong = ["None of the above", "All of the above", "I don't know"]
        for option in option_texts:
            if option in obvious_wrong:
                suggestions.append(
                    "Avoid using 'None/All of the above' - create specific distractors"
                )
                score *= 0.9

        # Check option length consistency
        lengths = [len(option) for option in option_texts]
        if max(lengths) > 2 * min(lengths):
            suggestions.append(
                "Options have very different lengths - consider making them more consistent"
            )
            score *= 0.95

        return score, issues, suggestions

    def _validate_answer(self, mcq_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Validate correct answer - CRITICAL FIX: Handle both field names"""
        issues = []
        score = 1.0

        # CRITICAL FIX: Check both possible field names
        correct = mcq_data.get("correct", mcq_data.get("correct_answer", ""))
        options = mcq_data.get("options", {})

        if correct not in options:
            issues.append(f"Correct answer '{correct}' not found in options")
            score = 0.0

        return score, issues

    def _validate_explanation(self, mcq_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Validate explanation quality"""
        issues = []
        score = 1.0

        explanation = mcq_data.get("explanation", "")

        # Length validation
        if len(explanation) < self.min_explanation_length:
            issues.append(
                f"Explanation too short: {len(explanation)} chars (min: {self.min_explanation_length})"
            )
            score *= 0.7

        # Check if explanation actually explains
        if not any(
            word in explanation.lower()
            for word in ["because", "since", "due to", "therefore", "thus"]
        ):
            issues.append(
                "Explanation should include reasoning words (because, since, therefore, etc.)"
            )
            score *= 0.9

        return score, issues


def validate_mcq_json(json_string: str) -> ValidationResult:
    """
    Validate MCQ from JSON string

    Args:
        json_string: JSON string containing MCQ data

    Returns:
        ValidationResult
    """
    try:
        mcq_data = json.loads(json_string)
        validator = MCQValidator()
        return validator.validate_mcq(mcq_data)
    except json.JSONDecodeError as e:
        return ValidationResult(
            False, 0.0, [f"Invalid JSON: {e}"], ["Ensure response is valid JSON"]
        )
    except Exception as e:
        return ValidationResult(False, 0.0, [f"Validation error: {e}"], [])


def fix_common_mcq_issues(mcq_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automatically fix common MCQ issues

    Args:
        mcq_data: MCQ data dictionary

    Returns:
        Fixed MCQ data
    """
    fixed_data = mcq_data.copy()

    try:
        # Fix question format
        if "question" in fixed_data:
            question = fixed_data["question"].strip()
            if not question.endswith("?"):
                question += "?"
            fixed_data["question"] = question

        # Fix correct answer format
        if "correct" in fixed_data:
            correct = str(fixed_data["correct"]).upper().strip()
            if correct in ["A", "B", "C", "D"]:
                fixed_data["correct"] = correct

        # Ensure options are strings
        if "options" in fixed_data and isinstance(fixed_data["options"], dict):
            for key, value in fixed_data["options"].items():
                fixed_data["options"][key] = str(value).strip()

        # Ensure explanation exists
        if "explanation" not in fixed_data or not fixed_data["explanation"]:
            fixed_data["explanation"] = "This is the correct answer based on the provided context."

        logger.info("Applied automatic MCQ fixes")

    except Exception as e:
        logger.error(f"Error fixing MCQ issues: {e}")

    return fixed_data