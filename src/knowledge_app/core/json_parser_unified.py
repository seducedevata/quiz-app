"""
Unified JSON Parser for MCQ Generation

This module provides a single, robust JSON parsing solution that replaces
all scattered JSON parsing logic across the codebase.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


class UnifiedJSONParser:
    """Single source of truth for JSON parsing in MCQ generation"""
    
    @staticmethod
    def parse_mcq_response(response: str) -> Optional[Dict[str, Any]]:
        """
        Parse MCQ response from model output
        
        Args:
            response: Raw text response from model
            
        Returns:
            Parsed MCQ dictionary or None if parsing fails
        """
        if not response:
            return None
            
        # Try direct JSON parsing first
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
            
        # Try to extract JSON from markdown blocks
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL | re.IGNORECASE)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
                
        # Try to extract JSON from plain text
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
                
        # Try structured parsing for non-JSON formats
        return UnifiedJSONParser._parse_structured_text(response)
    
    @staticmethod
    def _parse_structured_text(response: str) -> Optional[Dict[str, Any]]:
        """Parse structured text that isn't valid JSON"""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if not lines:
            return None
            
        # Look for question and options
        question = None
        options = []
        correct_answer = None
        
        for i, line in enumerate(lines):
            if not question and (line.startswith('Question:') or line.startswith('Q:')):
                question = line.split(':', 1)[1].strip()
            elif line.startswith(('A)', 'B)', 'C)', 'D)', 'E)')):
                option_text = line[2:].strip()
                options.append(option_text)
                
                # Check if this is marked as correct
                if '*' in line or line.lower().startswith('a)') and 'correct' in line.lower():
                    correct_answer = 'A'
                elif '*' in line or line.lower().startswith('b)') and 'correct' in line.lower():
                    correct_answer = 'B'
                elif '*' in line or line.lower().startswith('c)') and 'correct' in line.lower():
                    correct_answer = 'C'
                elif '*' in line or line.lower().startswith('d)') and 'correct' in line.lower():
                    correct_answer = 'D'
                elif '*' in line or line.lower().startswith('e)') and 'correct' in line.lower():
                    correct_answer = 'E'
                    
        if question and len(options) >= 2:
            # Map options to A, B, C, D, E
            option_map = {}
            for i, opt in enumerate(options[:5]):
                option_map[chr(65 + i)] = opt
                
            # If no correct answer found, default to first option
            if not correct_answer:
                correct_answer = 'A'
                
            return {
                'question': question,
                'options': option_map,
                'correct_answer': correct_answer,
                'explanation': 'Generated from structured text'
            }
            
        return None
    
    @staticmethod
    def validate_mcq_structure(mcq: Dict[str, Any]) -> bool:
        """Validate that parsed MCQ has required structure"""
        if not isinstance(mcq, dict):
            return False
            
        required_keys = ['question', 'options', 'correct_answer']
        if not all(key in mcq for key in required_keys):
            return False
            
        # Validate options structure
        options = mcq.get('options')
        if not isinstance(options, dict) or len(options) < 2:
            return False
            
        # Validate correct answer is in options
        correct = mcq.get('correct_answer')
        if correct not in options:
            return False
            
        return True
    
    @staticmethod
    def normalize_mcq_format(mcq: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize MCQ format to standard structure"""
        normalized = {
            'question': str(mcq.get('question', '')),
            'options': {},
            'correct_answer': str(mcq.get('correct_answer', 'A')),
            'explanation': str(mcq.get('explanation', ''))
        }
        
        # Normalize options
        options = mcq.get('options', {})
        if isinstance(options, dict):
            normalized['options'] = {str(k): str(v) for k, v in options.items()}
        elif isinstance(options, list):
            # Convert list to dict with A, B, C, D...
            for i, opt in enumerate(options):
                if i < 5:  # Max 5 options
                    normalized['options'][chr(65 + i)] = str(opt)
                    
        # Ensure we have at least 2 options
        if len(normalized['options']) < 2:
            normalized['options'] = {
                'A': 'True',
                'B': 'False'
            }
            normalized['correct_answer'] = 'A'
            
        return normalized


# Global instance for easy access
_parser = UnifiedJSONParser()


def parse_mcq_response(response: str) -> Optional[Dict[str, Any]]:
    """Convenience function to parse MCQ response"""
    return UnifiedJSONParser.parse_mcq_response(response)


def validate_and_normalize_mcq(mcq: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and normalize MCQ structure"""
    if UnifiedJSONParser.validate_mcq_structure(mcq):
        return UnifiedJSONParser.normalize_mcq_format(mcq)
    return None
