#!/usr/bin/env python3
"""
🔧 FIX: Centralized JSON parsing utility to eliminate code duplication

This module provides a single, robust JSON parsing implementation that can be
used by both online and offline MCQ generators, eliminating the DRY violation.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class CentralizedJSONParser:
    """
    🔧 FIX: Centralized JSON parser with multiple parsing strategies
    
    This replaces the duplicated JSON parsing logic found in both:
    - src/knowledge_app/core/online_mcq_generator.py
    - src/knowledge_app/core/offline_mcq_generator.py
    """
    
    def __init__(self):
        self.parsing_stats = {
            "total_attempts": 0,
            "successful_parses": 0,
            "method_1_success": 0,  # Code fence extraction
            "method_2_success": 0,  # Direct JSON parsing
            "method_3_success": 0,  # Cleaned parsing
            "method_4_success": 0,  # Fixed JSON issues
        }
    
    def parse_json_response_robust(self, content: str) -> Optional[Dict[str, Any]]:
        """
        🚀 ENHANCED: Robust JSON extraction with multiple parsing strategies
        
        This is the centralized implementation that replaces the duplicated logic.
        """
        self.parsing_stats["total_attempts"] += 1
        
        if not content or not content.strip():
            return None
        
        # Handle streaming/chunked responses
        accumulated_content = self._concatenate_streaming_chunks(content)
        
        # Method 1: Look for JSON blocks with code fences
        result = self._try_code_fence_extraction(accumulated_content)
        if result:
            self.parsing_stats["method_1_success"] += 1
            self.parsing_stats["successful_parses"] += 1
            return result
        
        # Method 2: Try direct JSON parsing
        result = self._try_direct_json_parsing(accumulated_content)
        if result:
            self.parsing_stats["method_2_success"] += 1
            self.parsing_stats["successful_parses"] += 1
            return result
        
        # Method 3: Try parsing after aggressive cleaning
        result = self._try_cleaned_parsing(accumulated_content)
        if result:
            self.parsing_stats["method_3_success"] += 1
            self.parsing_stats["successful_parses"] += 1
            return result
        
        # Method 4: Try to fix common JSON issues
        result = self._try_fixed_json_issues(accumulated_content)
        if result:
            self.parsing_stats["method_4_success"] += 1
            self.parsing_stats["successful_parses"] += 1
            return result
        
        logger.error(f"❌ All JSON parsing methods failed for content: {content[:200]}...")
        return None
    
    def _concatenate_streaming_chunks(self, content: str) -> str:
        """Handle streaming/chunked responses by concatenating all content"""
        # For now, just return the content as-is
        # This can be enhanced later to handle actual streaming chunks
        return content.strip()
    
    def _try_code_fence_extraction(self, content: str) -> Optional[Dict[str, Any]]:
        """Method 1: Look for JSON blocks with code fences"""
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_block_match:
            json_str = json_block_match.group(1)
            try:
                parsed = json.loads(json_str)
                if self._validate_json_structure_robust(parsed):
                    return parsed
            except:
                pass
        return None
    
    def _try_direct_json_parsing(self, content: str) -> Optional[Dict[str, Any]]:
        """Method 2: Try direct JSON parsing"""
        try:
            parsed = json.loads(content)
            if self._validate_json_structure_robust(parsed):
                return parsed
        except:
            pass
        return None
    
    def _try_cleaned_parsing(self, content: str) -> Optional[Dict[str, Any]]:
        """Method 3: Try parsing after aggressive cleaning"""
        try:
            # Clean response more aggressively
            cleaned = content.strip()
            
            # Remove common intro patterns
            intro_patterns = [
                r'Here is the (?:valid )?JSON object.*?:',
                r'Here is the (?:generated )?JSON.*?:',
                r'Here is the JSON structure.*?:',
                r'Based on.*?here is.*?:',
                r'The JSON object.*?:',
                r'Here\'s the.*?:',
                r'.*?PhD-level.*?question.*?:'
            ]
            for pattern in intro_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove code fences
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            elif cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Remove any leading/trailing text before/after JSON
            start_brace = cleaned.find('{')
            end_brace = cleaned.rfind('}')
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                json_part = cleaned[start_brace:end_brace + 1]
                parsed = json.loads(json_part)
                if self._validate_json_structure_robust(parsed):
                    return parsed
        except:
            pass
        return None
    
    def _try_fixed_json_issues(self, content: str) -> Optional[Dict[str, Any]]:
        """Method 4: Try to fix common JSON issues"""
        try:
            fixed_response = self._fix_common_json_issues(content)
            if fixed_response:
                parsed = json.loads(fixed_response)
                if self._validate_json_structure_robust(parsed):
                    return parsed
        except:
            pass
        return None
    
    def _fix_common_json_issues(self, content: str) -> Optional[str]:
        """Fix common JSON formatting issues"""
        if not content:
            return None
        
        # Extract JSON part
        start_brace = content.find('{')
        end_brace = content.rfind('}')
        if start_brace == -1 or end_brace == -1:
            return None
        
        json_part = content[start_brace:end_brace + 1]
        
        # Fix common issues
        fixes = [
            # Fix trailing commas
            (r',(\s*[}\]])', r'\1'),
            # Fix missing quotes around keys
            (r'(\w+):', r'"\1":'),
            # Fix single quotes to double quotes
            (r"'([^']*)'", r'"\1"'),
            # Fix unescaped quotes in strings
            (r'(?<!\\)"(?=[^"]*"[^"]*$)', r'\\"'),
        ]
        
        for pattern, replacement in fixes:
            json_part = re.sub(pattern, replacement, json_part)
        
        return json_part
    
    def _validate_json_structure_robust(self, parsed_json: Dict) -> bool:
        """Validate that the JSON has the required MCQ structure"""
        if not isinstance(parsed_json, dict):
            return False
        
        # Check required fields
        required_fields = ['question', 'options', 'correct', 'explanation']
        for field in required_fields:
            if field not in parsed_json:
                return False
        
        # Check question is non-empty string
        question = parsed_json.get('question', '')
        if not isinstance(question, str) or not question.strip():
            return False
        
        # Check options format (can be dict or list)
        options = parsed_json.get('options', [])
        if isinstance(options, dict):
            # Handle {"A": "...", "B": "...", "C": "...", "D": "..."} format
            if len(options) < 2:
                return False
            for key, value in options.items():
                if not isinstance(value, str) or not value.strip():
                    return False
        elif isinstance(options, list):
            # Handle ["...", "...", "...", "..."] format
            if len(options) < 2:
                return False
            for option in options:
                if not isinstance(option, str) or not option.strip():
                    return False
        else:
            return False
        
        # Check correct answer
        correct = parsed_json.get('correct', '')
        if not isinstance(correct, str) or not correct.strip():
            return False
        
        # Check explanation
        explanation = parsed_json.get('explanation', '')
        if not isinstance(explanation, str) or not explanation.strip():
            return False
        
        return True
    
    def get_parsing_stats(self) -> Dict[str, Any]:
        """Get statistics about parsing success rates"""
        total = self.parsing_stats["total_attempts"]
        if total == 0:
            return {"success_rate": 0.0, "method_breakdown": {}}
        
        return {
            "success_rate": self.parsing_stats["successful_parses"] / total,
            "total_attempts": total,
            "successful_parses": self.parsing_stats["successful_parses"],
            "method_breakdown": {
                "code_fence": self.parsing_stats["method_1_success"],
                "direct_json": self.parsing_stats["method_2_success"],
                "cleaned_parsing": self.parsing_stats["method_3_success"],
                "fixed_issues": self.parsing_stats["method_4_success"],
            }
        }


# Global instance for easy access
_global_parser = CentralizedJSONParser()


def parse_json_response_robust(content: str) -> Optional[Dict[str, Any]]:
    """
    🔧 FIX: Global function for robust JSON parsing
    
    This is the main entry point that both online and offline generators should use.
    """
    return _global_parser.parse_json_response_robust(content)


def get_json_parsing_stats() -> Dict[str, Any]:
    """Get global JSON parsing statistics"""
    return _global_parser.get_parsing_stats()


def validate_mcq_json_structure(parsed_json: Dict) -> bool:
    """Validate MCQ JSON structure"""
    return _global_parser._validate_json_structure_robust(parsed_json)
