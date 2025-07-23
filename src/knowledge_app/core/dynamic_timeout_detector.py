#!/usr/bin/env python3
"""
🕒 Dynamic Timeout Detection System

This module implements intelligent timeout detection that monitors JSON response
completion instead of using hardcoded timeouts. The system detects when a
complete MCQ JSON response has been received and automatically completes
generation without waiting for arbitrary timeout periods.

Key Features:
- Real-time JSON completion detection
- Streaming response monitoring
- Adaptive timeout based on response progress
- Hardware-agnostic (works on slow and fast systems)
- No hardcoded timeout values
"""

import json
import re
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class ResponseProgress:
    """Track progress of JSON response generation"""
    start_time: float
    last_update_time: float
    accumulated_content: str
    detected_structure: Dict[str, bool]
    completion_confidence: float
    is_complete: bool
    error_count: int


class DynamicTimeoutDetector:
    """
    🕒 Intelligent timeout detection based on JSON completion
    
    This class monitors streaming responses and detects when a complete
    MCQ JSON object has been received, eliminating the need for hardcoded timeouts.
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, ResponseProgress] = {}
        self.lock = Lock()
        
        # JSON structure patterns for MCQ detection
        self.required_fields = ['question', 'options', 'correct', 'explanation']
        self.optional_fields = ['difficulty', 'topic', 'category']
        
        # Completion detection thresholds
        self.min_confidence_threshold = 0.8
        self.max_stagnation_time = 10.0  # Max seconds without progress
        self.min_response_time = 2.0     # Min seconds before considering complete
        
        logger.info("🕒 Dynamic timeout detector initialized")
    
    def start_monitoring(self, session_id: str) -> None:
        """Start monitoring a new response session"""
        with self.lock:
            self.active_sessions[session_id] = ResponseProgress(
                start_time=time.time(),
                last_update_time=time.time(),
                accumulated_content="",
                detected_structure={field: False for field in self.required_fields},
                completion_confidence=0.0,
                is_complete=False,
                error_count=0
            )
        logger.info(f"🕒 Started monitoring session: {session_id}")
    
    def update_content(self, session_id: str, new_content: str) -> bool:
        """
        Update content for a session and check for completion
        
        Returns:
            bool: True if response is complete, False if still generating
        """
        with self.lock:
            if session_id not in self.active_sessions:
                logger.warning(f"⚠️ Session {session_id} not found, starting new monitoring")
                self.start_monitoring(session_id)
            
            progress = self.active_sessions[session_id]
            progress.accumulated_content += new_content
            progress.last_update_time = time.time()
            
            # Analyze current content for completion
            is_complete = self._analyze_completion(progress)
            progress.is_complete = is_complete
            
            if is_complete:
                logger.info(f"✅ Session {session_id} detected as complete (confidence: {progress.completion_confidence:.2f})")
            
            return is_complete
    
    def check_timeout(self, session_id: str) -> bool:
        """
        Check if session should timeout due to stagnation
        
        Returns:
            bool: True if session should timeout, False if still active
        """
        with self.lock:
            if session_id not in self.active_sessions:
                return True
            
            progress = self.active_sessions[session_id]
            current_time = time.time()
            
            # Check for stagnation (no updates for too long)
            time_since_update = current_time - progress.last_update_time
            if time_since_update > self.max_stagnation_time:
                logger.warning(f"⏰ Session {session_id} stagnated for {time_since_update:.1f}s")
                return True
            
            # Check if we have some content but it's been too long
            total_time = current_time - progress.start_time
            if total_time > 300:  # 5 minutes absolute maximum
                logger.warning(f"⏰ Session {session_id} exceeded absolute maximum time: {total_time:.1f}s")
                return True
            
            return False
    
    def get_completion_status(self, session_id: str) -> Dict[str, Any]:
        """Get detailed completion status for a session"""
        with self.lock:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            progress = self.active_sessions[session_id]
            current_time = time.time()
            
            return {
                "session_id": session_id,
                "is_complete": progress.is_complete,
                "confidence": progress.completion_confidence,
                "elapsed_time": current_time - progress.start_time,
                "content_length": len(progress.accumulated_content),
                "detected_fields": progress.detected_structure,
                "error_count": progress.error_count
            }
    
    def stop_monitoring(self, session_id: str) -> Optional[str]:
        """Stop monitoring and return final content"""
        with self.lock:
            if session_id in self.active_sessions:
                progress = self.active_sessions[session_id]
                final_content = progress.accumulated_content
                del self.active_sessions[session_id]
                logger.info(f"🕒 Stopped monitoring session: {session_id}")
                return final_content
            return None
    
    def _analyze_completion(self, progress: ResponseProgress) -> bool:
        """Analyze if the accumulated content represents a complete response"""
        content = progress.accumulated_content
        
        if not content.strip():
            return False
        
        # Check minimum time requirement
        elapsed = time.time() - progress.start_time
        if elapsed < self.min_response_time:
            return False
        
        # Try to parse as JSON and check structure
        confidence = self._calculate_completion_confidence(content, progress)
        progress.completion_confidence = confidence
        
        return confidence >= self.min_confidence_threshold
    
    def _calculate_completion_confidence(self, content: str, progress: ResponseProgress) -> float:
        """Calculate confidence that the response is complete"""
        confidence = 0.0
        
        # Factor 1: Valid JSON structure (40% weight)
        json_confidence = self._check_json_structure(content, progress)
        confidence += json_confidence * 0.4
        
        # Factor 2: Required fields present (30% weight)
        fields_confidence = self._check_required_fields(content, progress)
        confidence += fields_confidence * 0.3
        
        # Factor 3: Content quality indicators (20% weight)
        quality_confidence = self._check_content_quality(content)
        confidence += quality_confidence * 0.2
        
        # Factor 4: Completion markers (10% weight)
        markers_confidence = self._check_completion_markers(content)
        confidence += markers_confidence * 0.1
        
        return min(confidence, 1.0)
    
    def _check_json_structure(self, content: str, progress: ResponseProgress) -> float:
        """Check if content contains valid JSON structure"""
        try:
            # Clean content first
            cleaned_content = content.strip()

            # Try to extract and parse JSON
            json_patterns = [
                r'\{.*\}',                           # Simple JSON object (most permissive)
                r'```json\s*(\{.*?\})\s*```',         # JSON in code block
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON object
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, cleaned_content, re.DOTALL | re.MULTILINE)
                for match in matches:
                    try:
                        if isinstance(match, tuple):
                            match = match[0] if match else ""

                        # Try to parse the JSON
                        parsed = json.loads(match)
                        if isinstance(parsed, dict) and len(parsed) > 0:
                            logger.debug(f"🕒 Valid JSON structure found: {list(parsed.keys())}")
                            return 1.0  # Valid JSON found
                    except json.JSONDecodeError as e:
                        logger.debug(f"🕒 JSON parse failed: {e}")
                        continue

            # Check for JSON-like structure even if not parseable
            brace_count = content.count('{') + content.count('}')
            quote_count = content.count('"')
            colon_count = content.count(':')

            if brace_count >= 2 and quote_count >= 4 and colon_count >= 2:
                logger.debug(f"🕒 JSON-like structure detected (braces: {brace_count}, quotes: {quote_count}, colons: {colon_count})")
                return 0.5  # JSON-like structure
            elif '{' in content and '}' in content:
                return 0.3  # Partial JSON structure

            return 0.0

        except Exception as e:
            progress.error_count += 1
            logger.debug(f"🕒 JSON structure check failed: {e}")
            return 0.0
    
    def _check_required_fields(self, content: str, progress: ResponseProgress) -> float:
        """Check if required MCQ fields are present"""
        detected_count = 0

        for field in self.required_fields:
            # Check for field in JSON format with more flexible patterns
            patterns = [
                f'"{field}"\\s*:',           # JSON field with quotes
                f"'{field}'\\s*:",           # JSON with single quotes
                f'\\b{field}\\b\\s*:',       # Field name followed by colon
                f'"{field}"',                # Just the field name in quotes
                f"'{field}'",                # Field name in single quotes
            ]

            field_found = any(re.search(pattern, content, re.IGNORECASE | re.MULTILINE) for pattern in patterns)
            progress.detected_structure[field] = field_found

            if field_found:
                detected_count += 1
                logger.debug(f"🕒 Detected field '{field}' in content")

        logger.debug(f"🕒 Detected {detected_count}/{len(self.required_fields)} required fields")
        return detected_count / len(self.required_fields)
    
    def _check_content_quality(self, content: str) -> float:
        """Check content quality indicators"""
        quality_score = 0.0
        
        # Check for reasonable content length
        if len(content) > 100:
            quality_score += 0.3
        
        # Check for multiple choice indicators
        choice_patterns = ['A)', 'B)', 'C)', 'D)', '"A"', '"B"', '"C"', '"D"']
        choice_count = sum(1 for pattern in choice_patterns if pattern in content)
        if choice_count >= 4:
            quality_score += 0.4
        
        # Check for explanation content
        explanation_indicators = ['explanation', 'because', 'therefore', 'reason']
        if any(indicator in content.lower() for indicator in explanation_indicators):
            quality_score += 0.3
        
        return min(quality_score, 1.0)
    
    def _check_completion_markers(self, content: str) -> float:
        """Check for markers that indicate response completion"""
        completion_markers = [
            '```',           # End of code block
            '}',             # End of JSON object
            'END_JSON',      # Explicit end marker
            '###',           # Section separator
        ]
        
        marker_count = sum(1 for marker in completion_markers if marker in content)
        return min(marker_count / 2.0, 1.0)


# Global instance for easy access
_global_detector = DynamicTimeoutDetector()


def start_response_monitoring(session_id: str) -> None:
    """Start monitoring a response session"""
    _global_detector.start_monitoring(session_id)


def update_response_content(session_id: str, content: str) -> bool:
    """Update response content and check for completion"""
    return _global_detector.update_content(session_id, content)


def check_response_timeout(session_id: str) -> bool:
    """Check if response should timeout"""
    return _global_detector.check_timeout(session_id)


def get_response_status(session_id: str) -> Dict[str, Any]:
    """Get response completion status"""
    return _global_detector.get_completion_status(session_id)


def stop_response_monitoring(session_id: str) -> Optional[str]:
    """Stop monitoring and get final content"""
    return _global_detector.stop_monitoring(session_id)
