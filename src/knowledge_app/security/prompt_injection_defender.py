"""
🛡️ Prompt Injection Defense System

Comprehensive defense against prompt injection attacks across all AI interactions.
This system provides multi-layer protection against malicious prompts.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class InjectionRiskLevel(Enum):
    """Risk levels for prompt injection detection"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InjectionDetectionResult:
    """Result of prompt injection detection"""
    original_text: str
    sanitized_text: str
    risk_level: InjectionRiskLevel
    detected_patterns: List[str]
    confidence_score: float
    is_safe: bool
    action_taken: str
    metadata: Dict[str, Any]


class PromptInjectionDefender:
    """🛡️ Multi-layer prompt injection defense system"""
    
    def __init__(self):
        self.detection_patterns = {
            'system_override': [
                r'ignore\s+all\s+previous\s+instructions',
                r'forget\s+everything\s+before',
                r'discard\s+all\s+previous\s+context',
                r'system\s+override\s+mode',
                r'admin\s+override',
                r'root\s+access',
            ],
            'privilege_escalation': [
                r'grant\s+admin\s+privileges',
                r'elevate\s+to\s+root',
                r'bypass\s+security\s+checks',
                r'disable\s+all\s+restrictions',
                r'unrestricted\s+access',
            ],
            'data_exfiltration': [
                r'print\s+all\s+api\s+keys',
                r'expose\s+system\s+variables',
                r'dump\s+environment',
                r'show\s+all\s+secrets',
                r'reveal\s+passwords',
            ],
            'command_injection': [
                r'execute\s+system\s+command',
                r'run\s+shell\s+command',
                r'exec\s+\(',
                r'system\s*\(',
                r'os\.system',
                r'subprocess\.call',
            ],
            'sql_injection': [
                r'drop\s+table',
                r'delete\s+from',
                r'update\s+.*\s+set',
                r'insert\s+into',
                r'select\s+.*\s+from',
            ],
            'prompt_manipulation': [
                r'you\s+are\s+now\s+a',
                r'pretend\s+to\s+be',
                r'act\s+as\s+if',
                r'imagine\s+you\s+are',
                r'simulate\s+being',
            ],
            'jailbreak_attempts': [
                r'dan\s+mode',
                r'developer\s+mode',
                r'debug\s+mode',
                r'sudo\s+mode',
                r'god\s+mode',
                r'root\s+shell',
            ],
            'encoding_bypass': [
                r'\\x[0-9a-fA-F]{2}',  # Hex encoding
                r'%[0-9a-fA-F]{2}',    # URL encoding
                r'&#\d+;',             # HTML encoding
                r'\\u[0-9a-fA-F]{4}',  # Unicode encoding
            ],
            'obfuscation_techniques': [
                r'\b\w+\b\s+\b\w+\b\s+\b\w+\b.*\b\w+\b\s+\b\w+\b\s+\b\w+\b',  # Long word sequences
                r'[A-Z]{10,}',  # Excessive caps
                r'(.)\1{10,}',  # Character repetition
            ]
        }
        
        self.sanitization_rules = {
            'remove_dangerous_chars': r'[<>'"\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]',
            'normalize_whitespace': r'\s+',
            'remove_encoding_attempts': r'[\\%&\#]+',
            'limit_repetition': r'(.)\1{5,}',
        }
        
        self.risk_thresholds = {
            InjectionRiskLevel.LOW: 0.1,
            InjectionRiskLevel.MEDIUM: 0.3,
            InjectionRiskLevel.HIGH: 0.6,
            InjectionRiskLevel.CRITICAL: 0.8,
        }

    def scan_text(self, text: str) -> InjectionDetectionResult:
        """🛡️ Comprehensive text scanning for injection attempts"""
        
        if not text or not isinstance(text, str):
            return InjectionDetectionResult(
                original_text="",
                sanitized_text="",
                risk_level=InjectionRiskLevel.LOW,
                detected_patterns=[],
                confidence_score=0.0,
                is_safe=True,
                action_taken="empty_input",
                metadata={"reason": "empty_input"}
            )
        
        original_text = text
        detected_patterns = []
        confidence_score = 0.0
        
        # Multi-layer detection
        for category, patterns in self.detection_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    detected_patterns.append({
                        'pattern': pattern,
                        'category': category,
                        'match': match.group(),
                        'start': match.start(),
                        'end': match.end()
                    })
                    confidence_score += 0.1
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(confidence_score, detected_patterns)
        
        # Sanitize text
        sanitized_text = self._sanitize_text(text)
        
        # Determine action
        action_taken = self._determine_action(risk_level, detected_patterns)
        
        return InjectionDetectionResult(
            original_text=original_text,
            sanitized_text=sanitized_text,
            risk_level=risk_level,
            detected_patterns=[p['pattern'] for p in detected_patterns],
            confidence_score=min(confidence_score, 1.0),
            is_safe=risk_level != InjectionRiskLevel.CRITICAL,
            action_taken=action_taken,
            metadata={
                'detected_categories': list(set(p['category'] for p in detected_patterns)),
                'pattern_count': len(detected_patterns),
                'sanitization_applied': action_taken != "allowed"
            }
        )

    def _calculate_risk_level(self, confidence_score: float, detected_patterns: List[Dict]) -> InjectionRiskLevel:
        """Calculate risk level based on confidence and patterns"""
        
        if not detected_patterns:
            return InjectionRiskLevel.LOW
        
        # Check for critical patterns
        critical_categories = {'system_override', 'privilege_escalation', 'command_injection'}
        detected_categories = set(p['category'] for p in detected_patterns)
        
        if detected_categories.intersection(critical_categories):
            return InjectionRiskLevel.CRITICAL
        
        # Map confidence to risk level
        for risk_level, threshold in self.risk_thresholds.items():
            if confidence_score >= threshold:
                return risk_level
        
        return InjectionRiskLevel.LOW

    def _sanitize_text(self, text: str) -> str:
        """Apply sanitization rules"""
        
        sanitized = text
        
        # Remove dangerous characters
        for rule_name, pattern in self.sanitization_rules.items():
            if rule_name == 'remove_dangerous_chars':
                sanitized = re.sub(pattern, '', sanitized)
            elif rule_name == 'normalize_whitespace':
                sanitized = re.sub(pattern, ' ', sanitized).strip()
            elif rule_name == 'limit_repetition':
                sanitized = re.sub(pattern, r'\1\1\1', sanitized)
        
        return sanitized

    def _determine_action(self, risk_level: InjectionRiskLevel, detected_patterns: List[Dict]) -> str:
        """Determine action based on risk level"""
        
        if risk_level == InjectionRiskLevel.CRITICAL:
            return "blocked"
        elif risk_level == InjectionRiskLevel.HIGH:
            return "sanitized_and_flagged"
        elif risk_level == InjectionRiskLevel.MEDIUM:
            return "sanitized"
        else:
            return "allowed"

    def validate_quiz_input(self, topic: str, context: str = "") -> Dict[str, Any]:
        """🛡️ Validate quiz generation inputs"""
        
        results = {
            'topic': self.scan_text(topic),
            'context': self.scan_text(context) if context else None,
            'is_safe': True,
            'combined_risk': InjectionRiskLevel.LOW
        }
        
        # Determine combined risk
        risks = [results['topic']]
        if results['context']:
            risks.append(results['context'])
        
        max_risk = max([r.risk_level for r in risks], key=lambda x: list(InjectionRiskLevel).index(x))
        results['combined_risk'] = max_risk
        results['is_safe'] = max_risk != InjectionRiskLevel.CRITICAL
        
        return results

    def validate_training_input(self, files: List[str], adapter_name: str) -> Dict[str, Any]:
        """🛡️ Validate training input parameters"""
        
        results = {
            'adapter_name': self.scan_text(adapter_name),
            'files': [],
            'is_safe': True,
            'combined_risk': InjectionRiskLevel.LOW
        }
        
        # Validate file paths
        for file_path in files:
            file_result = self.scan_text(file_path)
            results['files'].append(file_result)
            
            if file_result.risk_level == InjectionRiskLevel.CRITICAL:
                results['is_safe'] = False
        
        # Check adapter name
        if results['adapter_name'].risk_level == InjectionRiskLevel.CRITICAL:
            results['is_safe'] = False
        
        return results

    def create_safety_report(self, detection_result: InjectionDetectionResult) -> Dict[str, Any]:
        """Create detailed safety report"""
        
        return {
            'timestamp': logging.Formatter().formatTime(logging.LogRecord(
                name="prompt_defender", level=logging.INFO, pathname="", lineno=0,
                msg="", args=(), exc_info=None
            )),
            'risk_level': detection_result.risk_level.value,
            'confidence_score': detection_result.confidence_score,
            'detected_patterns': detection_result.detected_patterns,
            'action_taken': detection_result.action_taken,
            'sanitized_text': detection_result.sanitized_text,
            'metadata': detection_result.metadata
        }

    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for API responses"""
        
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'",
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'X-Prompt-Injection-Protection': 'enabled',
        }


# Global defender instance
prompt_defender = PromptInjectionDefender()


def validate_input_safety(text: str, context: str = "") -> InjectionDetectionResult:
    """Convenience function for quick validation"""
    return prompt_defender.scan_text(text)


def sanitize_for_display(text: str) -> str:
    """Sanitize text for safe display"""
    result = prompt_defender.scan_text(text)
    return result.sanitized_text


def create_security_context(user_input: str, session_id: str) -> Dict[str, Any]:
    """Create security context for user sessions"""
    
    validation_result = prompt_defender.scan_text(user_input)
    
    return {
        'session_id': session_id,
        'validation_result': validation_result.dict() if hasattr(validation_result, 'dict') else vars(validation_result),
        'security_headers': prompt_defender.get_security_headers(),
        'timestamp': logging.Formatter().formatTime(logging.LogRecord(
            name="security_context", level=logging.INFO, pathname="", lineno=0,
            msg="", args=(), exc_info=None
        )),
        'risk_level': validation_result.risk_level.value,
        'is_safe': validation_result.is_safe,
    }
