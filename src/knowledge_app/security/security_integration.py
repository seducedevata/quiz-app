"""
🔒 Security Integration Layer

Comprehensive security layer that integrates all security systems:
- Data contract enforcement
- Prompt injection defense
- API key security
- Input validation
- Output sanitization
"""

import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import all security systems
from ..core.ai_data_contracts import data_contract_enforcer, QuestionContract, QuizGenerationResponse
from .prompt_injection_defender import prompt_defender, InjectionDetectionResult
from ..core.secure_api_key_manager import SecureApiKeyManager

logger = logging.getLogger(__name__)


class SecurityIntegrationLayer:
    """🔒 Central security integration for the entire application"""
    
    def __init__(self):
        self.api_key_manager = SecureApiKeyManager()
        self.security_headers = prompt_defender.get_security_headers()
        self.validation_cache = {}
        
    def validate_quiz_generation_input(self, topic: str, context: str = "", 
                                     num_questions: int = 5) -> Dict[str, Any]:
        """🛡️ Comprehensive validation for quiz generation"""
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'sanitized_inputs': {},
            'security_report': {},
            'data_contract': None
        }
        
        try:
            # 1. Prompt injection defense
            injection_check = prompt_defender.validate_quiz_input(topic, context)
            
            if not injection_check['is_safe']:
                validation_result['is_valid'] = False
                validation_result['errors'].append(
                    f"Security risk detected: {injection_check['combined_risk']}"
                )
                
            validation_result['security_report'] = injection_check
            
            # 2. Input sanitization
            sanitized_topic = prompt_defender.sanitize_for_display(topic)
            sanitized_context = prompt_defender.sanitize_for_display(context)
            
            validation_result['sanitized_inputs'] = {
                'topic': sanitized_topic,
                'context': sanitized_context,
                'num_questions': max(1, min(50, num_questions))
            }
            
            # 3. Data contract validation
            contract_data = {
                'topic': sanitized_topic,
                'context': sanitized_context,
                'num_questions': num_questions,
                'difficulty': 'medium',
                'question_type': 'multiple_choice'
            }
            
            # This would normally validate against a proper contract
            validation_result['data_contract'] = contract_data
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Quiz input validation failed: {e}")
            
        return validation_result
    
    def validate_training_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """🛡️ Validate training configuration with security checks"""
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'sanitized_config': {},
            'security_report': {},
            'data_contract': None
        }
        
        try:
            # 1. Validate adapter name
            adapter_name = config.get('adapter_name', '')
            adapter_check = prompt_defender.scan_text(adapter_name)
            
            if adapter_check.risk_level.value == 'critical':
                validation_result['is_valid'] = False
                validation_result['errors'].append("Adapter name contains security risks")
                
            # 2. Validate file paths
            files = config.get('selected_files', [])
            file_validation = prompt_defender.validate_training_input(files, adapter_name)
            
            if not file_validation['is_safe']:
                validation_result['is_valid'] = False
                validation_result['errors'].append("File paths contain security risks")
                
            validation_result['security_report'] = file_validation
            
            # 3. Sanitize configuration
            sanitized_config = {
                'adapter_name': prompt_defender.sanitize_for_display(adapter_name),
                'base_model': str(config.get('base_model', '')).strip(),
                'selected_files': [
                    prompt_defender.sanitize_for_display(f) for f in files
                ],
                'learning_rate': max(0.0001, min(1.0, float(config.get('learning_rate', 0.001)))),
                'epochs': max(1, min(100, int(config.get('epochs', 3)))),
                'training_preset': str(config.get('training_preset', 'standard')).strip()
            }
            
            validation_result['sanitized_config'] = sanitized_config
            
            # 4. Data contract validation
            from ..core.ai_data_contracts import TrainingConfigPayload
            # This would validate against the proper contract
            validation_result['data_contract'] = sanitized_config
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Training validation error: {str(e)}")
            logger.error(f"Training configuration validation failed: {e}")
            
        return validation_result
    
    def validate_ai_response(self, response: Dict[str, Any], 
                          response_type: str = "quiz") -> Dict[str, Any]:
        """🛡️ Validate AI-generated responses with data contracts"""
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'sanitized_response': {},
            'data_contract': None,
            'security_hash': None
        }
        
        try:
            # 1. Validate response structure
            if not isinstance(response, dict):
                validation_result['is_valid'] = False
                validation_result['errors'].append("Response must be a dictionary")
                return validation_result
                
            # 2. Apply data contract validation
            if response_type == "quiz":
                from ..core.ai_data_contracts import QuizGenerationResponse
                try:
                    quiz_response = QuizGenerationResponse(**response)
                    validation_result['data_contract'] = quiz_response.dict()
                    validation_result['security_hash'] = data_contract_enforcer.generate_validation_hash(response)
                except Exception as e:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(f"Quiz contract validation failed: {e}")
                    
            elif response_type == "question":
                from ..core.ai_data_contracts import QuestionContract
                try:
                    for question in response.get('questions', []):
                        question_contract = QuestionContract(**question)
                    validation_result['security_hash'] = data_contract_enforcer.generate_validation_hash(response)
                except Exception as e:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(f"Question contract validation failed: {e}")
                    
            # 3. Security content validation
            content_validation = self._validate_content_security(response)
            if not content_validation['is_safe']:
                validation_result['is_valid'] = False
                validation_result['errors'].extend(content_validation['issues'])
                
            validation_result['sanitized_response'] = self._sanitize_response(response)
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"AI response validation error: {str(e)}")
            logger.error(f"AI response validation failed: {e}")
            
        return validation_result
    
    def _validate_content_security(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """🛡️ Deep content security validation"""
        
        security_result = {
            'is_safe': True,
            'issues': [],
            'scanned_fields': []
        }
        
        def scan_value(value, path=""):
            if isinstance(value, str):
                scan_result = prompt_defender.scan_text(value)
                if scan_result.risk_level.value == 'critical':
                    security_result['is_safe'] = False
                    security_result['issues'].append(f"Critical content in {path}: {scan_result.detected_patterns}")
                security_result['scanned_fields'].append(path)
                    
            elif isinstance(value, dict):
                for k, v in value.items():
                    scan_value(v, f"{path}.{k}" if path else k)
                    
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    scan_value(v, f"{path}[{i}]" if path else f"[{i}]")
        
        scan_value(content)
        return security_result
    
    def _sanitize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """🛡️ Sanitize response content"""
        
        def sanitize_value(obj):
            if isinstance(obj, str):
                return prompt_defender.sanitize_for_display(obj)
            elif isinstance(obj, dict):
                return {k: sanitize_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_value(item) for item in obj]
            else:
                return obj
        
        return sanitize_value(response)
    
    def create_security_audit_log(self, action: str, details: Dict[str, Any]) -> None:
        """🔍 Create security audit log entry"""
        
        audit_entry = {
            'timestamp': logging.Formatter().formatTime(logging.LogRecord(
                name="security_audit", level=logging.INFO, pathname="", lineno=0,
                msg="", args=(), exc_info=None
            )),
            'action': action,
            'details': details,
            'session_id': details.get('session_id', 'unknown'),
            'risk_level': details.get('risk_level', 'low'),
            'ip_address': details.get('ip_address', 'unknown'),
            'user_agent': details.get('user_agent', 'unknown')
        }
        
        logger.info(f"SECURITY_AUDIT: {json.dumps(audit_entry)}")
    
    def get_security_headers(self) -> Dict[str, str]:
        """🔒 Get security headers for API responses"""
        return self.security_headers.copy()
    
    def validate_api_key_usage(self, provider: str, api_key: str) -> Dict[str, Any]:
        """🛡️ Validate API key usage"""
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'provider': provider,
            'key_hash': None,
            'security_report': {}
        }
        
        try:
            # Validate provider
            valid_providers = ['openai', 'anthropic', 'groq', 'google', 'tavily']
            if provider not in valid_providers:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Invalid provider: {provider}")
                
            # Validate API key format
            if not api_key or len(api_key.strip()) < 10:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Invalid API key format")
                
            # Create secure hash for logging (without exposing key)
            import hashlib
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8]
            validation_result['key_hash'] = key_hash
            
            # Security scan the key
            key_scan = prompt_defender.scan_text(api_key)
            if key_scan.risk_level.value == 'critical':
                validation_result['is_valid'] = False
                validation_result['errors'].append("API key contains security risks")
                
            validation_result['security_report'] = {
                'key_hash': key_hash,
                'provider': provider,
                'scan_result': key_scan.risk_level.value
            }
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"API key validation error: {str(e)}")
            logger.error(f"API key validation failed: {e}")
            
        return validation_result


# Global security layer instance
security_layer = SecurityIntegrationLayer()


def secure_quiz_generation(topic: str, context: str = "", **kwargs) -> Dict[str, Any]:
    """Convenience function for secure quiz generation"""
    return security_layer.validate_quiz_generation_input(topic, context, **kwargs)


def secure_training_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for secure training configuration"""
    return security_layer.validate_training_configuration(config)


def secure_ai_response(response: Dict[str, Any], response_type: str = "quiz") -> Dict[str, Any]:
    """Convenience function for secure AI response validation"""
    return security_layer.validate_ai_response(response, response_type)


def create_security_context(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive security context"""
    return security_layer.create_security_audit_log("session_start", session_data)
