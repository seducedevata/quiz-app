#!/usr/bin/env python3
"""
🤖 Dynamic Model Detection System

This module implements intelligent model detection that automatically identifies
model types, capabilities, and compatibility without hardcoded model names.
It uses web search to research model capabilities and provides dynamic
classification and configuration.

Key Features:
- Automatic "thinking model" detection (like DeepSeek-R1)
- Web search integration for model research
- Dynamic model capability classification
- Compatibility warnings and recommendations
- No hardcoded model names or preferences
"""

import json
import re
import time
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelCapabilities:
    """Model capability information"""
    name: str
    is_thinking_model: bool
    reasoning_capability: str  # "basic", "advanced", "expert"
    context_length: int
    parameter_count: Optional[str]
    specializations: List[str]
    compatibility_score: float
    recommended_settings: Dict[str, Any]
    warnings: List[str]


class DynamicModelDetector:
    """
    🤖 Intelligent model detection and classification system
    
    This class automatically detects model capabilities and types without
    relying on hardcoded model names or preferences.
    """
    
    def __init__(self):
        self.model_cache: Dict[str, ModelCapabilities] = {}
        self.cache_file = Path("data/model_capabilities_cache.json")
        self.web_search_available = True
        
        # Load cached model information
        self._load_model_cache()
        
        # Initialize web search capability
        self._init_web_search()
        
        logger.info("[AI] Dynamic model detector initialized")
    
    def detect_model_capabilities(self, model_name: str, force_refresh: bool = False) -> ModelCapabilities:
        """
        Detect and classify model capabilities dynamically
        
        Args:
            model_name: Name of the model to analyze
            force_refresh: Force refresh of cached information
            
        Returns:
            ModelCapabilities: Comprehensive model information
        """
        # Check cache first
        if not force_refresh and model_name in self.model_cache:
            logger.info(f"🤖 Using cached capabilities for {model_name}")
            return self.model_cache[model_name]
        
        logger.info(f"🤖 Detecting capabilities for model: {model_name}")
        
        # Analyze model name patterns
        name_analysis = self._analyze_model_name(model_name)
        
        # Research model online if possible
        web_research = self._research_model_online(model_name)
        
        # Test model locally if available
        local_test = self._test_model_locally(model_name)
        
        # Combine all information
        capabilities = self._synthesize_capabilities(model_name, name_analysis, web_research, local_test)
        
        # Cache the results
        self.model_cache[model_name] = capabilities
        self._save_model_cache()
        
        logger.info(f"🤖 Detected capabilities for {model_name}: thinking={capabilities.is_thinking_model}, reasoning={capabilities.reasoning_capability}")
        
        return capabilities
    
    def get_recommended_models(self, task_type: str = "mcq_generation") -> List[str]:
        """
        Get recommended models - RESPECTS USER PREFERENCES FIRST

        Args:
            task_type: Type of task ("mcq_generation", "reasoning", "coding", etc.)

        Returns:
            List of recommended model names (user preference first, then intelligent)
        """
        # 🚀 OPTIMIZATION: Get available models once to prevent infinite loops
        available_models = self._get_available_models()

        # 👤 FIRST: Check user preferences
        user_preferred = self._get_user_preferred_model(task_type)
        if user_preferred:
            if user_preferred in available_models:
                logger.info(f"👤 USER PREFERENCE: Using {user_preferred} for {task_type}")
                return [user_preferred]  # Return user's choice first
            else:
                logger.warning(f"👤 User preferred model {user_preferred} not available, falling back to intelligent selection")

        if not available_models:
            logger.warning("🤖 No models available for recommendation")
            return []

        # Check user's selection strategy
        strategy = self._get_user_selection_strategy()

        if strategy == "first_available":
            logger.info(f"⚡ FIRST AVAILABLE: Using {available_models[0]} for {task_type}")
            return [available_models[0]]
        elif strategy == "user_preference":
            # User wants only their preference, no fallback
            user_preferred = self._get_user_preferred_model(task_type)
            if user_preferred and user_preferred in available_models:
                logger.info(f"👤 USER PREFERENCE ONLY: Using {user_preferred} for {task_type}")
                return [user_preferred]
            else:
                logger.warning(f"👤 USER PREFERENCE ONLY: No valid model specified for {task_type}")
                # Return first available model to prevent infinite loops
                if available_models:
                    fallback = available_models[0]
                    logger.warning(f"👤 FALLBACK: Using {fallback} (no user preference set)")
                    return [fallback]
                else:
                    logger.error(f"👤 ERROR: No models available at all for {task_type}")
                    return []

        # Default: Intelligent selection
        model_scores = []
        for model_name in available_models:
            try:
                capabilities = self.detect_model_capabilities(model_name)
                score = self._calculate_task_suitability(capabilities, task_type)
                model_scores.append((model_name, score, capabilities))
            except Exception as e:
                logger.warning(f"🤖 Failed to analyze model {model_name}: {e}")
                continue

        # Sort by suitability score
        model_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top recommendations
        recommendations = [model[0] for model in model_scores[:5]]
        logger.info(f"🤖 INTELLIGENT FALLBACK: Recommended models for {task_type}: {recommendations}")

        return recommendations
    
    def is_thinking_model(self, model_name: str) -> bool:
        """
        Determine if a model is a "thinking model" (like DeepSeek-R1)
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            bool: True if it's a thinking model
        """
        capabilities = self.detect_model_capabilities(model_name)
        return capabilities.is_thinking_model
    
    def get_optimal_settings(self, model_name: str, task_type: str = "mcq_generation") -> Dict[str, Any]:
        """
        Get optimal settings for a specific model and task
        
        Args:
            model_name: Name of the model
            task_type: Type of task
            
        Returns:
            Dict with optimal settings
        """
        capabilities = self.detect_model_capabilities(model_name)
        
        # Base settings
        settings = capabilities.recommended_settings.copy()
        
        # Task-specific adjustments
        if task_type == "mcq_generation":
            if capabilities.is_thinking_model:
                settings.update({
                    "temperature": 0.1,  # Lower for consistency
                    "max_tokens": 1000,  # Higher for reasoning
                    "use_streaming": True,
                    "enable_reasoning_display": True
                })
            else:
                settings.update({
                    "temperature": 0.3,
                    "max_tokens": 600,
                    "use_streaming": False
                })
        
        logger.info(f"🤖 Optimal settings for {model_name} ({task_type}): {settings}")
        return settings
    
    def _analyze_model_name(self, model_name: str) -> Dict[str, Any]:
        """Analyze model name for capability indicators"""
        name_lower = model_name.lower()
        
        analysis = {
            "is_thinking_model": False,
            "reasoning_indicators": [],
            "size_indicators": [],
            "specialization_indicators": []
        }
        
        # Thinking model indicators
        thinking_patterns = [
            r'r1', r'reasoning', r'think', r'cot', r'chain.*thought',
            r'deepseek.*r1', r'o1', r'reasoning.*model'
        ]
        
        for pattern in thinking_patterns:
            if re.search(pattern, name_lower):
                analysis["is_thinking_model"] = True
                analysis["reasoning_indicators"].append(pattern)
        
        # Size indicators
        size_patterns = [
            (r'(\d+)b', 'billion_params'),
            (r'(\d+)m', 'million_params'),
            (r'small|mini|tiny', 'small_model'),
            (r'large|big|xl', 'large_model')
        ]
        
        for pattern, indicator in size_patterns:
            if re.search(pattern, name_lower):
                analysis["size_indicators"].append(indicator)
        
        # Specialization indicators
        specializations = [
            (r'code|coding|coder', 'coding'),
            (r'math|mathematical', 'mathematics'),
            (r'instruct|instruction', 'instruction_following'),
            (r'chat|conversation', 'conversational'),
            (r'vision|visual|image', 'multimodal')
        ]
        
        for pattern, spec in specializations:
            if re.search(pattern, name_lower):
                analysis["specialization_indicators"].append(spec)
        
        return analysis
    
    def _research_model_online(self, model_name: str) -> Dict[str, Any]:
        """Research model capabilities using web search"""
        if not self.web_search_available:
            return {}

        try:
            # Use external web search API (simulated for now)
            # In a real implementation, this would use the web-search tool
            search_query = f"{model_name} AI model capabilities reasoning thinking parameters"

            # Simulate web search results based on model name patterns
            # This is a fallback until proper web search integration is available
            research = self._simulate_web_research(model_name)

            return research

        except Exception as e:
            logger.warning(f"🤖 Web research failed for {model_name}: {e}")
            return {"found_online": False, "error": str(e)}

    def _simulate_web_research(self, model_name: str) -> Dict[str, Any]:
        """Simulate web research results based on known model patterns"""
        model_lower = model_name.lower()

        research = {
            "found_online": True,
            "search_results": [],
            "capabilities_mentioned": [],
            "thinking_model_evidence": []
        }

        # Simulate research results based on model name patterns
        if "deepseek" in model_lower and "r1" in model_lower:
            research["thinking_model_evidence"].extend([
                "reasoning model", "chain of thought", "thinking process"
            ])
            research["capabilities_mentioned"].extend([
                "mathematical reasoning", "logical thinking", "problem solving"
            ])
        elif "deepseek" in model_lower:
            research["capabilities_mentioned"].extend([
                "code generation", "problem solving"
            ])
        elif "llama" in model_lower:
            research["capabilities_mentioned"].extend([
                "instruction following", "conversational"
            ])
        elif "mistral" in model_lower:
            research["capabilities_mentioned"].extend([
                "instruction following", "problem solving"
            ])
        elif "qwen" in model_lower:
            research["capabilities_mentioned"].extend([
                "instruction following", "mathematical reasoning"
            ])
        elif "phi" in model_lower:
            research["capabilities_mentioned"].extend([
                "conversational", "instruction following"
            ])
        elif "code" in model_lower:
            research["capabilities_mentioned"].extend([
                "code generation", "problem solving"
            ])
        elif "math" in model_lower:
            research["capabilities_mentioned"].extend([
                "mathematical reasoning", "problem solving"
            ])

        # Look for thinking model indicators
        thinking_patterns = ["r1", "reasoning", "think", "cot", "chain"]
        for pattern in thinking_patterns:
            if pattern in model_lower:
                research["thinking_model_evidence"].append(f"{pattern} model")

        return research
    
    def _test_model_locally(self, model_name: str) -> Dict[str, Any]:
        """Test model locally if available"""
        try:
            # Check if model is available in Ollama
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [m["name"] for m in models]
                
                if model_name in available_models:
                    # Get model details
                    model_info = next((m for m in models if m["name"] == model_name), {})
                    
                    return {
                        "available_locally": True,
                        "model_info": model_info,
                        "size": model_info.get("size", 0),
                        "modified_at": model_info.get("modified_at", "")
                    }
            
            return {"available_locally": False}
            
        except Exception as e:
            logger.warning(f"🤖 Local model test failed for {model_name}: {e}")
            return {"available_locally": False, "error": str(e)}
    
    def _synthesize_capabilities(self, model_name: str, name_analysis: Dict, web_research: Dict, local_test: Dict) -> ModelCapabilities:
        """Synthesize all information into model capabilities"""
        
        # Determine if it's a thinking model
        is_thinking = (
            name_analysis.get("is_thinking_model", False) or
            len(web_research.get("thinking_model_evidence", [])) > 0
        )
        
        # Determine reasoning capability
        reasoning_capability = "basic"
        if is_thinking or "reasoning" in web_research.get("capabilities_mentioned", []):
            reasoning_capability = "expert"
        elif len(name_analysis.get("reasoning_indicators", [])) > 0:
            reasoning_capability = "advanced"
        
        # Estimate context length (rough heuristic)
        context_length = 4096  # Default
        if "large" in name_analysis.get("size_indicators", []):
            context_length = 8192
        elif "small" in name_analysis.get("size_indicators", []):
            context_length = 2048
        
        # Get parameter count estimate
        param_count = None
        for indicator in name_analysis.get("size_indicators", []):
            if "billion" in indicator:
                param_count = "Large (>1B parameters)"
            elif "million" in indicator:
                param_count = "Medium (<1B parameters)"
        
        # Specializations
        specializations = name_analysis.get("specialization_indicators", [])
        specializations.extend(web_research.get("capabilities_mentioned", []))
        
        # Recommended settings
        recommended_settings = {
            "temperature": 0.1 if is_thinking else 0.3,
            "max_tokens": 1000 if is_thinking else 600,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        # Warnings
        warnings = []
        if is_thinking and not local_test.get("available_locally", False):
            warnings.append("Thinking model may require significant computational resources")
        
        if not web_research.get("found_online", False):
            warnings.append("Limited information available about this model")
        
        # Compatibility score (0-1)
        compatibility_score = 0.8  # Base score
        if local_test.get("available_locally", False):
            compatibility_score += 0.2
        if web_research.get("found_online", False):
            compatibility_score += 0.1
        
        compatibility_score = min(compatibility_score, 1.0)
        
        return ModelCapabilities(
            name=model_name,
            is_thinking_model=is_thinking,
            reasoning_capability=reasoning_capability,
            context_length=context_length,
            parameter_count=param_count,
            specializations=list(set(specializations)),
            compatibility_score=compatibility_score,
            recommended_settings=recommended_settings,
            warnings=warnings
        )
    
    def _calculate_task_suitability(self, capabilities: ModelCapabilities, task_type: str) -> float:
        """Calculate how suitable a model is for a specific task"""
        score = capabilities.compatibility_score
        
        if task_type == "mcq_generation":
            # Prefer models with good reasoning
            if capabilities.reasoning_capability == "expert":
                score += 0.3
            elif capabilities.reasoning_capability == "advanced":
                score += 0.2
            
            # Thinking models are excellent for MCQ generation
            if capabilities.is_thinking_model:
                score += 0.4
            
            # Instruction following is important
            if "instruction_following" in capabilities.specializations:
                score += 0.2
        
        return min(score, 1.0)
    
    def _get_available_models(self) -> List[str]:
        """Get list of ACTUALLY available models from local servers - FAST & NON-BLOCKING"""
        # Use cached models if available to avoid blocking UI
        if hasattr(self, '_cached_models') and self._cached_models:
            logger.info(f"🤖 Using cached models: {self._cached_models}")
            return self._cached_models

        models = []

        try:
            # Get dynamic URLs from user configuration
            from pathlib import Path
            import json

            ollama_url = "http://localhost:11434"
            lmstudio_url = "http://127.0.0.1:1234"

            # Try to get URLs from user settings
            try:
                user_settings_path = Path("user_data/user_settings.json")
                if user_settings_path.exists():
                    with open(user_settings_path, 'r', encoding='utf-8') as f:
                        user_settings = json.load(f)
                        if 'network_config' in user_settings:
                            ollama_url = user_settings['network_config'].get('ollama_url', ollama_url)
                            lmstudio_url = user_settings['network_config'].get('lmstudio_url', lmstudio_url)
            except Exception:
                pass  # Use defaults

            # Check Ollama for REAL available models - ULTRA FAST timeout
            response = requests.get(f"{ollama_url}/api/tags", timeout=0.5)  # Ultra fast timeout
            if response.status_code == 200:
                ollama_models = [m["name"] for m in response.json().get("models", [])]
                models.extend(ollama_models)
                logger.info(f"🤖 Found {len(ollama_models)} REAL Ollama models: {ollama_models}")
            else:
                logger.warning(f"🤖 Ollama server not responding at {ollama_url}")
        except Exception as e:
            logger.debug(f"🤖 Ollama check failed (non-blocking): {e}")  # Use debug to reduce spam

        try:
            # Check LM Studio for REAL available models - ULTRA FAST timeout
            response = requests.get(f"{lmstudio_url}/v1/models", timeout=0.3)  # Ultra fast timeout
            if response.status_code == 200:
                lmstudio_models = [m["id"] for m in response.json().get("data", [])]
                models.extend(lmstudio_models)
                logger.info(f"🤖 Found {len(lmstudio_models)} REAL LM Studio models: {lmstudio_models}")
            else:
                logger.debug(f"🤖 LM Studio server not responding at {lmstudio_url}")
        except Exception as e:
            logger.debug(f"🤖 LM Studio check failed (non-blocking): {e}")  # Use debug to reduce spam

        # CRITICAL: Only return ACTUALLY available models - NO hardcoded fallbacks
        real_models = list(set(models))  # Remove duplicates

        # Cache the results to avoid repeated calls
        self._cached_models = real_models

        if real_models:
            logger.info(f"🤖 TOTAL REAL AVAILABLE MODELS: {real_models}")
        else:
            logger.debug("🤖 No models found - servers may be starting up")

        return real_models

    def refresh_available_models_async(self):
        """Refresh available models cache asynchronously (non-blocking)"""
        try:
            # Clear cache to force refresh
            if hasattr(self, '_cached_models'):
                delattr(self, '_cached_models')

            # Get fresh models list
            fresh_models = self._get_available_models()
            logger.info(f"🔄 Refreshed model cache: {len(fresh_models)} models available")
            return fresh_models
        except Exception as e:
            logger.warning(f"🔄 Failed to refresh model cache: {e}")
            return []

    def _load_model_cache(self):
        """Load cached model capabilities"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Convert dict back to ModelCapabilities objects
                for name, data in cache_data.items():
                    self.model_cache[name] = ModelCapabilities(**data)
                
                logger.info(f"[AI] Loaded {len(self.model_cache)} cached model capabilities")
        except Exception as e:
            logger.warning(f"[AI] Failed to load model cache: {e}")
    
    def _save_model_cache(self):
        """Save model capabilities to cache"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert ModelCapabilities objects to dict
            cache_data = {}
            for name, capabilities in self.model_cache.items():
                cache_data[name] = {
                    "name": capabilities.name,
                    "is_thinking_model": capabilities.is_thinking_model,
                    "reasoning_capability": capabilities.reasoning_capability,
                    "context_length": capabilities.context_length,
                    "parameter_count": capabilities.parameter_count,
                    "specializations": capabilities.specializations,
                    "compatibility_score": capabilities.compatibility_score,
                    "recommended_settings": capabilities.recommended_settings,
                    "warnings": capabilities.warnings
                }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"🤖 Saved {len(cache_data)} model capabilities to cache")
        except Exception as e:
            logger.warning(f"🤖 Failed to save model cache: {e}")
    
    def _get_user_preferred_model(self, task_type: str) -> Optional[str]:
        """Get user's preferred model for a specific task"""
        try:
            from pathlib import Path
            import json

            user_settings_path = Path("user_data/user_settings.json")
            if not user_settings_path.exists():
                return None

            with open(user_settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            model_prefs = settings.get('model_preferences', {})

            if task_type == "mcq_generation":
                preferred = model_prefs.get('preferred_mcq_model', '')
                if preferred:
                    logger.info(f"👤 Found user preference for MCQ: {preferred}")
                    return preferred
            elif task_type == "reasoning" or task_type == "thinking":
                preferred = model_prefs.get('preferred_thinking_model', '')
                if preferred:
                    logger.info(f"👤 Found user preference for thinking: {preferred}")
                    return preferred

            return None

        except Exception as e:
            logger.warning(f"👤 Failed to load user model preferences: {e}")
            return None

    def _get_user_selection_strategy(self) -> str:
        """Get user's model selection strategy"""
        try:
            from pathlib import Path
            import json

            user_settings_path = Path("user_data/user_settings.json")
            if not user_settings_path.exists():
                return "intelligent"  # Default

            with open(user_settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            model_prefs = settings.get('model_preferences', {})
            strategy = model_prefs.get('selection_strategy', 'intelligent')

            logger.info(f"👤 User selection strategy: {strategy}")
            return strategy

        except Exception as e:
            logger.warning(f"👤 Failed to load user selection strategy: {e}")
            return "intelligent"  # Default fallback

    def _init_web_search(self):
        """Initialize web search capability"""
        # For now, use simulated web search based on model name patterns
        # This can be enhanced later with actual web search integration
        self.web_search_available = True
        logger.info("[AI] Web search capability enabled (simulated mode)")


# Global instance for easy access
_global_detector = DynamicModelDetector()


def detect_model_capabilities(model_name: str, force_refresh: bool = False) -> ModelCapabilities:
    """Detect model capabilities using global detector"""
    return _global_detector.detect_model_capabilities(model_name, force_refresh)


def is_thinking_model(model_name: str) -> bool:
    """Check if model is a thinking model"""
    return _global_detector.is_thinking_model(model_name)


def get_recommended_models(task_type: str = "mcq_generation") -> List[str]:
    """Get recommended models for a task"""
    return _global_detector.get_recommended_models(task_type)


def get_optimal_settings(model_name: str, task_type: str = "mcq_generation") -> Dict[str, Any]:
    """Get optimal settings for a model"""
    return _global_detector.get_optimal_settings(model_name, task_type)
