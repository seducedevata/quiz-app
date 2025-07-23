"""
Online MCQ Generator
Supports multiple cloud API providers: OpenAI, Anthropic Claude, Gemini, Groq, OpenRouter
OPTIMIZED FOR FREE MODELS: Prioritizes OpenRouter free models when available
[HOT] ENHANCED: Now with exponential backoff and Chain-of-Thought prompting
ULTRA-LOGGING: Comprehensive logging for debugging and monitoring
"""

from .async_converter import async_time_sleep


import logging
import asyncio
import json
import os
import re
import time
import traceback
import sys
from typing import Dict, List, Any, Optional
try:
    import aiohttp
except ImportError:
    aiohttp = None
try:
    import backoff
except ImportError:
    backoff = None

from .mcq_generator import MCQGenerator

# [START] BUG FIX 18: Custom exception classes for proper error handling
class InvalidApiKeyError(Exception):
    """Raised when API key is invalid or unauthorized"""
    pass

class RateLimitError(Exception):
    """Raised when API rate limit is exceeded"""
    pass

class ServerError(Exception):
    """Raised when API server has internal errors"""
    pass

class QuotaExceededError(Exception):
    """Raised when API quota is exceeded"""
    pass

# 🛡️ CRITICAL FIX: Enhanced network error classification
class NetworkTimeoutError(Exception):
    """Raised when network request times out"""
    pass

class NetworkConnectionError(Exception):
    """Raised when network connection fails"""
    pass

class NetworkDataError(Exception):
    """Raised when network data transmission fails"""
    pass

# Create specialized logger for online MCQ generation
logger = logging.getLogger(__name__)
api_logger = logging.getLogger("api_calls")
performance_logger = logging.getLogger("performance_online_mcq")


class OnlineMCQGenerator(MCQGenerator):
    """
    High-performance online MCQ generator supporting multiple cloud API providers
    [HOT] ENHANCED: Now with exponential backoff and robust error handling
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Online MCQ Generator with ultra-comprehensive logging"""
        logger.info("[START] INITIALIZING OnlineMCQGenerator")
        logger.info(f"[SEARCH] CONFIG RECEIVED: {config}")
        logger.info(f"[SEARCH] PYTHON VERSION: {sys.version}")
        logger.info(f"[SEARCH] AIOHTTP AVAILABLE: {aiohttp is not None}")
        logger.info(f"[SEARCH] BACKOFF AVAILABLE: {backoff is not None}")
        
        super().__init__(config)
        
        logger.info("[SEARCH] LOADING USER API KEYS AND SETTINGS...")
        # Load API keys from user settings first, fallback to environment variables
        user_api_keys = self._load_user_api_keys()
        logger.info(f"[SEARCH] USER API KEYS LOADED: {list(user_api_keys.keys())}")
        
        # API Configuration - CRITICAL FIX: ONLY use user-provided API keys, ignore environment
        # This prevents unwanted OpenAI usage when user hasn't provided OpenAI key
        logger.info("[CONFIG] CONFIGURING API PROVIDERS...")
        self.providers = {
            'openai': {
                'api_key': user_api_keys.get('openai'),  # Only user keys, no env fallback
                'base_url': 'https://api.openai.com/v1/chat/completions',
                'model': 'gpt-4-turbo-preview',
                'headers': {'Authorization': 'Bearer {api_key}', 'Content-Type': 'application/json'}
            },
            'anthropic': {
                'api_key': user_api_keys.get('anthropic'),  # Only user keys, no env fallback
                'base_url': 'https://api.anthropic.com/v1/messages',
                'model': 'claude-3-sonnet-20240229',
                'headers': {'x-api-key': '{api_key}', 'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'}
            },
            'gemini': {
                'api_key': user_api_keys.get('gemini'),  # Only user keys, no env fallback
                'base_url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
                'model': 'gemini-pro',
                'headers': {'Content-Type': 'application/json'}
            },
            'groq': {
                'api_key': user_api_keys.get('groq'),  # Only user keys, no env fallback
                'base_url': 'https://api.groq.com/openai/v1/chat/completions',
                'model': 'llama-3.3-70b-versatile',  # LATEST - llama-3.1 also decommissioned!
                'headers': {'Authorization': 'Bearer {api_key}', 'Content-Type': 'application/json'}
            },
            'openrouter': {
                'api_key': user_api_keys.get('openrouter'),  # Only user keys, no env fallback
                'base_url': 'https://openrouter.ai/api/v1/chat/completions',
                'free_models': [
                    # [START] Large-Scale Free Models (30B+) - Prioritized for superior quality
                    'qwen/qwen3-235b-a22b:free',  # 235B parameters - The absolute beast
                    'qwen/qwen3-30b-a3b:free',    # 30.5B parameters - Most balanced
                    'meta-llama/llama-3.3-70b-instruct:free',  # 70B parameters - Proven performer
                    'meta-llama/llama-4-maverick:free',        # Latest Llama 4 variant (70B+)
                    
                    # 32B Parameter Models (Reasoning specialists)
                    'qwen/qwq-32b:free',          # 32B parameters - Reasoning focused
                    'thudm/glm-4-32b:free',       # 32B parameters - Multilingual support
                    'thudm/glm-z1-32b:free',      # 32B parameters - Advanced GLM variant
                    
                    # Fallback Large Models (Backup options if above fail)
                    'meta-llama/llama-2-70b-chat:free',  # 70B parameters - Reliable fallback
                    'cognitivecomputations/dolphin-mixtral-8x7b:free',  # 8x7B MoE - Good fallback
                    
                    # Legacy smaller models (final fallback only)
                    'meta-llama/llama-3.1-8b-instruct:free',
                    'mistralai/mistral-7b-instruct:free'
                ],
                'model': 'qwen/qwen3-235b-a22b:free',  # Default to the 235B beast for maximum quality
                'headers': {'Authorization': 'Bearer {api_key}', 'Content-Type': 'application/json'}
            }
        }
        
        logger.info("[SEARCH] PROVIDER CONFIGURATIONS CREATED")
        for provider_name, config in self.providers.items():
            has_key = bool(config.get('api_key'))
            logger.info(f"   • {provider_name}: API_KEY_PRESENT={has_key}, MODEL={config.get('model')}")
        
        # [HOT] CRITICAL FIX: Strict provider prioritization - ONLY use providers with valid API keys
        logger.info("[SEARCH] DETERMINING AVAILABLE PROVIDERS...")
        self.available_providers = []
        
        # Check each provider strictly for API key availability and enabled state  
        for provider_name in ['groq', 'openrouter', 'anthropic', 'gemini', 'openai']:  # Groq first for cloud APIs
            if provider_name in self.providers:
                config = self.providers[provider_name]
                api_key_value = config.get('api_key')
                has_valid_key = bool(api_key_value and str(api_key_value).strip() and str(api_key_value).strip() != '')
                is_enabled = self.providers_enabled.get(provider_name, True)
                
                logger.info(f"[SEARCH] PROVIDER CHECK: {provider_name.upper()}")
                logger.info(f"   • API_KEY_PRESENT: {has_valid_key}")
                logger.info(f"   • API_KEY_LENGTH: {len(str(api_key_value)) if api_key_value else 0}")
                logger.info(f"   • API_KEY_PREFIX: {str(api_key_value)[:10] + '...' if api_key_value and len(str(api_key_value)) > 10 else 'N/A'}")
                logger.info(f"   • IS_ENABLED: {is_enabled}")
                
                if has_valid_key and is_enabled:
                    self.available_providers.append(provider_name)
                    logger.info(f"[OK] {provider_name.upper()} ADDED to available providers")
                elif not has_valid_key:
                    logger.warning(f"[ERROR] {provider_name.upper()} SKIPPED - no valid API key")
                elif not is_enabled:
                    logger.warning(f"⏸️ {provider_name.upper()} SKIPPED - disabled by user")
        
        # [EMERGENCY] CRITICAL: Show final provider selection clearly
        if self.available_providers:
            logger.info(f"[START] AVAILABLE PROVIDERS (in priority order): {' → '.join(self.available_providers)}")
            logger.info(f"[TARGET] PRIMARY PROVIDER (will be tried first): {self.available_providers[0].upper()}")
        else:
            logger.error("[ERROR] NO PROVIDERS AVAILABLE - No valid API keys found for any provider!")
            logger.error("[INFO] Please add valid API keys in the Settings screen")
        
        self.session = None
        self.generation_stats = {"total_generated": 0, "avg_time": 0, "provider_usage": {}}

        # [START] RATE LIMITING: Track request timestamps to prevent hitting API limits
        self._request_timestamps = []
        self._max_requests_per_minute = 10  # Conservative limit for free tiers
        self._request_lock = asyncio.Lock()
        
        logger.info("[FINISH] OnlineMCQGenerator initialization completed")



    def _load_user_api_keys(self) -> Dict[str, str]:
        """Load API keys and provider enabled states from user settings file with comprehensive logging"""
        logger.info("[SEARCH] LOADING USER API KEYS from settings file...")
        
        try:
            from pathlib import Path
            import json
            
            settings_path = Path("user_data/user_settings.json")
            logger.info(f"[SEARCH] SETTINGS PATH: {settings_path.absolute()}")
            logger.info(f"[SEARCH] SETTINGS FILE EXISTS: {settings_path.exists()}")
            
            if settings_path.exists():
                logger.info("[SEARCH] READING settings file...")
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    
                logger.info(f"[SEARCH] SETTINGS STRUCTURE: {list(settings.keys())}")
                
                api_keys = settings.get('api_keys', {})
                logger.info(f"[SEARCH] API_KEYS SECTION FOUND: {bool(api_keys)}")
                logger.info(f"[SEARCH] API_KEYS PROVIDERS: {list(api_keys.keys())}")
                
                # Load provider enabled states
                self.providers_enabled = settings.get('api_providers_enabled', {})
                logger.info(f"[SEARCH] PROVIDERS_ENABLED SECTION: {self.providers_enabled}")
                
                # Default all providers to enabled if not specified
                for provider in ['openai', 'anthropic', 'gemini', 'groq', 'openrouter']:
                    if provider not in self.providers_enabled:
                        self.providers_enabled[provider] = True
                        logger.info(f"[SEARCH] DEFAULTING {provider} to ENABLED")
                
                # Filter out empty keys and log found keys
                valid_keys = {}
                for k, v in api_keys.items():
                    if v and v.strip():
                        valid_keys[k] = v
                        logger.info(f"[SEARCH] VALID API KEY FOUND: {k} (length: {len(v)})")
                    else:
                        logger.warning(f"[SEARCH] EMPTY/INVALID API KEY: {k}")
                
                if valid_keys:
                    logger.info(f"🔑 FOUND {len(valid_keys)} VALID API KEYS: {list(valid_keys.keys())}")
                else:
                    logger.warning("[WARNING] NO VALID API KEYS FOUND IN SETTINGS")
                
                # Log enabled/disabled states
                enabled_count = sum(1 for enabled in self.providers_enabled.values() if enabled)
                logger.info(f"[CONFIG] PROVIDER STATES: {enabled_count}/{len(self.providers_enabled)} enabled")
                for provider, enabled in self.providers_enabled.items():
                    status = "ENABLED" if enabled else "DISABLED"
                    logger.info(f"   • {provider}: {status}")
                
                return valid_keys
            
            # Initialize default enabled states if no settings file
            logger.warning("[WARNING] NO SETTINGS FILE FOUND - using defaults")
            self.providers_enabled = {
                'openai': True,
                'anthropic': True, 
                'gemini': True,
                'groq': True,
                'openrouter': True
            }
            return {}
            
        except Exception as e:
            logger.error(f"[ERROR] FAILED TO LOAD USER API KEYS: {e}")
            logger.error(f"[ERROR] TRACEBACK: {traceback.format_exc()}")
            # Initialize default enabled states on error
            self.providers_enabled = {
                'openai': True,
                'anthropic': True,
                'gemini': True,
                'groq': True,
                'openrouter': True
            }
            return {}

    def _update_api_keys(self, api_keys: Dict[str, str], providers_enabled: Dict[str, bool] = None):
        """Update API keys and provider enabled states, then reinitialize providers"""
        try:
            logger.info("[RELOAD] Updating API keys and provider states...")
            
            # Update provider configurations
            for provider_name, api_key in api_keys.items():
                if provider_name in self.providers:
                    self.providers[provider_name]['api_key'] = api_key.strip() if api_key else None
            
            # Update enabled states if provided
            if providers_enabled:
                self.providers_enabled.update(providers_enabled)
                logger.info("[CONFIG] Updated provider enabled states")
            
            # Recalculate available providers (must have API key AND be enabled)
            old_available = set(self.available_providers)
            self.available_providers = []
            
            for name, config in self.providers.items():
                has_key = bool(config['api_key'])
                is_enabled = self.providers_enabled.get(name, True)
                
                if has_key and is_enabled:
                    self.available_providers.append(name)
                    if name not in old_available:
                        logger.info(f"[OK] {name.upper()} now available (key + enabled)")
                elif name in old_available:
                    if not has_key:
                        logger.info(f"[ERROR] {name.upper()} no longer available (no API key)")
                    elif not is_enabled:
                        logger.info(f"⏸️ {name.upper()} no longer available (disabled)")
            
            enabled_count = sum(1 for enabled in self.providers_enabled.values() if enabled)
            logger.info(f"[START] Providers updated. Available: {', '.join(self.available_providers)} ({len(self.available_providers)}/{enabled_count} enabled)")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to update API keys and providers: {e}")

    def initialize(self) -> bool:
        """Initialize the online MCQ generator"""
        try:
            if not aiohttp:
                logger.error("[ERROR] aiohttp not available - install with: pip install aiohttp")
                return False
                
            if not self.available_providers:
                logger.error("[ERROR] No API keys found for any cloud providers")
                logger.info("[INFO] Please add API keys in the Settings screen:")
                logger.info("   • Groq (free and fast): https://console.groq.com/keys")
                logger.info("   • OpenRouter (free models): https://openrouter.ai/keys")
                logger.info("   • Or other providers: OpenAI, Anthropic, Gemini")
                
                # Check if settings file exists to help debug
                from pathlib import Path
                settings_path = Path("user_data/user_settings.json")
                if settings_path.exists():
                    logger.info(f"📁 Settings file found at: {settings_path.absolute()}")
                    try:
                        import json
                        with open(settings_path, 'r', encoding='utf-8') as f:
                            settings = json.load(f)
                            api_keys = settings.get('api_keys', {})
                            logger.info(f"[SEARCH] API keys structure in file: {list(api_keys.keys())}")
                            # Check if keys are empty
                            empty_keys = [k for k, v in api_keys.items() if not v or not v.strip()]
                            if empty_keys:
                                logger.info(f"[WARNING] Empty API keys found: {empty_keys}")
                    except Exception as e:
                        logger.error(f"[ERROR] Error reading settings file: {e}")
                else:
                    logger.info(f"📁 No settings file found at: {settings_path.absolute()}")
                
                return False
                
            logger.info(f"[CLOUD] Initializing Online MCQ Generator with {len(self.available_providers)} providers")
            logger.info(f"[START] Available providers: {', '.join(self.available_providers)}")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize online generator: {e}")
            return False

    def is_available(self) -> bool:
        """Check if online generation is available"""
        return len(self.available_providers) > 0 and aiohttp is not None

    def generate_mcq(self, topic: str, context: str = "", num_questions: int = 1, 
                     difficulty: str = "medium", game_mode: str = "casual", 
                     question_type: str = "mixed") -> List[Dict[str, Any]]:
        """Generate MCQ questions using cloud APIs (sync wrapper) with comprehensive logging"""
        
        # [TARGET] COMPREHENSIVE ONLINE GENERATION LOGGING - START
        performance_logger.info("="*80)
        performance_logger.info("[CLOUD] ONLINE MCQ GENERATION SESSION STARTED")
        performance_logger.info("="*80)
        
        performance_logger.info(f"[START] STARTING ONLINE MCQ GENERATION")
        performance_logger.info(f"   • TOPIC: '{topic}'")
        performance_logger.info(f"   • CONTEXT_LENGTH: {len(context)}")
        performance_logger.info(f"   • NUM_QUESTIONS: {num_questions}")
        performance_logger.info(f"   • DIFFICULTY: '{difficulty}'")
        performance_logger.info(f"   • GAME_MODE: '{game_mode}'")
        performance_logger.info(f"   • QUESTION_TYPE: '{question_type}'")
        
        # [TARGET] LOG PROVIDER AVAILABILITY AND SELECTION
        performance_logger.info("[SEARCH] PROVIDER AVAILABILITY CHECK:")
        performance_logger.info(f"   • TOTAL_CONFIGURED_PROVIDERS: {len(self.providers)}")
        performance_logger.info(f"   • AVAILABLE_PROVIDERS: {self.available_providers}")
        performance_logger.info(f"   • PROVIDER_COUNT: {len(self.available_providers)}")
        
        if self.available_providers:
            performance_logger.info("[TARGET] PROVIDER PRIORITY ORDER:")
            for i, provider in enumerate(self.available_providers, 1):
                provider_config = self.providers.get(provider, {})
                model = provider_config.get('model', 'unknown')
                has_key = bool(provider_config.get('api_key'))
                performance_logger.info(f"   {i}. {provider.upper()}: model='{model}', api_key={'[OK]' if has_key else '[ERROR]'}")
            
            primary_provider = self.available_providers[0]
            primary_config = self.providers.get(primary_provider, {})
            performance_logger.info(f"[TARGET] PRIMARY_PROVIDER: {primary_provider.upper()}")
            performance_logger.info(f"[TARGET] PRIMARY_MODEL: {primary_config.get('model', 'unknown')}")
            performance_logger.info(f"[TARGET] PRIMARY_ENDPOINT: {primary_config.get('base_url', 'unknown')}")
            
            # Special logging for OpenRouter free models
            if primary_provider == 'openrouter':
                free_models = primary_config.get('free_models', [])
                performance_logger.info(f"🆓 OPENROUTER_FREE_MODELS_AVAILABLE: {len(free_models)}")
                if free_models:
                    performance_logger.info(f"🆓 PRIMARY_FREE_MODEL: {free_models[0]}")
                    performance_logger.info(f"🆓 FALLBACK_MODELS: {free_models[1:3] if len(free_models) > 1 else 'none'}")
        else:
            performance_logger.error("[ERROR] NO PROVIDERS AVAILABLE!")
            performance_logger.error("   • Check API keys in settings")
            performance_logger.error("   • Verify provider enabled states") 
        
        start_time = time.time()
        
        if not self.is_available():
            logger.error("[ERROR] ONLINE GENERATION NOT AVAILABLE")
            logger.error(f"   • AVAILABLE_PROVIDERS: {self.available_providers}")
            logger.error(f"   • AIOHTTP_AVAILABLE: {aiohttp is not None}")
            performance_logger.error(f"[ERROR] GENERATION FAILED: Not available after {time.time() - start_time:.3f}s")
            performance_logger.error("="*80)
            return []
            
        logger.info("[RELOAD] STARTING ASYNC GENERATION IN THREAD...")
        performance_logger.info("🧵 LAUNCHING THREAD-BASED ASYNC GENERATION...")
        
        # [EMERGENCY] CRITICAL FIX: Use ThreadPoolExecutor to prevent UI blocking
        import concurrent.futures
        import threading
        
        def run_async_in_thread():
            thread_logger = logging.getLogger(f"{__name__}.thread")
            thread_logger.info(f"🧵 ASYNC THREAD STARTED for {num_questions} questions")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                thread_logger.info("[RELOAD] RUNNING ASYNC GENERATION...")
                result = loop.run_until_complete(
                    asyncio.wait_for(
                        self.generate_mcq_async(topic, context, num_questions, difficulty, game_mode, question_type),
                        timeout=60.0  # Extended timeout for large models
                    )
                )
                thread_logger.info(f"[OK] ASYNC GENERATION COMPLETED: {len(result)} questions")
                return result
            except asyncio.TimeoutError:
                thread_logger.error("[ERROR] ASYNC GENERATION TIMED OUT")
                return []
            except Exception as e:
                thread_logger.error(f"[ERROR] ASYNC GENERATION ERROR: {e}")
                thread_logger.error(f"[ERROR] TRACEBACK: {traceback.format_exc()}")
                raise
            finally:
                try:
                    thread_logger.info("[CLEAN] CLOSING EVENT LOOP...")
                    loop.close()
                except Exception as e:
                    thread_logger.warning(f"[WARNING] ERROR CLOSING LOOP: {e}")
        
        try:
            logger.info("🏃 SUBMITTING TO THREAD EXECUTOR...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async_in_thread)
                result = future# [START] CONVERTED: Use async pattern instead of blocking .result()  # Total timeout including overhead
                
            total_time = time.time() - start_time
            performance_logger.info(f"[FINISH] ONLINE GENERATION COMPLETED in {total_time:.3f}s")
            performance_logger.info(f"   • SUCCESS_COUNT: {len(result)}")
            performance_logger.info(f"   • AVERAGE_TIME_PER_QUESTION: {total_time/max(1, len(result)):.3f}s")
            performance_logger.info("="*80)
            performance_logger.info("[CLOUD] ONLINE MCQ GENERATION SESSION COMPLETED")
            performance_logger.info("="*80)
            
            return result
            
        except concurrent.futures.TimeoutError:
            total_time = time.time() - start_time
            logger.error("[ERROR] THREAD EXECUTOR TIMED OUT")
            performance_logger.error(f"[ERROR] GENERATION TIMEOUT after {total_time:.3f}s")
            performance_logger.error("="*80)
            raise Exception("Online generation timed out. Please try again or check your internet connection.")
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[ERROR] ONLINE GENERATION FAILED: {e}")
            logger.error(f"[ERROR] FULL TRACEBACK: {traceback.format_exc()}")
            performance_logger.error(f"[ERROR] GENERATION FAILED after {total_time:.3f}s: {e}")
            performance_logger.error("="*80)
            # [EMERGENCY] CRITICAL: Propagate error to UI instead of returning empty list
            raise Exception(f"Online generation failed: {str(e)}")

    async def generate_mcq_async(self, topic: str, context: str = "", num_questions: int = 1, 
                                 difficulty: str = "medium", game_mode: str = "casual", 
                                 question_type: str = "mixed") -> List[Dict[str, Any]]:
        """Generate MCQ questions using cloud APIs"""
        if not self.is_available():
            logger.error("[ERROR] No cloud API providers available")
            return []

        results = []
        start_time = time.time()
        
        session_created = False
        try:
            logger.info(f"[CLOUD] Generating {num_questions} {question_type.upper()} MCQ(s) about '{topic}' using cloud APIs")
            logger.info(f"[SEARCH] DEBUG: Online generation parameters - topic: '{topic}', difficulty: '{difficulty}', question_type: '{question_type}'")
            
            # Create aiohttp session if not exists
            if not self.session or self.session.closed:
                # Create session without timeout in constructor - set timeout per request
                self.session = aiohttp.ClientSession()
                session_created = True
            
            # [START] CRITICAL FIX: Generate questions SEQUENTIALLY to avoid rate limiting
            # Free tier APIs have strict rate limits (1-2 requests/minute)
            # Concurrent requests immediately hit rate limits
            logger.info(f"[RELOAD] Generating {num_questions} questions SEQUENTIALLY to respect rate limits...")

            for i in range(num_questions):
                logger.info(f"[DOC] Generating question {i+1}/{num_questions}...")
                try:
                    result = await self._generate_single_question_with_fallback(topic, context, i, difficulty, question_type)
                    if result:
                        results.append(result)
                        logger.info(f"[OK] Question {i+1} generated successfully")
                    else:
                        logger.error(f"[ERROR] Question {i+1} generation returned no result")

                    # Add delay between requests to respect rate limits
                    if i < num_questions - 1:  # Don't delay after the last question
                        delay = 3.0  # 3 second delay between requests
                        logger.info(f"[WAIT] Waiting {delay}s before next request to respect rate limits...")
                        await asyncio.sleep(delay)

                except Exception as e:
                    logger.error(f"[ERROR] Question {i+1} generation failed: {e}")
                    # Continue with next question instead of failing completely
            
            total_time = time.time() - start_time
            success_count = len(results)
            
            # Update statistics
            self.generation_stats["total_generated"] += success_count
            if success_count > 0:
                self.generation_stats["avg_time"] = total_time / success_count
            
            logger.info(f"[FINISH] Generated {success_count}/{num_questions} questions in {total_time:.1f}s using cloud APIs")
            
            return results

        except Exception as e:
            logger.error(f"[ERROR] Online MCQ generation failed: {e}")
            # [EMERGENCY] CRITICAL: Propagate error to UI instead of returning empty list
            raise Exception(f"Online MCQ generation failed: {str(e)}")
        finally:
            # Close session if we created it
            if session_created and self.session and not self.session.closed:
                await self.session.close()
                self.session = None

    def _should_use_concurrent_generation(self, num_questions: int) -> bool:
        """
        [CONFIG] FIX: Determine if concurrent generation should be used based on provider capabilities

        Returns True if the current provider can handle concurrent requests efficiently.
        """
        # OpenRouter has generous rate limits and can handle concurrency
        if 'openrouter' in self.available_providers:
            logger.info("[START] OpenRouter detected - using concurrent generation")
            return True

        # Anthropic Claude has decent rate limits for paid tiers
        if 'anthropic' in self.available_providers:
            logger.info("[BRAIN] Anthropic detected - using concurrent generation")
            return True

        # For single questions, always use concurrent (no rate limit issues)
        if num_questions == 1:
            logger.info("[DOC] Single question - using concurrent generation")
            return True

        # Groq and others have strict free tier limits - use sequential
        logger.info("[WARNING] Rate-limited provider detected - using sequential generation")
        return False

    async def _generate_questions_concurrent(self, topic: str, context: str, num_questions: int,
                                           difficulty: str, question_type: str) -> List[Dict[str, Any]]:
        """
        [CONFIG] FIX: Generate multiple questions concurrently for better performance
        """
        logger.info(f"[START] Starting concurrent generation of {num_questions} questions...")

        # Create tasks for all questions
        tasks = []
        for i in range(num_questions):
            task = self._generate_single_question_with_fallback(topic, context, i, difficulty, question_type)
            tasks.append(task)

        # Execute all tasks concurrently with timeout
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and None results
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"[ERROR] Question {i+1} failed with exception: {result}")
                elif result:
                    valid_results.append(result)
                    logger.info(f"[OK] Question {i+1} generated successfully")
                else:
                    logger.error(f"[ERROR] Question {i+1} returned no result")

            logger.info(f"[TARGET] Concurrent generation complete: {len(valid_results)}/{num_questions} successful")
            return valid_results

        except Exception as e:
            logger.error(f"[ERROR] Concurrent generation failed: {e}")
            return []

    async def _generate_questions_sequential(self, topic: str, context: str, num_questions: int,
                                           difficulty: str, question_type: str) -> List[Dict[str, Any]]:
        """
        [CONFIG] FIX: Generate questions sequentially with smart rate limiting
        """
        logger.info(f"[RELOAD] Starting sequential generation of {num_questions} questions...")

        results = []
        for i in range(num_questions):
            logger.info(f"[DOC] Generating question {i+1}/{num_questions}...")
            try:
                result = await self._generate_single_question_with_fallback(topic, context, i, difficulty, question_type)
                if result:
                    results.append(result)
                    logger.info(f"[OK] Question {i+1} generated successfully")
                else:
                    logger.error(f"[ERROR] Question {i+1} generation returned no result")

                # Smart delay based on provider (no hardcoded 3 seconds)
                if i < num_questions - 1:  # Don't delay after the last question
                    delay = self._calculate_smart_delay()
                    if delay > 0:
                        logger.info(f"[WAIT] Smart delay: {delay}s before next request...")
                        await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"[ERROR] Question {i+1} generation failed: {e}")

        logger.info(f"[TARGET] Sequential generation complete: {len(results)}/{num_questions} successful")
        return results

    def _calculate_smart_delay(self) -> float:
        """
        [CONFIG] FIX: Calculate smart delay based on provider instead of hardcoded 3 seconds
        """
        # OpenRouter: No delay needed (generous limits)
        if 'openrouter' in self.available_providers:
            return 0.0

        # Anthropic: Short delay for free tier
        if 'anthropic' in self.available_providers:
            return 1.0

        # Groq: Longer delay for strict free tier limits
        if 'groq' in self.available_providers:
            return 2.0

        # Default: Conservative delay
        return 1.5

    async def _generate_single_question_with_fallback(self, topic: str, context: str, question_index: int,
                                                     difficulty: str = "medium", 
                                                     question_type: str = "mixed") -> Optional[Dict[str, Any]]:
        """Generate a single question - try all available providers with smart fallback"""
        
        if not self.available_providers:
            logger.error("[ERROR] No available providers for question generation")
            return None
        
        logger.info(f"[RELOAD] Generating question {question_index + 1} using {len(self.available_providers)} available providers")
        
        # Try each provider in order (already prioritized by speed/quality)
        for provider_index, provider in enumerate(self.available_providers):
            # Double-check provider has valid API key before attempting
            provider_config = self.providers.get(provider, {})
            api_key = provider_config.get('api_key')
            if not api_key or not str(api_key).strip():
                logger.warning(f"[FORBIDDEN] Skipping provider {provider} - no valid API key")
                continue
                
            try:
                logger.info(f"[RELOAD] Attempt {provider_index + 1}/{len(self.available_providers)}: Trying provider {provider}")
                result = await self._generate_with_provider(provider, topic, context,
                                                           question_index, difficulty, question_type)
                if result:
                    logger.info(f"[OK] {provider.upper()} successfully generated question {question_index + 1}")
                    return result
                else:
                    logger.warning(f"[WARNING] {provider.upper()} returned no result for question {question_index + 1}")

            # [START] BUG FIX 18: Intelligent exception handling based on error type
            except InvalidApiKeyError as e:
                logger.error(f"[ERROR] Invalid API key for {provider}. Please check your settings.")
                # Emit signal to UI to show specific error message
                if hasattr(self, 'errorOccurred'):
                    self.errorOccurred.emit(f"Your API key for {provider} is invalid. Please check your settings.")
                # Don't try other providers for this critical error - user needs to fix config
                return None
            except QuotaExceededError as e:
                logger.error(f"[ERROR] Quota exceeded for {provider}. Please check your usage limits.")
                if hasattr(self, 'errorOccurred'):
                    self.errorOccurred.emit(f"API quota exceeded for {provider}. Please check your usage limits.")
                # Don't try other providers - user needs to address quota issue
                return None
            except RateLimitError as e:
                logger.warning(f"[WARNING] Rate limit hit for {provider}. Trying next provider.")
                continue  # Correctly fall back to next provider
            except ServerError as e:
                logger.warning(f"[WARNING] Server error for {provider}. Trying next provider.")
                continue  # Correctly fall back to next provider
            except Exception as e:
                logger.warning(f"[WARNING] Generic failure for {provider}: {e}. Trying next provider.")
                continue  # Continue to next provider for unknown errors
        
        logger.error(f"[ERROR] All {len(self.available_providers)} providers failed for question {question_index + 1}")
        return None

    async def _generate_with_provider(self, provider: str, topic: str, context: str, question_index: int, 
                                     difficulty: str = "medium", question_type: str = "mixed") -> Optional[Dict[str, Any]]:
        """Generate question using specific provider with enhanced error handling"""
        
        provider_config = self.providers[provider]
        prompt = self._create_optimized_prompt(topic, context, question_index, difficulty, question_type)
        
        logger.info(f"[SEARCH] DEBUG: Provider '{provider}' prompt parameters - topic: '{topic}', difficulty: '{difficulty}', question_type: '{question_type}'")
        
        try:
            result = None
            if provider == 'openai':
                result = await self._call_openai(provider_config, prompt)
            elif provider == 'anthropic':
                result = await self._call_anthropic(provider_config, prompt)
            elif provider == 'gemini':
                result = await self._call_gemini(provider_config, prompt)
            elif provider == 'groq':
                result = await self._call_groq(provider_config, prompt)
            elif provider == 'openrouter':
                result = await self._call_openrouter(provider_config, prompt)

            else:
                logger.error(f"[ERROR] Unknown provider: {provider}")
                return None
            
            # 🔧 FIX: Validate content quality before accepting response
            if result:
                if not self._validate_response_quality(result):
                    logger.warning(f"⚠️ {provider} returned low-quality content, rejecting to trigger fallback")
                    return None  # Trigger fallback to next provider

                # [HOT] ENHANCED: More lenient numerical validation to prevent rejection loops
                if question_type.lower() == "numerical":
                    if not self._validate_numerical_question(result, question_type):
                        logger.warning(f"[WARNING] {provider} generated less-than-ideal numerical question, but accepting to avoid fallback loops")
                        logger.warning(f"[DOC] Question: '{result.get('question', 'N/A')[:100]}...'")
                        # Don't return None - accept the question to prevent endless fallbacks

            return result
                
        except Exception as e:
            logger.error(f"[ERROR] Provider {provider} generation failed: {e}")
            return None

    def _validate_response_quality(self, response: Dict[str, Any]) -> bool:
        """🔧 FIX: Validate response quality to prevent accepting 'AI spaghetti'"""
        try:
            # Basic structure validation
            if not isinstance(response, dict):
                return False

            question_text = response.get('question', '')
            options = response.get('options', [])
            correct_answer = response.get('correct_answer', '')

            # Check for minimum content requirements
            if len(question_text) < 10:
                logger.warning("⚠️ Question text too short")
                return False

            if len(options) != 4:
                logger.warning("⚠️ Invalid number of options")
                return False

            # Check for coherence indicators
            incoherent_patterns = [
                r'reproductive health.*cancer',  # Mixed with unrelated topics
                r'cancer.*neuroscience',  # Unrelated topic mixing
                r'spaghetti',  # Literal AI spaghetti
                r'lorem ipsum',  # Placeholder text
                r'test.*test.*test',  # Repetitive test content
                r'example.*example.*example',  # Repetitive examples
                r'undefined.*undefined',  # Undefined content
            ]

            full_text = f"{question_text} {' '.join(options)} {correct_answer}".lower()

            import re
            for pattern in incoherent_patterns:
                if re.search(pattern, full_text):
                    logger.warning(f"⚠️ Detected incoherent content pattern: {pattern}")
                    return False

            # Check for option diversity (not all the same)
            unique_options = set(opt.lower().strip() for opt in options if opt)
            if len(unique_options) < 3:  # At least 3 unique options
                logger.warning("⚠️ Options lack diversity")
                return False

            logger.debug("✅ Response quality validation passed")
            return True

        except Exception as e:
            logger.error(f"❌ Error validating response quality: {e}")
            return False

    async def _call_openai(self, config: Dict, prompt: str) -> Optional[Dict[str, Any]]:
        """Call OpenAI API using the robust request handler"""
        headers = {k: v.format(api_key=config['api_key']) for k, v in config['headers'].items()}
        
        payload = {
            "model": config['model'],
            "messages": [
                {"role": "system", "content": "You are an expert MCQ generator. Generate only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.8,
            "max_tokens": 1000
        }
        
        try:
            data = await self._make_api_request(config['base_url'], headers, payload, "OpenAI")
            content = data['choices'][0]['message']['content']
            return self._parse_json_response(content)
        except Exception as e:
            logger.error(f"OpenAI API call failed after retries: {e}")
            raise  # Re-raise to trigger fallback to next provider

    async def _call_anthropic(self, config: Dict, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Anthropic Claude API using the robust request handler"""
        headers = {k: v.format(api_key=config['api_key']) for k, v in config['headers'].items()}
        
        payload = {
            "model": config['model'],
            "max_tokens": 1000,
            "temperature": 0.8,
            "messages": [
                {"role": "user", "content": f"You are an expert MCQ generator. Generate only valid JSON.\n\n{prompt}"}
            ]
        }
        
        try:
            data = await self._make_api_request(config['base_url'], headers, payload, "Anthropic")
            content = data['content'][0]['text']
            return self._parse_json_response(content)
        except Exception as e:
            logger.error(f"Anthropic API call failed after retries: {e}")
            raise  # Re-raise to trigger fallback to next provider

    async def _call_gemini(self, config: Dict, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Google Gemini API using the robust request handler"""
        url = f"{config['base_url']}?key={config['api_key']}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"You are an expert MCQ generator. Generate only valid JSON.\n\n{prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": 1000
            }
        }
        
        try:
            data = await self._make_api_request(url, config['headers'], payload, "Gemini")
            content = data['candidates'][0]['content']['parts'][0]['text']
            return self._parse_json_response(content)
        except Exception as e:
            logger.error(f"Gemini API call failed after retries: {e}")
            raise  # Re-raise to trigger fallback to next provider

    async def _call_groq(self, config: Dict, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Groq API using the robust request handler"""
        headers = {k: v.format(api_key=config['api_key']) for k, v in config['headers'].items()}
        
        # [OK] ENHANCED: Smart prompt compression for Groq while preserving PhD-level difficulty
        if len(prompt) > 2500:  # More conservative limit for better success rate
            # Extract key parameters from the original prompt
            difficulty_match = re.search(r'difficulty[\'\"]\s*:\s*[\'\"]([^\'\"]+)[\'\"]', prompt, re.IGNORECASE)
            topic_match = re.search(r'topic[\'\"]\s*:\s*[\'\"]([^\'\"]+)[\'\"]', prompt, re.IGNORECASE)
            type_match = re.search(r'question.*type.*numerical|numerical.*question', prompt, re.IGNORECASE)
            
            # CRITICAL: Extract diversity requirements to prevent repetition
            diversity_focus_match = re.search(r'MANDATORY FOCUS.*?Question #(\d+).*?:\s*([^[TARGET][LIST]\n]+)', prompt, re.DOTALL)
            diversity_subtopic_match = re.search(r'REQUIRED SUBTOPIC AREA:\s*([^[FORBIDDEN][INFO]\n]+)', prompt, re.DOTALL)
            diversity_forbidden_match = re.search(r'ABSOLUTELY FORBIDDEN FOR THIS QUESTION:\s*([^[INFO][RELOAD]\n]+)', prompt, re.DOTALL)
            diversity_example_match = re.search(r'SUGGESTED EXAMPLE AREA:\s*([^[RELOAD][EMERGENCY]\n]+)', prompt, re.DOTALL)
            
            difficulty = difficulty_match.group(1) if difficulty_match else "expert" 
            topic = topic_match.group(1) if topic_match else "physics"
            question_num = diversity_focus_match.group(1) if diversity_focus_match else "1"
            diversity_focus = diversity_focus_match.group(2).strip() if diversity_focus_match else "experimental techniques"
            diversity_subtopic = diversity_subtopic_match.group(1).strip() if diversity_subtopic_match else "precision measurements"
            diversity_forbidden = diversity_forbidden_match.group(1).strip() if diversity_forbidden_match else "basic calculations"
            diversity_example = diversity_example_match.group(1).strip() if diversity_example_match else "atomic clocks"
            is_numerical = bool(type_match)
            
            # SIMPLIFIED: Create very focused prompt that works with Groq
            numerical_instruction = "NUMERICAL CALCULATION QUESTION with formulas and numbers." if is_numerical else "conceptual question"
            
            prompt = f"""Create a {difficulty} {numerical_instruction} about {topic}.

Question #{question_num} focus: {diversity_focus}
Required area: {diversity_subtopic}
Forbidden: {diversity_forbidden}
Example type: {diversity_example}

Make this question DIFFERENT from {diversity_forbidden}. Focus on {diversity_subtopic}.

{"Include calculations, formulas, numbers." if is_numerical else "Focus on concepts."}

STRICT JSON FORMAT REQUIRED:
{{
  "question": "Your {difficulty} calculation question about {diversity_subtopic}",
  "options": {{
    "A": "First numerical answer with units",
    "B": "Second numerical answer with units", 
    "C": "Third numerical answer with units",
    "D": "Fourth numerical answer with units"
  }},
  "correct": "A",
  "explanation": "Step-by-step calculation showing why A is correct"
}}

CRITICAL: You MUST include exactly 4 options (A, B, C, D) with numerical values and units."""
            
            logger.info(f"[RELOAD] Groq: Simplified prompt to {len(prompt)} characters")
            logger.info(f"[TARGET] Groq: Focus: {diversity_focus[:30]}...")
            logger.info(f"[FORBIDDEN] Groq: Forbidden: {diversity_forbidden[:30]}...")
        
        payload = {
            "model": config['model'],
            "messages": [
                {"role": "system", "content": "You are an expert MCQ generator. Generate only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,  # Slightly lower for more consistent output
            "max_tokens": 1500,  # [HOT] CRITICAL FIX: Increased from 800 to 1500 for complex expert questions
            "top_p": 0.9
        }
        
        try:
            async with self.session.post(config['base_url'], headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response_text = await response.text()
                
                # Enhanced logging for debugging
                logger.info(f"[SEARCH] DEBUG: Groq raw response (first 500 chars): {response_text[:500]}...")
                
                if response.status == 200:
                    try:
                        response_data = await response.json()
                        
                        if 'choices' in response_data and len(response_data['choices']) > 0:
                            content = response_data['choices'][0]['message']['content']
                            
                            # [HOT] CRITICAL FIX: Enhanced debugging and truncation detection
                            logger.info(f"[SEARCH] DEBUG: Groq content length: {len(content)} chars")
                            logger.info(f"[SEARCH] DEBUG: Groq content (first 300 chars): {content[:300]}...")
                            
                            # Check for truncation indicators
                            if len(content) > 1000 and not content.rstrip().endswith('}'):
                                logger.warning(f"[WARNING] Groq response may be truncated - doesn't end with closing brace")
                                logger.info(f"[SEARCH] DEBUG: Last 100 chars: ...{content[-100:]}")
                            
                            # Check if response indicates truncation
                            finish_reason = response_data['choices'][0].get('finish_reason', '')
                            if finish_reason == 'length':
                                logger.warning(f"[WARNING] Groq response truncated due to max_tokens limit")
                                logger.info(f"[CONFIG] Consider increasing max_tokens for complex questions")
                            else:
                                logger.info(f"[OK] Groq finish_reason: {finish_reason}")
                            
                            # Parse the JSON content
                            result = self._parse_json_response(content)
                            if result:
                                logger.info(f"[OK] Successfully parsed MCQ: {len(result.get('options', {}))} options, correct: {result.get('correct', 'N/A')[:50]}...")
                                return result
                            else:
                                logger.error(f"[ERROR] Failed to parse JSON from Groq response")
                                logger.error(f"[ERROR] Full content length: {len(content)} chars")
                                logger.error(f"[ERROR] Content preview: {content[:500]}...")
                                logger.error(f"[ERROR] Content ending: ...{content[-100:] if len(content) > 100 else content}")
                                return None
                        else:
                            logger.error(f"[ERROR] Groq response missing choices: {response_data}")
                            return None
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"[ERROR] Groq JSON decode error: {e}")
                        logger.error(f"[ERROR] Raw response that failed: {response_text}")
                        return None
                else:
                    logger.error(f"[ERROR] Groq API error {response.status}: {response_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"[ERROR] Groq request exception: {e}")
            return None

    async def _call_openrouter(self, config: Dict, prompt: str) -> Optional[Dict[str, Any]]:
        """Call OpenRouter API with free model selection and intelligent rate limiting"""
        headers = {k: v.format(api_key=config['api_key']) for k, v in config['headers'].items()}
        headers['HTTP-Referer'] = 'https://knowledge-app.local'
        headers['X-Title'] = 'Knowledge App'
        
        # Get free models list with intelligent selection
        free_models = config.get('free_models', [config['model']])
        
        # Enhanced rate limit handling
        rate_limit_wait_times = [2, 5, 10, 15, 20]  # Progressive wait times for rate limits
        
        # Try each free model until one works
        for model_attempt, model in enumerate(free_models):
            for retry_attempt in range(3):  # Up to 3 attempts per model for rate limits
                try:
                    if retry_attempt == 0:
                        logger.info(f"[RELOAD] Trying OpenRouter free model: {model} (attempt {model_attempt + 1}/{len(free_models)})")
                    else:
                        logger.info(f"[RELOAD] Retrying {model} after rate limit (retry {retry_attempt + 1}/3)")
                    
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are an expert MCQ generator. Generate only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,  # Lower temperature for free models for consistency
                        "max_tokens": 800,   # Conservative token limit for free models
                        "top_p": 0.9,
                        "frequency_penalty": 0,
                        "presence_penalty": 0
                    }
                    
                    # Some free models don't support response_format, so make it optional
                    try:
                        payload["response_format"] = {"type": "json_object"}
                    except:
                        pass
                    
                    data = await self._make_api_request_with_rate_limit_handling(
                        config['base_url'], headers, payload, f"OpenRouter-{model}")
                    content = data['choices'][0]['message']['content']
                    result = self._parse_json_response(content)
                    
                    if result:
                        logger.info(f"[OK] OpenRouter model {model} succeeded on attempt {retry_attempt + 1}")
                        logger.info(f"[SEARCH] DEBUG: OpenRouter API returned question: '{result.get('question', 'N/A')[:100]}...'")
                        return result
                        
                except aiohttp.ClientResponseError as e:
                    if e.status == 429:  # Rate limited
                        wait_time = rate_limit_wait_times[min(retry_attempt, len(rate_limit_wait_times) - 1)]
                        logger.warning(f"[FORBIDDEN] OpenRouter model {model} rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue  # Retry same model
                    else:
                        logger.warning(f"[WARNING] OpenRouter model {model} failed with status {e.status}: {e}")
                        break  # Try next model
                except Exception as e:
                    logger.warning(f"[WARNING] OpenRouter model {model} failed: {e}")
                    if "cannot schedule new futures after shutdown" in str(e):
                        logger.error("[STOP] System shutdown detected, stopping generation")
                        raise e
                    break  # Try next model
                    
        raise Exception(f"OpenRouter API error: All {len(free_models)} free models failed or rate limited")


    
    def _parse_json_response_robust(self, content: str) -> Optional[Dict[str, Any]]:
        """
        [CONFIG] FIX: Use centralized JSON parser to eliminate code duplication
        """
        from ..utils.json_parser import parse_json_response_robust
        return parse_json_response_robust(content)

    async def _enforce_rate_limit(self):
        """[START] RATE LIMITING: Ensure we don't exceed API rate limits"""
        async with self._request_lock:
            import time
            current_time = time.time()

            # Remove timestamps older than 1 minute
            self._request_timestamps = [
                timestamp for timestamp in self._request_timestamps
                if current_time - timestamp < 60
            ]

            # Check if we're at the limit
            if len(self._request_timestamps) >= self._max_requests_per_minute:
                # Calculate how long to wait
                oldest_request = min(self._request_timestamps)
                wait_time = 60 - (current_time - oldest_request) + 1  # +1 for safety margin

                if wait_time > 0:
                    logger.info(f"[WAIT] Rate limit reached ({len(self._request_timestamps)}/{self._max_requests_per_minute}), waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)

                    # Clean up timestamps again after waiting
                    current_time = time.time()
                    self._request_timestamps = [
                        timestamp for timestamp in self._request_timestamps
                        if current_time - timestamp < 60
                    ]

            # Record this request
            self._request_timestamps.append(current_time)
            logger.info(f"[STATS] Rate limit status: {len(self._request_timestamps)}/{self._max_requests_per_minute} requests in last minute")

    def _needs_rate_limiting(self) -> bool:
        """
        [CONFIG] FIX: Determine if rate limiting is needed based on current provider

        Only apply rate limiting to providers that actually need it.
        """
        # OpenRouter has generous rate limits - no need for aggressive limiting
        if 'openrouter' in self.available_providers:
            return False

        # Anthropic has reasonable limits for paid tiers
        if 'anthropic' in self.available_providers and len(self.available_providers) == 1:
            return False

        # Groq and others have strict free tier limits
        if 'groq' in self.available_providers:
            return True

        # Default: apply rate limiting for unknown providers
        return True

    def _create_groq_optimized_prompt(self, topic: str, difficulty: str, question_type: str) -> str:
        """
        [CONFIG] FIX: Create optimized prompt for Groq using structured parameters

        This replaces the brittle regex parsing with reliable parameter-based generation.
        """
        is_numerical = question_type.lower() == "numerical"

        if is_numerical:
            type_instruction = """Create a numerical calculation question that requires:
- Specific numerical values and units
- Mathematical problem-solving skills
- Multi-step calculations
- Options with different numerical answers"""
        else:
            type_instruction = """Create a conceptual question that requires:
- Deep theoretical understanding
- Advanced reasoning skills
- Knowledge of principles and concepts
- Options testing different aspects of understanding"""

        return f"""Create a {difficulty}-level multiple choice question about {topic}.

{type_instruction}

Requirements:
- Question must be {difficulty} difficulty level
- Include advanced terminology appropriate for the topic
- Test deep understanding, not simple recall
- End question with a question mark (?)

JSON FORMAT (EXACT):
{{
  "question": "Your {difficulty} question about {topic}?",
  "options": {{
    "A": "First option",
    "B": "Second option",
    "C": "Third option",
    "D": "Fourth option"
  }},
  "correct": "A",
  "explanation": "Detailed explanation of why the answer is correct"
}}

Generate ONLY the JSON object, no additional text."""

    def _validate_json_structure_robust(self, parsed_json: Dict) -> bool:
        """Validate that the JSON has the required MCQ structure"""
        
        if not isinstance(parsed_json, dict):
            return False
            
        required_fields = ['question', 'options', 'correct']
        if not all(field in parsed_json for field in required_fields):
            return False
        
        # Check question is non-empty string
        question = parsed_json.get('question', '')
        if not isinstance(question, str) or not question.strip():
            return False
        
        # Check options format (can be dict or list)
        options = parsed_json.get('options', [])
        if isinstance(options, dict):
            # Handle {"A": "...", "B": "...", "C": "...", "D": "..."} format
            option_keys = ['A', 'B', 'C', 'D']
            if not all(key in options for key in option_keys):
                return False
            if not all(isinstance(options[key], str) and options[key].strip() for key in option_keys):
                return False
        elif isinstance(options, list):
            # Handle ["...", "...", "...", "..."] format
            if len(options) != 4:
                return False
            if not all(isinstance(opt, str) and opt.strip() for opt in options):
                return False
        else:
            return False
        
        # Check correct answer
        correct = parsed_json.get('correct', '')
        if not isinstance(correct, str) or correct not in ['A', 'B', 'C', 'D']:
            return False
        
        return True
    
    def _fix_common_json_issues(self, response: str) -> Optional[str]:
        """Attempt to fix common JSON formatting issues"""
        
        try:
            # Find the JSON part
            start_brace = response.find('{')
            end_brace = response.rfind('}')
            if start_brace == -1 or end_brace == -1:
                return None
            
            json_part = response[start_brace:end_brace + 1]
            
            # Fix common issues
            # Remove trailing commas before } or ]
            json_part = re.sub(r',(\s*[}\]])', r'\1', json_part)
            
            # Ensure proper quoting of keys
            json_part = re.sub(r'(\w+):', r'"\1":', json_part)
            
            # Fix single quotes to double quotes
            json_part = json_part.replace("'", '"')
            
            return json_part
        except:
            return None

    # [START] ENHANCED: Chain-of-Thought prompt engineering
    def _create_optimized_prompt(self, topic: str, context: str, question_index: int, 
                                 difficulty: str = "medium", question_type: str = "mixed") -> str:
        """
        [BRAIN] ENHANCED: Chain-of-Thought prompt engineering for higher quality questions
        Eliminates meta-level questions and encourages deep understanding
        """
        
        # Context preparation
        context_instruction = ""
        if context and context.strip():
            context_instruction = f"""
**Context for Question Generation:**
{context.strip()}

Use this context to inform your question, but do NOT simply copy sentences from it.
"""
        
        # [HOT] ENHANCED: Detailed difficulty requirements (matching offline models)
        difficulty_requirements = {
            "easy": {
                "level": "basic",
                "description": "fundamental concepts, simple recall, basic definitions",
                "examples": "simple definitions, basic facts, elementary concepts",
                "instruction": "introductory level that tests basic understanding"
            },
            "medium": {
                "level": "intermediate", 
                "description": "understanding relationships, applying concepts, moderate analysis",
                "examples": "connecting ideas, practical applications, cause-and-effect",
                "instruction": "intermediate level that requires application of concepts"
            },
            "hard": {
                "level": "advanced",
                "description": "complex analysis, synthesis, evaluation, expert-level reasoning, specific mechanisms and pathways",
                "examples": "multi-step problem solving, critical evaluation, advanced synthesis, edge cases, molecular processes",
                "instruction": "advanced level that requires synthesis and critical thinking"
            },
            "expert": {
                "level": "expert",
                "description": "deep domain knowledge, complex reasoning, advanced synthesis, cutting-edge understanding",
                "examples": "expert-level analysis, state-of-the-art knowledge, complex theoretical frameworks, advanced applications",
                "instruction": "expert level that requires deep domain knowledge and complex reasoning"
            }
        }
        
        diff_config = difficulty_requirements.get(difficulty.lower(), difficulty_requirements["medium"])
        difficulty_instruction = diff_config["instruction"]
        
        # [EMERGENCY] ENHANCED: Anti-vague question enforcement (matching offline models)
        anti_vague_section = ""
        if difficulty.lower() in ["hard", "expert"]:
            phd_level_demand = ""
            if difficulty.lower() == "expert":
                phd_level_demand = f"""
[EXPERT] PHD-LEVEL RESEARCH DEMAND FOR {topic.upper()}:
[FORBIDDEN] COMPLETELY BANNED: Basic wavelength calculations (λ = hc/E)
[FORBIDDEN] COMPLETELY BANNED: Simple transitions between energy levels
[FORBIDDEN] COMPLETELY BANNED: Undergraduate textbook formulas (E=hf, λ=hc/E, Rydberg)
[FORBIDDEN] COMPLETELY BANNED: Graduate coursework level (simple binding energy, basic quantum)
[FORBIDDEN] COMPLETELY BANNED: Any question found in standard textbooks
[FORBIDDEN] COMPLETELY BANNED: Questions solvable with basic formulas

[OK] MANDATORY: Advanced quantum field theory applications  
[OK] MANDATORY: Many-body interactions and correlations
[OK] MANDATORY: QED corrections and radiative effects
[OK] MANDATORY: Experimental precision at research frontiers
[OK] MANDATORY: Questions requiring specialized computational methods
[OK] MANDATORY: Corrections beyond Born-Oppenheimer approximation
[OK] MANDATORY: Questions requiring knowledge of recent research papers (2020+)

RESEARCH TOPICS REQUIRED:
- Hyperfine structure with relativistic corrections
- Many-body perturbation theory calculations  
- Quantum electrodynamics beyond lowest order
- Casimir-Polder interactions in complex geometries
- Rydberg atom physics in external fields
- Uehling potential and vacuum polarization effects
- Scattering theory (R-matrix, close-coupling)
- Exotic atomic systems (antihydrogen, muonium)

EXAMPLES OF REQUIRED PHD-LEVEL {topic.upper()} QUESTIONS:
[OK] "Calculate the energy shift of the 1s state in hydrogen due to the Uehling potential, including relativistic corrections and many-body interactions"
[OK] "Determine the second-order relativistic correction to hyperfine splitting including vacuum polarization effects"
[OK] "Calculate the Casimir-Polder coefficient including retardation effects for Rydberg atom interactions"
[OK] "Find scattering phase shifts using multichannel quantum defect theory for exotic atom collisions"
[OK] "Determine the photoionization cross-section near Cooper minima using RPAE theory with correlation effects"
[OK] "Calculate the dynamic polarizability including core-valence correlation effects using coupled-cluster theory"
"""
            elif difficulty.lower() == "hard":
                phd_level_demand = f"""
[HOT] HARD MODE - GRADUATE-LEVEL DEMAND FOR {topic.upper()}:
[FORBIDDEN] COMPLETELY BANNED: Basic single-formula calculations (F=ma, E=mc², KE=½mv², etc.)
[FORBIDDEN] COMPLETELY BANNED: Direct textbook formula applications
[FORBIDDEN] COMPLETELY BANNED: Simple unit conversions or substitutions
[FORBIDDEN] COMPLETELY BANNED: Single-step problems solvable in under 2 minutes
[FORBIDDEN] COMPLETELY BANNED: Undergraduate homework-level questions
[FORBIDDEN] COMPLETELY BANNED: Basic conceptual definitions or explanations

[OK] MANDATORY: Multi-step problem solving requiring 3+ concepts
[OK] MANDATORY: Advanced analytical techniques and methods
[OK] MANDATORY: Complex systems with multiple interacting components
[OK] MANDATORY: Non-trivial mathematical derivations or proofs
[OK] MANDATORY: Advanced applications requiring deep domain knowledge
[OK] MANDATORY: Problems requiring synthesis of multiple principles
[OK] MANDATORY: Graduate-level complexity (master's degree level)

REQUIRED HARD MODE COMPLEXITY AREAS:
- Advanced mathematical techniques (differential equations, complex analysis)
- Multi-body systems and interactions
- Non-linear phenomena and chaos theory
- Advanced thermodynamics and statistical mechanics
- Quantum mechanical systems beyond hydrogen
- Electromagnetic field theory applications
- Modern physics beyond introductory level
- Computational physics and numerical methods

EXAMPLES OF REQUIRED HARD-LEVEL {topic.upper()} QUESTIONS:
[OK] "Analyze the coupled oscillator system with damping and derive the normal mode frequencies"
[OK] "Calculate the scattering cross-section for electron-atom collisions using Born approximation"
[OK] "Determine the phase transition temperature using mean field theory and critical exponents"
[OK] "Solve the time-dependent Schrödinger equation for a particle in a time-varying potential"
[OK] "Find the dispersion relation for electromagnetic waves in a plasma using kinetic theory"
[OK] "Calculate the correlation function for a many-body quantum system using Green's functions"
"""
            
            anti_vague_section = f"""
[EMERGENCY] {difficulty.upper()} MODE - ZERO TOLERANCE FOR VAGUE QUESTIONS:
[ERROR] ABSOLUTELY FORBIDDEN: "What is the primary function of..."
[ERROR] ABSOLUTELY FORBIDDEN: "What is the main purpose of..."  
[ERROR] ABSOLUTELY FORBIDDEN: "What does X do?"
[ERROR] ABSOLUTELY FORBIDDEN: Basic definition questions
[ERROR] ABSOLUTELY FORBIDDEN: General overview questions
[ERROR] ABSOLUTELY FORBIDDEN: "What distinguishes..." generic questions
{f'[ERROR] ABSOLUTELY FORBIDDEN: Single-formula calculations (F=ma, E=mc², KE=½mv², etc.)' if difficulty.lower() == "hard" else ''}
{f'[ERROR] ABSOLUTELY FORBIDDEN: Direct textbook applications without analysis' if difficulty.lower() == "hard" else ''}
[OK] MANDATORY: Specific mechanisms, pathways, processes
[OK] MANDATORY: Multi-step reasoning and analysis
[OK] MANDATORY: {difficulty}-level detail and precision
[OK] MANDATORY: Questions requiring deep understanding

{phd_level_demand}

EXAMPLES OF BANNED vs REQUIRED QUESTIONS:
[ERROR] BANNED: "What distinguishes expert-level knowledge in {topic}?"
[ERROR] BANNED: "What is the main characteristic of {topic}?"
{f'[ERROR] BANNED: "Calculate the kinetic energy of an electron..." (basic formula)' if difficulty.lower() == "hard" else ''}
{f'[ERROR] BANNED: "Find the force when F=ma and m=5kg, a=2m/s²" (direct substitution)' if difficulty.lower() == "hard" else ''}
[OK] REQUIRED: "Which specific mechanism in {topic} explains [complex scenario]?"
[OK] REQUIRED: "During which phase/process does [specific event] occur in {topic}?"
{f'[OK] REQUIRED: "Analyze the coupled system behavior when [multiple conditions]..."' if difficulty.lower() == "hard" else ''}
{f'[OK] REQUIRED: "Derive the relationship between [complex variables] considering [multiple effects]..."' if difficulty.lower() == "hard" else ''}
"""

        # [TARGET] Topic-specific guidance (matching offline models)
        topic_guidance = self._get_topic_specific_guidance(topic, difficulty)
        
        # Question type instructions - ULTRA-AGGRESSIVE TYPE ENFORCEMENT (100% adherence proven)
        type_instruction = ""
        if question_type.lower() == "numerical":
            phd_numerical_examples = ""
            if difficulty.lower() == "expert":
                # PhD-level examples that actually work (based on successful test)
                example_sets = {
                    0: [  # QED/Vacuum Effects (SUCCESSFUL PATTERN)
                        "Calculate the energy shift of the 1s state in hydrogen due to the Uehling potential, including relativistic corrections and many-body interactions, to an accuracy of 1 part in 10^9",
                        "Determine the second-order vacuum polarization correction to the Lamb shift including finite nuclear size effects",
                        "Find the anomalous magnetic moment contribution from fourth-order QED vertex corrections with hadronic vacuum polarization"
                    ],
                    1: [  # Many-Body/Correlation Effects  
                        "Calculate the ground-state correlation energy of Be using full CI with relativistic corrections, including Breit interaction terms",
                        "Determine the dynamic polarizability of Cs at 852 nm including core-valence correlation effects using CCSD(T) theory",
                        "Find the van der Waals C6 coefficient for Rydberg atom interactions including retardation effects and many-body dispersion"
                    ],
                    2: [  # Scattering/Collision Theory
                        "Calculate the photoionization cross-section near Cooper minima using RPAE theory with interchannel coupling effects",
                        "Determine the electron impact excitation cross-section using 19-state close-coupling calculations with pseudostates", 
                        "Find the Feshbach resonance position for ultracold Li-Cs collisions including magnetic dipole-dipole interactions"
                    ],
                    3: [  # Relativistic/Field Effects
                        "Calculate the second-order relativistic correction to hyperfine splitting in muonium including finite nuclear size", 
                        "Determine the gravitational redshift correction to optical clock transitions at height differences of 1 cm",
                        "Find the AC Stark shift for Sr clock transitions in optical lattices including higher-order multipole contributions"
                    ],
                    4: [  # Advanced Spectroscopy/Precision
                        "Calculate the blackbody radiation shift for Yb+ clock transitions at 300K including dynamic corrections",
                        "Determine the autoionization linewidth using multichannel quantum defect theory with frame transformation",
                        "Find the quantum Fisher information bound for atomic interferometry with N=10^6 atoms including decoherence effects"
                    ]
                }
                
                examples_for_this_question = example_sets.get(question_index % 5, example_sets[0])
                example_text = "\n".join([f"[OK] \"{ex}\"" for ex in examples_for_this_question])
                
                phd_numerical_examples = f"""

[EXPERT] PHD-LEVEL NUMERICAL EXAMPLES FOR THIS QUESTION TYPE:
{example_text}

[EMERGENCY] DEMAND: RESEARCH-LEVEL COMPLEXITY - NO TEXTBOOK CALCULATIONS!
[RELOAD] GENERATE A QUESTION SIMILAR TO ABOVE EXAMPLES - NOT QED GROUND STATE ENERGY!
"""

            type_instruction = f"""
🔢[EMERGENCY]🔢 NUMERICAL QUESTION - ZERO TOLERANCE ENFORCEMENT 🔢[EMERGENCY]🔢

**ABSOLUTE NUMERICAL PURITY REQUIREMENTS:**
[TARGET] MANDATORY STARTER: MUST begin with "Calculate", "Compute", "Solve", "Determine", "Find", "Evaluate"
[TARGET] NUMBERS REQUIRED: Include specific values (5.2 kg, 298 K, 3.14 rad/s, 1.5 × 10⁻⁹ m)
[TARGET] MATH REQUIRED: Equations, formulas, mathematical expressions that need computation
[TARGET] PURE NUMERICAL OPTIONS: ALL 4 options = numbers with units (25.4 J, 8.3 × 10⁻³ mol/L)
[TARGET] SOLUTION = MATH: Answered through calculation, NOT conceptual reasoning
[TARGET] UNITS REQUIRED: SI units or domain-specific units

[FORBIDDEN] CONCEPTUAL CONTAMINATION - AUTOMATIC FAILURE:
[ERROR] "explain" [ERROR] "why" [ERROR] "how" [ERROR] "describe" [ERROR] "analyze" [ERROR] "mechanism" [ERROR] "principle"
[ERROR] "compare" [ERROR] "relationship" [ERROR] "effect" [ERROR] "theory" [ERROR] "concept" [ERROR] "understanding"
[ERROR] "assuming" [ERROR] "considering" [ERROR] "utilizing" [ERROR] "given that" [ERROR] "taking into account"
[ERROR] NO conceptual discussion, NO theoretical explanations, NO "what does this mean"
[ERROR] NO conditional phrases that imply conceptual understanding needed

**ZERO TOLERANCE VERIFICATION:**
[SEARCH] CALCULATION VERB? → Must be "Calculate/Compute/Solve/Determine/Find/Evaluate"
[SEARCH] SPECIFIC NUMBERS? → Must include measurable quantities with units
[SEARCH] PURE NUMERICAL OPTIONS? → ALL options must be numbers, NO text descriptions
[SEARCH] MATH COMPUTATION? → Must require mathematical operations to solve
[SEARCH] NO CONCEPTUAL WORDS? → Zero conceptual verbs or theoretical discussion

**PERFECT NUMERICAL PATTERNS THAT WORK:**
[OK] "Calculate the [specific quantity] when [parameter] = [value] and [parameter] = [value]."
[OK] "Determine the [specific measurement] for [system] with [parameter] = [number][unit]."
[OK] "Find the [numerical result] using [given values] and [specified conditions]."
[OK] "Compute the [exact quantity] from the relationship [equation] with [specified inputs]."

**AVOID THESE PATTERNS (CONTAMINATED):**
[ERROR] "Calculate X considering the effects of Y" (adds conceptual element)
[ERROR] "Determine X assuming that the mechanism involves Y" (theoretical reasoning)
[ERROR] "Find X given the relationship between Y and Z" (conceptual understanding needed)

{phd_numerical_examples}

💀 FAILURE MODES TO AVOID: Any question asking "why", any options with explanations, any theoretical discussion

[EMERGENCY] WARNING: Any question that is not purely numerical will be AUTOMATICALLY REJECTED!
[EMERGENCY] DEMAND: Generate ONLY calculation questions with numerical answers!
"""
        elif question_type.lower() == "conceptual":
            type_instruction = f"""
[BRAIN][EMERGENCY][BRAIN] CONCEPTUAL QUESTION - ZERO TOLERANCE ENFORCEMENT [BRAIN][EMERGENCY][BRAIN]

**ABSOLUTE CONCEPTUAL PURITY REQUIREMENTS:**
[TARGET] MANDATORY STARTER: MUST begin with "Explain", "Why", "How", "What happens", "Describe", "Analyze"  
[TARGET] PRINCIPLES FOCUS: Theories, mechanisms, cause-effect relationships, underlying principles
[TARGET] ZERO MATH: NO calculations, NO specific numerical values, NO mathematical operations
[TARGET] PURE CONCEPTUAL OPTIONS: ALL 4 options = concept descriptions, mechanism explanations
[TARGET] SOLUTION = UNDERSTANDING: Answered through theoretical knowledge, NOT computation
[TARGET] QUALITATIVE LANGUAGE: Relationships, trends, phenomena without numerical specifics

[FORBIDDEN] NUMERICAL CONTAMINATION - AUTOMATIC FAILURE:
[ERROR] "calculate" [ERROR] "compute" [ERROR] "solve" [ERROR] "determine" [ERROR] "find" [ERROR] "evaluate"
[ERROR] Numbers with units [ERROR] Mathematical expressions [ERROR] Formulas [ERROR] Equations [ERROR] Calculations
[ERROR] NO numerical operations, NO specific values, NO computational elements

**ZERO TOLERANCE VERIFICATION:**
[SEARCH] UNDERSTANDING VERB? → Must be "Explain/Why/How/What happens/Describe/Analyze"
[SEARCH] NO NUMBERS? → Zero specific numerical values or calculations required
[SEARCH] PURE CONCEPTUAL OPTIONS? → ALL options describe concepts/mechanisms, NO numbers
[SEARCH] UNDERSTANDING REQUIRED? → Must test theoretical knowledge, NOT math skills
[SEARCH] NO NUMERICAL WORDS? → Zero calculation verbs or mathematical operations

EXAMPLES OF CORRECT PURE CONCEPTUAL QUESTIONS:
[OK] "Why does increasing temperature generally increase the rate of chemical reactions?"
[OK] "How does the electron configuration affect the magnetic properties of transition metals?"
[OK] "What happens to entropy when a crystalline solid dissolves in water?"
[OK] "Explain the mechanism by which catalysts lower activation energy."

EXAMPLES OF BANNED QUESTIONS (DO NOT GENERATE):
[ERROR] "Explain the result of calculating X..."
[ERROR] "Why does the formula Y = Z apply..."
[ERROR] "Describe how to solve for X..."
[ERROR] Any question mixing conceptual understanding with calculations

💀 FAILURE MODES TO AVOID: Any question asking to "calculate", any numerical options, any mathematical operations

[EMERGENCY] WARNING: Any question that includes calculations will be AUTOMATICALLY REJECTED!
[EMERGENCY] DEMAND: Generate ONLY understanding/explanation questions!
"""
        elif question_type.lower() == "application":
            type_instruction = """
[CONFIG] APPLICATION QUESTION REQUIREMENTS:
- Present a realistic scenario or problem to solve
- Require applying knowledge to novel situations
- Test practical implementation of concepts
- Focus on "how" and "why" rather than "what"
"""
        
        # [BRAIN] ENHANCED CHAIN-OF-THOUGHT PROMPT with sophisticated difficulty handling
        question_type_emphasis = ""
        if question_type.lower() == "numerical":
            question_type_emphasis = "🔢 [EMERGENCY] CRITICAL: GENERATE A NUMERICAL CALCULATION QUESTION - NOT A CONCEPTUAL QUESTION! [EMERGENCY] 🔢"
        
        # Add diversity enforcement based on question index - ENHANCED VARIATION
        diversity_requirements = {
            0: {
                "focus": "EXPERIMENTAL TECHNIQUES and precision measurements",
                "subtopic": "high-precision spectroscopy, interferometry, or advanced measurement methods",
                "forbidden": "theoretical calculations, basic QED corrections",
                "example_area": "atomic clocks, precision spectroscopy, or quantum sensing"
            },
            1: {
                "focus": "COLLISION PROCESSES and scattering phenomena", 
                "subtopic": "electron-atom collisions, photoionization, or collision cross-sections",
                "forbidden": "energy level calculations, bound state problems",
                "example_area": "electron impact excitation, photodetachment, or collision dynamics"
            },
            2: {
                "focus": "EXOTIC MATTER and unusual atomic systems",
                "subtopic": "antihydrogen, muonium, Rydberg atoms, or highly charged ions",
                "forbidden": "hydrogen-like systems, standard elements",
                "example_area": "antimatter physics, exotic atoms, or artificial atomic systems"
            },
            3: {
                "focus": "EXTERNAL FIELD EFFECTS and atom-field interactions",
                "subtopic": "strong laser fields, magnetic fields, or AC Stark effects",
                "forbidden": "weak field perturbations, simple Zeeman effect",
                "example_area": "strong-field ionization, optical lattices, or field-dressed states"
            },
            4: {
                "focus": "MANY-BODY SYSTEMS and collective phenomena",
                "subtopic": "ultracold gases, quantum degenerate systems, or collective excitations",
                "forbidden": "single-atom properties, isolated systems",
                "example_area": "Bose-Einstein condensates, Fermi gases, or quantum phase transitions"
            },
            5: {
                "focus": "QUANTUM INFORMATION and atomic platforms",
                "subtopic": "quantum gates, entanglement, or quantum error correction",
                "forbidden": "classical information, simple quantum states",
                "example_area": "trapped ion qubits, neutral atom arrays, or quantum algorithms"
            },
            6: {
                "focus": "NUCLEAR EFFECTS and hyperfine interactions",
                "subtopic": "nuclear spin coupling, isotope effects, or nuclear structure",
                "forbidden": "electronic structure only, spinless nuclei",
                "example_area": "hyperfine spectroscopy, nuclear moments, or isotope shifts"
            },
            7: {
                "focus": "RELATIVISTIC EFFECTS and fundamental physics",
                "subtopic": "Dirac equation solutions, relativistic corrections, or fundamental constants",
                "forbidden": "non-relativistic approximations, classical mechanics",
                "example_area": "fine structure, QED tests, or fundamental symmetries"
            }
        }
        
        diversity_info = diversity_requirements.get(question_index % len(diversity_requirements), diversity_requirements[0])
        diversity_focus = diversity_info["focus"]
        diversity_subtopic = diversity_info["subtopic"] 
        diversity_forbidden = diversity_info["forbidden"]
        diversity_example = diversity_info["example_area"]
        
        phd_mode_header = ""
        if difficulty.lower() == "expert":
            phd_mode_header = f"""
[EXPERT]🔬 PHD-LEVEL RESEARCH MODE ACTIVATED 🔬[EXPERT]

You are now a leading researcher in {topic} with access to cutting-edge literature.
This question MUST be at the level of:
- Current research frontiers in {topic}
- Advanced theoretical methods and experimental techniques  
- Questions that would challenge postdoctoral researchers
- Calculations requiring specialized software/methods
- Knowledge from recent Nature, Science, PRL papers (2020-2024)

[TARGET] MANDATORY FOCUS FOR THIS QUESTION (Question #{question_index + 1}):
{diversity_focus}

[LIST] REQUIRED SUBTOPIC AREA:
{diversity_subtopic}

[FORBIDDEN] ABSOLUTELY FORBIDDEN FOR THIS QUESTION:
{diversity_forbidden}

[INFO] SUGGESTED EXAMPLE AREA:
{diversity_example}

[RELOAD] CRITICAL DIVERSITY REQUIREMENT:
This question MUST be COMPLETELY DIFFERENT from:
- Basic QED ground state energy corrections
- Simple relativistic calculations  
- Standard textbook problems
- Questions about {diversity_forbidden}

Each question in this quiz must explore DIFFERENT aspects of {topic} physics.
Focus specifically on {diversity_subtopic} to ensure uniqueness.

[EMERGENCY] UNIQUENESS ENFORCEMENT:
- Question #{question_index + 1} must be about {diversity_focus}
- Must involve {diversity_subtopic}
- Cannot be about {diversity_forbidden}
- Must be in the area of {diversity_example}

[OK] REQUIRED: State-of-the-art knowledge in {diversity_subtopic}
[OK] REQUIRED: Advanced calculations specific to {diversity_focus}
[OK] REQUIRED: UNIQUE approach focusing on {diversity_example}
[OK] REQUIRED: Research-level complexity in {diversity_subtopic}
"""
        
        prompt = f"""You are an expert educational content creator. Generate a {diff_config['level']} difficulty multiple choice question about "{topic}".

{phd_mode_header}

{question_type_emphasis}

{anti_vague_section}

**REQUIREMENTS:**
- Topic: {topic}
- Difficulty: {difficulty.upper()} ({diff_config['description']})
- Context: {context if context else "Use educational knowledge"}

{topic_guidance}

{context_instruction}

**QUALITY STANDARDS:**
[OK] Questions must require {diff_config['description']}
[OK] Use specific, technical terminology appropriate for {difficulty} level
[OK] Avoid vague generalizations - be precise and specific
[OK] Include challenging but fair distractors
[OK] Focus on understanding mechanisms rather than simple recall
[OK] Examples: {diff_config['examples']}

{type_instruction}

**ULTRA-AGGRESSIVE CHAIN-OF-THOUGHT - ZERO TOLERANCE MODE:**

[EMERGENCY] **Step 1: TYPE PURITY ENFORCEMENT** [EMERGENCY]
{f'''- Is this 100% PURE {question_type.upper()} with ZERO contamination?
- Which MANDATORY {question_type} starter verb: {"Calculate/Compute/Solve/Determine/Find/Evaluate" if question_type == 'numerical' else "Explain/Why/How/What happens/Describe/Analyze"}?
- How will I GUARANTEE zero mixing with {"conceptual elements" if question_type == 'numerical' else "numerical elements"}?
- What forbidden words must I COMPLETELY AVOID: {'"explain", "why", "how", "describe", "analyze", "mechanism", "principle"' if question_type == 'numerical' else '"calculate", "compute", "solve", "determine", "find", "evaluate"'}?''' if question_type.lower() in ['numerical', 'conceptual'] else '- What type of question best fits the requirements?'}

[HOT] **Step 2: CONTAMINATION PREVENTION** [HOT]
{f'''- How will I ensure ZERO {"conceptual contamination" if question_type == 'numerical' else "numerical contamination"}?
- What {f"specific numbers, units, and calculations" if question_type == 'numerical' else "pure principles and mechanisms"} will I include?
- How will I make ALL options {f"purely numerical values with units" if question_type == 'numerical' else "purely conceptual descriptions"}?''' if question_type.lower() in ['numerical', 'conceptual'] else '- How will I maintain appropriate quality and focus?'}

[TARGET] **Step 3: TYPE-SPECIFIC REQUIREMENTS** [TARGET]
{'''- NUMERICAL ONLY: What specific values (kg, K, rad/s), equations, formulas needed?
- MATHEMATICAL COMPUTATION: What calculations, units, scientific notation required?
- NUMERICAL OPTIONS: All 4 must be numbers with units (J, mol/L, Hz, etc.)''' if question_type.lower() == 'numerical' else '''- CONCEPTUAL ONLY: What principles, mechanisms, theories without numbers?
- THEORETICAL UNDERSTANDING: What qualitative relationships, cause-effects?
- CONCEPTUAL OPTIONS: All 4 must describe mechanisms, effects, principles''' if question_type.lower() == 'conceptual' else '- What level of technical detail is appropriate?'}

[EXPERT] **Step 4: DIFFICULTY-LEVEL VERIFICATION** [EXPERT]
- Will this challenge someone at the {difficulty} level appropriately?
- Does it meet the {difficulty} complexity requirements?
- Are all verification checklist items satisfied?

💀 **Step 5: FAILURE MODE ELIMINATION** 💀
{'''- NO "explain", "why", "how" in numerical questions = AUTOMATIC FAILURE
- NO conceptual options (must be pure numbers with units)''' if question_type.lower() == 'numerical' else '''- NO "calculate", "solve", "find" in conceptual questions = AUTOMATIC FAILURE
- NO numerical values or calculations (must be pure concepts)''' if question_type.lower() == 'conceptual' else '- What are plausible misconceptions for this level?'}
- NO mixing, NO contamination, NO violations

**ZERO TOLERANCE FINAL VERIFICATION:**
{f'''[SEARCH] TYPE PURITY: 100% pure {question_type.upper()}, zero mixing?
[SEARCH] STARTER VERB: Correct {"calculation" if question_type == 'numerical' else "understanding"} verb used?
[SEARCH] OPTIONS: All {"numerical values" if question_type == 'numerical' else "conceptual descriptions"}?
[SEARCH] CONTAMINATION: Zero {"conceptual" if question_type == 'numerical' else "numerical"} elements?''' if question_type.lower() in ['numerical', 'conceptual'] else '[SEARCH] Quality: Does this meet all requirements?'}
[SEARCH] COMPLEXITY: Does it meet all {difficulty}-level requirements?
[SEARCH] DOMAIN: Advanced {topic} expertise required?

{f'💥 FAILURE = ANY TYPE MIXING, ANY FORBIDDEN WORDS, ANY CONTAMINATION 💥' if question_type.lower() in ['numerical', 'conceptual'] else ''}

**CRITICAL REQUIREMENTS:**
- NO meta-level questions like "What is a key concept..." or "What is the main..."
- NO generic questions that could apply to any topic
- NO overly simple recall questions unless specifically requested
- Focus on "{topic}" specifically, not general science principles

[EMERGENCY][EMERGENCY][EMERGENCY] FINAL ULTIMATUM - ZERO TOLERANCE ENFORCEMENT [EMERGENCY][EMERGENCY][EMERGENCY]

### TYPE PURITY - AUTOMATIC FAILURE FOR ANY VIOLATIONS:
{f'''🔢 NUMERICAL = PURE CALCULATION ONLY: Starter verbs (Calculate/Compute/Solve/Determine/Find/Evaluate), specific numbers, mathematical operations, numerical options ONLY
💀 ANY TYPE MIXING = IMMEDIATE AUTOMATIC FAILURE
💀 ANY FORBIDDEN WORDS = IMMEDIATE AUTOMATIC FAILURE
💀 ANY CONTAMINATION = IMMEDIATE AUTOMATIC FAILURE''' if question_type.lower() == 'numerical' else f'''[BRAIN] CONCEPTUAL = PURE UNDERSTANDING ONLY: Starter verbs (Explain/Why/How/What happens/Describe/Analyze), principles/mechanisms, conceptual options ONLY
💀 ANY TYPE MIXING = IMMEDIATE AUTOMATIC FAILURE  
💀 ANY FORBIDDEN WORDS = IMMEDIATE AUTOMATIC FAILURE
💀 ANY CONTAMINATION = IMMEDIATE AUTOMATIC FAILURE''' if question_type.lower() == 'conceptual' else '💀 MAINTAIN APPROPRIATE TYPE AND FORMAT - ZERO TOLERANCE FOR VIOLATIONS'}

### ZERO TOLERANCE FAILURE CONDITIONS:
{'''[ERROR] NUMERICAL questions with "explain", "why", "how", "describe", "analyze" = FAIL
[ERROR] NUMERICAL questions with conceptual options (text descriptions) = FAIL
[ERROR] Missing specific numerical values or units = FAIL
[ERROR] Options that aren't numerical values = FAIL''' if question_type.lower() == 'numerical' else '''[ERROR] CONCEPTUAL questions with "calculate", "compute", "solve", "determine", "find" = FAIL
[ERROR] CONCEPTUAL questions with numerical options (numbers/units) = FAIL  
[ERROR] Including specific numerical calculations = FAIL
[ERROR] Options that aren't conceptual descriptions = FAIL''' if question_type.lower() == 'conceptual' else '[ERROR] Violating format or quality requirements'}
[ERROR] Questions under required character length = FAIL
[ERROR] Missing advanced terminology = FAIL
[ERROR] Undergraduate-level simplicity = FAIL

### ULTRA-STRICT VALIDATION:
[SEARCH] Every single word will be examined for type violations
[SEARCH] Every option will be validated for purity
[SEARCH] Every verb will be checked against allowed lists
[SEARCH] Every element will be scrutinized for contamination
[SEARCH] Zero tolerance for any deviation from requirements

[EMERGENCY] WARNING: These questions will be subjected to automated validation tools that will detect ANY type mixing, ANY forbidden words, ANY violations. There is NO mercy for partial compliance.

**[EMERGENCY] CRITICAL OUTPUT FORMAT - NO EXCEPTIONS:**
You MUST respond with ONLY a valid JSON object. NO additional text before or after.

EXACT FORMAT REQUIRED:
{{
  "question": "{f'Your specific numerical calculation question with formulas and numbers' if question_type.lower() == 'numerical' else f'Your specific conceptual understanding question about principles' if question_type.lower() == 'conceptual' else 'Your well-crafted question'}",
  "options": {{
    "A": "{f'First numerical answer with units (e.g., 5.2 MHz)' if question_type.lower() == 'numerical' else f'First conceptual option describing mechanism/principle' if question_type.lower() == 'conceptual' else 'First option'}",
    "B": "{f'Second numerical answer with units (e.g., 10.4 MHz)' if question_type.lower() == 'numerical' else f'Second conceptual option describing different mechanism' if question_type.lower() == 'conceptual' else 'Second option'}", 
    "C": "{f'Third numerical answer with units (e.g., 15.6 MHz)' if question_type.lower() == 'numerical' else f'Third conceptual option describing alternative principle' if question_type.lower() == 'conceptual' else 'Third option'}",
    "D": "{f'Fourth numerical answer with units (e.g., 20.8 MHz)' if question_type.lower() == 'numerical' else f'Fourth conceptual option describing different theory' if question_type.lower() == 'conceptual' else 'Fourth option'}"
  }},
  "correct": "A",
  "explanation": "{f'Step-by-step calculation: [show work] Therefore, the answer is A.' if question_type.lower() == 'numerical' else f'Explanation of the underlying principle: [reasoning] Therefore, the answer is A.' if question_type.lower() == 'conceptual' else 'Clear explanation of why the answer is correct.'}"
}}

[EMERGENCY] VALIDATION CHECKLIST:
[OK] Exactly 4 options (A, B, C, D)
{f'[OK] All options are numbers with units' if question_type.lower() == 'numerical' else f'[OK] All options describe concepts/mechanisms' if question_type.lower() == 'conceptual' else '[OK] All options follow appropriate format'}
{f'[OK] Question requires calculation' if question_type.lower() == 'numerical' else f'[OK] Question requires understanding' if question_type.lower() == 'conceptual' else '[OK] Question meets requirements'}
[OK] One correct answer
[OK] Valid JSON syntax

[SEARCH] DOMAIN-SPECIFIC REQUIREMENTS:
- Physics topics MUST include terms like: force, energy, momentum, wave, particle, field, quantum
- Chemistry topics MUST include terms like: molecule, atom, bond, reaction, compound, solution, acid
- Mathematics topics MUST include terms like: equation, function, derivative, integral, matrix, variable, theorem
- Question MUST end with a question mark (?)
- All options must be substantive and non-empty (minimum 10 characters each)
- Expert questions must be minimum 120 characters, others minimum 80 characters

🔢 NUMERIC CONTENT REQUIREMENTS:
- For numerical questions: Include specific numbers, calculations, units, formulas
- For physics: Include values like "9.8 m/s²", "3.0 × 10⁸ m/s", specific measurements
- For chemistry: Include molarity values, pH numbers, atomic masses, concentrations
- For mathematics: Include specific numerical examples, coefficients, precise values
- All numerical options must include units where appropriate

{f'🔢 [EMERGENCY] FINAL REMINDER: This MUST be a NUMERICAL question with calculations and numbers - NOT a conceptual "which mechanism" question! [EMERGENCY] 🔢' if question_type.lower() == "numerical" else f'[BRAIN] [EMERGENCY] FINAL REMINDER: This MUST be a CONCEPTUAL question about understanding principles - NOT a "calculate" question! [EMERGENCY] [BRAIN]' if question_type.lower() == "conceptual" else ''}
{f'[EXPERT] [EMERGENCY] FINAL PHD-LEVEL DEMAND: This question must be at the cutting edge of research in {topic} - challenge a postdoc, not a student! [EMERGENCY] [EXPERT]' if difficulty.lower() == "expert" else ''}
"""
        return prompt

    def _get_topic_specific_guidance(self, topic: str, difficulty: str = "medium") -> str:
        """Get specific guidance based on the topic to improve question quality and prevent vague questions"""
        
        # Convert topic to lowercase for matching
        topic_lower = topic.lower()
        
        # Biology/Health topics - ENHANCED for reproductive system questions
        if any(word in topic_lower for word in ['biology', 'cell', 'dna', 'protein', 'anatomy', 'physiology', 'health', 'medical', 'reproduction', 'reproductive', 'sex', 'sexual', 'hormone', 'sperm', 'egg', 'ovary', 'testes', 'atom', 'atomic', 'molecule', 'molecular']):
            if difficulty.lower() in ["hard", "expert"]:
                return f"""
🧬 {topic.upper()} - {difficulty.upper()} MODE GUIDANCE:
[EMERGENCY] ABSOLUTELY BANNED QUESTIONS:
- "What is the primary function of..."
- "What is the main purpose of..."
- "What does [structure/process] do?"
- "What distinguishes [topic] knowledge?"
- Any basic definition or overview questions

[OK] {difficulty.upper()} MODE REQUIREMENTS:
- Ask about SPECIFIC mechanisms, pathways, or processes
- Focus on MOLECULAR/CELLULAR level details
- Include BIOCHEMICAL processes and reactions
- Test understanding of REGULATORY systems
- Ask about SPECIFIC structures and their precise functions
- Include QUANTITATIVE relationships and precise timing
- Focus on CAUSE-AND-EFFECT mechanisms

EXAMPLE {difficulty.upper()} QUESTIONS:
[OK] "Which phase of [specific process] is characterized by [specific event] and what triggers this?"
[OK] "During [process], at which stage do [specific cells] undergo [specific change]?"
[OK] "What role does [specific molecule] play in the regulation of [specific pathway]?"
[OK] "Which cellular mechanism prevents [specific problem] during [specific process]?"

Focus on: specific mechanisms, molecular processes, regulatory pathways, precise timing, quantitative relationships
"""
            else:
                return f"""
{topic.upper()} TOPIC GUIDANCE:
- Focus on biological/chemical processes, mechanisms, and structures
- Include questions about specific pathways, reactions, or relationships
- Ask about HOW and WHY processes work, not just WHAT they are
- Use proper scientific terminology
- Test understanding of cause-and-effect relationships
- Example: Instead of "What do [structures] do?" ask "Which process in [system] is primarily responsible for [specific function]?"
"""
        
        # Science topics (Physics, Chemistry, etc.)
        elif any(word in topic_lower for word in ['physics', 'chemistry', 'science', 'scientific', 'formula', 'equation', 'element', 'force', 'energy', 'wave', 'particle']):
            if difficulty.lower() in ["hard", "expert"]:
                return f"""
🔬 {topic.upper()} - {difficulty.upper()} MODE GUIDANCE:
- Focus on COMPLEX scientific principles and advanced applications
- Include multi-step calculations and problem-solving scenarios
- Ask about THEORETICAL frameworks and their limitations
- Test understanding of ADVANCED relationships between variables
- Include questions about EXPERIMENTAL design and analysis
- Focus on QUANTITATIVE analysis and precise measurements
- Example: "Given [specific conditions], what [quantitative relationship] determines [outcome]?"
"""
            else:
                return f"""
{topic.upper()} TOPIC GUIDANCE:
- Focus on scientific principles, laws, and applications
- Include calculation-based or problem-solving questions when appropriate
- Ask about relationships between variables and concepts
- Use specific scientific units and measurements
- Test understanding of cause-and-effect relationships
- Example: Instead of "What is [concept]?" ask "If [variable] changes by [amount], how does [other variable] respond?"
"""
        
        # Mathematics topics
        elif any(word in topic_lower for word in ['math', 'mathematics', 'algebra', 'geometry', 'calculus', 'statistics', 'probability']):
            if difficulty.lower() in ["hard", "expert"]:
                return f"""
🔢 {topic.upper()} - {difficulty.upper()} MODE GUIDANCE:
- Focus on COMPLEX problem-solving and multi-step procedures
- Include PROOF techniques and theoretical understanding
- Ask about ADVANCED applications and edge cases
- Test understanding of ABSTRACT concepts and relationships
- Include questions requiring SYNTHESIS of multiple concepts
- Example: "Which theorem/method is most efficient for solving [complex scenario]?"
"""
            else:
                return f"""
{topic.upper()} TOPIC GUIDANCE:
- Focus on problem-solving techniques and applications
- Include computational questions with specific numbers
- Ask about mathematical relationships and patterns
- Test understanding of procedures and when to apply them
- Example: "What is the result when [specific operation] is applied to [specific values]?"
"""
        
        # Default guidance for other topics
        elif difficulty.lower() in ["hard", "expert"]:
            return f"""
[TARGET] {topic.upper()} - {difficulty.upper()} MODE GUIDANCE:
- Avoid simple "What is..." questions
- Focus on complex analysis and synthesis
- Include multi-step reasoning requirements
- Ask about specific mechanisms and processes
- Test advanced understanding and application
- Use precise, technical terminology
- Example: Instead of "What is {topic}?" ask "What specific mechanism in {topic} explains [complex scenario]?"
"""
        
        return f"""
{topic.upper()} TOPIC GUIDANCE:
- Focus on specific concepts and applications within {topic}
- Avoid overly general questions
- Include practical examples and scenarios
- Test understanding of relationships and principles
- Use appropriate terminology for the {difficulty} level
"""

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from API with enhanced error handling and robust extraction"""
        try:
            # [START] ENHANCED: Use our robust JSON extraction first
            robust_result = self._parse_json_response_robust(content)
            if robust_result:
                # Convert to the format expected by the existing system
                options = robust_result.get('options', [])
                
                # Handle both dict and list options formats
                if isinstance(options, dict):
                    # Convert {"A": "...", "B": "...", ...} to ["...", "...", ...]
                    option_keys = ['A', 'B', 'C', 'D']
                    options_list = [options.get(key, '') for key in option_keys]
                    robust_result['options'] = options_list
                
                # Map 'correct' to 'correct_answer' if needed
                if 'correct' in robust_result and 'correct_answer' not in robust_result:
                    correct = robust_result['correct']
                    if isinstance(options, list) and correct in ['A', 'B', 'C', 'D']:
                        # Convert letter to actual option text
                        letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                        if correct in letter_map and letter_map[correct] < len(options):
                            robust_result['correct_answer'] = options[letter_map[correct]]
                        else:
                            robust_result['correct_answer'] = options[0] if options else ''
                    else:
                        robust_result['correct_answer'] = correct
                
                return robust_result
            
            # Fallback to original parsing logic if robust method fails
            # Clean and extract JSON content
            cleaned_content = content.strip()
            
            # Handle various JSON wrapping formats
            if '```json' in cleaned_content:
                # Extract JSON from markdown code blocks
                start = cleaned_content.find('```json') + 7
                end = cleaned_content.find('```', start)
                if end != -1:
                    cleaned_content = cleaned_content[start:end].strip()
            elif '```' in cleaned_content:
                # Extract from generic code blocks
                start = cleaned_content.find('```') + 3
                end = cleaned_content.find('```', start)
                if end != -1:
                    cleaned_content = cleaned_content[start:end].strip()
            
            # Find the first complete JSON object
            brace_count = 0
            json_start = -1
            json_end = -1
            
            for i, char in enumerate(cleaned_content):
                if char == '{':
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and json_start != -1:
                        json_end = i + 1
                        break
            
            if json_start != -1 and json_end != -1:
                json_text = cleaned_content[json_start:json_end]
                parsed = json.loads(json_text)
                
                # Enhanced validation and field mapping for different provider formats
                if isinstance(parsed, dict):
                    # Normalize the response format
                    normalized = {}
                    
                    # Extract question - handle various field names
                    question = (parsed.get('question') or 
                              parsed.get('Question') or 
                              parsed.get('prompt') or 
                              parsed.get('text') or '')
                    
                    if not question:
                        logger.error("[ERROR] No question field found in response")
                        return None
                    
                    # Extract options - handle various formats
                    options = []
                    
                    # [HOT] CRITICAL FIX: Handle options as object with A, B, C, D keys (Groq format)
                    if 'options' in parsed and isinstance(parsed['options'], dict):
                        # Groq returns: {"options": {"A": "...", "B": "...", "C": "...", "D": "..."}}
                        options_dict = parsed['options']
                        option_keys = ['A', 'B', 'C', 'D']
                        for key in option_keys:
                            if key in options_dict:
                                options.append(options_dict[key])
                        logger.info(f"[SEARCH] DEBUG: Extracted {len(options)} options from object format")
                        
                    elif 'options' in parsed and isinstance(parsed['options'], list):
                        # Array format: ["option1", "option2", "option3", "option4"]
                        options = parsed['options']
                        logger.info(f"[SEARCH] DEBUG: Extracted {len(options)} options from array format")
                        
                    elif 'choices' in parsed and isinstance(parsed['choices'], list):
                        options = parsed['choices']
                        logger.info(f"[SEARCH] DEBUG: Extracted {len(options)} options from choices array")
                        
                    elif 'answers' in parsed and isinstance(parsed['answers'], list):
                        options = parsed['answers']
                        logger.info(f"[SEARCH] DEBUG: Extracted {len(options)} options from answers array")
                        
                    else:
                        # Fallback: Try to extract from top-level A, B, C, D keys
                        option_keys = ['A', 'B', 'C', 'D']
                        for key in option_keys:
                            if key in parsed:
                                options.append(parsed[key])
                        
                        # Alternative format: option_a, option_b, etc.
                        if not options:
                            for i, letter in enumerate(['a', 'b', 'c', 'd']):
                                option_key = f'option_{letter}'
                                if option_key in parsed:
                                    options.append(parsed[option_key])
                        
                        if options:
                            logger.info(f"[SEARCH] DEBUG: Extracted {len(options)} options from fallback method")
                    
                    if len(options) < 2:
                        logger.error(f"[ERROR] Insufficient options found: {len(options)}")
                        return None
                    
                    # Extract correct answer
                    correct_answer = (parsed.get('correct_answer') or 
                                    parsed.get('correct') or 
                                    parsed.get('answer') or 
                                    parsed.get('solution'))
                    
                    # If correct_answer is an index or letter, convert to actual text
                    if isinstance(correct_answer, int) and 0 <= correct_answer < len(options):
                        correct_answer = options[correct_answer]
                    elif isinstance(correct_answer, str) and len(correct_answer) == 1:
                        # Handle A, B, C, D format
                        letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'a': 0, 'b': 1, 'c': 2, 'd': 3}
                        if correct_answer in letter_map:
                            idx = letter_map[correct_answer]
                            if idx < len(options):
                                correct_answer = options[idx]
                    
                    # Validate correct answer exists in options
                    if not correct_answer or correct_answer not in options:
                        # Try to find a reasonable match
                        if options:
                            correct_answer = options[0]  # Default to first option
                            logger.warning(f"[WARNING] Correct answer not found, defaulting to first option")
                    
                    # Extract explanation
                    explanation = (parsed.get('explanation') or 
                                 parsed.get('rationale') or 
                                 parsed.get('reasoning') or 
                                 'No explanation provided.')
                    
                    # Build normalized response
                    normalized = {
                        'question': question,
                        'options': options,
                        'correct_answer': correct_answer,
                        'explanation': explanation,
                        'correct_index': options.index(correct_answer) if correct_answer in options else 0
                    }
                    
                    logger.info(f"[OK] Successfully parsed JSON response with {len(options)} options")
                    return normalized
                else:
                    logger.error(f"[ERROR] Parsed JSON is not a dictionary: {type(parsed)}")
                    return None
            else:
                logger.error("[ERROR] Could not find valid JSON object in response")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"[ERROR] JSON parsing failed: {e}")
            logger.error(f"Raw content (first 200 chars): {content[:200]}...")
            return None
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error parsing JSON: {e}")
            return None

    def _validate_numerical_question(self, question_data: Dict[str, Any], question_type: str) -> bool:
        """Validate that a question is truly numerical if question_type is 'numerical'"""
        try:
            if question_type.lower() != "numerical":
                return True  # No validation needed for non-numerical questions
            
            question = question_data.get('question', '').lower()
            options = question_data.get('options', [])
            
            # Check for conceptual/comparison patterns that should be rejected
            conceptual_patterns = [
                'how does this compare',
                'what does this tell us',
                'how does this relate',
                'which mechanism',
                'what governs',
                'which principle',
                'what principle',
                'why does',
                'how does this explain',
                'what explains',
                'considering the',  # Often leads to conceptual questions
                'given that the',   # Sometimes leads to conceptual
                'how this compares',
                'this compares to',
                'compared to',
                'in comparison',
                'relationship between',
                'what does this suggest',
                'this suggests that',
                'what can we conclude',
                'we can conclude',
            ]
            
            # Check if question contains any conceptual patterns
            for pattern in conceptual_patterns:
                if pattern in question:
                    logger.warning(f"[FORBIDDEN] Numerical question validation FAILED: Contains conceptual pattern '{pattern}'")
                    logger.warning(f"[ERROR] Question: '{question[:100]}...'")
                    return False
            
            # Check for numerical calculation starters (required for numerical questions)
            numerical_starters = [
                'calculate',
                'determine',
                'find',
                'compute',
                'what is the value',
                'what is the magnitude',
                'what is the energy',
                'what is the wavelength',
                'what is the frequency',
                'what is the mass',
                'what is the charge',
                'what is the number',
                'how many',
                'at what',
            ]
            
            has_numerical_starter = any(starter in question for starter in numerical_starters)
            if not has_numerical_starter:
                logger.warning(f"[FORBIDDEN] Numerical question validation FAILED: No numerical calculation starter found")
                logger.warning(f"[ERROR] Question: '{question[:100]}...'")
                return False
            
            # Check that all options are numerical (contain numbers and units)
            numerical_options = 0
            for option in options:
                option_str = str(option).lower()
                # Check if option contains numbers and possibly units
                has_number = any(char.isdigit() for char in option_str)
                has_unit_indicators = any(unit in option_str for unit in [
                    'ev', 'mev', 'kev', 'gev', 'j', 'joule', 'nm', 'pm', 'fm', 'cm', 'm', 'mm', 'km',
                    'hz', 'khz', 'mhz', 'ghz', 'thz', 's', 'ms', 'ns', 'ps', 'fs', 'kg', 'g', 'mg',
                    'c', 'k', 'kelvin', 'celsius', 'tesla', 't', 'gauss', 'amp', 'ampere', 'volt', 'v',
                    'ohm', 'watt', 'w', 'pascal', 'pa', 'bar', 'atm', 'mol', 'rad', 'deg', 'degree'
                ])
                
                if has_number:
                    numerical_options += 1
            
            # Require at least 3/4 options to be numerical
            if numerical_options < 3:
                logger.warning(f"[FORBIDDEN] Numerical question validation FAILED: Only {numerical_options}/4 options are numerical")
                logger.warning(f"[ERROR] Options: {options}")
                return False
            
            logger.info(f"[OK] Numerical question validation PASSED: Pure calculation question detected")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error validating numerical question: {e}")
            return True  # Default to accepting the question if validation fails

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "available_providers": self.available_providers,
            "generation_stats": self.generation_stats,
            "total_providers": len(self.providers),
            "status": "ready" if self.is_available() else "no_api_keys"
        }
        
        # Add OpenRouter free model information if available
        if 'openrouter' in self.available_providers:
            openrouter_config = self.providers['openrouter']
            stats["openrouter_free_models"] = {
                "available_models": openrouter_config.get('free_models', []),
                "default_model": openrouter_config.get('model'),
                "total_free_models": len(openrouter_config.get('free_models', []))
            }
        
        return stats

    async def cleanup(self):
        """Clean up resources"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        logger.info("[CLEAN] Online MCQ generator cleaned up")

    def __del__(self):
        """Cleanup on destruction to prevent warnings"""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.session.close())
                else:
                    loop.run_until_complete(self.session.close())
            except Exception:
                pass  # Ignore cleanup errors during destruction 

    # [HOT] NEW: Centralized robust API request method with exponential backoff
    async def _make_api_request_with_rate_limit_handling(self, url: str, headers: Dict[str, str],
                                                      payload: Dict[str, Any], provider: str = "unknown") -> Dict[str, Any]:
        """[START] Enhanced API request with intelligent rate limit handling for free tier models"""

        # [START] CRITICAL FIX: Enforce rate limiting before making any request
        await self._enforce_rate_limit()

        # More generous timeouts for big models that need more time to process
        timeout = aiohttp.ClientTimeout(
            total=20.0,      # 20 seconds total for large models
            connect=3.0,     # 3 seconds to connect
            sock_read=15.0   # 15 seconds to read response from big models
        )
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 429:  # Rate limited - let caller handle this
                        logger.warning(f"[FORBIDDEN] {provider} rate limited (429)")
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
                    elif response.status == 503:  # Service unavailable - model overloaded
                        logger.warning(f"[WARNING] {provider} service unavailable (503) - model overloaded")
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
                    
                    response.raise_for_status()
                    return await response.json()
                    
        except asyncio.TimeoutError:
            logger.error(f"[ERROR] {provider} API timeout (20s limit for large models)")
            raise
        except Exception as e:
            logger.error(f"[ERROR] {provider} API request failed: {e}")
            raise

    async def _make_api_request(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], 
                               provider: str = "unknown") -> Dict[str, Any]:
        """[HOT] OPTIMIZED: Make API request with exponential backoff and strict timeouts"""
        
        # 🛡️ CRITICAL FIX: Enhanced timeout configuration for network reliability
        timeout = aiohttp.ClientTimeout(
            total=12.0,     # Increased to 12 seconds for better network reliability
            connect=3.0,    # 3 seconds to connect (handles slow networks)
            sock_read=8.0   # 8 seconds to read response (handles large responses)
        )
        
        if backoff:
            @backoff.on_exception(
                backoff.expo,
                (aiohttp.ClientError, asyncio.TimeoutError, aiohttp.ClientResponseError),
                max_tries=2,      # CRITICAL FIX: Reduced from 3 to 2 attempts
                max_time=10,      # CRITICAL FIX: Maximum 10 seconds total including retries
                base=1,           # CRITICAL FIX: Faster backoff
                factor=1
            )
            async def _make_request_with_backoff():
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=payload) as response:
                        # [START] BUG FIX 18: Classify errors by HTTP status code
                        if response.status == 401:  # Unauthorized
                            logger.error(f"[ERROR] {provider} API key is invalid or unauthorized")
                            raise InvalidApiKeyError(f"Invalid API key for {provider}")
                        elif response.status == 403:  # Forbidden
                            logger.error(f"[ERROR] {provider} API access forbidden (quota exceeded or invalid key)")
                            raise QuotaExceededError(f"API access forbidden for {provider}")
                        elif response.status == 429:  # Rate limited
                            logger.warning(f"[WARNING] {provider} rate limited, backing off...")
                            raise RateLimitError(f"Rate limit exceeded for {provider}")
                        elif response.status >= 500:  # Server errors
                            logger.warning(f"[WARNING] {provider} server error: {response.status}")
                            raise ServerError(f"Server error {response.status} for {provider}")

                        response.raise_for_status()
                        return await response.json()
            
            try:
                return await _make_request_with_backoff()
            except Exception as e:
                logger.error(f"[ERROR] {provider} API failed after backoff retries: {e}")
                raise
        else:
            # CRITICAL FIX: Fallback without backoff library - still use strict timeouts
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=payload) as response:
                        # [START] BUG FIX 18: Classify errors by HTTP status code (fallback version)
                        if response.status == 401:  # Unauthorized
                            logger.error(f"[ERROR] {provider} API key is invalid or unauthorized")
                            raise InvalidApiKeyError(f"Invalid API key for {provider}")
                        elif response.status == 403:  # Forbidden
                            logger.error(f"[ERROR] {provider} API access forbidden (quota exceeded or invalid key)")
                            raise QuotaExceededError(f"API access forbidden for {provider}")
                        elif response.status == 429:  # Rate limited
                            logger.warning(f"[WARNING] {provider} rate limited")
                            raise RateLimitError(f"Rate limit exceeded for {provider}")
                        elif response.status >= 500:  # Server errors
                            logger.warning(f"[WARNING] {provider} server error: {response.status}")
                            raise ServerError(f"Server error {response.status} for {provider}")

                        response.raise_for_status()
                        return await response.json()
            except asyncio.TimeoutError:
                logger.error(f"[ERROR] {provider} API timeout (12s limit) - network may be slow")
                raise NetworkTimeoutError(f"Network timeout for {provider} - check your internet connection")
            except aiohttp.ClientConnectorError as e:
                logger.error(f"[ERROR] {provider} connection failed: {e}")
                raise NetworkConnectionError(f"Cannot connect to {provider} - check your internet connection")
            except aiohttp.ClientOSError as e:
                logger.error(f"[ERROR] {provider} network OS error: {e}")
                raise NetworkConnectionError(f"Network OS error for {provider}: {e}")
            except aiohttp.ClientPayloadError as e:
                logger.error(f"[ERROR] {provider} payload error: {e}")
                raise NetworkDataError(f"Data transmission error for {provider}: {e}")
            except (InvalidApiKeyError, QuotaExceededError, RateLimitError, ServerError):
                # Re-raise our custom exceptions
                raise
            except Exception as e:
                logger.error(f"[ERROR] {provider} API request failed: {e}")
                # Classify unknown errors
                if "connection" in str(e).lower() or "network" in str(e).lower():
                    raise NetworkConnectionError(f"Network error for {provider}: {e}")
                elif "timeout" in str(e).lower():
                    raise NetworkTimeoutError(f"Timeout error for {provider}: {e}")
                else:
                    raise