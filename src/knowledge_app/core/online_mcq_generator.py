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
                    # [START] OpenRouter Free Models - Correct model IDs from official list
                    'qwen/qwq-32b-preview',       # 32B reasoning specialist with thinking tokens
                    'meta-llama/llama-3.1-8b-instruct',
                    'meta-llama/llama-3.1-70b-instruct',
                    'meta-llama/llama-3.2-1b-instruct',
                    'meta-llama/llama-3.2-3b-instruct',
                    'meta-llama/llama-3.2-11b-vision-instruct',
                    'meta-llama/llama-3.2-90b-vision-instruct',
                    'microsoft/phi-3-mini-128k-instruct',
                    'microsoft/phi-3-medium-128k-instruct',
                    'mistralai/mistral-7b-instruct',
                    'huggingface/zephyr-7b-beta',
                    'openchat/openchat-7b',
                    'undi95/toppy-m-7b',
                    'gryphe/mythomist-7b',
                    'nousresearch/nous-capybara-7b',
                    'teknium/openhermes-2-mistral-7b',
                    'togethercomputer/redpajama-incite-7b-chat',
                    'psyfighter2/psyfighter-13b-2',
                    'koboldai/psyfighter-13b-2',
                    'intel/neural-chat-7b-v3-1',
                    'pygmalionai/mythalion-13b',
                    'jondurbin/airoboros-l2-70b-gpt4-1.4.1',
                    'austism/chronos-hermes-13b'
                ],
                'model': 'qwen/qwq-32b-preview',  # Default to QwQ for reasoning capabilities
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
            logger.info(f"[SEARCH] CHECKING PROVIDER: {provider_name.upper()}")
            
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
            else:
                logger.warning(f"[ERROR] {provider_name.upper()} NOT FOUND in providers config!")
        
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
        """🔧 BUG FIX #25: Use unified config manager to eliminate redundant file I/O"""
        logger.info("[SEARCH] LOADING USER API KEYS via unified config manager...")
        
        try:
            # 🔧 FIX: Use unified config manager instead of direct file access
            from .unified_config_manager import get_unified_config_manager
            
            config_manager = get_unified_config_manager()
            logger.info("[SEARCH] Using unified config manager - no redundant file I/O")
            
            # Get API keys from unified config
            api_keys = config_manager.get_api_keys()
            logger.info(f"[SEARCH] Retrieved {len(api_keys)} API keys from unified config")
            
            # Get provider enabled states from unified config
            self.providers_enabled = config_manager.get_provider_states()
            logger.info(f"[SEARCH] Retrieved provider states from unified config")
            
            # Log enabled/disabled states
            enabled_count = sum(1 for enabled in self.providers_enabled.values() if enabled)
            logger.info(f"[CONFIG] PROVIDER STATES: {enabled_count}/{len(self.providers_enabled)} enabled")
            for provider, enabled in self.providers_enabled.items():
                status = "ENABLED" if enabled else "DISABLED"
                logger.info(f"   • {provider}: {status}")
            
            if api_keys:
                logger.info(f"🔑 FOUND {len(api_keys)} VALID API KEYS: {list(api_keys.keys())}")
            else:
                logger.warning("[WARNING] NO VALID API KEYS FOUND IN UNIFIED CONFIG")
            
            return api_keys
            
        except Exception as e:
            logger.error(f"[ERROR] FAILED TO LOAD USER API KEYS via unified config: {e}")
            import traceback
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
            # [INFO] Important: Propagate error to UI for user awareness
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
            # [INFO] Important: Propagate error to UI for user awareness
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
        """Call OpenRouter API with dynamic model configuration and intelligent rate limiting"""
        headers = {k: v.format(api_key=config['api_key']) for k, v in config['headers'].items()}
        headers['HTTP-Referer'] = 'https://knowledge-app.local'
        headers['X-Title'] = 'Knowledge App'
        
        # 🔧 DYNAMIC MODEL LOADING: Use configurable model list instead of hardcoded
        try:
            from .openrouter_config import get_openrouter_models, get_preferred_openrouter_model
            
            # Get models from dynamic configuration
            free_models = get_openrouter_models(use_free_only=True)
            
            # If no models configured, fallback to config or default
            if not free_models:
                free_models = config.get('free_models', [config.get('model', 'qwen/qwq-32b-preview')])
                logger.warning("⚠️ No OpenRouter models in config, using fallback list")
            else:
                logger.info(f"✅ Using {len(free_models)} OpenRouter models from configuration")
                
        except ImportError:
            # Fallback to hardcoded list if config module not available
            free_models = config.get('free_models', [config['model']])
            logger.info("ℹ️ Using hardcoded OpenRouter model list (config module not available)")
        
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

    # [START] UNIFIED PROMPT SYSTEM: Use centralized prompt generation
    def _create_optimized_prompt(self, topic: str, context: str, question_index: int, 
                                 difficulty: str = "medium", question_type: str = "mixed") -> str:
        """
        � REVOLUTIONARY PROMPT SYSTEM: Use unified trust-based prompt generation
        
        This now delegates to the unified prompt system to eliminate code duplication
        and leverage the trust-based AI collaboration approach instead of constraints.
        """
        try:
            from .unified_prompt_builder import UnifiedPromptBuilder
            
            # Use the balanced prompt system
            builder = UnifiedPromptBuilder()
            prompt = builder.build_unified_prompt(
                topic=topic,
                difficulty=difficulty,
                question_type=question_type,
                context=context
            )
            
            logger.debug(f"Generated balanced online prompt for {topic} ({difficulty}, {question_type})")
            return prompt
            
        except ImportError as e:
            logger.error(f"Failed to import prompt system: {e}")
            # Fallback to a simple prompt if unified system not available
            return self._create_gentle_fallback_prompt(topic, context, difficulty, question_type)
    
    def _create_gentle_fallback_prompt(self, topic: str, context: str, difficulty: str, question_type: str) -> str:
        """Simple fallback prompt that avoids harsh constraints"""
        context_section = f"Context: {context}\n\n" if context and context.strip() else ""
        
        prompt = f"""You are an expert educator creating a quality multiple choice question.

Create exactly 1 multiple choice question about: {topic}

{context_section}Difficulty: {difficulty}
Question Type: {question_type}

Return ONLY a JSON object with this structure:
{{
    "question": "Your question here?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_answer": "A",
    "explanation": "Detailed explanation"
}}

Requirements:
- Question must end with "?"
- All options must be substantial (20+ characters)
- Use technical terminology appropriately
- Ensure one clearly correct answer

Generate the question now:"""
        
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
- Use appropriate terminology for the {difficulty} level
"""

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from API using unified parser"""
        try:
            # ARCHITECTURAL FIX: Use unified JSON parser instead of scattered implementations
            from .json_parser_unified import parse_mcq_response
            return parse_mcq_response(content)
        except Exception as e:
            logger.error(f"[ERROR] Unified JSON parsing failed: {e}")
            
            # Fallback to original parsing logic if unified method fails
            try:
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
                        
                        # Handle options as object with A, B, C, D keys (Groq format)
                        if 'options' in parsed and isinstance(parsed['options'], dict):
                            options_dict = parsed['options']
                            option_keys = ['A', 'B', 'C', 'D']
                            for key in option_keys:
                                if key in options_dict:
                                    options.append(options_dict[key])
                            logger.info(f"[SEARCH] DEBUG: Extracted {len(options)} options from object format")
                            
                        elif 'options' in parsed and isinstance(parsed['options'], list):
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
            
            # Check if question contains conceptual patterns (for quality guidance)
            for pattern in conceptual_patterns:
                if pattern in question:
                    logger.info(f"[QUALITY] Numerical question enhancement: Question contains conceptual pattern '{pattern}'")
                    logger.info(f"[GUIDANCE] Question: '{question[:100]}...' could benefit from more quantitative focus")
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
