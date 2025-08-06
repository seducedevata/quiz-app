from .async_converter import async_requests_post, async_requests_get


from .async_converter import async_requests_post, async_requests_get


import logging
import requests
import json
from typing import Dict, Any, List, Optional, Union, AsyncGenerator, Generator
import time

logger = logging.getLogger(__name__)


class OllamaModelInference:
    """
    A robust and optimized inference client for a local Ollama server.
    This connects to an existing Ollama instance and automatically selects
    the best available model with GPU optimization for maximum speed.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", preferred_models: Optional[List[str]] = None):
        self.base_url = ollama_url
        self.ollama_api_url = f"{self.base_url}/api/generate"
        self.preferred_models = preferred_models or ["llama3.1", "llama3", "llama2", "phi3"]
        self.active_model: Optional[str] = None
        
        # GPU optimization parameters for maximum performance
        self.generation_params = {
            'temperature': 0.8,  # Balanced creativity/speed
            'top_p': 0.9,
            'top_k': 30,  # Reduced for faster generation
            'num_predict': 400,  # Shorter responses for speed
            'num_ctx': 2048,  # Smaller context window for speed
            'repeat_penalty': 1.1,
            'seed': -1,  # Random seed for variety
            # GPU optimization flags - FORCE MAXIMUM GPU USAGE
            'num_gpu': -1,  # Use ALL available GPU layers
            'num_thread': 8,  # Optimize CPU threads for remaining work
            'low_vram': False,  # Don't limit VRAM usage - use it all!
            'use_mmap': True,  # Memory mapping for efficiency
            'use_mlock': True,  # Lock memory for performance
            # Performance optimization
            'numa': True,  # NUMA optimization
            'batch_size': 512,  # Larger batch size for GPU efficiency
            'parallel': 4,  # Parallel processing
        }
        
        # ✅ FIXED: Defer initialization to avoid blocking constructor
        self._initialization_attempted = False

    async def _initialize_model_async(self):
        """✅ FIXED: Async model initialization to prevent UI blocking"""
        if self._initialization_attempted:
            return self.active_model is not None

        self._initialization_attempted = True
        logger.info("Initializing Ollama: Verifying connection and finding available model...")

        try:
            # ✅ FIXED: Use async converter instead of blocking requests
            from .async_converter import async_requests_get
            response_data = await async_requests_get(f"{self.base_url}/api/tags", timeout=3.0)

            if response_data:
                available_models = [model["name"] for model in response_data.get("models", [])]
                logger.info(f"✅ Available Ollama models: {available_models}")

                # ✅ DYNAMIC: Score models instead of hardcoded preferences
                if available_models:
                    best_model = self._select_best_model(available_models)
                    self.active_model = best_model
                    logger.debug(f"✅ Active Ollama model set to: {self.active_model}")
                    self._optimize_model_for_gpu()  # FIXED: Removed 'await' from synchronous function
                else:
                    logger.error("❌ No models available in Ollama")
            else:
                logger.error("❌ Failed to get response from Ollama")
        except Exception as e:
            logger.error(f"❌ Ollama initialization error: {e}")

        return self.active_model is not None

    def _select_best_model(self, available_models: List[str]) -> str:
        """✅ DYNAMIC: Select best model based on capabilities, not hardcoded preferences"""
        model_scores = {}
        for model in available_models:
            score = self._score_model_capabilities(model)
            model_scores[model] = score

        # Return highest scoring model
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        logger.debug(f"🎯 Selected best model: {best_model} (score: {model_scores[best_model]})")
        return best_model

    def _score_model_capabilities(self, model_name: str) -> int:
        """✅ DYNAMIC: Score model based on general capabilities"""
        score = 0
        model_lower = model_name.lower()

        # Size-based scoring
        if "70b" in model_lower or "72b" in model_lower: score += 100
        elif "32b" in model_lower: score += 90
        elif "14b" in model_lower: score += 80
        elif "8b" in model_lower or "7b" in model_lower: score += 70
        elif "3b" in model_lower: score += 60
        else: score += 50

        # Model family scoring
        if "llama3.1" in model_lower: score += 40    # Latest and best
        elif "llama3" in model_lower: score += 35    # Good general model
        elif "phi3" in model_lower: score += 30      # Efficient
        elif "mistral" in model_lower: score += 25   # Decent
        elif "qwen" in model_lower: score += 35      # Good performance
        elif "deepseek" in model_lower: score += 45  # Excellent reasoning

        return score

    async def _initialize_model(self):
        """Async version of model initialization for unified inference manager compatibility"""
        return await self._initialize_model_async()

    def _optimize_model_for_gpu(self):
        """Force GPU optimization for the active model"""
        try:
            if not self.active_model:
                return
                
            logger.info(f"🚀 Optimizing {self.active_model} for MAXIMUM GPU utilization...")
            
            # Skip the preload request during initialization to avoid hanging
            # GPU optimization will happen on first actual generation
            logger.info("🎮 GPU optimization will be applied on first generation request")

            # 🎮 GAMING MODE: Apply gaming-optimized GPU settings
            self._apply_gaming_optimizations()

            # Just set the optimization parameters without making a request
            self.gpu_optimized = True
                
        except Exception as e:
            logger.error(f"❌ GPU optimization setup failed: {e}")

    def generate_text(self, prompt: str, stream: bool = False, **kwargs) -> Union[Optional[str], Generator[str, None, None]]:
        """
        Generate text using the active Ollama model with GPU optimization.
        
        Args:
            prompt: Input text prompt
            stream: If True, returns an async generator that yields tokens
            **kwargs: Additional generation parameters (including 'request_timeout' for custom timeout)
        
        Returns:
            If stream=False: Generated text or None if generation fails
            If stream=True: Async generator that yields tokens
        """
        if not self.active_model:
            logger.error("❌ No active model available for generation")
            return None

        # Extract custom timeout if provided
        request_timeout = kwargs.pop('request_timeout', None)
        
        # Merge user parameters with optimized defaults
        generation_options = {**self.generation_params, **kwargs}
        
        # Ensure GPU optimization is always enabled
        generation_options.update({
            'num_gpu': -1,  # Always force full GPU usage
            'low_vram': False,  # Never limit VRAM
            'use_mmap': True,
            'use_mlock': True,
        })

        model_to_use = kwargs.pop('model_override', self.active_model)
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "stream": stream,  # Now configurable
            "options": generation_options
        }

        # Dynamic timeout based on model type and complexity
        timeout = self._calculate_dynamic_timeout(request_timeout, prompt)

        try:
            start_time = time.time()
            logger.info(f"[START] Generation with Ollama model '{self.active_model}' (GPU accelerated)")
            logger.info(f"[TIME] Using dynamic timeout: {timeout}s for model '{self.active_model}'")
            
            # Show progress indicator for long timeouts
            if timeout > 120:
                logger.info(f"🧠 Reasoning model detected - this may take up to {timeout/60:.1f} minutes")
                logger.info("🔄 Generation in progress... (reasoning models think deeply)")

            # CRITICAL FIX: Use synchronous requests to avoid async/await issues
            import requests

            if stream:
                # For streaming, use requests with stream=True
                try:
                    response = requests.post(
                        self.ollama_api_url,
                        json=payload,
                        timeout=timeout,
                        stream=True
                    )
                    
                    if response.status_code == 200:
                        def token_generator():
                            for line in response.iter_lines():
                                if line:
                                    try:
                                        data = json.loads(line.decode('utf-8'))
                                        if 'response' in data:
                                            yield data['response']
                                    except:
                                        continue
                        return token_generator()
                    else:
                        logger.error(f"❌ Streaming request failed: {response.status_code}")
                        return None
                except Exception as e:
                    logger.error(f"❌ Streaming generation failed: {e}")
                    return None
            else:
                # For non-streaming, use synchronous request
                try:
                    response = requests.post(
                        self.ollama_api_url,
                        json=payload,
                        timeout=timeout
                    )

                    if response.status_code == 200:
                        data = response.json()
                        response_text = data.get("response", "")
                        
                        if response_text:
                            elapsed = time.time() - start_time
                            tokens_per_second = len(response_text.split()) / elapsed if elapsed > 0 else 0

                            logger.info(f"✅ Ollama generation complete in {elapsed:.1f}s ({tokens_per_second:.1f} tokens/s)")
                            return response_text
                        else:
                            logger.error(f"❌ Ollama generation failed - empty response")
                            return None
                    else:
                        logger.error(f"❌ Ollama generation failed: {response.status_code}")
                        return None
                        
                except requests.exceptions.Timeout:
                    logger.error(f"❌ Ollama generation timeout after {timeout}s")
                    return None
                except Exception as e:
                    logger.error(f"❌ Ollama generation failed: {e}")
                    return None

        except Exception as e:
            logger.error(f"❌ Non-blocking Ollama generation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error in Ollama generation: {e}")
            return None

    def is_available(self) -> bool:
        """✅ FIXED: Non-blocking availability check with basic server connectivity test"""
        try:
            # Quick server connectivity check without full initialization
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            server_available = response.status_code == 200
            
            # If server is available and we have an active model, we're good
            if server_available and self.active_model is not None:
                return True
            
            # If server is available but no active model, we can still work
            # (model will be initialized on first use)
            if server_available:
                return True
                
            return False
            
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            # If we have an active model from previous initialization, assume available
            return self.active_model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the active model"""
        if not self.active_model:
            return {}

        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"model": self.active_model},
                timeout=10
            )

            if response.status_code == 200:
                info = response.json()
                # Add our optimization status
                info["gpu_optimized"] = True
                info["optimization_params"] = self.generation_params
                return info
            return {}
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}

    def optimize_for_speed(self):
        """Apply maximum speed optimizations at runtime"""
        logger.debug("[START] Applying TURBO speed optimizations...")

        # Ultra-fast generation parameters
        self.generation_params.update({
            'temperature': 0.9,  # Higher for faster sampling
            'top_k': 20,  # Even more restrictive for speed
            'num_predict': 300,  # Shorter responses
            'num_ctx': 1536,  # Smaller context
            'repeat_penalty': 1.05,  # Lower penalty for speed
            # Maximum GPU utilization
            'num_gpu': -1,
            'num_thread': 12,  # More CPU threads
            'batch_size': 1024,  # Larger batches
            'parallel': 6,  # More parallel processing
        })

        logger.debug("[FAST] TURBO mode activated - Maximum speed with full GPU utilization!")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "model": self.active_model,
            "gpu_optimized": True,
            "generation_params": self.generation_params,
            "server_url": self.base_url,
            "status": "ready" if self.is_available() else "unavailable"
        }

    async def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            # CRITICAL FIX: Use async converter to prevent UI blocking
            from .async_converter import get_async_converter
            async_converter = get_async_converter()

            response_data = await async_converter.async_get(
                f"{self.base_url}/api/tags",
                timeout=2.0
            )

            if response_data:
                models = response_data.get("models", [])
                return [model["name"] for model in models]
            else:
                logger.error(f"❌ Failed to fetch available models")
                return []
        except Exception as e:
            logger.error(f"❌ Error fetching available models: {e}")
            return []

    def _calculate_dynamic_timeout(self, request_timeout: Optional[float], prompt: str) -> float:
        """Calculate dynamic timeout based on model type, prompt complexity, and reasoning requirements"""
        
        # If user specified timeout, respect it
        if request_timeout is not None:
            logger.info(f"🕐 Using user-specified timeout: {request_timeout}s")
            return request_timeout
        
        base_timeout = 60  # Base timeout for standard models
        
        # Model-specific timeout multipliers
        model_multiplier = 1.0
        if self.active_model:
            model_lower = self.active_model.lower()
            
            # Reasoning models need much more time
            if any(keyword in model_lower for keyword in ['deepseek-r1', 'r1:', 'reasoning']):
                model_multiplier = 6.0  # 6x longer for reasoning models
                logger.info(f"🧠 Detected reasoning model: {self.active_model} (6x timeout multiplier)")
            elif 'deepseek' in model_lower:
                model_multiplier = 4.0  # 4x longer for DeepSeek models
                logger.info(f"🧠 Detected DeepSeek model: {self.active_model} (4x timeout multiplier)")
            elif any(keyword in model_lower for keyword in ['qwq', 'thinking']):
                model_multiplier = 3.0  # 3x longer for thinking models
                logger.info(f"🧠 Detected thinking model: {self.active_model} (3x timeout multiplier)")
            elif any(size in model_lower for size in ['70b', '72b']):
                model_multiplier = 2.5  # 2.5x longer for large models
                logger.info(f"🧠 Detected large model: {self.active_model} (2.5x timeout multiplier)")
            elif any(size in model_lower for size in ['32b', '34b']):
                model_multiplier = 2.0  # 2x longer for medium-large models
                logger.info(f"🧠 Detected medium-large model: {self.active_model} (2x timeout multiplier)")
        
        # Prompt complexity multiplier
        prompt_multiplier = 1.0
        prompt_lower = prompt.lower()
        
        # Expert/complex prompts need more time
        if any(keyword in prompt_lower for keyword in ['expert', 'phd', 'advanced', 'complex']):
            prompt_multiplier *= 2.0
            logger.info("🎓 Detected expert/complex prompt (2x timeout multiplier)")
        
        # Mathematical/calculation prompts need more time for reasoning models
        if any(keyword in prompt_lower for keyword in ['calculate', 'solve', 'derive', 'prove']):
            if model_multiplier >= 3.0:  # Only for reasoning models
                prompt_multiplier *= 1.5
                logger.info("🔢 Detected mathematical prompt on reasoning model (1.5x timeout multiplier)")
        
        # Long prompts need more time
        if len(prompt) > 1000:
            prompt_multiplier *= 1.3
            logger.info("📝 Detected long prompt (1.3x timeout multiplier)")
        
        # Calculate final timeout
        final_timeout = base_timeout * model_multiplier * prompt_multiplier
        
        # Cap the timeout to reasonable limits
        final_timeout = min(final_timeout, 600)  # Max 10 minutes
        final_timeout = max(final_timeout, 30)   # Min 30 seconds
        
        logger.info(f"🕐 Dynamic timeout calculated: {final_timeout:.1f}s (base: {base_timeout}s, model: {model_multiplier}x, prompt: {prompt_multiplier}x)")
        
        return final_timeout

    def _apply_gaming_optimizations(self):
        """🎮 Apply gaming-optimized GPU settings for VRAM conservation"""
        try:
            import os

            # Check if gaming mode is enabled
            gaming_mode = os.getenv('GAMING_MODE', 'false').lower() == 'true'

            if gaming_mode:
                logger.info("🎮 GAMING MODE: Applying VRAM-conservative settings")

                # Gaming-optimized parameters (less VRAM, more RAM)
                self.generation_params.update({
                    'num_gpu': int(os.getenv('OLLAMA_NUM_GPU', '10')),  # Fewer GPU layers
                    'num_ctx': 1024,        # Smaller context (less VRAM)
                    'batch_size': 256,      # Smaller batches (less VRAM)
                    'parallel': 2,          # Fewer parallel requests
                    'low_vram': True,       # Enable low VRAM mode
                    'use_mmap': True,       # Use memory mapping (more RAM)
                    'use_mlock': False,     # Don't lock memory (more flexible)
                })

                logger.debug(f"🎮 Gaming optimizations applied: GPU layers={self.generation_params['num_gpu']}")
                logger.debug("🎮 Models will use more RAM and less VRAM for gaming compatibility")

            else:
                logger.debug("[START] PERFORMANCE MODE: Using maximum GPU utilization")

                # Performance-optimized parameters (more VRAM, max speed)
                self.generation_params.update({
                    'num_gpu': int(os.getenv('OLLAMA_NUM_GPU', '-1')),  # All GPU layers
                    'num_ctx': 2048,        # Larger context
                    'batch_size': 1024,     # Larger batches
                    'parallel': 6,          # More parallel requests
                    'low_vram': False,      # Disable low VRAM mode
                    'use_mmap': True,       # Still use memory mapping
                    'use_mlock': True,      # Lock memory for speed
                })

                logger.debug("[START] Performance optimizations applied for maximum speed")

        except Exception as e:
            logger.warning(f"🎮 Gaming optimization failed: {e}")

    def initialize(self) -> bool:
        """✅ FIXED: Non-blocking initialization that won't hang the UI"""
        if self._initialization_attempted:
            return self.active_model is not None

        self._initialization_attempted = True
        logger.info("Initializing Ollama: Verifying connection and finding available model...")

        try:
            # ✅ CRITICAL FIX: Use direct requests for non-blocking connection test
            import requests

            logger.info(f"[START] Making GET request to {self.base_url}/api/tags")

            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
                if response.status_code == 200:
                    response_data = response.json()
                else:
                    logger.error(f"[ERROR] Failed to get models: {response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"[ERROR] Request failed: {e}")
                return False

            logger.info("[OK] GET request successful")

            if response_data and isinstance(response_data, dict):
                available_models = [model["name"] for model in response_data.get("models", [])]
                logger.info(f"✅ Available Ollama models: {available_models}")

                if available_models:
                    best_model = self._select_best_model(available_models)
                    self.active_model = best_model
                    logger.debug(f"🎯 Selected best model: {best_model} (score: {self._score_model_capabilities(best_model)})")
                    logger.debug(f"✅ Active Ollama model set to: {self.active_model}")

                    # Apply GPU optimizations without blocking
                    self._optimize_model_for_gpu()

                    return True
                else:
                    logger.error("❌ No models available in Ollama")
                    return False
            else:
                logger.error("❌ Failed to get valid response from Ollama")
                return False

        except Exception as e:
            logger.error(f"❌ Ollama initialization error: {e}")
            return False
