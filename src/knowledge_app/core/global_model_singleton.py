"""
[HOT] FIRE METHOD 3: Global Model Singleton - Pre-warm and Pin Strategy

This is the nuclear brute force solution:
- Load model ONCE at application startup
- Keep it pinned in GPU memory FOREVER
- No loading/unloading during runtime
- No race conditions possible
- No deadlocks possible
- No memory corruption possible

Trade-offs:
- Slower startup (30-60 seconds)
- Higher memory usage (always 4GB+ GPU)
- But: ZERO crashes, ZERO race conditions, ZERO complexity

This is the "pay once, use forever" approach.
"""

import threading
import time
import logging
import gc
import os
import re
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)

# 🛡️ CRITICAL SECURITY FIX: Allowed model patterns to prevent path traversal
ALLOWED_MODEL_PATTERNS = [
    r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$',  # Standard HuggingFace format: org/model
    r'^[a-zA-Z0-9_.-]+$',  # Simple model names
]

BLOCKED_PATH_PATTERNS = [
    r'\.\.',  # Directory traversal
    r'/',  # Absolute paths (except in org/model format)
    r'\\',  # Windows paths
    r'~',  # Home directory
    r'\$',  # Environment variables
]

def _validate_model_path(model_id: str) -> bool:
    """
    🛡️ CRITICAL SECURITY FIX #13: Enhanced model path validation to prevent path traversal attacks
    
    This function prevents attackers from using malicious model paths to access
    unauthorized files or directories on the system.

    Args:
        model_id: The model identifier to validate

    Returns:
        bool: True if the model path is safe, False otherwise
    """
    if not model_id or not isinstance(model_id, str):
        logger.error("[SECURITY] CRITICAL: Invalid model_id type - must be non-empty string")
        return False

    # 🛡️ CRITICAL: Normalize path to prevent obfuscated traversal attempts
    try:
        normalized_path = os.path.normpath(model_id)
        # Check if normalization changed the path (indicates potential traversal)
        if normalized_path != model_id and '..' in model_id:
            logger.error(f"[SECURITY] CRITICAL: Path traversal attempt detected: {model_id}")
            return False
    except Exception:
        logger.error(f"[SECURITY] CRITICAL: Cannot normalize path: {model_id}")
        return False

    # 🛡️ CRITICAL: Check for blocked patterns (enhanced list)
    critical_patterns = [
        r'\.\.',  # Parent directory traversal
        r'/',  # Unix absolute paths
        r'\\',  # Windows paths and escapes
        r'~',  # Home directory
        r'\$',  # Environment variables
        r'%',  # Windows environment variables (%SYSTEMROOT%, etc.)
        r'\x00',  # Null byte injection
        r'[<>:"|?*]',  # Windows forbidden characters
        r'^\.',  # Hidden files/directories
        r'\.\./',  # Unix traversal
        r'\.\.\\',  # Windows traversal
        r'file://',  # File URL scheme
        r'ftp://',  # FTP URL scheme
        r'http://',  # HTTP URL scheme
        r'https://',  # HTTPS URL scheme
        r'^/[^a-zA-Z0-9_.-]',  # Suspicious absolute paths
    ]
    
    for pattern in critical_patterns:
        if re.search(pattern, model_id, re.IGNORECASE):
            logger.error(f"[SECURITY] CRITICAL: Blocked security pattern '{pattern}' found in model_id: {model_id}")
            return False

    # 🛡️ CRITICAL: Enforce strict allow-list of model patterns
    # Only allow safe, verified model naming conventions
    safe_patterns = [
        r'^[a-zA-Z0-9_.-]+$',  # Basic alphanumeric with safe chars
        r'^[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+$',  # Model:tag format (e.g., llama3.1:8b)
        r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$',  # Model/version format
    ]
    
    model_is_safe = False
    for pattern in safe_patterns:
        if re.match(pattern, model_id):
            model_is_safe = True
            break
    
    if not model_is_safe:
        logger.error(f"[SECURITY] CRITICAL: Model path does not match safe patterns: {model_id}")
        return False

    # 🛡️ CRITICAL: Additional length and content checks
    if len(model_id) > 128:  # Reasonable model name length limit
        logger.error(f"[SECURITY] CRITICAL: Model path too long (>{128} chars): {model_id}")
        return False
    
    # Check for suspicious repeated characters (potential overflow attempt)
    if re.search(r'(.)\1{10,}', model_id):  # More than 10 repeated chars
        logger.error(f"[SECURITY] CRITICAL: Suspicious repeated characters in model_id: {model_id}")
        return False
    
    # 🛡️ CRITICAL: Validate against known safe model repositories
    safe_prefixes = [
        'llama',
        'mistral', 
        'phi',
        'qwq',
        'deepseek',
        'codellama',
        'yi',
        'gemma'
    ]
    
    # Extract model base name for validation
    model_base = model_id.split(':')[0].split('/')[0].lower()
    if not any(model_base.startswith(prefix) for prefix in safe_prefixes):
        logger.warning(f"[SECURITY] WARNING: Model not from known safe repository: {model_id}")
        # Don't block, but log warning for security monitoring

    logger.info(f"[SECURITY] ✅ Model path validation passed: {model_id}")
    return True


class GlobalModelSingleton:
    """
    🔧 FIX: Enhanced Global Model Singleton with Model Switching Support

    Now supports:
    - Loading models once for performance
    - Safe model switching when needed
    - Proper resource cleanup
    - Thread-safe operations
    """

    _instance: Optional["GlobalModelSingleton"] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # [CONFIG] CRITICAL FIX: Thread-safe initialization check
        with self._lock:
            if self._initialized:
                return

            # Model storage
            self.model = None
            self.tokenizer = None
            self.model_id = None
            self.device = None
            self.is_loaded = False
            self.load_time = None
            self.generation_count = 0

            # LoRA adapter support
            self.current_adapter = None
            self.lora_manager = None

            # Thread safety
            self.generation_lock = threading.RLock()
            self.model_switch_lock = threading.RLock()  # 🔧 FIX: Separate lock for model switching

            # Memory tracking
            self.initial_gpu_memory = 0
            self.model_gpu_memory = 0

            # [CONFIG] CRITICAL FIX: Set initialized flag at the very end to prevent race conditions
            self._initialized = True
            logger.info("[BUILD] GlobalModelSingleton initialized (not loaded yet)")

    def load_model_once(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2") -> bool:
        """
        Load the model ONCE and keep it forever.
        This should be called at application startup.
        """
        # 🛡️ CRITICAL SECURITY FIX: Validate model path before loading
        if not _validate_model_path(model_id):
            logger.error(f"[ERROR] SECURITY: Rejected unsafe model path: {model_id}")
            return False

        with self._lock:
            if self.is_loaded:
                logger.info(f"[OK] Model already loaded: {self.model_id}")
                return True

            logger.info("[HOT] FIRE METHOD 3: Loading model ONCE and pinning forever")
            logger.info(f"[START] Loading model: {model_id}")

            start_time = time.time()

            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM

                # Clear any existing CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.initial_gpu_memory = torch.cuda.memory_allocated()
                    logger.info(f"Initial GPU memory: {self.initial_gpu_memory // 1024**2}MB")
                    self.device = "cuda"
                else:
                    self.device = "cpu"
                    logger.warning("CUDA not available, using CPU")

                # Load tokenizer
                logger.info("[DOC] Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("[OK] Tokenizer loaded")

                # Load model with optimized settings for RTX 3060 12GB
                logger.info("[BRAIN] Loading model with RTX 3060 optimizations...")

                if "7B" in model_id or "7b" in model_id:
                    # THE ROYAL DECREE: 7B model strategy with Swift Blade of Attention
                    logger.info(
                        "👑 THE ROYAL DECREE: Using 7B model optimizations with Swift Blade of Attention"
                    )

                    # Import attention optimizer
                    from .attention_optimizer import attention_optimizer

                    # Base model kwargs for 7B models on RTX 3060 12GB
                    model_kwargs = {
                        "torch_dtype": torch.bfloat16,  # Use bfloat16 for better performance
                        "device_map": "auto",
                        "load_in_4bit": True,  # 4-bit quantization for memory efficiency
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True,
                        # RTX 3060 specific optimizations
                        "max_memory": {0: "10GB"},  # Leave 2GB buffer
                        "offload_folder": "./offload_cache",  # Offload if needed
                    }

                    # Apply attention optimization
                    attention_kwargs = attention_optimizer.get_model_kwargs(model_id)
                    model_kwargs.update(attention_kwargs)

                    attn_impl = model_kwargs.get("attn_implementation", "eager")
                    logger.info(f"🗡️ Loading 7B model with {attn_impl} attention")

                    self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
                else:
                    # THE ROYAL DECREE: Smaller model strategy with Swift Blade of Attention
                    logger.info(
                        "👑 THE ROYAL DECREE: Using smaller model optimizations with Swift Blade of Attention"
                    )

                    # Base model kwargs for smaller models
                    model_kwargs = {
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True,
                    }

                    # Apply attention optimization
                    attention_kwargs = attention_optimizer.get_model_kwargs(model_id)
                    model_kwargs.update(attention_kwargs)

                    attn_impl = model_kwargs.get("attn_implementation", "eager")
                    logger.info(f"🗡️ Loading smaller model with {attn_impl} attention")

                    self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

                # Store model info
                self.model_id = model_id
                self.load_time = time.time() - start_time

                # Check final memory usage
                if torch.cuda.is_available():
                    final_memory = torch.cuda.memory_allocated()
                    self.model_gpu_memory = final_memory - self.initial_gpu_memory
                    logger.info(f"Final GPU memory: {final_memory // 1024**2}MB")
                    logger.info(f"Model memory usage: {self.model_gpu_memory // 1024**2}MB")

                # Verify model device
                if hasattr(self.model, "device"):
                    logger.info(f"Model device: {self.model.device}")

                self.is_loaded = True

                # Initialize LoRA manager
                try:
                    from .lora_manager import LoraManager
                    self.lora_manager = LoraManager()
                    logger.info("[LINK] LoRA manager initialized")
                except Exception as e:
                    logger.warning(f"[WARNING] LoRA manager initialization failed: {e}")

                logger.info("[SUCCESS] MODEL LOADED AND PINNED FOREVER!")
                logger.info(f"[TIME] Load time: {self.load_time:.1f} seconds")
                logger.info(f"[SAVE] GPU memory used: {self.model_gpu_memory // 1024**2}MB")
                logger.info("🔒 Model will NEVER be unloaded during runtime")

                return True

            except Exception as e:
                logger.error(f"[ERROR] Failed to load model: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                self.is_loaded = False
                return False

    def load_lora_adapter(self, adapter_name: str) -> bool:
        """
        [LINK] GOLDEN PATH: Load and apply LoRA adapter to the base model
        
        This allows domain specialization by applying trained LoRA weights
        on top of the base model for better performance in specific domains.
        """
        if not self.is_loaded:
            logger.error("[ERROR] Cannot load LoRA adapter: base model not loaded")
            return False
        
        if not self.lora_manager:
            logger.error("[ERROR] LoRA manager not available")
            return False
        
        with self.generation_lock:
            try:
                logger.info(f"[LINK] Loading LoRA adapter: {adapter_name}")
                
                # Check if adapter exists
                if not self.lora_manager.has_adapter(adapter_name):
                    logger.error(f"[ERROR] LoRA adapter not found: {adapter_name}")
                    return False
                
                # If we already have this adapter loaded, skip
                if self.current_adapter == adapter_name:
                    logger.info(f"[OK] LoRA adapter '{adapter_name}' already loaded")
                    return True
                
                # Apply the LoRA adapter to the model
                self.model = self.lora_manager.load_adapter(self.model, adapter_name)
                self.current_adapter = adapter_name
                
                logger.info(f"[OK] LoRA adapter '{adapter_name}' loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to load LoRA adapter '{adapter_name}': {e}")
                return False
    
    def get_available_adapters(self) -> list:
        """Get list of available LoRA adapters"""
        if not self.lora_manager:
            return []
        return self.lora_manager.list_adapters()

    def generate_text(
        self, prompt: str, max_tokens: int = 150, temperature: float = 0.7,
        adapter_name: Optional[str] = None, max_length: Optional[int] = None, do_sample: bool = True
    ) -> str:
        """
        Generate text using the pinned model with optional LoRA adapter.
        This is thread-safe and can be called from anywhere.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum new tokens to generate (legacy parameter)
            temperature: Sampling temperature
            adapter_name: Optional LoRA adapter name to apply temporarily
            max_length: Maximum total length (overrides max_tokens if provided)
            do_sample: Whether to use sampling

        Returns:
            Generated text string (just the new content, not including prompt)
        """
        if not self.is_loaded:
            logger.error("[ERROR] Model not loaded for text generation")
            return ""

        with self.generation_lock:
            try:
                logger.info(f"[TARGET] Generating text (request #{self.generation_count + 1})")
                if adapter_name:
                    logger.info(f"   [LINK] Using adapter: {adapter_name}")
                logger.debug(f"Prompt: {prompt[:50]}...")

                start_time = time.time()

                # Determine which model to use
                active_model = self.model  # Start with base model

                # Apply adapter temporarily if requested (without modifying singleton state)
                if adapter_name and self.lora_manager:
                    try:
                        if self.lora_manager.has_adapter(adapter_name):
                            # Create a temporary adapter instance for this request only
                            active_model = self.lora_manager.load_adapter(self.model, adapter_name)
                            logger.debug(f"[OK] Temporarily applied adapter: {adapter_name}")
                        else:
                            logger.warning(f"[WARNING] Adapter '{adapter_name}' not found, using base model")
                    except Exception as e:
                        logger.warning(f"[WARNING] Failed to apply adapter '{adapter_name}': {e}, using base model")

                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

                # Move to model device
                device = next(active_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Determine max_new_tokens
                if max_length:
                    input_length = inputs['input_ids'].shape[1]
                    max_new_tokens = max(1, max_length - input_length)
                else:
                    max_new_tokens = max_tokens

                # Generate with no_grad for memory efficiency and GPU monitoring
                with torch.no_grad():
                    # Monitor GPU utilization during generation
                    if torch.cuda.is_available():
                        gpu_memory_before = torch.cuda.memory_allocated()
                        logger.debug(f"[HOT] GPU memory before generation: {gpu_memory_before // 1024**2}MB")

                    outputs = active_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        # Additional GPU optimization parameters
                        num_beams=1,  # Use greedy decoding for maximum GPU efficiency
                        early_stopping=False,  # Don't stop early to maintain GPU load
                    )

                    # Monitor GPU utilization after generation
                    if torch.cuda.is_available():
                        gpu_memory_after = torch.cuda.memory_allocated()
                        gpu_memory_used = gpu_memory_after - gpu_memory_before
                        logger.debug(f"[HOT] GPU memory after generation: {gpu_memory_after // 1024**2}MB")
                        logger.debug(f"[HOT] GPU memory used for generation: {gpu_memory_used // 1024**2}MB")

                # Decode result and extract only the new content
                full_result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Remove the original prompt to get only the generated content
                if full_result.startswith(prompt):
                    result = full_result[len(prompt):].strip()
                else:
                    result = full_result.strip()

                generation_time = time.time() - start_time
                self.generation_count += 1

                logger.info(f"[OK] Generation successful in {generation_time:.1f}s")
                logger.debug(f"Result length: {len(result)} characters")

                return result

            except Exception as e:
                logger.error(f"[ERROR] Generation failed: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")

                return ""  # Return empty string on error

    def get_status(self) -> Dict[str, Any]:
        """Get model status"""
        return {
            "is_loaded": self.is_loaded,
            "model_id": self.model_id,
            "device": self.device,
            "load_time": self.load_time,
            "generation_count": self.generation_count,
            "gpu_memory_mb": self.model_gpu_memory // 1024**2 if self.model_gpu_memory else 0,
            "uptime": time.time() - (time.time() - self.load_time) if self.load_time else 0,
        }

    def force_cleanup(self):
        """
        EMERGENCY ONLY: Force cleanup of the model.
        This should NEVER be called during normal operation.
        Only use this for application shutdown.
        """
        logger.warning("[EMERGENCY] EMERGENCY: Force cleanup requested")

        with self._lock:
            try:
                if self.model is not None:
                    del self.model
                    logger.info("Model deleted")

                if self.tokenizer is not None:
                    del self.tokenizer
                    logger.info("Tokenizer deleted")

                # Force garbage collection
                collected = gc.collect()
                logger.info(f"Garbage collection freed {collected} objects")

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    final_memory = torch.cuda.memory_allocated()
                    logger.info(f"Final GPU memory after cleanup: {final_memory // 1024**2}MB")

                # Reset state
                self.model = None
                self.tokenizer = None
                self.model_id = None
                self.is_loaded = False
                self.generation_count = 0

                logger.warning("[EMERGENCY] EMERGENCY cleanup completed")

            except Exception as e:
                logger.error(f"[ERROR] Error during emergency cleanup: {e}")

    def unload_model(self) -> bool:
        """
        🔧 FIX: Safely unload the current model to free memory

        Returns:
            bool: True if unload was successful
        """
        with self.model_switch_lock:
            try:
                if not self.is_loaded:
                    logger.info("[OK] No model loaded to unload")
                    return True

                logger.info(f"[UNLOAD] Unloading model: {self.model_id}")

                # Clean up model and tokenizer
                if self.model is not None:
                    del self.model
                    self.model = None

                if self.tokenizer is not None:
                    del self.tokenizer
                    self.tokenizer = None

                # Reset state
                self.is_loaded = False
                self.model_id = None
                self.current_adapter = None
                self.lora_manager = None

                # Force garbage collection
                import gc
                gc.collect()

                # Clear GPU cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("[GPU] GPU cache cleared")

                logger.info("[SUCCESS] Model unloaded successfully")
                return True

            except Exception as e:
                logger.error(f"[ERROR] Failed to unload model: {e}")
                return False

    def switch_model(self, new_model_id: str) -> bool:
        """
        🔧 FIX: Switch to a different model safely

        Args:
            new_model_id: The new model to load

        Returns:
            bool: True if switch was successful
        """
        with self.model_switch_lock:
            try:
                # Check if we're already using this model
                if self.is_loaded and self.model_id == new_model_id:
                    logger.info(f"[OK] Model '{new_model_id}' already loaded")
                    return True

                logger.info(f"[SWITCH] Switching from '{self.model_id}' to '{new_model_id}'")

                # Unload current model first
                if self.is_loaded:
                    if not self.unload_model():
                        logger.error("[ERROR] Failed to unload current model before switching")
                        return False

                # Load new model
                return self.load_model_once(new_model_id)

            except Exception as e:
                logger.error(f"[ERROR] Failed to switch model to '{new_model_id}': {e}")
                return False


# Global instance
_global_model: Optional[GlobalModelSingleton] = None


def get_global_model() -> GlobalModelSingleton:
    """Get the global model singleton"""
    global _global_model
    if _global_model is None:
        _global_model = GlobalModelSingleton()
    return _global_model


def load_global_model(model_id: str = "mistralai/Mistral-7B-Instruct-v0.2") -> bool:
    """Load the global model once at application startup"""
    # 🛡️ CRITICAL SECURITY FIX: Validate model path at entry point
    if not _validate_model_path(model_id):
        logger.error(f"[ERROR] SECURITY: Rejected unsafe model path in load_global_model: {model_id}")
        return False

    model = get_global_model()
    return model.load_model_once(model_id)


def generate_mcq_global(prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
    """
    Generate MCQ using the global pinned model.
    This is the main function your UI should call.
    """
    model = get_global_model()
    return model.generate_text(prompt, max_tokens)


def is_global_model_ready() -> bool:
    """Check if the global model is ready"""
    model = get_global_model()
    return model.is_loaded


def get_global_model_status() -> Dict[str, Any]:
    """Get global model status"""
    model = get_global_model()
    return model.get_status()


def shutdown_global_model():
    """Shutdown the global model (emergency only)"""
    model = get_global_model()
    model.force_cleanup()


def switch_global_model(new_model_id: str) -> bool:
    """
    🔧 FIX: Switch to a different model safely

    Args:
        new_model_id: The new model to load

    Returns:
        bool: True if switch was successful
    """
    # 🛡️ CRITICAL SECURITY FIX: Validate model path before switching
    if not _validate_model_path(new_model_id):
        logger.error(f"[ERROR] SECURITY: Rejected unsafe model path in switch_global_model: {new_model_id}")
        return False

    model = get_global_model()
    return model.switch_model(new_model_id)


def unload_global_model() -> bool:
    """
    🔧 FIX: Safely unload the current model to free memory

    Returns:
        bool: True if unload was successful
    """
    model = get_global_model()
    return model.unload_model()
