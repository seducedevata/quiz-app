"""
Model manager service implementation
"""

# CRITICAL MEMORY FIX: Import only lightweight modules during startup
import logging
import os
import time
from typing import Dict, Any, Optional, Callable
from .interfaces import IModelManager
from .gpu_check import get_gpu_info

# CRITICAL MEMORY FIX: Heavy ML imports will be done lazily when model operations are first used

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/opt-125m"  # Smaller, publicly available model

# Custom exceptions for better error handling
class ModelLoadingError(Exception):
    """Base exception for model loading errors"""
    pass

class InsufficientMemoryError(ModelLoadingError):
    """Raised when there's insufficient memory to load the model"""
    pass

class ModelNotFoundError(ModelLoadingError):
    """Raised when the specified model cannot be found"""
    pass

class ModelCorruptedError(ModelLoadingError):
    """Raised when the model files are corrupted"""
    pass

class NetworkError(ModelLoadingError):
    """Raised when there are network issues downloading the model"""
    pass


class ModelManagerService(IModelManager):
    """Implementation of the model manager service"""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._current_model_path = None
        self._error_callback: Optional[Callable[[str], None]] = None

    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback function for error reporting to UI"""
        self._error_callback = callback

    def _report_error(self, error_message: str):
        """Report error to UI if callback is set"""
        if self._error_callback:
            try:
                self._error_callback(error_message)
            except Exception as e:
                logger.error(f"Error reporting to UI: {e}")

    def _retry_with_exponential_backoff(self, func: Callable, max_retries: int = 3, base_delay: float = 1.0) -> Any:
        """Retry a function with exponential backoff"""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise  # Re-raise on final attempt

                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                self._report_error(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
                time.sleep(delay)

    def load_model(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
        quantization_config: Any = None,
        torch_dtype: Any = None,
        timeout_seconds: int = 300,
    ) -> bool:
        """
        Load a model

        Args:
            model_name_or_path: Model name or path
            adapter_path: Optional path to adapter weights
            device_map: Device mapping strategy
            quantization_config: Optional quantization configuration
            torch_dtype: Optional torch data type
            timeout_seconds: Timeout in seconds

        Returns:
            bool: True if successful

        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info(f"Loading model {model_name_or_path}")

            # Set up quantization if not provided
            if quantization_config is None and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )

            # Set up dtype if not provided
            if torch_dtype is None:
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # Load tokenizer with retry logic
            def _load_tokenizer():
                return AutoTokenizer.from_pretrained(
                    model_name_or_path, trust_remote_code=True
                )

            self._tokenizer = self._retry_with_exponential_backoff(_load_tokenizer)

            # Load model with retry logic
            def _load_model():
                return AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    device_map=device_map,
                    quantization_config=quantization_config,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )

            self._model = self._retry_with_exponential_backoff(_load_model)

            # Load adapter if provided
            if adapter_path and os.path.exists(adapter_path):
                if not os.path.exists(adapter_path):
                    raise ModelNotFoundError(f"Adapter path not found: {adapter_path}")
                self._model.load_adapter(adapter_path)

            self._current_model_path = model_name_or_path
            self._device = next(self._model.parameters()).device

            logger.info(f"Model loaded successfully on {self._device}")
            self._report_error(f"✅ Model loaded successfully: {model_name_or_path}")
            return True

        except torch.cuda.OutOfMemoryError as e:
            error_msg = f"Insufficient GPU memory to load model {model_name_or_path}"
            logger.error(error_msg)
            self._report_error(error_msg)
            self.cleanup_model_resources()
            raise InsufficientMemoryError(error_msg) from e

        except FileNotFoundError as e:
            error_msg = f"Model not found: {model_name_or_path}"
            logger.error(error_msg)
            self._report_error(error_msg)
            self.cleanup_model_resources()
            raise ModelNotFoundError(error_msg) from e

        except OSError as e:
            if "Connection error" in str(e) or "Network" in str(e):
                error_msg = f"Network error loading model {model_name_or_path}: {e}"
                logger.error(error_msg)
                self._report_error(error_msg)
                self.cleanup_model_resources()
                raise NetworkError(error_msg) from e
            elif "corrupted" in str(e).lower() or "invalid" in str(e).lower():
                error_msg = f"Model files corrupted: {model_name_or_path}"
                logger.error(error_msg)
                self._report_error(error_msg)
                self.cleanup_model_resources()
                raise ModelCorruptedError(error_msg) from e
            else:
                error_msg = f"OS error loading model {model_name_or_path}: {e}"
                logger.error(error_msg)
                self._report_error(error_msg)
                self.cleanup_model_resources()
                raise ModelLoadingError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error loading model {model_name_or_path}: {e}"
            logger.error(error_msg)
            self._report_error(error_msg)
            self.cleanup_model_resources()
            raise ModelLoadingError(error_msg) from e

    def unload_model(self) -> bool:
        """
        Unload the current model

        Returns:
            bool: True if successful
        """
        try:
            self.cleanup_model_resources()
            return True
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False

    def get_optimal_model_config(self, device_type: str = "auto") -> Dict[str, Any]:
        """
        Get optimal model configuration based on available hardware

        Args:
            device_type: Target device type

        Returns:
            Dict containing optimal configuration
        """
        try:
            gpu_info = get_gpu_info() if device_type in ["auto", "cuda"] else None

            config = {
                "model_name_or_path": DEFAULT_MODEL,
                "device_map": "auto",
                "torch_dtype": torch.float32,
                "quantization_config": None,
            }

            if gpu_info and gpu_info["cuda_available"]:
                # GPU is available
                free_memory = gpu_info["free_memory"]
                if free_memory > 8 * 1024:  # More than 8GB free
                    config.update({"torch_dtype": torch.float16, "quantization_config": None})
                else:
                    # Use 4-bit quantization for limited memory
                    config.update(
                        {
                            "torch_dtype": torch.float16,
                            "quantization_config": BitsAndBytesConfig(
                                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                            ),
                        }
                    )

            return config

        except Exception as e:
            logger.error(f"Error getting optimal config: {e}")
            return {}

    def cleanup_model_resources(
        self, model: Any = None, offload_folder: Optional[str] = None
    ) -> None:
        """
        Clean up model resources

        Args:
            model: Optional model to clean up
            offload_folder: Optional folder for offloading
        """
        try:
            if model is None:
                model = self._model

            if model is not None:
                # Move model to CPU
                if hasattr(model, "cpu"):
                    model.cpu()

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Delete model
                del model

            self._model = None
            self._tokenizer = None
            self._current_model_path = None
            self._device = "cpu"

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded

        Returns:
            bool: True if model is loaded
        """
        return self._model is not None and self._tokenizer is not None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 350,
        temperature: float = 0.5,
        top_p: float = 0.85,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        timeout_seconds: int = 60,
    ) -> str:
        """
        Generate text using the model

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            timeout_seconds: Timeout in seconds

        Returns:
            Generated text

        Raises:
            Exception: If generation fails
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")

        try:
            # Tokenize input with proper attention mask handling
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,  # No padding for single prompt
                truncation=True,
                max_length=512,
                return_attention_mask=True,
            )

            # Move to device
            inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

            # Handle attention mask for models where pad_token == eos_token
            if self._tokenizer.pad_token_id == self._tokenizer.eos_token_id:
                # Create explicit attention mask
                input_length = inputs["input_ids"].shape[1]
                attention_mask = torch.ones(
                    (1, input_length), dtype=torch.long, device=self._device
                )
                inputs["attention_mask"] = attention_mask

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            # Decode output
            generated_text = self._tokenizer.decode(
                outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            # Extract only the generated part
            if prompt in generated_text:
                generated_text = generated_text[len(prompt) :].strip()

            return generated_text

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise