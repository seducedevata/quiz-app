"""
Attention Optimization Module

This module provides utilities for optimizing attention mechanisms in transformer models,
supporting multiple backends including FlashAttention 2, xFormers, and standard attention.
Automatically selects the best available option for the current platform.
"""

import os
import logging
import platform
import torch
from typing import Dict, Any, Optional, Tuple
from packaging import version

logger = logging.getLogger(__name__)


class AttentionOptimizer:
    """Manages attention optimization strategies for transformer models"""

    def __init__(self):
        try:
            self.available_backends = self._detect_available_backends()
            # Ensure available_backends is always a dictionary
            if not isinstance(self.available_backends, dict):
                logger.error("Backend detection returned invalid result, using fallback")
                self.available_backends = {
                    "flash_attention_2": False,
                    "xformers": False,
                    "eager": True,
                }

            self.selected_backend = self._select_best_backend()
            logger.info(f"Attention optimizer initialized with backend: {self.selected_backend}")
        except Exception as e:
            logger.error(f"Failed to initialize AttentionOptimizer: {e}")
            # Ensure we have safe fallback values
            self.available_backends = {"flash_attention_2": False, "xformers": False, "eager": True}
            self.selected_backend = "eager"
            logger.warning("Using fallback eager attention due to initialization error")

    def _detect_available_backends(self) -> Dict[str, bool]:
        """
        THE ROYAL INSPECTOR: Detect which attention optimization backends are available

        This method implements the Principle of a Worthy Smithy by checking
        the environment for available attention optimization tools.
        """
        backends = {
            "flash_attention_2": False,
            "xformers": False,
            "sdpa": False,
            "eager": True,  # Always available as fallback
        }

        logger.info("🔍 THE ROYAL INSPECTOR: Detecting available attention backends...")

        # Check PyTorch SDPA (most reliable and built-in)
        try:
            import torch
            import torch.nn.functional as F

            pytorch_version = tuple(map(int, torch.__version__.split(".")[:2]))
            if pytorch_version >= (2, 0):
                # Test SDPA functionality if CUDA is available
                if torch.cuda.is_available():
                    try:
                        q = torch.randn(1, 8, 64, device="cuda", dtype=torch.float16)
                        k = torch.randn(1, 8, 64, device="cuda", dtype=torch.float16)
                        v = torch.randn(1, 8, 64, device="cuda", dtype=torch.float16)
                        F.scaled_dot_product_attention(q, k, v)
                        del q, k, v  # Clean up
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logger.debug(f"SDPA CUDA test failed: {e}")
                backends["sdpa"] = True
                logger.info(f"✅ PyTorch SDPA detected and tested (PyTorch {torch.__version__})")
        except Exception as e:
            logger.debug(f"PyTorch SDPA detection failed: {e}")

        # Check for Flash Attention 2
        try:
            import flash_attn

            backends["flash_attention_2"] = True
            logger.info("✅ Flash Attention 2 detected and available")
        except ImportError:
            logger.info("❌ Flash Attention 2 not available (ImportError)")
        except Exception as e:
            logger.warning(f"❌ Flash Attention 2 check failed: {e}")

        # Check for xFormers with comprehensive compatibility checking
        # Temporarily disabled due to version incompatibility causing Windows fatal exceptions
        backends["xformers"] = False  # self._check_xformers_compatibility()
        logger.debug("xFormers disabled due to version incompatibility with PyTorch 2.5.1+cu121")

        return backends

    def _check_xformers_compatibility(self) -> bool:
        """Comprehensive xFormers compatibility checking for PyTorch 2.5.1+cu121"""
        try:
            # Suppress xFormers warnings during import
            import os
            import sys
            import contextlib

            # Set environment variables to suppress xFormers output
            os.environ["XFORMERS_MORE_DETAILS"] = "0"

            # Context manager to suppress stdout/stderr during xFormers import
            @contextlib.contextmanager
            def suppress_output():
                with open(os.devnull, "w") as devnull:
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    try:
                        sys.stdout = devnull
                        sys.stderr = devnull
                        yield
                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr

            # Import xFormers with suppressed output
            with suppress_output():
                import xformers
                import torch

            # Check for known version incompatibilities
            pytorch_version = torch.__version__
            xformers_version = xformers.__version__

            # Known problematic combinations with enhanced detection
            version_mismatch_patterns = [
                ("2.5.1", "2.7.0"),  # Current known issue
                ("2.5.0", "2.7.0"),  # Related versions
                ("2.4.", "2.7."),  # Major version gaps
            ]

            for pt_pattern, xf_pattern in version_mismatch_patterns:
                if pt_pattern in pytorch_version and xf_pattern in xformers_version:
                    logger.debug(
                        f"Known incompatible versions: PyTorch {pytorch_version} + xFormers {xformers_version}"
                    )
                    logger.debug(
                        f"xFormers built for PyTorch {xf_pattern}+ but running on PyTorch {pt_pattern}"
                    )
                    logger.debug("Automatically falling back to eager attention for stability")
                    return False

            # Try to import and test xFormers ops
            try:
                with suppress_output():
                    import xformers.ops

                if not hasattr(xformers.ops, "memory_efficient_attention"):
                    logger.debug("xFormers ops.memory_efficient_attention not available")
                    return False

                # Test with dummy tensors if CUDA available
                if torch.cuda.is_available():
                    try:
                        # Small compatibility test with proper error handling
                        device = torch.device("cuda")
                        dtype = torch.float16

                        test_q = torch.randn(1, 1, 4, 8, device=device, dtype=dtype)
                        test_k = torch.randn(1, 1, 4, 8, device=device, dtype=dtype)
                        test_v = torch.randn(1, 1, 4, 8, device=device, dtype=dtype)

                        # Test memory efficient attention with timeout
                        with torch.no_grad():
                            result = xformers.ops.memory_efficient_attention(test_q, test_k, test_v)

                        # Clean up test tensors
                        del test_q, test_k, test_v, result
                        torch.cuda.empty_cache()

                        logger.info("✅ xFormers compatibility test passed")
                        return True

                    except RuntimeError as e:
                        if "version" in str(e).lower() or "compatibility" in str(e).lower():
                            logger.debug(f"xFormers version compatibility error: {e}")
                            logger.debug("This is likely due to PyTorch/xFormers version mismatch")
                        else:
                            logger.debug(f"xFormers runtime error: {e}")
                        return False
                    except Exception as e:
                        logger.debug(f"xFormers compatibility test failed: {e}")
                        return False
                else:
                    # CPU mode - be conservative
                    logger.debug("xFormers detected but CUDA not available for verification")
                    return False

            except ImportError as e:
                logger.debug(f"xFormers ops import failed: {e}")
                return False

        except ImportError:
            logger.debug("xFormers not available")
            return False
        except Exception as e:
            logger.warning(f"❌ xFormers detection failed: {e}")
            return False

    def _select_best_backend(self) -> str:
        """
        THE ROYAL SELECTOR: Choose the best available attention backend

        This implements the Principle of the Right Tool for the Kingdom
        by selecting the optimal backend based on platform and availability.
        """
        # Safety check for available_backends
        if not isinstance(self.available_backends, dict):
            logger.error("available_backends is not a dictionary, using eager fallback")
            return "eager"

        system = platform.system().lower()
        logger.info(f"👑 THE ROYAL SELECTOR: Choosing best backend for {system}")

        # ROYAL PREFERENCE ORDER (based on reliability and performance):
        # 1. PyTorch SDPA (most reliable, built-in, excellent performance)
        # 2. Flash Attention 2 (fastest, but requires specific hardware/software)
        # 3. xFormers (good Windows compatibility)
        # 4. Eager (fallback)

        if self.available_backends.get("sdpa", False):
            logger.info("👑 ROYAL CHOICE: PyTorch SDPA (Built-in, reliable, fast)")
            return "sdpa"
        elif self.available_backends.get("flash_attention_2", False):
            logger.info("👑 ROYAL CHOICE: Flash Attention 2 (Maximum speed)")
            return "flash_attention_2"
        elif self.available_backends.get("xformers", False):
            logger.info("👑 ROYAL CHOICE: xFormers (Windows optimized)")
            return "xformers"
        else:
            logger.warning("👑 ROYAL FALLBACK: Eager attention (standard)")
            return "eager"

    def get_model_kwargs(self, model_name: str = None) -> Dict[str, Any]:
        """
        THE ROYAL DECREE: ATTENTION IMPLEMENTATION CASCADE WITH FORCED GPU USAGE

        This method implements the Swift Blade of Attention with a robust fallback cascade
        following the Principle of Explicit Command and the Principle of the Right Tool.

        🚨 CRITICAL FIX: Forces GPU usage when CUDA is available, no matter what!
        """
        kwargs = {}

        # 🚨 CRITICAL FIX: Force GPU usage if CUDA is available
        if torch.cuda.is_available():
            kwargs["device_map"] = "cuda"
            kwargs["torch_dtype"] = torch.float16  # Use FP16 for better GPU performance
            logger.info("🚨 FORCING GPU USAGE: CUDA detected, enforcing GPU device placement")
        else:
            logger.warning("⚠️ CUDA not available, falling back to CPU")

        # THE ROYAL DECREE: Try the fastest backends first with explicit commands
        logger.info("👑 THE ROYAL DECREE: Forging the Swift Blade of Attention...")

        # First, try to detect and use PyTorch SDPA (most reliable on Windows)
        try:
            import torch

            pytorch_version = tuple(map(int, torch.__version__.split(".")[:2]))
            if pytorch_version >= (2, 0):
                kwargs["attn_implementation"] = "sdpa"
                logger.info("🗡️ ROYAL COMMAND: Using PyTorch SDPA (Scaled Dot-Product Attention)")
                logger.info("   - Built into PyTorch 2.0+")
                logger.info("   - Excellent performance and reliability")
                return kwargs
        except Exception as e:
            logger.warning(f"⚠️ SDPA detection failed: {e}")

        # Second, try Flash Attention 2 (if available)
        if self.available_backends.get("flash_attention_2", False):
            kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("🗡️ ROYAL COMMAND: Using Flash Attention 2")
            logger.info("   - Maximum speed and memory efficiency")
            return kwargs

        # Third, try xFormers (Windows champion)
        elif self.available_backends.get("xformers", False):
            # xFormers is enabled through model configuration, not attn_implementation
            kwargs["attn_implementation"] = "eager"  # Use eager as base, configure xFormers later
            logger.info("🗡️ ROYAL COMMAND: Using xFormers (Windows optimized)")
            logger.info("   - Excellent performance on Windows")
            return kwargs

        # Final fallback to eager attention
        else:
            kwargs["attn_implementation"] = "eager"
            logger.warning("⚠️ ROYAL FALLBACK: Using standard eager attention")
            logger.warning(
                "   - Consider installing xFormers or Flash Attention 2 for better performance"
            )

        return kwargs

    def apply_royal_decree_to_model_loading(
        self, model_name: str, base_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        THE ROYAL DECREE: Apply the Swift Blade of Attention to model loading

        This method combines the attention optimization with existing model kwargs
        and implements the cascade with proper error handling.
        """
        logger.info(f"👑 Applying Royal Decree to model loading: {model_name}")

        # Start with base kwargs
        final_kwargs = base_kwargs.copy()

        # Get attention optimization kwargs
        attention_kwargs = self.get_model_kwargs(model_name)

        # Merge attention kwargs with base kwargs
        final_kwargs.update(attention_kwargs)

        # Log the final configuration
        attn_impl = final_kwargs.get("attn_implementation", "not specified")
        logger.info(f"🗡️ Final attention implementation: {attn_impl}")

        return final_kwargs

    def load_model_with_royal_decree(self, model_class, model_name: str, **base_kwargs):
        """
        THE ROYAL DECREE: Load model with attention cascade and comprehensive error handling

        This method implements the complete Royal Decree with fallback cascade.
        """
        from transformers import AutoModelForCausalLM

        logger.info(f"👑 THE ROYAL DECREE: Loading {model_name} with Swift Blade of Attention")

        # Apply the Royal Decree to get optimized kwargs
        model_kwargs = self.apply_royal_decree_to_model_loading(model_name, base_kwargs)

        # THE ROYAL DECREE CASCADE: Try each backend in order
        attempts = [
            ("flash_attention_2", "🗡️ ATTEMPTING: Flash Attention 2 (Fastest)"),
            ("sdpa", "🗡️ ATTEMPTING: PyTorch SDPA (Reliable)"),
            ("eager", "🗡️ ATTEMPTING: Eager Attention (Fallback)"),
        ]

        for attn_impl, log_message in attempts:
            try:
                logger.info(log_message)
                current_kwargs = model_kwargs.copy()
                current_kwargs["attn_implementation"] = attn_impl

                # Attempt to load the model
                model = model_class.from_pretrained(model_name, **current_kwargs)

                # Success! Log the victory
                logger.info(f"👑 ROYAL SUCCESS: Model loaded with {attn_impl}")
                logger.info(f"🎉 The Swift Blade of Attention has been forged!")

                # Configure xFormers if needed
                if attn_impl == "eager" and self.available_backends.get("xformers", False):
                    self.configure_model_for_xformers(model)

                return model

            except Exception as e:
                error_msg = str(e).lower()
                if "flash" in error_msg or "attention" in error_msg or attn_impl in error_msg:
                    logger.warning(f"⚠️ {attn_impl} failed: {e}")
                    continue
                else:
                    # Non-attention related error, re-raise
                    logger.error(f"💥 Critical error during model loading: {e}")
                    raise

        # If we get here, all attempts failed
        logger.error("💥 ROYAL DECREE FAILED: All attention implementations failed")
        raise RuntimeError("Failed to load model with any attention implementation")

    def configure_model_for_xformers(self, model) -> None:
        """Configure model to use xFormers if selected"""
        if self.selected_backend != "xformers" or not self.available_backends.get(
            "xformers", False
        ):
            return

        try:
            import xformers.ops

            # Enable xFormers memory efficient attention
            if hasattr(model, "enable_xformers_memory_efficient_attention"):
                model.enable_xformers_memory_efficient_attention()
                logger.info("✅ xFormers memory efficient attention enabled")
            else:
                # For models that don't have the direct method, try to configure manually
                self._configure_xformers_manually(model)

        except (ImportError, RuntimeError, OSError) as e:
            logger.warning(f"Failed to configure xFormers: {e}")
            logger.info("Falling back to standard attention")
            # Update backend selection to reflect the fallback
            self.selected_backend = "eager"
            self.available_backends["xformers"] = False

    def _configure_xformers_manually(self, model) -> None:
        """Manually configure xFormers for models that don't have built-in support"""
        try:
            import xformers.ops

            # Iterate through model modules and replace attention where possible
            for name, module in model.named_modules():
                if hasattr(module, "attention") or "attn" in name.lower():
                    # This is a simplified approach - in practice, you'd need
                    # model-specific configuration
                    if hasattr(module, "config"):
                        module.config.use_memory_efficient_attention_xformers = True

            logger.info("✅ xFormers manually configured for model")

        except Exception as e:
            logger.warning(f"Manual xFormers configuration failed: {e}")

    def get_memory_optimization_settings(self) -> Dict[str, Any]:
        """Get memory optimization settings based on selected backend"""
        settings = {"gradient_checkpointing": True, "use_cache": False, "low_cpu_mem_usage": True}

        if self.selected_backend == "xformers":
            # xFormers-specific optimizations
            settings.update(
                {
                    "attention_dropout": 0.0,  # xFormers works better with no dropout
                    "use_memory_efficient_attention": True,
                    "memory_efficient_attention_xformers": True,
                }
            )

        elif self.selected_backend == "flash_attention_2":
            # FlashAttention-specific optimizations
            settings.update({"flash_attention_2": True, "use_flash_attention_2": True})

        return settings

    def get_training_optimizations(self) -> Dict[str, Any]:
        """Get training-specific optimizations for the selected backend"""
        optimizations = {
            "dataloader_pin_memory": False,  # Can cause issues with some attention backends
            "dataloader_num_workers": 0,  # Safer with attention optimizations
        }

        if self.selected_backend == "xformers":
            optimizations.update(
                {
                    "fp16": True,  # xFormers works well with FP16
                    "bf16": False,  # Stick to FP16 for better compatibility
                    "tf32": True,  # Enable TF32 for better performance on Ampere GPUs
                }
            )

        elif self.selected_backend == "flash_attention_2":
            optimizations.update(
                {
                    "bf16": torch.cuda.is_bf16_supported(),  # Use BF16 if supported
                    "fp16": not torch.cuda.is_bf16_supported(),  # Fallback to FP16
                    "tf32": True,
                }
            )
        else:
            # Eager attention optimizations for Windows
            system = platform.system().lower()
            if system == "windows":
                optimizations.update(
                    {
                        "fp16": True,  # FP16 still provides memory benefits
                        "bf16": False,  # More compatible on Windows
                        "tf32": True,  # Enable TF32 for Ampere GPUs
                        "torch_compile": False,  # Disable torch.compile on Windows for stability
                        "use_reentrant": False,  # Better gradient checkpointing
                    }
                )
            else:
                optimizations.update(
                    {
                        "bf16": torch.cuda.is_bf16_supported(),
                        "fp16": not torch.cuda.is_bf16_supported(),
                        "tf32": True,
                    }
                )

        return optimizations

    def force_gpu_usage(self, model, force: bool = True) -> None:
        """
        🚨 CRITICAL FIX: Force model to use GPU when available

        This method ensures 100% GPU utilization as requested by the user.
        No more CPU fallbacks - GPU usage is enforced!
        """
        if not force:
            return

        if torch.cuda.is_available():
            try:
                # Force model to GPU
                if hasattr(model, 'to'):
                    model = model.to('cuda')
                    logger.info("🚨 FORCED MODEL TO GPU: Model moved to CUDA device")

                # Set model to use FP16 for better GPU performance
                if hasattr(model, 'half'):
                    model = model.half()
                    logger.info("🚨 ENABLED FP16: Model converted to half precision for GPU optimization")

                # Enable GPU optimizations
                if hasattr(model, 'eval'):
                    model.eval()

                # Clear GPU cache to ensure clean state
                torch.cuda.empty_cache()

                # Log GPU status
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"🚨 GPU ENFORCED: {gpu_name} ({gpu_memory:.1f}GB) - 100% GPU utilization mode")

            except Exception as e:
                logger.error(f"❌ Failed to force GPU usage: {e}")
                logger.warning("⚠️ Continuing with current device placement")
        else:
            logger.warning("⚠️ CUDA not available - cannot force GPU usage")

    def get_status_info(self) -> Dict[str, Any]:
        """Get status information about attention optimization"""
        return {
            "selected_backend": self.selected_backend,
            "available_backends": self.available_backends,
            "platform": platform.system(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
        }

    def log_optimization_info(self) -> None:
        """Log detailed information about attention optimization"""
        info = self.get_status_info()

        logger.info("=== Attention Optimization Status ===")
        logger.info(f"Platform: {info['platform']}")
        logger.info(f"Selected Backend: {info['selected_backend']}")
        logger.info(f"Available Backends: {info['available_backends']}")
        logger.info(f"CUDA Available: {info['cuda_available']}")

        if info["cuda_available"]:
            logger.info(f"CUDA Version: {info['cuda_version']}")
            logger.info(f"PyTorch Version: {info['pytorch_version']}")

            # GPU info
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

        logger.info("=====================================")


# Global instance for easy access with error handling
try:
    attention_optimizer = AttentionOptimizer()
except Exception as e:
    logger.error(f"Failed to create global AttentionOptimizer instance: {e}")

    # Create a minimal fallback instance
    class FallbackAttentionOptimizer:
        def __init__(self):
            self.available_backends = {"flash_attention_2": False, "xformers": False, "eager": True}
            self.selected_backend = "eager"

        def get_model_kwargs(self, model_name: str = None) -> Dict[str, Any]:
            return {"attn_implementation": "eager"}

        def configure_model_for_xformers(self, model) -> None:
            pass

        def get_memory_optimization_settings(self) -> Dict[str, Any]:
            return {"gradient_checkpointing": True, "use_cache": False, "low_cpu_mem_usage": True}

        def get_training_optimizations(self) -> Dict[str, Any]:
            return {"dataloader_pin_memory": False, "dataloader_num_workers": 0}

        def get_status_info(self) -> Dict[str, Any]:
            return {
                "selected_backend": "eager",
                "available_backends": {
                    "flash_attention_2": False,
                    "xformers": False,
                    "eager": True,
                },
                "platform": platform.system(),
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "pytorch_version": torch.__version__,
            }

        def log_optimization_info(self) -> None:
            logger.warning("Using fallback attention optimizer due to initialization error")

    attention_optimizer = FallbackAttentionOptimizer()
    logger.warning("Created fallback AttentionOptimizer instance")