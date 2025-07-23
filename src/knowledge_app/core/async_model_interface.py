"""
Async Model Interface for Intelligent MCQ Generation

This module provides an interface between the intelligent MCQ generator
and the loaded models, handling text generation in a clean, async way.
"""

import logging
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# 🚨 CRITICAL FIX: Import attention optimizer for GPU enforcement
try:
    from .attention_optimizer import attention_optimizer
except ImportError:
    logger.warning("⚠️ Could not import attention_optimizer, GPU enforcement may not work")
    attention_optimizer = None


class AsyncModelInterface:
    """Interface for async model operations"""

    def __init__(self, model, tokenizer):
        """
        Initialize the model interface with FORCED GPU USAGE.

        Args:
            model: Loaded transformer model
            tokenizer: Loaded tokenizer

        🚨 CRITICAL FIX: Forces GPU usage when CUDA is available!
        """
        self.model = model
        self.tokenizer = tokenizer

        # 🚨 CRITICAL FIX: Force GPU usage if available
        if attention_optimizer and torch.cuda.is_available():
            logger.info("🚨 FORCING GPU USAGE: Moving model to GPU...")
            attention_optimizer.force_gpu_usage(self.model, force=True)

            # Ensure tokenizer also uses GPU when possible
            if hasattr(self.tokenizer, 'to'):
                try:
                    self.tokenizer = self.tokenizer.to('cuda')
                    logger.info("🚨 TOKENIZER MOVED TO GPU")
                except:
                    logger.warning("⚠️ Could not move tokenizer to GPU")

        self.device = next(model.parameters()).device

        # Log final device placement
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚨 ENFORCED GPU USAGE: Model on {self.device} ({gpu_name})")
        else:
            logger.info(f"✅ AsyncModelInterface initialized on device: {self.device}")

    async def generate_text(
        self, prompt: str, max_tokens: int = 600, temperature: float = 0.7, top_p: float = 0.9
    ) -> str:
        """
        Generate text from prompt using the loaded model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Generated text
        """
        try:
            logger.debug(f"🔄 Generating text for prompt: {prompt[:100]}...")

            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.device)

            # Generate with proper parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the original prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()

            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.debug(f"✅ Generated {len(response)} characters")
            return response

        except Exception as e:
            logger.error(f"❌ Text generation failed: {e}")
            raise

    def is_ready(self) -> bool:
        """Check if the model interface is ready for generation"""
        return self.model is not None and self.tokenizer is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        try:
            return {
                "model_type": type(self.model).__name__,
                "device": str(self.device),
                "vocab_size": self.tokenizer.vocab_size,
                "model_max_length": getattr(self.tokenizer, "model_max_length", "unknown"),
                "pad_token": self.tokenizer.pad_token,
                "eos_token": self.tokenizer.eos_token,
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}