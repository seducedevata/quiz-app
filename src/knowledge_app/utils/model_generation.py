"""
Model generation utilities for the Knowledge App.

This module provides functions for generating text with AI models,
with proper error handling, timeout functionality, and post-processing.
"""

import logging
import torch
from knowledge_app.utils.timeout import timeout
from knowledge_app.utils.question_parser import extract_mcq_blocks

logger = logging.getLogger(__name__)


@timeout(60, "Model generation timed out")
def generate_with_model(
    model,
    tokenizer,
    prompt,
    max_new_tokens=350,
    temperature=0.5,
    top_p=0.85,
    top_k=50,
    repetition_penalty=1.1,
):
    """
    Generate text using a loaded model with timeout and error handling.

    Args:
        model: The loaded model
        tokenizer: The tokenizer for the model
        prompt: The input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens

    Returns:
        str: The generated text
    """
    if not model or not tokenizer:
        logger.error("Model or tokenizer not provided")
        return "Error: Model not loaded or failed to load."

    try:
        # Tokenize input with proper attention mask handling
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,  # No padding for single prompt
            truncation=True,
            max_length=2048,
            return_attention_mask=True,
        )
        input_ids = inputs["input_ids"].to(model.device)

        # Handle attention mask properly for models where pad_token == eos_token
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            # Create explicit attention mask
            input_length = input_ids.shape[1]
            attention_mask = torch.ones((1, input_length), dtype=torch.long, device=model.device)
        else:
            attention_mask = inputs["attention_mask"].to(model.device)

        logger.debug(f"Generating with prompt (length: {len(prompt)})")

        # Generate with no_grad for efficiency
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Extract only the newly generated tokens
        prompt_tokens_len = input_ids.shape[1]
        generated_tokens = outputs[0][prompt_tokens_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return generated_text

    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        return f"Error during generation: {str(e)}"


def process_generated_text(text):
    """
    Process generated text to extract and clean up MCQ blocks.

    Args:
        text: The generated text

    Returns:
        str: The processed text containing a clean MCQ block
    """
    if not text:
        return text

    # Check for custom stop sequence
    custom_stop_sequence = "{{END_OF_MCQ_BLOCK}}"
    stop_marker_index = text.find(custom_stop_sequence)
    if stop_marker_index != -1:
        text = text[: stop_marker_index + len(custom_stop_sequence)].strip()
        logger.debug("Truncated output at custom stop sequence")
        # Remove the stop sequence itself
        text = text.replace(custom_stop_sequence, "").strip()
        return text

    # Try to extract the first complete MCQ block
    blocks = extract_mcq_blocks(text)
    if blocks:
        logger.debug("Extracted first complete MCQ block")
        return blocks[0]

    logger.warning("Could not find complete MCQ block. Using full output.")
    return text