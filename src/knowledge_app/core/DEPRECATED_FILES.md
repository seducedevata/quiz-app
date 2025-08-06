# DEPRECATED FILES - DO NOT USE

The following files are deprecated and should not be used. They are kept for reference only.

## Deprecated Model Managers (Use UnifiedInferenceManager instead)
- `global_model_singleton.py` - Use `unified_inference_manager.py`
- `offline_mcq_generator.py` - Use `unified_inference_manager.py`
- `online_mcq_generator.py` - Use `unified_inference_manager.py`

## Deprecated Prompt Systems (Use unified_prompt_builder.py instead)
- `inquisitor_prompt_legacy.py` - Use `unified_prompt_builder.py`
- `inquisitor_prompt_new.py` - Use `unified_prompt_builder.py`
- `simplified_prompt_system.py` - Use `unified_prompt_builder.py`
- `intelligent_prompt_generator.py` - Use `unified_prompt_builder.py`

## Migration Guide
1. Replace all calls to GlobalModelSingleton with UnifiedInferenceManager
2. Replace all prompt generation with UnifiedPromptBuilder
3. Use the unified interface for all model operations

## Files to be removed in next major version:
- global_model_singleton.py
- offline_mcq_generator.py (legacy)
- online_mcq_generator.py (legacy)
- inquisitor_prompt_legacy.py
- inquisitor_prompt_new.py
- simplified_prompt_system.py
