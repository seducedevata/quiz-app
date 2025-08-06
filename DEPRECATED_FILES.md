# Deprecated Files & Systems

## Overview
This document lists deprecated files and systems that have been superseded by unified components to eliminate architectural conflicts.

## Deprecated Configuration Systems
- `app_config.py` - Replaced by `unified_config_manager.py`
- `proper_config_manager.py` - Replaced by `unified_config_manager.py`
- `config_migration_manager.py` - Migration tool, no longer needed post-migration

## Deprecated Model Management
- `global_model_singleton.py` - Replaced by `unified_inference_manager.py`
- Legacy model selection patterns in individual components

## Deprecated JSON Parsing
- Individual JSON parsers in `offline_mcq_generator.py`, `online_mcq_generator.py`
- `enhanced_mcq_parser.py` - Replaced by unified JSON parsing

## Deprecated Prompt Systems
- `inquisitor_prompt_legacy.py` - Replaced by `unified_prompt_builder.py`
- `simplified_prompt_system.py` - Replaced by `unified_prompt_builder.py`

## Migration Status
✅ UnifiedConfigManager - Active replacement for all configuration access
✅ UnifiedInferenceManager - Active replacement for model management
✅ UnifiedPromptBuilder - Active replacement for prompt generation

## Usage Guidelines
- All new code should use the unified components
- Legacy systems are maintained for backward compatibility during transition
- Gradual migration path provided through compatibility layers
