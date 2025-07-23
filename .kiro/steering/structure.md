# Knowledge App Project Structure

## Core Directories

- **src/knowledge_app/**: Main application source code
  - **core/**: Core business logic and functionality
    - **mcq_generator.py**: Base MCQ generation interface
    - **offline_mcq_generator.py**: Local model MCQ generation
    - **ollama_model_inference.py**: Ollama integration
    - **unified_inference_manager.py**: Manages different inference backends
    - **inquisitor_prompt.py**: Prompt engineering and templates
  - **webengine_app.py**: Web-based UI application
  - **loading_screen.py**: Application splash screen
  - **ui/**: UI components and assets
  - **utils/**: Utility functions and helpers

- **training/**: Standalone training module
  - **core/**: Core training functionality
  - **data/**: Training data management
  - **models/**: Model storage and adapters
  - **config/**: Training configurations
  - **scripts/**: Training utility scripts

- **tests/**: Consolidated test suite
  - **conftest.py**: Test configuration and fixtures
  - **test_core_functionality.py**: Core functionality tests
  - **test_config_manager.py**: Configuration tests
  - **test_model_manager.py**: Model management tests

## Data Directories

- **config/**: Application configuration files
  - **api_keys.json**: API key storage
  - **app_settings.json**: Application settings
  - **unified_config.json**: Unified configuration

- **user_data/**: User-specific data
  - **user_settings.json**: User preferences
  - **question_history.sqlite**: Question history database

- **data/**: Application data
  - **cache/**: Cached data
  - **training_output/**: Training output files

- **uploaded_books/**: User uploaded documents
  - **[document files]**: PDF and text documents

## Test Files

- **test_*.py**: Individual test files for specific features
  - **test_expert_mode_fix.py**: Expert mode fixes
  - **test_streaming_fixes.py**: Streaming functionality tests
  - **test_ui_fixes.py**: UI-related tests

## Code Organization Patterns

- **Core Logic Separation**: Business logic separated from UI
- **Manager Classes**: Singleton managers for key subsystems
- **Factory Pattern**: For creating appropriate generators/models
- **Strategy Pattern**: For different question generation strategies
- **Observer Pattern**: For progress updates and notifications

## Naming Conventions

- **Classes**: PascalCase (e.g., `MCQGenerator`, `OllamaModelInference`)
- **Functions/Methods**: snake_case (e.g., `generate_mcq`, `is_available`)
- **Variables**: snake_case (e.g., `model_name`, `is_initialized`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_TIMEOUT`, `MAX_QUESTIONS`)
- **Private Members**: Prefixed with underscore (e.g., `_initialize`, `_parse_response`)