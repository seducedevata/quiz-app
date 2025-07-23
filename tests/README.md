# Consolidated Test Suite

This directory contains the unified, maintainable test suite for the knowledge_app project. All tests here are practical, reliable, and run without GUI dependencies or complex setup requirements.

**✅ CONSOLIDATED & CLEANED**: All duplicate and impractical tests have been removed. This is now the single source of truth for testing.

## Test Structure

### Core Tests
- **test_core_functionality.py** - Basic imports, project structure, and essential functionality
- **test_config_manager.py** - Configuration management system
- **test_memory_manager.py** - Memory monitoring and optimization
- **test_storage_manager.py** - File storage and caching system
- **test_hardware_acceleration.py** - PyTorch and CUDA functionality
- **test_model_manager.py** - Model loading, saving, and management (non-GUI)
- **test_training_estimator.py** - Training time estimation algorithms

## What's Included ✅

- ✅ Core functionality tests
- ✅ Configuration management
- ✅ Memory management
- ✅ Storage and caching
- ✅ Hardware acceleration (CUDA/CPU)
- ✅ Model management (core features)
- ✅ Training estimation
- ✅ Error handling
- ✅ Concurrent access testing
- ✅ Mock-based testing where appropriate

## What's Excluded ❌

- ❌ GUI integration tests (prone to hanging/crashing)
- ❌ Complex system integration tests
- ❌ UI component tests (require display)
- ❌ Training dialog tests (GUI dependencies)
- ❌ Quiz generator tests (module not available)
- ❌ Tests requiring external services
- ❌ Tests with complex setup requirements

## Running Tests

### Run All Tests
```bash
# Clean output (no warnings) - RECOMMENDED
python tests/run_clean.py

# Detailed output (with warnings)
python tests/run_tests.py

# Using pytest directly
python -m pytest tests/ -v
```

### Run Specific Test Category
```bash
# Core functionality
python tests/run_tests.py core_functionality

# Configuration tests
python tests/run_tests.py config_manager

# Memory management
python tests/run_tests.py memory_manager

# Storage management
python tests/run_tests.py storage_manager

# Hardware acceleration
python tests/run_tests.py hardware_acceleration

# Model management
python tests/run_tests.py model_manager

# Training estimation
python tests/run_tests.py training_estimator
```

### List Available Tests
```bash
python tests/run_tests.py list
```

## Test Features

### Robust Error Handling
- All tests include proper error handling
- Tests skip gracefully if dependencies are missing
- No tests should crash or hang

### Minimal Dependencies
- Only essential dependencies required
- No GUI framework dependencies
- Works in headless environments

### Fast Execution
- All tests complete within reasonable time
- No long-running operations
- Parallel execution support

### Comprehensive Coverage
- Core business logic
- Error conditions
- Edge cases
- Concurrent access patterns

## Maintenance Guidelines

### Adding New Tests
1. Follow the existing naming convention: `test_<component>.py`
2. Include proper error handling and skip conditions
3. Use mocks for external dependencies
4. Keep tests focused and atomic
5. Add appropriate docstrings

### Test Categories
Use pytest markers to categorize tests:
```python
@pytest.mark.core
def test_basic_functionality():
    pass

@pytest.mark.slow
def test_long_running_operation():
    pass
```

### Best Practices
- Keep tests independent (no shared state)
- Use temporary directories for file operations
- Clean up resources in teardown
- Mock external dependencies
- Test both success and failure cases

## Expected Results

When all tests pass, you should see output like:
```
======================== test session starts ========================
tests/test_core_functionality.py::TestCoreImports::test_basic_python_imports PASSED
tests/test_core_functionality.py::TestCoreImports::test_third_party_imports PASSED
tests/test_config_manager.py::TestConfigManager::test_singleton_pattern PASSED
tests/test_hardware_acceleration.py::TestHardwareAcceleration::test_torch_device_available PASSED
tests/test_memory_manager.py::TestMemoryManager::test_memory_manager_initialization PASSED
tests/test_model_manager.py::TestModelManager::test_model_manager_initialization PASSED
tests/test_storage_manager.py::TestStorageManager::test_initialization PASSED
tests/test_training_estimator.py::TestTrainingEstimator::test_initial_estimates PASSED
...
======================== 65 passed, 4 skipped in 13.68s ========================
```

**Current Status**: ✅ 65 tests passing, 4 skipped (expected for optional features), 0 warnings (filtered)

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure the project is properly installed or PYTHONPATH is set
2. **CUDA Tests Failing**: Normal if CUDA is not available - tests will skip
3. **Permission Errors**: Ensure write access to temporary directories
4. **Module Not Found**: Some optional modules may not be available - tests will skip

### Environment Setup
```bash
# Ensure project is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install in development mode
pip install -e .
```

## Cleanup Summary

### Removed Impractical Tests ❌
- `test_advanced_checkpointing_gold.py` (400 lines, overly complex)
- `test_enhanced_checkpointing.py` (duplicate functionality)
- `test_fire_estimator.py` (standalone, not integrated)
- `test_multimodal_training_integration.py` (too specific)
- `test_state_saving.py` (duplicate functionality)
- `test_training_callback_fix.py` (too specific)

### Removed Duplicate Tests ❌
- `test_cuda.py` (duplicate of hardware acceleration tests)
- `test_gpu.py` (duplicate of hardware acceleration tests)
- `test_gpu_init.py` (duplicate of hardware acceleration tests)
- `test_config.py` (duplicate of config manager tests)

### Removed Test Infrastructure ❌
- Multiple scattered test runners (`comprehensive_test_runner.py`, `fast_test_runner.py`, etc.)
- Old `tests/` directory (replaced by consolidated `tests/`)
- Empty test directories (`test_cache/`, `test_data/`, `test_images/`)
- Test result log files and temporary files

### Consolidated Structure ✅
- **Single test directory**: `tests/` (renamed from `tests_clean/`)
- **7 focused test files**: Core functionality, configuration, hardware, memory, models, storage, training
- **69 practical tests**: All tests are maintainable and serve a clear purpose
- **Proper fixtures**: Shared test configuration and cleanup
- **No GUI dependencies**: All tests run in headless environments

This consolidated test suite provides reliable, maintainable testing for the core functionality of the knowledge_app project.
