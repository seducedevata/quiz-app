# Knowledge App Training Module

🚀 **Standalone Training System** - Completely separated from the main Knowledge App for focused training operations.

## 📁 Directory Structure

```
training/
├── __init__.py                    # Main training module entry point
├── main.py                        # Standalone CLI entry point
├── README.md                      # This documentation
├── core/                          # Core training logic
│   ├── __init__.py
│   ├── training_orchestrator.py   # Main training orchestration
│   ├── training_manager.py        # Training session management
│   ├── training_controller.py     # Training flow control
│   ├── training_service.py        # Training services
│   ├── training_worker.py         # Training worker threads
│   ├── training_thread.py         # Threading utilities
│   ├── training_callbacks.py      # Training progress callbacks
│   ├── training_metrics.py        # Training metrics tracking
│   ├── training_estimator.py      # Training time/performance estimation
│   ├── training_data_processor.py # Training data processing
│   ├── training_management.py     # Training lifecycle management
│   ├── auto_training_manager.py   # Automated training management
│   └── golden_path_trainer.py     # Optimized training path
├── data/                          # Training data management
│   ├── __init__.py
│   ├── processors/               # Data processing utilities
│   └── processed_training/       # Processed training datasets
├── tests/                         # Training tests and validation
│   ├── __init__.py
│   ├── check_training_data.py    # Data quality validation
│   ├── check_real_training_data.py # Real data validation
│   ├── comprehensive_training_test.py # Full training tests
│   ├── test_single_book_output.py # Single book processing test
│   ├── test_enhanced_processor.py # Enhanced processor tests
│   ├── test_one_book.py          # One book validation
│   └── run_unified_test_suite.py # Complete test suite
├── scripts/                       # Training utility scripts
│   ├── __init__.py
│   ├── generate_quality_training_data.py # Quality data generation
│   ├── regenerate_training_data.py # Data regeneration
│   └── trainingloc.py            # Training code analysis
├── config/                        # Training configurations
│   ├── __init__.py
│   └── training_config.yaml      # Training parameters
└── models/                        # Training models and adapters
    ├── __init__.py
    ├── lora_adapters/            # LoRA adapter storage
    └── lora_adapters_mistral/    # Mistral-specific adapters
```

## 🚀 Quick Start

### Standalone Training CLI

```bash
# Show help
python training/main.py --help

# Start model training with default settings
python training/main.py train

# Custom training configuration
python training/main.py train --epochs 3 --batch-size 4 --learning-rate 0.0001

# Test single book processing
python training/main.py test --single-book data/uploaded_books/sample.txt

# Run comprehensive tests
python training/main.py test --comprehensive

# Generate high-quality training data
python training/main.py generate-data --books data/uploaded_books/ --quality

# Analyze training code
python training/main.py scripts --locate
```

### Import as Module

```python
# Import main training components
from training import TrainingOrchestrator, TrainingManager, TrainingController

# Import specific training utilities
from training.core.training_estimator import TrainingEstimator
from training.core.training_callbacks import TrainingCallback
from training.core.golden_path_trainer import GoldenPathTrainer

# Use training services
training_manager = TrainingManager()
success = training_manager.start_training(config)
```

## 🎯 Key Features

### ✅ Complete Separation
- **Standalone Module**: Works independently from main Knowledge App
- **Clean Architecture**: Organized by functionality (core, tests, scripts, config)
- **Modular Design**: Each component can be used independently
- **CLI Interface**: Command-line interface for headless training

### 🚀 Training Capabilities
- **GPU Optimization**: Optimized for GPU training with 7B models
- **LoRA Fine-tuning**: Efficient parameter-efficient training
- **Multiple Backends**: Support for LM Studio, Ollama, and direct model training
- **Progress Tracking**: Real-time training progress and estimation
- **Quality Control**: Advanced data quality validation and filtering

### 🧪 Testing Framework
- **Comprehensive Tests**: Full training pipeline validation
- **Single Book Testing**: Quick validation on individual books
- **Data Quality Checks**: Automated data quality validation
- **Performance Benchmarks**: Training performance measurement

### 📊 Data Processing
- **Enhanced Processing**: Advanced document processing pipeline
- **Quality Generation**: High-quality training data generation
- **Batch Processing**: Efficient processing of multiple books
- **Cache Management**: Intelligent caching for processed data

## 🔧 Configuration

### Training Configuration (`training/config/training_config.yaml`)

```yaml
# Basic training settings
epochs: 2
batch_size: 8
learning_rate: 0.0002
force_gpu: true
fp16: true

# LoRA settings
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1

# Advanced settings
gradient_accumulation_steps: 2
warmup_steps: 50
max_grad_norm: 1.0
dataloader_num_workers: 4
```

## 📋 Training Presets

The system includes optimized training presets:

- **Quick Training**: Fast experimentation (1 epoch, ~5-15 min)
- **Standard Training**: Balanced results (2 epochs, ~15-30 min)
- **High Quality Training**: Best results (3 epochs, ~30-60 min)
- **GPU Optimized**: Maximum GPU utilization (recommended, ~20-45 min)

## 🎓 Usage Examples

### 1. Basic Model Training

```bash
# Quick training session
python training/main.py train --epochs 1 --batch-size 8

# High-quality training
python training/main.py train --epochs 3 --batch-size 4 --learning-rate 0.0001
```

### 2. Data Quality Testing

```bash
# Test single book processing
python training/main.py test --single-book data/uploaded_books/physics_book.txt

# Check all training data quality
python training/main.py test --check-data

# Run comprehensive validation
python training/main.py test --comprehensive
```

### 3. Data Generation

```bash
# Generate quality training data
python training/main.py generate-data --quality

# Process specific book directory
python training/main.py generate-data --books /path/to/books --output /path/to/output
```

### 4. Training Analysis

```bash
# Analyze training code distribution
python training/main.py scripts --locate

# Regenerate training datasets
python training/main.py scripts --regenerate
```

## 🔍 Integration with Main App

The training module is still integrated with the main Knowledge App:

- **Web UI Integration**: Training can be started from the web interface
- **Progress Updates**: Real-time progress updates to the main app
- **Configuration Sync**: Training settings synchronized with main app
- **Resource Sharing**: Shared access to uploaded books and processed data

## 🚨 Important Notes

### Performance Optimization
- **GPU Required**: Training is optimized for GPU usage
- **Memory Management**: Efficient memory usage for large models
- **Batch Processing**: Optimized batch sizes for different hardware
- **Caching Strategy**: Intelligent caching to avoid reprocessing

### Data Quality
- **Educational Focus**: Optimized for educational content (JEE, Physics, Math)
- **Mathematical Content**: Special handling for mathematical expressions
- **Structure Preservation**: Maintains document structure and context
- **Quality Validation**: Automated quality scoring and filtering

## 📈 Monitoring and Logging

Training operations are fully logged with:
- **Progress Tracking**: Real-time progress updates
- **Performance Metrics**: Training speed and efficiency metrics
- **Error Handling**: Comprehensive error logging and recovery
- **Resource Monitoring**: GPU/CPU utilization tracking

## 🔄 Migration Notes

Files moved from original locations:
- `src/knowledge_app/core/training_*.py` → `training/core/`
- Root training test files → `training/tests/`
- Training scripts → `training/scripts/`
- Training config → `training/config/`
- LoRA adapters → `training/models/`

Import statements updated throughout the codebase to use the new `training.` module prefix.

---

**🎉 Training Module is now completely separated and ready for independent operation!** 