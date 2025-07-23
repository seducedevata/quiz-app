# Knowledge App Technical Stack

## Core Technologies

- **Python**: Primary programming language (3.8+)
- **PyQt5**: GUI framework
- **QtWebEngine**: Web-based UI rendering
- **Ollama**: Local model inference backend
- **LLM Models**: Local models for question generation (llama3.1, deepseek, etc.)
- **HTML/CSS/JS**: Frontend UI components

## Key Libraries

- **PyQtWebEngine**: Web content rendering
- **requests**: API communication
- **torch**: Optional for advanced ML features
- **transformers**: Hugging Face transformers library
- **sentence-transformers**: Semantic search capabilities
- **llama-cpp-python**: For GGUF model support
- **pdfplumber**: PDF text extraction
- **Pillow/OpenCV**: Image processing
- **pytesseract**: OCR capabilities
- **faiss-cpu**: Vector similarity search
- **cryptography**: Secure API key storage

## Build & Development

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# For pinned versions
pip install -r requirements-pinned.txt
```

### Running the Application

```bash
# Start the application
python main.py

# Force GPU mode
python force_gpu_config.bat  # Windows
```

### Testing

```bash
# Run all tests
python tests/run_clean.py

# Run specific test category
python tests/run_tests.py core_functionality
```

### Training Module

```bash
# Start model training
python training/main.py train

# Test single book processing
python training/main.py test --single-book data/uploaded_books/sample.txt
```

## Configuration

- **config/**: Application configuration files
- **user_data/**: User settings and preferences
- **training/config/**: Training-specific configurations

## Performance Considerations

- **GPU Acceleration**: Optimized for GPU when available
- **Dynamic Timeouts**: Adjusts based on model complexity
- **Memory Management**: Efficient resource utilization
- **Caching**: Intelligent caching for processed data