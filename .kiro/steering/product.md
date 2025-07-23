# Knowledge App

The Knowledge App is an educational application designed to generate quiz questions on various topics. It features a modern web-based UI built with PyQt5's QtWebEngine.

## Core Features

- **Quiz Generation**: Creates multiple-choice questions (MCQs) on user-specified topics
- **Difficulty Levels**: Supports various difficulty levels including expert mode
- **Question Types**: Generates different types of questions (conceptual, numerical, mixed)
- **Online/Offline Modes**: Works with both online API providers and local models
- **Document Processing**: Processes uploaded documents (PDFs, text) for quiz generation
- **Training System**: Includes a standalone training module for fine-tuning models

## Key Components

- **MCQ Generator**: Core question generation system
- **Unified Inference Manager**: Manages different inference backends
- **Ollama Integration**: Local model inference using Ollama
- **Batch Two Model Pipeline**: Advanced pipeline for expert-level questions
- **Dynamic Timeout System**: Adapts timeouts based on model and prompt complexity
- **Training Module**: Standalone system for model fine-tuning

## User Experience

- **Web-based UI**: Clean, modern interface using QtWebEngine
- **Token Streaming**: Real-time display of generated content
- **Progress Indicators**: Visual feedback during long operations
- **User Preferences**: Remembers user settings and model preferences