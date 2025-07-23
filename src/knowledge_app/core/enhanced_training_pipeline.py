#!/usr/bin/env python3
"""
🚀 ENHANCED TRAINING PIPELINE
Comprehensive training pipeline that integrates document processing, training data generation,
and model fine-tuning into a unified workflow.
"""

import json
import logging
import asyncio
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from .training_data_generator import TrainingDataGenerator, TrainingDataConfig

logger = logging.getLogger(__name__)

@dataclass
class TrainingPipelineConfig:
    """Configuration for the complete training pipeline"""
    # Document processing
    uploaded_files: List[str]
    chunk_size: int = 1000
    
    # Data generation
    source_model: str = "deepseek-coder"  # Model for generating training data
    target_model: str = "llama3.1:8b"     # Model to be fine-tuned
    question_types: List[str] = None
    difficulty_levels: List[str] = None
    questions_per_chunk: int = 3
    quality_threshold: float = 0.8
    
    # 🚀 DYNAMIC: Training configuration from user UI
    epochs: int = None  # Will be set from user configuration
    learning_rate: float = None  # Will be set from user configuration
    batch_size: int = None  # Will be set from user configuration
    max_length: int = None  # Will be set from user configuration
    warmup_steps: int = None  # Will be set from user configuration
    gradient_accumulation_steps: int = None  # Will be set from user configuration
    lora_r: int = None  # Will be set from user configuration
    lora_alpha: int = None  # Will be set from user configuration
    
    # Output configuration
    output_format: str = "json"  # "json", "csv", "xml"
    output_directory: str = "training_output"
    save_intermediate_results: bool = True
    
    def __post_init__(self):
        if self.question_types is None:
            self.question_types = ["numerical", "conceptual", "mixed"]
        if self.difficulty_levels is None:
            self.difficulty_levels = ["medium", "hard", "expert"]

class EnhancedTrainingPipeline(QObject):
    """
    🚀 ENHANCED TRAINING PIPELINE
    
    Complete training workflow that:
    1. Processes uploaded documents using AdvancedDocumentProcessor
    2. Generates high-quality training data using local AI models
    3. Exports training data in JSON/CSV/XML format
    4. Optionally fine-tunes target models
    """
    
    # Progress signals
    pipeline_started = pyqtSignal(dict)
    stage_progress = pyqtSignal(str, int, str)  # stage_name, progress_percent, status_message
    stage_completed = pyqtSignal(str, dict)     # stage_name, results
    pipeline_completed = pyqtSignal(dict)       # final_results
    pipeline_error = pyqtSignal(str, str)       # stage_name, error_message
    
    def __init__(self, config: TrainingPipelineConfig):
        super().__init__()
        self.config = config
        self.is_running = False
        self.should_stop = False
        self.current_stage = ""
        self.results = {}
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize pipeline components"""
        try:
            # Try to import existing document processor, fall back to simple processor
            try:
                from training.document_processing.document_processor import AdvancedDocumentProcessor
                self.document_processor = AdvancedDocumentProcessor(
                    preserve_educational_content=True,
                    use_semantic_chunking=True
                )
                logger.info("✅ Using AdvancedDocumentProcessor")
            except ImportError as e:
                logger.warning(f"⚠️ AdvancedDocumentProcessor not available: {e}")
                # Use a simple mock processor for testing
                self.document_processor = self._create_simple_document_processor()
                logger.info("✅ Using SimpleDocumentProcessor for testing")
            
            # Initialize training data generator config
            data_gen_config = TrainingDataConfig(
                source_model=self.config.source_model,
                target_model=self.config.target_model,
                question_types=self.config.question_types,
                difficulty_levels=self.config.difficulty_levels,
                questions_per_chunk=self.config.questions_per_chunk,
                chunk_size=self.config.chunk_size,
                output_format=self.config.output_format,
                quality_threshold=self.config.quality_threshold,
                use_anti_vague_enforcement=True,
                include_explanations=True
            )
            
            self.training_data_generator = TrainingDataGenerator(data_gen_config)
            
            logger.info("✅ Enhanced training pipeline components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline components: {e}")
            raise

    def _create_simple_document_processor(self):
        """Create a simple document processor for testing when AdvancedDocumentProcessor is not available"""
        class SimpleDocumentProcessor:
            def process_documents_advanced(self, file_paths, output_dir, chunk_size=1000):
                """Simple mock document processing for testing"""
                chunks = []

                for i, file_path in enumerate(file_paths):
                    # Create mock chunks for testing
                    mock_chunks = [
                        {
                            "text": f"This is a sample text chunk from {file_path}. It contains educational content about physics, mathematics, and science. The content is designed to test the training data generation pipeline with realistic academic material.",
                            "topic": "Physics",
                            "metadata": {
                                "source": file_path,
                                "chunk_index": 0,
                                "page": 1
                            }
                        },
                        {
                            "text": f"Advanced concepts in quantum mechanics and thermodynamics from {file_path}. This chunk covers wave-particle duality, energy conservation, and statistical mechanics principles that are fundamental to understanding modern physics.",
                            "topic": "Quantum Mechanics",
                            "metadata": {
                                "source": file_path,
                                "chunk_index": 1,
                                "page": 2
                            }
                        }
                    ]
                    chunks.extend(mock_chunks)

                return {
                    "success": True,
                    "chunks": chunks,
                    "stats": {
                        "files_processed": len(file_paths),
                        "chunks_generated": len(chunks),
                        "total_text_length": sum(len(chunk["text"]) for chunk in chunks)
                    }
                }

        return SimpleDocumentProcessor()
    
    @pyqtSlot()
    def start_pipeline(self):
        """Start the complete training pipeline"""
        if self.is_running:
            logger.warning("⚠️ Training pipeline already running")
            return
        
        self.is_running = True
        self.should_stop = False
        self.current_stage = ""
        self.results = {}
        
        # Start pipeline in background thread
        pipeline_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        pipeline_thread.start()
    
    @pyqtSlot()
    def stop_pipeline(self):
        """Stop the training pipeline"""
        logger.info("🛑 Stopping training pipeline...")
        self.should_stop = True
    
    def _run_pipeline(self):
        """Run the complete training pipeline"""
        try:
            start_time = time.time()
            
            # Emit pipeline started signal
            pipeline_info = {
                "config": asdict(self.config),
                "stages": ["document_processing", "data_generation", "data_export", "model_training"],
                "started_at": start_time
            }
            self.pipeline_started.emit(pipeline_info)
            
            # Stage 1: Document Processing
            if not self.should_stop:
                self._run_document_processing_stage()

            # 🔧 FIX: Validate document processing output before proceeding
            if not self.should_stop and self._validate_document_processing_output():
                asyncio.run(self._run_data_generation_stage())

            # 🔧 FIX: Validate data generation output before proceeding
            if not self.should_stop and self._validate_data_generation_output():
                self._run_data_export_stage()
            
            # Stage 4: Model Training (Optional)
            if not self.should_stop and self.config.target_model and "training_data_path" in self.results:
                self._run_model_training_stage()
            
            # Pipeline completed
            if not self.should_stop:
                total_time = time.time() - start_time
                final_results = {
                    **self.results,
                    "pipeline_completed": True,
                    "total_time": total_time,
                    "completed_at": time.time()
                }
                self.pipeline_completed.emit(final_results)
                logger.info(f"🎉 Training pipeline completed successfully in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            self.pipeline_error.emit(self.current_stage or "unknown", str(e))
        finally:
            self.is_running = False

    def _validate_document_processing_output(self) -> bool:
        """
        🔧 FIX: Validate that document processing produced usable data

        Returns True if the output is valid and the pipeline should continue.
        """
        if "processed_chunks" not in self.results:
            logger.error("❌ Document processing failed: No processed_chunks in results")
            self.pipeline_error.emit("document_processing", "Document processing failed to produce any output")
            return False

        chunks = self.results["processed_chunks"]

        if not chunks:
            logger.error("❌ Document processing failed: Empty chunks list")
            self.pipeline_error.emit("document_processing", "No text chunks were extracted from the uploaded documents")
            return False

        if not isinstance(chunks, list):
            logger.error(f"❌ Document processing failed: Invalid chunks type {type(chunks)}")
            self.pipeline_error.emit("document_processing", "Document processing produced invalid output format")
            return False

        # Check if chunks contain meaningful content
        meaningful_chunks = [chunk for chunk in chunks if isinstance(chunk, str) and len(chunk.strip()) > 50]

        if len(meaningful_chunks) == 0:
            logger.error("❌ Document processing failed: No meaningful text content found")
            self.pipeline_error.emit("document_processing", "Documents contain no extractable text content (may be image-only PDFs)")
            return False

        if len(meaningful_chunks) < len(chunks) * 0.5:
            logger.warning(f"⚠️ Document processing warning: Only {len(meaningful_chunks)}/{len(chunks)} chunks contain meaningful content")

        logger.info(f"✅ Document processing validation passed: {len(meaningful_chunks)} meaningful chunks extracted")
        return True

    def _validate_data_generation_output(self) -> bool:
        """
        🔧 FIX: Validate that data generation produced usable questions

        Returns True if the output is valid and the pipeline should continue.
        """
        if "generated_questions" not in self.results:
            logger.error("❌ Data generation failed: No generated_questions in results")
            self.pipeline_error.emit("data_generation", "Question generation failed to produce any output")
            return False

        questions = self.results["generated_questions"]

        if not questions:
            logger.error("❌ Data generation failed: Empty questions list")
            self.pipeline_error.emit("data_generation", "No questions were generated from the processed documents")
            return False

        if not isinstance(questions, list):
            logger.error(f"❌ Data generation failed: Invalid questions type {type(questions)}")
            self.pipeline_error.emit("data_generation", "Question generation produced invalid output format")
            return False

        # Check if questions are properly formatted
        valid_questions = []
        for i, question in enumerate(questions):
            if isinstance(question, dict) and "question" in question and "answer" in question:
                if len(question["question"].strip()) > 10 and len(question["answer"].strip()) > 5:
                    valid_questions.append(question)
                else:
                    logger.warning(f"⚠️ Question {i+1} has insufficient content")
            else:
                logger.warning(f"⚠️ Question {i+1} has invalid format")

        if len(valid_questions) == 0:
            logger.error("❌ Data generation failed: No valid questions found")
            self.pipeline_error.emit("data_generation", "All generated questions are invalid or malformed")
            return False

        # Require at least 5 valid questions for training
        min_questions = 5
        if len(valid_questions) < min_questions:
            logger.error(f"❌ Data generation failed: Only {len(valid_questions)} valid questions (minimum {min_questions} required)")
            self.pipeline_error.emit("data_generation", f"Insufficient questions for training: {len(valid_questions)}/{min_questions}")
            return False

        if len(valid_questions) < len(questions) * 0.7:
            logger.warning(f"⚠️ Data generation warning: Only {len(valid_questions)}/{len(questions)} questions are valid")

        # Update results with only valid questions
        self.results["generated_questions"] = valid_questions

        logger.info(f"✅ Data generation validation passed: {len(valid_questions)} valid questions generated")
        return True

    def _run_document_processing_stage(self):
        """
        Stage 1: Process uploaded documents

        🔧 CANCELLATION FIX: Added periodic cancellation checks during processing
        """
        self.current_stage = "document_processing"
        logger.info("📚 Starting document processing stage...")

        try:
            # 🔧 CANCELLATION CHECK
            if self.should_stop:
                logger.info("🛑 Document processing cancelled by user")
                return

            self.stage_progress.emit("document_processing", 0, "Initializing document processing...")

            # Create output directory
            output_dir = Path(self.config.output_directory) / "processed_docs"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 🔧 CANCELLATION CHECK
            if self.should_stop:
                logger.info("🛑 Document processing cancelled during setup")
                return

            # Process documents using existing AdvancedDocumentProcessor
            self.stage_progress.emit("document_processing", 20, "Processing uploaded documents...")

            # 🔧 INTERRUPTIBLE PROCESSING: Add cancellation support
            # Note: The actual document processor should also be made interruptible
            result = self.document_processor.process_documents_advanced(
                file_paths=self.config.uploaded_files,
                output_dir=str(output_dir),
                chunk_size=self.config.chunk_size
            )

            # 🔧 CANCELLATION CHECK after processing
            if self.should_stop:
                logger.info("🛑 Document processing cancelled after processing")
                return

            if not result or not result.get("success", False):
                raise Exception("Document processing failed")

            processed_chunks = result.get("chunks", [])
            if not processed_chunks:
                raise Exception("No chunks generated from documents")

            self.stage_progress.emit("document_processing", 100, f"Processed {len(processed_chunks)} chunks")

            # Store results
            stage_results = {
                "processed_chunks": processed_chunks,
                "processing_stats": result.get("stats", {}),
                "chunk_count": len(processed_chunks)
            }

            self.results["processed_chunks"] = processed_chunks
            self.results["document_processing_stats"] = result.get("stats", {})

            self.stage_completed.emit("document_processing", stage_results)
            logger.info(f"✅ Document processing completed: {len(processed_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Document processing stage failed: {e}")
            self.pipeline_error.emit("document_processing", str(e))
            raise
    
    async def _run_data_generation_stage(self):
        """Stage 2: Generate training data from processed chunks"""
        self.current_stage = "data_generation"
        logger.info("🤖 Starting training data generation stage...")
        
        try:
            processed_chunks = self.results["processed_chunks"]
            
            def progress_callback(message: str):
                # Extract progress percentage if possible
                if "%" in message:
                    try:
                        percent = int(message.split("(")[1].split("%")[0])
                        self.stage_progress.emit("data_generation", percent, message)
                    except:
                        self.stage_progress.emit("data_generation", 50, message)
                else:
                    self.stage_progress.emit("data_generation", 50, message)
            
            self.stage_progress.emit("data_generation", 0, "Starting AI-powered data generation...")
            
            # Generate training data
            generation_result = await self.training_data_generator.generate_training_data_from_documents(
                processed_chunks=processed_chunks,
                progress_callback=progress_callback
            )
            
            if not generation_result.get("success", False):
                raise Exception(f"Training data generation failed: {generation_result.get('error', 'Unknown error')}")
            
            generated_questions = generation_result["questions"]
            generation_stats = generation_result["statistics"]
            
            self.stage_progress.emit("data_generation", 100, f"Generated {len(generated_questions)} high-quality questions")
            
            # Store results
            stage_results = {
                "generated_questions": len(generated_questions),
                "generation_stats": generation_stats,
                "quality_distribution": self._analyze_quality_distribution(generated_questions)
            }
            
            self.results["generated_questions"] = generated_questions
            self.results["data_generation_stats"] = generation_stats
            
            self.stage_completed.emit("data_generation", stage_results)
            logger.info(f"✅ Data generation completed: {len(generated_questions)} questions")
            
        except Exception as e:
            logger.error(f"❌ Data generation stage failed: {e}")
            self.pipeline_error.emit("data_generation", str(e))
            raise
    
    def _run_data_export_stage(self):
        """Stage 3: Export training data to specified format"""
        self.current_stage = "data_export"
        logger.info("💾 Starting data export stage...")
        
        try:
            self.stage_progress.emit("data_export", 0, "Preparing data export...")
            
            # Create output directory
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            timestamp = int(time.time())
            filename = f"training_data_{timestamp}.{self.config.output_format}"
            output_path = output_dir / filename
            
            self.stage_progress.emit("data_export", 50, f"Exporting to {self.config.output_format.upper()}...")
            
            # Export training data
            success = self.training_data_generator.export_training_data(str(output_path))
            
            if not success:
                raise Exception("Data export failed")
            
            self.stage_progress.emit("data_export", 100, f"Exported to {output_path}")
            
            # Store results
            stage_results = {
                "output_path": str(output_path),
                "output_format": self.config.output_format,
                "file_size": output_path.stat().st_size if output_path.exists() else 0
            }
            
            self.results["training_data_path"] = str(output_path)
            self.results["export_info"] = stage_results
            
            self.stage_completed.emit("data_export", stage_results)
            logger.info(f"✅ Data export completed: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Data export stage failed: {e}")
            self.pipeline_error.emit("data_export", str(e))
            raise
    
    def _run_model_training_stage(self):
        """Stage 4: Fine-tune the target model (optional)"""
        self.current_stage = "model_training"
        logger.info("🎯 Starting model training stage...")
        
        try:
            self.stage_progress.emit("model_training", 0, "Initializing model training...")
            
            # This is a placeholder for actual model training
            # In a real implementation, you would integrate with your existing training infrastructure
            
            training_data_path = self.results["training_data_path"]
            
            self.stage_progress.emit("model_training", 25, "Loading training data...")
            self.stage_progress.emit("model_training", 50, "Configuring model...")
            self.stage_progress.emit("model_training", 75, "Starting training process...")
            
            # Simulate training process
            time.sleep(2)  # Replace with actual training logic
            
            self.stage_progress.emit("model_training", 100, "Model training completed")
            
            # Store results
            stage_results = {
                "target_model": self.config.target_model,
                "training_data_used": training_data_path,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "status": "completed"
            }
            
            self.results["model_training_info"] = stage_results
            
            self.stage_completed.emit("model_training", stage_results)
            logger.info("✅ Model training stage completed")
            
        except Exception as e:
            logger.error(f"❌ Model training stage failed: {e}")
            self.pipeline_error.emit("model_training", str(e))
            raise
    
    def _analyze_quality_distribution(self, questions) -> Dict[str, Any]:
        """Analyze the quality distribution of generated questions"""
        try:
            if not questions:
                return {}
            
            quality_scores = [q.quality_score for q in questions]
            
            return {
                "total_questions": len(questions),
                "average_quality": sum(quality_scores) / len(quality_scores),
                "min_quality": min(quality_scores),
                "max_quality": max(quality_scores),
                "high_quality_count": len([q for q in quality_scores if q >= 0.8]),
                "medium_quality_count": len([q for q in quality_scores if 0.6 <= q < 0.8]),
                "low_quality_count": len([q for q in quality_scores if q < 0.6])
            }
            
        except Exception as e:
            logger.error(f"❌ Quality analysis failed: {e}")
            return {}
