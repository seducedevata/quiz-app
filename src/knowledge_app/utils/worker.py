from PyQt5.QtCore import QRunnable, QObject, pyqtSignal, pyqtSlot
import os
import sys
import io
import traceback
import logging
from knowledge_app.app_health import setup_logging, check_dependencies
import torch
from typing import Any, Dict, Optional
from pathlib import Path

# Setup logging at app/module import time
setup_logging()


# Dependency check (can be called at app startup)
def check_app_health():
    ok, msg = check_dependencies()
    if not ok:
        logging.error(msg)
        print(msg)
        # Optionally, show a dialog in the UI here
        raise RuntimeError(msg)
    else:
        logging.info(msg)


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(dict)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.is_killed = False

    def kill(self):
        """Mark the worker for termination"""
        self.is_killed = True

    @pyqtSlot()
    def run(self):
        """
        Initialize the runner function with passed args, kwargs.
        """
        try:
            if self.is_killed:
                logger.info("Worker was killed before execution")
                return

            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)

        except Exception as e:
            logger.error(f"Error in worker thread: {e}", exc_info=True)
            self.signals.error.emit((type(e), e, traceback.format_exc()))
            self.signals.result.emit(f"Error: {e}")

        finally:
            if not self.is_killed:
                self.signals.finished.emit()


# New TrainingWorker class for the AI training pipeline
class TrainingWorker(QRunnable):
    def __init__(
        self, domain_name: str, training_params: Dict[str, Any], model_path: Optional[str] = None
    ):
        super().__init__()

        self.domain_name = domain_name
        self.training_params = training_params
        self.model_path = model_path
        self.signals = WorkerSignals()
        self.is_killed = False
        self.device = None
        self.last_memory_check = 0
        self.memory_check_interval = 50  # Check memory every 50 steps

    def kill(self):
        """Mark the worker for termination"""
        self.is_killed = True

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.device and self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Remove temporary paths from sys.path
            app_root_path = str(Path(__file__).parent.parent.parent)
            if app_root_path in sys.path:
                sys.path.remove(app_root_path)

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def handle_error(self, error: Exception, context: str):
        """Handle errors during training"""
        error_msg = f"{context}: {str(error)}"
        logger.error(error_msg, exc_info=True)

        self.signals.error.emit((type(error), error, traceback.format_exc()))
        self.signals.result.emit(error_msg)

    def _setup_gpu(self):
        """Set up GPU for training if available."""
        import torch

        try:
            if torch.cuda.is_available():
                # Clear GPU cache
                torch.cuda.empty_cache()

                # Get GPU info
                device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                memory_cached = torch.cuda.memory_cached(0) / 1024**2

                logger.info(f"Using GPU: {gpu_name}")

                # Optimize CUDA settings
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                # Set memory split for training
                if total_memory > 8000:  # If GPU has more than 8GB
                    max_split = int(total_memory * 0.85)  # Use up to 85% of GPU memory
                    torch.cuda.set_per_process_memory_fraction(max_split / total_memory)

                return device
            else:
                logger.info("No GPU available, using CPU")
                return torch.device("cpu")
        except Exception as e:
            logger.warning(f"WARNING: GPU initialization failed: {e}. Falling back to CPU.")
            logger.warning(f"Error details: {str(e)}")
            return torch.device("cpu")

    def _check_gpu_memory(self, step):
        """Monitor and manage GPU memory during training."""
        if step % self.memory_check_interval == 0 and self.device.type == "cuda":
            import torch

            try:
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                memory_cached = torch.cuda.memory_cached(0) / 1024**2
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2

                # If memory usage is above 90%, trigger cleanup
                if memory_allocated / total_memory > 0.9:
                    torch.cuda.empty_cache()
                    logger.info(
                        f"Memory cleanup triggered at step {step}. New allocation: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB"
                    )

                return {
                    "allocated": memory_allocated,
                    "cached": memory_cached,
                    "total": total_memory,
                }
            except Exception as e:
                logger.warning(f"WARNING: Error checking GPU memory: {e}")
                return None

    @pyqtSlot()
    def run(self):
        """Execute the training process"""
        if self.is_killed:
            logger.info("Training worker was killed before execution")
            return

        try:
            # Add app root to Python path temporarily
            app_root_path = str(Path(__file__).parent.parent.parent)
            if app_root_path not in sys.path:
                sys.path.append(app_root_path)

            # Import training modules
from training.core.training_manager import TrainingManager
            from knowledge_app.core.preprocessing import preprocess_data

            try:
                # Set up device
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Using device: {self.device}")

                # Preprocess data
                self.signals.progress.emit({"status": "Preprocessing data...", "value": 10})
                preprocessed_data = preprocess_data(
                    self.domain_name, self.training_params, self.device
                )

                if self.is_killed:
                    logger.info("Training worker was killed after preprocessing")
                    return

                # Train model
                self.signals.progress.emit({"status": "Training model...", "value": 30})
                final_status = train_model(
                    preprocessed_data,
                    self.training_params,
                    self.model_path,
                    self.device,
                    progress_callback=self.signals.progress.emit,
                )

                if self.is_killed:
                    logger.info("Training worker was killed after training")
                    return

                logger.info(f"Training completed: {final_status}")
                self.signals.progress.emit({"status": final_status, "value": 100})
                self.signals.result.emit(final_status)

            except Exception as e:
                self.handle_error(e, "Training error")

        except Exception as e:
            self.handle_error(e, "Initialization error")

        finally:
            if not self.is_killed:
                self.cleanup()
                self.signals.finished.emit()
            logger.debug(f"Training worker finished for domain: {self.domain_name}")


# === RAG Integration Workers ===
class RAGIngestWorker(QRunnable):
    """Worker to ingest a document (PDF or text) into the RAG index."""

    def __init__(self, file_path, file_type="pdf", meta=None):
        super().__init__()
        self.file_path = file_path
        self.file_type = file_type
        self.meta = meta or {}
        self.signals = WorkerSignals()
        self.is_killed = False

    def kill(self):
        """Mark the worker for termination"""
        self.is_killed = True

    @pyqtSlot()
    def run(self):
        if self.is_killed:
            logger.info("RAG ingest worker was killed before execution")
            return

        rag = None
        try:
            from knowledge_app.rag_engine import RAGEngine

            rag = RAGEngine()

            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")

            if self.file_type == "pdf":
                self.signals.progress.emit({"status": "Ingesting PDF...", "value": 10})
                num_pages = rag.ingest_pdf(self.file_path, meta=self.meta)
                result = f"Ingested {num_pages} pages from PDF: {os.path.basename(self.file_path)}"
            else:
                self.signals.progress.emit({"status": "Ingesting text...", "value": 10})
                num_chunks = rag.ingest_text(self.file_path, meta=self.meta)
                result = f"Ingested text file with {num_chunks} chunks: {os.path.basename(self.file_path)}"

            self.signals.progress.emit({"status": "Completed", "value": 100})
            self.signals.result.emit(result)

        except Exception as e:
            error_msg = f"Error ingesting document: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.signals.error.emit((type(e), e, traceback.format_exc()))
            self.signals.result.emit(error_msg)

        finally:
            if rag:
                try:
                    rag.cleanup()
                except Exception as e:
                    logger.error(f"Error during RAG cleanup: {e}")

            if not self.is_killed:
                self.signals.finished.emit()


class RAGQAWorker(QRunnable):
    """Worker to perform question-answering using the RAG system."""

    def __init__(self, question, top_k_retriever=5, top_k_reader=1):
        super().__init__()
        self.question = question
        self.top_k_retriever = top_k_retriever
        self.top_k_reader = top_k_reader
        self.signals = WorkerSignals()
        self.is_killed = False

    def kill(self):
        """Mark the worker for termination"""
        self.is_killed = True

    @pyqtSlot()
    def run(self):
        if self.is_killed:
            logger.info("RAG QA worker was killed before execution")
            return

        rag = None
        try:
            from knowledge_app.rag_engine import RAGEngine

            rag = RAGEngine()

            self.signals.progress.emit({"status": "Retrieving relevant passages...", "value": 30})
            passages = rag.retrieve(self.question, top_k=self.top_k_retriever)

            if not passages:
                result = "No relevant information found in the knowledge base."
                self.signals.result.emit(result)
                return

            self.signals.progress.emit({"status": "Generating answer...", "value": 70})
            answer = rag.generate_answer(
                question=self.question, passages=passages, top_k=self.top_k_reader
            )

            self.signals.progress.emit({"status": "Completed", "value": 100})
            self.signals.result.emit(answer)

        except Exception as e:
            error_msg = f"Error during question answering: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.signals.error.emit((type(e), e, traceback.format_exc()))
            self.signals.result.emit(error_msg)

        finally:
            if rag:
                try:
                    rag.cleanup()
                except Exception as e:
                    logger.error(f"Error during RAG cleanup: {e}")

            if not self.is_killed:
                self.signals.finished.emit()


class ModelLoadingWorker(QRunnable):
    """Worker to handle model loading operations."""

    def __init__(self, model_name, model_type="qa", device=None):
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.signals = WorkerSignals()
        self.is_killed = False

    def kill(self):
        """Mark the worker for termination"""
        self.is_killed = True

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    @pyqtSlot()
    def run(self):
        if self.is_killed:
            logger.info("Model loading worker was killed before execution")
            return

        try:
            from knowledge_app.core.unified_inference_manager import get_inference_manager

            self.signals.progress.emit({"status": f"Loading {self.model_name}...", "value": 10})

            manager = get_inference_manager()
            model = manager.load_model(
                model_name=self.model_name, model_type=self.model_type, device=self.device
            )

            self.signals.progress.emit({"status": "Model loaded successfully", "value": 100})
            self.signals.result.emit(model)

        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.signals.error.emit((type(e), e, traceback.format_exc()))
            self.signals.result.emit(error_msg)

        finally:
            if not self.is_killed:
                self.cleanup()
                self.signals.finished.emit()