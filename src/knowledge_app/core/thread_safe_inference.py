"""
🔥 THREAD-SAFE INFERENCE WRAPPER - Prevents UI Blocking
Isolates ALL AI operations from the main UI thread for responsive user experience
"""

import logging
import threading
import concurrent.futures
from typing import Optional, Dict, Any, Callable
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
import time

from .unified_inference_manager import get_unified_inference_manager, generate_mcq_unified

logger = logging.getLogger(__name__)


class ThreadSafeInferenceWrapper(QObject):
    """
    🔥 CRITICAL COMPONENT: Thread-Safe AI Inference Wrapper
    
    Ensures ALL AI operations are executed in background threads to prevent UI freezing.
    Uses proper Qt signals for thread-safe communication back to the UI.
    """
    
    # Qt signals for thread-safe communication
    mcq_generated = pyqtSignal(dict)  # Success: MCQ data
    generation_failed = pyqtSignal(str)  # Error: error message
    generation_progress = pyqtSignal(str)  # Progress updates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Thread pool for AI operations - isolated from UI
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=8,  # Increased for better parallelization
            thread_name_prefix="ThreadSafeInference"
        )
        
        # Track active operations
        self._active_operations = {}
        self._operation_counter = 0
        self._lock = threading.Lock()
        
        logger.info("🛡️ ThreadSafeInferenceWrapper initialized")
    
    def generate_mcq_async(self, topic: str, difficulty: str = "medium", 
                          question_type: str = "mixed", mode: str = "auto", timeout: float = 60.0) -> str:
        """
        🚀 MAIN PUBLIC API: Generate MCQ asynchronously without blocking UI
        
        Returns: operation_id that can be used to track the operation
        """
        # 🔥 DEBUG: Log the parameters received by thread-safe inference
        logger.info(f"🔍 DEBUG: ThreadSafeInference.generate_mcq_async called with difficulty='{difficulty}', topic='{topic}', question_type='{question_type}', mode='{mode}'")
        
        with self._lock:
            self._operation_counter += 1
            operation_id = f"mcq_op_{self._operation_counter}"
        
        # Submit to thread pool immediately
        future = self._executor.submit(
            self._generate_mcq_worker,
            operation_id, topic, difficulty, question_type, mode, timeout
        )
        
        # Track the operation
        with self._lock:
            self._active_operations[operation_id] = {
                "future": future,
                "start_time": time.time(),
                "topic": topic,
                "mode": mode,
                "timeout": timeout
            }
        
        # Start monitoring for completion
        self._start_operation_monitor(operation_id)
        
        logger.info(f"🚀 Started async MCQ generation: {operation_id} for '{topic}'")
        return operation_id

    def get_result(self, operation_id: str, timeout: float = 120.0) -> Optional[Dict[str, Any]]:
        """
        🚀 Get the result of an async operation

        Args:
            operation_id: The operation ID returned by generate_mcq_async
            timeout: Maximum time to wait for the result

        Returns:
            The generated MCQ result or None if failed/timeout
        """
        try:
            with self._lock:
                if operation_id not in self._active_operations:
                    logger.error(f"❌ Operation {operation_id} not found")
                    return None

                future = self._active_operations[operation_id]['future']

            # Wait for the result with timeout
            result = future.result(timeout=timeout)
            logger.info(f"✅ Operation {operation_id} completed successfully")
            return result

        except concurrent.futures.TimeoutError:
            logger.error(f"⏰ Operation {operation_id} timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"❌ Operation {operation_id} failed: {e}")
            return None
        finally:
            # Clean up the operation
            with self._lock:
                if operation_id in self._active_operations:
                    del self._active_operations[operation_id]

    def _generate_mcq_worker(self, operation_id: str, topic: str, difficulty: str,
                           question_type: str, mode: str, timeout: float) -> Optional[Dict[str, Any]]:
        """
        Worker function that runs in background thread - NEVER blocks UI
        """
        try:
            # 🔥 DEBUG: Log worker parameters
            logger.info(f"🔄 Worker {operation_id}: Generating MCQ for '{topic}' at '{difficulty}' difficulty in '{mode}' mode")
            
            # 🚀 CRITICAL FIX: Set inference mode BEFORE generation
            manager = get_unified_inference_manager()
            manager.set_inference_mode(mode)
            logger.info(f"🎯 Worker {operation_id}: Set inference mode to '{mode}'")
            
            # Emit progress signal (thread-safe)
            mode_text = "🎮 LOCAL" if mode == "offline" else "🌐 CLOUD" if mode == "online" else "🔄 AUTO"
            self.generation_progress.emit(f"{mode_text} Generating {difficulty} question about {topic}...")
            
            # 🔥 DEBUG: Log call to unified inference
            logger.info(f"🔍 DEBUG: Worker {operation_id} calling generate_mcq_unified with difficulty='{difficulty}', mode='{mode}'")
            
            # Use unified inference manager - this is the ONLY place AI is called
            result = generate_mcq_unified(
                topic=topic,
                difficulty=difficulty,
                question_type=question_type,
                timeout=timeout
            )
            
            if result:
                logger.info(f"✅ Worker {operation_id}: MCQ generated successfully")
                # Emit success signal (thread-safe)
                self.mcq_generated.emit(result)
                return result
            else:
                logger.error(f"❌ Worker {operation_id}: No MCQ generated")
                # Emit failure signal (thread-safe)
                self.generation_failed.emit(f"Failed to generate question about {topic}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Worker {operation_id}: Exception: {e}")
            # Emit failure signal (thread-safe)
            self.generation_failed.emit(f"Error generating question: {str(e)}")
            return None
        finally:
            # Clean up operation tracking
            with self._lock:
                if operation_id in self._active_operations:
                    del self._active_operations[operation_id]
    
    def _start_operation_monitor(self, operation_id: str):
        """Start monitoring operation for timeout and completion"""
        def check_operation():
            with self._lock:
                if operation_id not in self._active_operations:
                    return  # Operation completed or cancelled
                
                op_data = self._active_operations[operation_id]
                elapsed = time.time() - op_data["start_time"]
                
                if elapsed > op_data["timeout"]:
                    # Operation timed out
                    logger.error(f"⏰ Operation {operation_id} timed out after {elapsed:.1f}s")
                    
                    # Cancel the future
                    future = op_data["future"]
                    if not future.done():
                        future.cancel()
                    
                    # Emit timeout error
                    self.generation_failed.emit(
                        f"Question generation timed out after {op_data['timeout']}s"
                    )
                    
                    # Clean up
                    del self._active_operations[operation_id]
                else:
                    # Check again in 1 second
                    QTimer.singleShot(1000, check_operation)
        
        # Start monitoring after a short delay
        QTimer.singleShot(100, check_operation)
    
    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel an active operation"""
        with self._lock:
            if operation_id not in self._active_operations:
                return False
            
            op_data = self._active_operations[operation_id]
            future = op_data["future"]
            
            if not future.done():
                cancelled = future.cancel()
                if cancelled:
                    logger.info(f"🚫 Cancelled operation {operation_id}")
                    del self._active_operations[operation_id]
                    return True
            
            return False
    
    def get_active_operations(self) -> Dict[str, Dict[str, Any]]:
        """Get info about currently active operations"""
        with self._lock:
            return {
                op_id: {
                    "topic": op_data["topic"],
                    "mode": op_data["mode"],
                    "elapsed": time.time() - op_data["start_time"],
                    "timeout": op_data["timeout"]
                }
                for op_id, op_data in self._active_operations.items()
            }
    
    def shutdown(self):
        """Gracefully shutdown the thread pool"""
        logger.info("🔄 Shutting down ThreadSafeInferenceWrapper...")
        
        # Cancel all active operations
        with self._lock:
            for operation_id, op_data in self._active_operations.items():
                future = op_data["future"]
                if not future.done():
                    future.cancel()
                    logger.info(f"🚫 Cancelled operation {operation_id} during shutdown")
            self._active_operations.clear()
        
        # Shutdown executor
        self._executor.shutdown(wait=True, timeout=5.0)
        logger.info("✅ ThreadSafeInferenceWrapper shutdown complete")


# Global singleton instance
_thread_safe_inference = None
_lock = threading.Lock()


def get_thread_safe_inference() -> ThreadSafeInferenceWrapper:
    """Get the global thread-safe inference wrapper instance"""
    global _thread_safe_inference
    
    if _thread_safe_inference is None:
        with _lock:
            if _thread_safe_inference is None:
                _thread_safe_inference = ThreadSafeInferenceWrapper()
    
    return _thread_safe_inference


def shutdown_thread_safe_inference():
    """Shutdown the global thread-safe inference wrapper"""
    global _thread_safe_inference
    
    if _thread_safe_inference is not None:
        _thread_safe_inference.shutdown()
        _thread_safe_inference = None