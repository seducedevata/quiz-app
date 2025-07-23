"""
Lazy RAG Engine Loader

This module provides lazy loading for the RAG engine to prevent startup delays.
The RAG engine is only initialized when first needed, dramatically improving
application startup time.

Features:
- Lazy initialization of RAG components
- Background loading with progress indicators
- Fallback mechanisms for failed initialization
- Memory-efficient loading strategies
"""

from .async_converter import async_time_sleep


from .async_converter import async_time_sleep


import logging
import threading
import time
from typing import Optional, Callable, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class LazyRAGLoader:
    """
    Lazy loader for RAG engine components that defers heavy initialization
    until first use, dramatically improving startup performance.
    """

    def __init__(
        self,
        db_path: str = "faiss_index.db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.db_path = db_path
        self.embedding_model = embedding_model

        # Lazy loading state
        self._rag_engine = None
        self._is_loading = False
        self._load_error = None
        self._load_thread = None
        self._load_callbacks = []

        # Performance tracking
        self._load_start_time = None
        self._load_end_time = None

        logger.info(f"🔄 Lazy RAG loader initialized (db_path={db_path})")

    @property
    def is_loaded(self) -> bool:
        """Check if RAG engine is loaded and ready"""
        return self._rag_engine is not None

    @property
    def is_loading(self) -> bool:
        """Check if RAG engine is currently loading"""
        return self._is_loading

    @property
    def load_error(self) -> Optional[Exception]:
        """Get any loading error that occurred"""
        return self._load_error

    @property
    def load_time(self) -> Optional[float]:
        """Get the time taken to load the RAG engine"""
        if self._load_start_time and self._load_end_time:
            return self._load_end_time - self._load_start_time
        return None

    def add_load_callback(self, callback: Callable[[bool, Optional[Exception]], None]):
        """Add a callback to be called when loading completes"""
        self._load_callbacks.append(callback)

    def get_rag_engine(self, timeout: float = 30.0):
        """
        Get the RAG engine, loading it if necessary.

        Args:
            timeout: Maximum time to wait for loading to complete

        Returns:
            RAG engine instance or None if loading failed
        """
        if self._rag_engine is not None:
            return self._rag_engine

        if self._load_error is not None:
            logger.warning(f"RAG engine previously failed to load: {self._load_error}")
            return None

        if not self._is_loading:
            self._start_background_loading()

        # Wait for loading to complete
        start_wait = time.time()
        while self._is_loading and (time.time() - start_wait) < timeout:
            await async_time_sleep(0.1)

        if self._is_loading:
            logger.warning(f"RAG engine loading timed out after {timeout}s")
            return None

        return self._rag_engine

    def start_background_loading(self):
        """Start loading the RAG engine in the background"""
        if not self._is_loading and self._rag_engine is None:
            self._start_background_loading()

    def _start_background_loading(self):
        """Internal method to start background loading"""
        if self._is_loading:
            return

        self._is_loading = True
        self._load_error = None
        self._load_start_time = time.time()

        logger.info("🔄 Starting background RAG engine loading...")

        self._load_thread = threading.Thread(
            target=self._load_rag_engine, name="RAGEngineLoader", daemon=True
        )
        self._load_thread.start()

    def _load_rag_engine(self):
        """Load the RAG engine in a background thread"""
        try:
            logger.info("📚 Initializing RAG engine components...")

            # Import RAG engine lazily to avoid startup delays
            from ..rag_engine import RAGEngine

            # Create RAG engine with minimal initialization
            self._rag_engine = RAGEngine(db_path=self.db_path, embedding_model=self.embedding_model)

            self._load_end_time = time.time()
            load_time = self.load_time

            logger.info(f"✅ RAG engine loaded successfully in {load_time:.2f}s")

            # Call success callbacks
            for callback in self._load_callbacks:
                try:
                    callback(True, None)
                except Exception as e:
                    logger.warning(f"RAG load callback failed: {e}")

        except Exception as e:
            self._load_error = e
            self._load_end_time = time.time()

            logger.error(f"❌ Failed to load RAG engine: {e}")

            # Call error callbacks
            for callback in self._load_callbacks:
                try:
                    callback(False, e)
                except Exception as cb_error:
                    logger.warning(f"RAG load callback failed: {cb_error}")

        finally:
            self._is_loading = False

    def cleanup(self):
        """Clean up resources"""
        if self._rag_engine:
            try:
                if hasattr(self._rag_engine, "cleanup"):
                    self._rag_engine.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up RAG engine: {e}")

            self._rag_engine = None

        if self._load_thread and self._load_thread.is_alive():
            # Note: We can't force-stop the thread, but we can mark it as daemon
            logger.debug("RAG loader thread still running (daemon)")


# Global lazy RAG loader instance
_global_rag_loader: Optional[LazyRAGLoader] = None


def get_lazy_rag_loader(
    db_path: str = "faiss_index.db", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> LazyRAGLoader:
    """
    Get the global lazy RAG loader instance.

    Args:
        db_path: Path to FAISS database
        embedding_model: Embedding model to use

    Returns:
        LazyRAGLoader instance
    """
    global _global_rag_loader

    if _global_rag_loader is None:
        _global_rag_loader = LazyRAGLoader(db_path, embedding_model)

    return _global_rag_loader


def preload_rag_engine():
    """Start preloading the RAG engine in the background"""
    loader = get_lazy_rag_loader()
    loader.start_background_loading()


def get_rag_engine_when_ready(timeout: float = 30.0):
    """
    Get the RAG engine, waiting for it to load if necessary.

    Args:
        timeout: Maximum time to wait for loading

    Returns:
        RAG engine instance or None if loading failed
    """
    loader = get_lazy_rag_loader()
    return loader.get_rag_engine(timeout)


def cleanup_lazy_rag_loader():
    """Clean up the global lazy RAG loader"""
    global _global_rag_loader

    if _global_rag_loader:
        _global_rag_loader.cleanup()
        _global_rag_loader = None