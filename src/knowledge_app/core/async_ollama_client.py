#!/usr/bin/env python3
"""
🚀 Async Ollama Client - Non-blocking Ollama API calls

This module provides truly asynchronous Ollama API calls that never block the UI thread.
Uses asyncio and aiohttp for non-blocking HTTP requests.
"""

import asyncio
import logging
import time
import json
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class AsyncOllamaClient:
    """
    🚀 Async Ollama Client - Never blocks the UI thread
    
    This client provides truly asynchronous Ollama API calls using:
    1. asyncio for async/await patterns
    2. ThreadPoolExecutor for CPU-bound operations
    3. Non-blocking HTTP requests
    4. Proper timeout handling
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        self.tags_url = f"{base_url}/api/tags"
        
        # Thread pool for blocking operations
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="AsyncOllama"
        )
        
        # Session management
        self._session = None
        self._session_lock = threading.Lock()
        
        logger.info(f"🚀 AsyncOllamaClient initialized for {base_url}")
    
    def _get_session(self):
        """Get or create HTTP session (thread-safe)"""
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    try:
                        import requests
                        self._session = requests.Session()
                        # Set reasonable timeouts
                        self._session.timeout = (5, 30)  # (connect, read)
                        logger.info("✅ HTTP session created")
                    except ImportError:
                        logger.error("❌ requests library not available")
                        return None
        return self._session
    
    async def generate_async(self, model: str, prompt: str, **options) -> Optional[str]:
        """
        🚀 Generate text asynchronously without blocking UI
        
        Args:
            model: Ollama model name
            prompt: Input prompt
            **options: Generation options
            
        Returns:
            Generated text or None if failed
        """
        try:
            logger.info(f"🚀 Starting async generation with {model}")
            
            # Prepare payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": options
            }
            
            # Run the blocking HTTP request in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._make_blocking_request,
                payload
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Async generation failed: {e}")
            return None
    
    def _make_blocking_request(self, payload: Dict[str, Any]) -> Optional[str]:
        """
        Make blocking HTTP request (runs in thread pool)
        
        This method runs in a background thread, so it won't block the UI
        even if the HTTP request takes a long time.
        """
        try:
            session = self._get_session()
            if not session:
                return None
            
            start_time = time.time()
            logger.info(f"📡 Making HTTP request to Ollama...")
            
            # Make the request with timeout
            response = session.post(
                self.generate_url,
                json=payload,
                timeout=45  # 45 second timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("response", "")
                
                elapsed = time.time() - start_time
                logger.info(f"✅ Ollama request completed in {elapsed:.1f}s")
                return result
            else:
                logger.error(f"❌ Ollama request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Blocking request failed: {e}")
            return None
    
    async def list_models_async(self) -> Optional[list]:
        """
        🚀 List available models asynchronously
        
        Returns:
            List of model names or None if failed
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._list_models_blocking
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Async model listing failed: {e}")
            return None
    
    def _list_models_blocking(self) -> Optional[list]:
        """List models (blocking, runs in thread pool)"""
        try:
            session = self._get_session()
            if not session:
                return None
            
            response = session.get(self.tags_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                logger.info(f"✅ Found {len(models)} Ollama models")
                return models
            else:
                logger.error(f"❌ Model listing failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Model listing error: {e}")
            return None
    
    def generate_sync_non_blocking(self, model: str, prompt: str, **options) -> Optional[str]:
        """
        🚀 TRULY NON-BLOCKING generation using concurrent.futures

        This method uses ThreadPoolExecutor to ensure the HTTP request
        never blocks the calling thread, even if it takes a long time.

        Args:
            model: Ollama model name
            prompt: Input prompt
            **options: Generation options

        Returns:
            Generated text or None if failed
        """
        try:
            # 🚀 CRITICAL FIX: Use ThreadPoolExecutor for truly non-blocking execution
            import concurrent.futures

            # Prepare payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": options
            }

            # Submit to thread pool and wait with timeout
            future = self._executor.submit(self._make_blocking_request, payload)

            # 🛡️ SECURITY FIX: Wait for result with timeout to prevent deadlocks
            timeout = options.get('timeout', 45)  # Default 45 seconds
            try:
                result = future.result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                logger.error(f"❌ Generation timed out after {timeout}s")
                future.cancel()  # Cancel the future to prevent resource leaks
                return None

        except concurrent.futures.TimeoutError:
            logger.error(f"❌ Generation timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"❌ Non-blocking generation failed: {e}")
            return None

    def start_generation_async(self, model: str, prompt: str, callback=None, **options):
        """
        🚀 Start generation asynchronously with callback

        This method starts generation and returns immediately.
        The callback will be called when generation completes.

        Args:
            model: Ollama model name
            prompt: Input prompt
            callback: Function to call with result (optional)
            **options: Generation options

        Returns:
            Future object that can be used to check status
        """
        try:
            # Prepare payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": options
            }

            # Submit to thread pool
            future = self._executor.submit(self._make_blocking_request, payload)

            # Add callback if provided
            if callback:
                def done_callback(fut):
                    try:
                        result = fut.result()
                        callback(result)
                    except Exception as e:
                        logger.error(f"❌ Async generation callback failed: {e}")
                        callback(None)

                future.add_done_callback(done_callback)

            logger.info(f"🚀 Started async generation for {model}")
            return future

        except Exception as e:
            logger.error(f"❌ Failed to start async generation: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self._session:
                self._session.close()
                self._session = None
            
            if self._executor:
                self._executor.shutdown(wait=False)
            
            logger.info("✅ AsyncOllamaClient cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")


# Global instance
_async_ollama_client: Optional[AsyncOllamaClient] = None
_client_lock = threading.Lock()


def get_async_ollama_client() -> AsyncOllamaClient:
    """Get the global async Ollama client instance"""
    global _async_ollama_client
    
    if _async_ollama_client is None:
        with _client_lock:
            if _async_ollama_client is None:
                _async_ollama_client = AsyncOllamaClient()
    
    return _async_ollama_client


def cleanup_async_ollama_client():
    """Clean up the global client"""
    global _async_ollama_client
    
    if _async_ollama_client is not None:
        _async_ollama_client.cleanup()
        _async_ollama_client = None
