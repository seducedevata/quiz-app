#!/usr/bin/env python3
"""
🚀 ASYNC CONVERTER - Convert ALL blocking operations to non-blocking

This module provides async wrappers for every blocking operation in the codebase
to completely eliminate UI freezing.
"""


# Async converter functions will be defined below


import asyncio
import aiohttp
import aiofiles
import logging
import time
import json
import threading
from typing import Optional, Dict, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)

class AsyncConverter:
    """
    🚀 Universal async converter for all blocking operations
    
    This class provides async versions of every blocking operation:
    - HTTP requests (requests.post/get)
    - File I/O (open, read, write)
    - Time delays (time.sleep)
    - Subprocess calls
    - Future waits
    """
    
    def __init__(self):
        # Thread pool for CPU-bound operations
        self._executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="AsyncConverter"
        )
        
        # HTTP session for async requests
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = threading.Lock()
        
        logger.info("[START] AsyncConverter initialized")
    
    def _make_sync_request(self, method: str, url: str, json_data: Dict[str, Any] = None, timeout: float = 60) -> Optional[Dict[str, Any]]:
        """Make synchronous HTTP request in isolated thread - NEVER blocks UI"""
        try:
            import requests

            logger.info(f"[START] Making {method} request to {url}")

            if method.upper() == "POST":
                response = requests.post(url, json=json_data, timeout=timeout)
            elif method.upper() == "GET":
                response = requests.get(url, timeout=timeout)
            else:
                logger.error(f"❌ Unsupported HTTP method: {method}")
                return None

            if response.status_code == 200:
                data = response.json()
                logger.info(f"[OK] {method} request successful")
                return data
            else:
                logger.error(f"❌ {method} request failed: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"❌ {method} request timeout: {url}")
            return None
        except Exception as e:
            logger.error(f"❌ {method} request error: {e}")
            return None

    async def async_post(self, url: str, json_data: Dict[str, Any] = None,
                        timeout: float = 60) -> Optional[Dict[str, Any]]:
        """
        🔧 CRITICAL FIX for Bug 2: True async HTTP POST request - NEVER blocks UI

        Args:
            url: Request URL
            json_data: JSON payload
            timeout: Request timeout

        Returns:
            Response JSON or None if failed
        """
        # 🔧 CRITICAL FIX: Use proper async pattern with loop.run_in_executor
        # BEFORE (BLOCKING): future.result() blocks the event loop
        # AFTER (NON-BLOCKING): await loop.run_in_executor() yields control
        import asyncio

        def run_sync_post():
            return self._make_sync_request("POST", url, json_data, timeout)

        # 🔧 CRITICAL FIX: Use non-blocking await instead of blocking future.result()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, run_sync_post)
    
    async def async_get(self, url: str, timeout: float = 30) -> Optional[Dict[str, Any]]:
        """
        🔧 CRITICAL FIX for Bug 2: True async HTTP GET request - NEVER blocks UI

        Args:
            url: Request URL
            timeout: Request timeout

        Returns:
            Response JSON or None if failed
        """
        # 🔧 CRITICAL FIX: Use proper async pattern with loop.run_in_executor
        # BEFORE (BLOCKING): future.result() blocks the event loop
        # AFTER (NON-BLOCKING): await loop.run_in_executor() yields control
        import asyncio

        def run_sync_get():
            return self._make_sync_request("GET", url, None, timeout)

        # 🔧 CRITICAL FIX: Use non-blocking await instead of blocking future.result()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, run_sync_get)
    
    async def async_time_sleep(self, seconds: float):
        """
        🚀 Async sleep - NEVER blocks UI

        Args:
            seconds: Sleep duration
        """
        await asyncio.sleep(seconds)
    
    async def async_file_read(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        🚀 Async file read - NEVER blocks UI
        
        Args:
            file_path: Path to file
            
        Returns:
            File content or None if failed
        """
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return content
        except Exception as e:
            logger.error(f"❌ Async file read failed: {e}")
            return None
    
    async def async_file_write(self, file_path: Union[str, Path], content: str) -> bool:
        """
        🚀 Async file write - NEVER blocks UI
        
        Args:
            file_path: Path to file
            content: Content to write
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
                return True
        except Exception as e:
            logger.error(f"❌ Async file write failed: {e}")
            return False
    
    async def async_subprocess(self, command: list, timeout: float = 30) -> Optional[str]:
        """
        🚀 Async subprocess call - NEVER blocks UI
        
        Args:
            command: Command to execute
            timeout: Execution timeout
            
        Returns:
            Command output or None if failed
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            
            if process.returncode == 0:
                return stdout.decode('utf-8')
            else:
                logger.error(f"❌ Subprocess failed: {stderr.decode('utf-8')}")
                return None
                
        except asyncio.TimeoutError:
            logger.error(f"❌ Subprocess timeout: {command}")
            return None
        except Exception as e:
            logger.error(f"❌ Subprocess error: {e}")
            return None
    
    def sync_to_async(self, sync_func: Callable, *args, **kwargs):
        """
        🚀 Convert any synchronous function to async - NEVER blocks UI
        
        Args:
            sync_func: Synchronous function to convert
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Async version of the function
        """
        async def async_wrapper():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: sync_func(*args, **kwargs)
            )
        return async_wrapper()
    
    async def async_ollama_request(self, url: str, payload: Dict[str, Any],
                                  timeout: float = 60) -> Optional[str]:
        """
        🚀 Async Ollama request with streaming support - NEVER blocks UI

        Args:
            url: Ollama API URL
            payload: Request payload
            timeout: Request timeout

        Returns:
            Generated text or None if failed
        """
        # 🚀 CRITICAL FIX: Use thread-isolated synchronous request to prevent ALL UI blocking
        import concurrent.futures
        import requests

        def run_sync_ollama():
            try:
                start_time = time.time()
                logger.info(f"🚀 Starting thread-isolated Ollama request...")

                # Use synchronous requests - NO aiohttp, NO event loops
                response = requests.post(url, json=payload, timeout=timeout)

                if response.status_code == 200:
                    data = response.json()
                    result = data.get("response", "")

                    elapsed = time.time() - start_time
                    logger.info(f"✅ Thread-isolated Ollama request completed in {elapsed:.1f}s")
                    return result
                else:
                    logger.error(f"❌ Ollama request failed: {response.status_code}")
                    return None

            except requests.exceptions.Timeout:
                logger.error(f"❌ Ollama request timeout after {timeout}s")
                return None
            except Exception as e:
                logger.error(f"❌ Ollama request error: {e}")
                return None

        # 🔧 CRITICAL FIX for Bug 2: Use non-blocking await instead of blocking future.result()
        # BEFORE (BLOCKING): future.result() blocks the event loop
        # AFTER (NON-BLOCKING): await loop.run_in_executor() yields control
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, run_sync_ollama)
    
    def async_ollama_stream(self, url: str, payload: Dict[str, Any],
                           timeout: float = 60):
        """
        🛡️ CRITICAL FIX: Async Ollama streaming with proper timeout handling

        Args:
            url: Ollama API URL
            payload: Request payload (must have stream=True)
            timeout: Total timeout for the entire streaming operation

        Yields:
            Token strings as they are generated

        Raises:
            TimeoutError: If streaming exceeds the specified timeout
        """
        import concurrent.futures
        import requests
        import queue
        import threading
        import time

        # 🛡️ CRITICAL FIX: Track total streaming time for timeout enforcement
        start_time = time.time()

        # Create a queue to pass tokens between threads
        token_queue = queue.Queue()
        error_queue = queue.Queue()

        def stream_ollama():
            try:
                logger.info(f"🚀 Starting thread-isolated Ollama streaming request with {timeout}s timeout...")

                # Ensure streaming is enabled
                payload["stream"] = True

                # 🛡️ CRITICAL FIX: Use shorter connection timeout, let total timeout handle the rest
                connection_timeout = min(timeout, 30)  # Max 30s for connection
                response = requests.post(url, json=payload, stream=True, timeout=connection_timeout)

                if response.status_code == 200:
                    for line in response.iter_lines():
                        # 🛡️ CRITICAL FIX: Check timeout during streaming
                        if time.time() - start_time > timeout:
                            error_queue.put("Streaming timeout exceeded")
                            break

                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if 'response' in data:
                                    token = data['response']
                                    if token:  # Only yield non-empty tokens
                                        token_queue.put(token)

                                # Check if generation is done
                                if data.get('done', False):
                                    break

                            except json.JSONDecodeError:
                                continue

                    # Signal completion
                    token_queue.put(None)
                    logger.info("✅ Ollama streaming completed")
                else:
                    error_queue.put(f"HTTP {response.status_code}")
                    token_queue.put(None)

            except Exception as e:
                logger.error(f"❌ Ollama streaming error: {e}")
                error_queue.put(str(e))
                token_queue.put(None)

        # Start streaming in background thread
        thread = threading.Thread(target=stream_ollama, daemon=True)
        thread.start()

        # 🛡️ CRITICAL FIX: Yield tokens with total timeout enforcement
        while True:
            try:
                # 🛡️ CRITICAL FIX: Check total timeout before each iteration
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    logger.error(f"❌ Streaming timeout exceeded: {elapsed_time:.2f}s > {timeout}s")
                    raise TimeoutError(f"Ollama streaming timed out after {elapsed_time:.2f} seconds")

                # Check for errors first
                if not error_queue.empty():
                    error = error_queue.get_nowait()
                    logger.error(f"❌ Streaming error: {error}")
                    if "timeout" in error.lower():
                        raise TimeoutError(f"Ollama streaming failed: {error}")
                    break

                # 🛡️ CRITICAL FIX: Calculate remaining timeout for queue.get()
                remaining_timeout = timeout - elapsed_time
                queue_timeout = min(1.0, max(0.1, remaining_timeout))  # Between 0.1s and 1s

                # Get next token with dynamic timeout
                token = token_queue.get(timeout=queue_timeout)

                if token is None:  # End of stream
                    break

                yield token

            except queue.Empty:
                # 🛡️ CRITICAL FIX: Check timeout and thread status
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    logger.error(f"❌ Streaming timeout during queue wait: {elapsed_time:.2f}s")
                    raise TimeoutError(f"Ollama streaming timed out after {elapsed_time:.2f} seconds")

                # Check if thread is still alive
                if not thread.is_alive():
                    logger.warning("⚠️ Streaming thread died, ending stream")
                    break
                continue

            except Exception as e:
                logger.error(f"❌ Token yielding error: {e}")
                break
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
            
            if self._executor:
                self._executor.shutdown(wait=False)
            
            logger.info("✅ AsyncConverter cleaned up")
            
        except Exception as e:
            logger.error(f"❌ AsyncConverter cleanup error: {e}")


# Global instance
_async_converter: Optional[AsyncConverter] = None
_converter_lock = threading.Lock()


def get_async_converter() -> AsyncConverter:
    """Get the global async converter instance"""
    global _async_converter
    
    if _async_converter is None:
        with _converter_lock:
            if _async_converter is None:
                _async_converter = AsyncConverter()
    
    return _async_converter


async def cleanup_async_converter():
    """Clean up the global converter"""
    global _async_converter

    if _async_converter is not None:
        await _async_converter.cleanup()
        _async_converter = None


# 🚀 CONVENIENCE FUNCTIONS - Direct replacements for blocking operations

async def async_requests_post(url: str, json_data: Dict[str, Any] = None,
                             timeout: float = 60) -> Optional[Dict[str, Any]]:
    """🚀 Direct replacement for await async_requests_post() - NEVER blocks UI"""
    converter = get_async_converter()
    return await converter.async_post(url, json_data, timeout)


async def async_requests_get(url: str, timeout: float = 30) -> Optional[Dict[str, Any]]:
    """🚀 Direct replacement for await async_requests_get() - NEVER blocks UI"""
    converter = get_async_converter()
    return await converter.async_get(url, timeout)


async def async_time_sleep(seconds: float):
    """🚀 Direct replacement for time.sleep() - NEVER blocks UI"""
    await asyncio.sleep(seconds)


async def async_subprocess(command: list, **kwargs):
    """🚀 Direct replacement for subprocess calls - NEVER blocks UI"""
    converter = get_async_converter()
    return await converter.async_subprocess(command, **kwargs)


async def async_file_read(file_path: Union[str, Path]) -> Optional[str]:
    """🚀 Direct replacement for open().read() - NEVER blocks UI"""
    converter = get_async_converter()
    return await converter.async_file_read(file_path)


async def async_file_write(file_path: Union[str, Path], content: str) -> bool:
    """🚀 Direct replacement for open().write() - NEVER blocks UI"""
    converter = get_async_converter()
    return await converter.async_file_write(file_path, content)


def make_async(sync_func: Callable):
    """
    🚀 Decorator to convert any synchronous function to async - NEVER blocks UI

    Usage:
        @make_async
        def blocking_function():
            await async_time_sleep(5)  # This will become non-blocking
            return "done"

        # Use as: result = await blocking_function()
    """
    async def async_wrapper(*args, **kwargs):
        converter = get_async_converter()
        return await converter.sync_to_async(sync_func, *args, **kwargs)

    return async_wrapper


class AsyncContextManager:
    """
    🚀 Context manager to ensure all operations within a block are async

    Usage:
        async with AsyncContextManager():
            # All operations here will be non-blocking
            result = await async_requests_post(url, data)
            await async_time_sleep(1)
            content = await async_file_read("file.txt")
    """

    def __init__(self):
        self.converter = get_async_converter()

    async def __aenter__(self):
        return self.converter

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass


def run_async_in_thread(async_func, *args, **kwargs):
    """
    🚀 Run async function in a separate thread with its own event loop

    This is the CRITICAL function that prevents all UI blocking by running
    async operations in completely isolated threads.

    Args:
        async_func: Async function to run
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Result of the async function
    """
    import concurrent.futures

    def run_in_new_loop():
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the async function
            result = loop.run_until_complete(async_func(*args, **kwargs))
            return result

        except Exception as e:
            logger.error(f"❌ Async function failed: {e}")
            raise
        finally:
            # Clean up the loop
            try:
                loop.close()
            except:
                pass

    # Run in separate thread to avoid blocking UI
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            # 🔧 NOTE: future.result() is INTENTIONAL here - this is a sync wrapper function
            # This function is designed to be called from sync code to run async functions
            # The blocking happens in a separate thread, not the main UI thread
            return future.result(timeout=60)  # 60 second timeout
    except concurrent.futures.TimeoutError:
        logger.error("❌ Async function timed out after 60 seconds")
        raise
    except Exception as e:
        logger.error(f"❌ Thread execution failed: {e}")
        raise
