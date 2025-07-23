"""
🔥 FIRE METHOD 2: Model Server Client

This is the UI-side interface to the isolated model server.
Your UI uses this instead of touching the model directly.
All model operations are now process-isolated and crash-proof.

Key Features:
- Process isolation prevents memory corruption
- Queue-based communication eliminates race conditions
- Automatic server management and recovery
- Thread-safe client interface
- Timeout handling for robustness
"""

from .async_converter import async_time_sleep


from .async_converter import async_time_sleep


import multiprocessing as mp
import threading
import time
import logging
import uuid
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import sys
import os

logger = logging.getLogger(__name__)


class ModelServerClient:
    """
    Client interface to the isolated model server process.
    This is what your UI will use instead of direct model access.
    """

    def __init__(self):
        self.server_process: Optional[mp.Process] = None
        self.request_queue: Optional[mp.Queue] = None
        self.response_queue: Optional[mp.Queue] = None
        self.is_running = False
        self.lock = threading.RLock()

        # Response tracking
        self.pending_requests: Dict[str, threading.Event] = {}
        self.responses: Dict[str, Dict[str, Any]] = {}
        self.response_thread: Optional[threading.Thread] = None
        self.stop_response_thread = threading.Event()

        # Server stats
        self.start_time = None
        self.last_status_check = 0

    def start_server(self) -> bool:
        """Start the isolated model server process"""
        with self.lock:
            if self.is_running:
                logger.warning("Model server already running")
                return True

            try:
                logger.info("🚀 Starting isolated model server process...")

                # Create communication queues
                self.request_queue = mp.Queue(maxsize=100)
                self.response_queue = mp.Queue(maxsize=100)

                # Import and start server process
                from model_server import run_model_server

                self.server_process = mp.Process(
                    target=run_model_server,
                    args=(self.request_queue, self.response_queue),
                    name="ModelServer",
                    daemon=True,
                )

                self.server_process.start()
                self.start_time = time.time()

                # Start response handling thread
                self.stop_response_thread.clear()
                self.response_thread = threading.Thread(
                    target=self._response_handler, name="ModelServerResponseHandler", daemon=True
                )
                self.response_thread.start()

                # Wait for server to be ready (with timeout)
                ready = self._wait_for_server_ready(timeout=60)

                if ready:
                    self.is_running = True
                    logger.info("✅ Model server started successfully")
                    return True
                else:
                    logger.error("❌ Model server failed to start within timeout")
                    self.stop_server()
                    return False

            except Exception as e:
                logger.error(f"❌ Failed to start model server: {e}")
                self.stop_server()
                return False

    async def _wait_for_server_ready(self, timeout: float = 60) -> bool:
        """
        🚀 BUG FIX 14: Wait for server to be ready (properly async)

        This method is correctly async and uses proper await patterns
        to avoid blocking the event loop.
        """
        import asyncio
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # 🚀 BUG FIX 14: Run blocking get_status in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                status = await loop.run_in_executor(None, lambda: self.get_status(timeout=5))
                if status and status.get("is_loaded"):
                    return True
            except:
                pass

            # 🚀 BUG FIX 14: Proper async sleep
            await asyncio.sleep(1)

        return False

    def _response_handler(self):
        """Handle responses from the server in a separate thread"""
        logger.info("Response handler thread started")

        while not self.stop_response_thread.is_set():
            try:
                # Get response with timeout
                response = self.response_queue.get(timeout=1.0)
                request_id = response.get("request_id")

                if request_id:
                    # Store response and signal waiting thread
                    self.responses[request_id] = response

                    if request_id in self.pending_requests:
                        self.pending_requests[request_id].set()

            except:
                continue  # Timeout or queue error

        logger.info("Response handler thread stopped")

    def _send_request(
        self, request: Dict[str, Any], timeout: float = 30
    ) -> Optional[Dict[str, Any]]:
        """Send a request to the server and wait for response"""
        if not self.is_running:
            logger.error("Model server not running")
            return None

        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request["request_id"] = request_id

        # Create event for this request
        event = threading.Event()
        self.pending_requests[request_id] = event

        try:
            # Send request
            self.request_queue.put(request, timeout=5)

            # Wait for response
            if event.wait(timeout):
                response = self.responses.pop(request_id, None)
                return response
            else:
                logger.error(f"Request {request_id} timed out")
                return None

        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return None

        finally:
            # Cleanup
            self.pending_requests.pop(request_id, None)
            self.responses.pop(request_id, None)

    def generate_mcq(
        self, prompt: str, max_tokens: int = 150, timeout: float = 60
    ) -> Optional[Dict[str, Any]]:
        """
        Generate MCQ using the isolated model server.
        This is the main method your UI will call.
        """
        logger.info(f"🎯 Requesting MCQ generation: {prompt[:50]}...")

        request = {"action": "generate", "prompt": prompt, "max_tokens": max_tokens}

        response = self._send_request(request, timeout)

        if response and response.get("success"):
            logger.info("✅ MCQ generation successful")
            return {
                "success": True,
                "result": response.get("result"),
                "prompt": response.get("prompt"),
                "length": response.get("length"),
            }
        else:
            error = response.get("error", "Unknown error") if response else "No response"
            logger.error(f"❌ MCQ generation failed: {error}")
            return {"success": False, "error": error, "result": None}

    def get_status(self, timeout: float = 10) -> Optional[Dict[str, Any]]:
        """Get server status"""
        request = {"action": "status"}
        response = self._send_request(request, timeout)

        if response:
            return response
        else:
            return None

    def is_server_healthy(self) -> bool:
        """Check if server is healthy"""
        try:
            status = self.get_status(timeout=5)
            return status is not None and status.get("is_loaded", False)
        except:
            return False

    def stop_server(self):
        """
        🚀 BUG FIX 14: Stop the model server process (synchronous)

        This method is intentionally synchronous as it's called during shutdown
        and needs to complete before the application exits.
        """
        with self.lock:
            if not self.is_running:
                return

            logger.info("🛑 Stopping model server...")

            try:
                # Send shutdown command
                if self.request_queue:
                    shutdown_request = {"action": "shutdown"}
                    self.request_queue.put(shutdown_request, timeout=5)

                    # Wait a bit for graceful shutdown (synchronous)
                    import time
                    time.sleep(2)

            except:
                pass

            # Stop response handler
            self.stop_response_thread.set()
            if self.response_thread and self.response_thread.is_alive():
                self.response_thread.join(timeout=5)

            # Terminate server process
            if self.server_process and self.server_process.is_alive():
                self.server_process.terminate()
                self.server_process.join(timeout=10)

                if self.server_process.is_alive():
                    logger.warning("Force killing model server process")
                    self.server_process.kill()
                    self.server_process.join(timeout=5)

            # Cleanup
            self.server_process = None
            self.request_queue = None
            self.response_queue = None
            self.is_running = False
            self.pending_requests.clear()
            self.responses.clear()

            logger.info("✅ Model server stopped")

    def restart_server(self) -> bool:
        """
        🚀 BUG FIX 14: Restart the server (synchronous for reliability)
        """
        logger.info("🔄 Restarting model server...")
        self.stop_server()
        import time
        time.sleep(2)  # Give time for cleanup
        return self.start_server()

    def __enter__(self):
        """Context manager entry"""
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_server()

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.stop_server()
        except:
            pass


# Global singleton instance
_global_client: Optional[ModelServerClient] = None
_client_lock = threading.Lock()


def get_model_client() -> ModelServerClient:
    """Get the global model server client instance"""
    global _global_client

    with _client_lock:
        if _global_client is None:
            _global_client = ModelServerClient()
        return _global_client


def start_global_model_server() -> bool:
    """Start the global model server"""
    client = get_model_client()
    return client.start_server()


def stop_global_model_server():
    """Stop the global model server"""
    global _global_client

    with _client_lock:
        if _global_client:
            _global_client.stop_server()
            _global_client = None


def generate_mcq_isolated(prompt: str, max_tokens: int = 150) -> Optional[Dict[str, Any]]:
    """
    Generate MCQ using the isolated model server.
    This is the main function your UI should call instead of direct model access.
    """
    client = get_model_client()

    # Auto-start server if not running
    if not client.is_running:
        logger.info("Auto-starting model server...")
        if not client.start_server():
            return {"success": False, "error": "Failed to start model server", "result": None}

    return client.generate_mcq(prompt, max_tokens)