"""
Enhanced shutdown manager for Knowledge App
Pure Python shutdown handling - no QtWidgets bloatware
"""

import sys
import os
import logging
import signal
import threading
import time
import atexit
from typing import List, Callable, Optional

logger = logging.getLogger(__name__)


class ShutdownManager:
    """Enhanced shutdown manager with immediate termination capability"""
    
    def __init__(self):
        self.shutdown_callbacks: List[Callable] = []
        self.is_shutting_down = False
        self.shutdown_lock = threading.Lock()
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, self._signal_handler)
            atexit.register(self.shutdown)
        except Exception as e:
            logger.error(f"Failed to setup signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
    
    def register_shutdown_callback(self, callback: Callable) -> None:
        """Register a callback to be called during shutdown"""
        if callback not in self.shutdown_callbacks:
            self.shutdown_callbacks.append(callback)
            logger.debug(f"Registered shutdown callback: {callback.__name__}")
    
    def unregister_shutdown_callback(self, callback: Callable) -> None:
        """Unregister a shutdown callback"""
        if callback in self.shutdown_callbacks:
            self.shutdown_callbacks.remove(callback)
            logger.debug(f"Unregistered shutdown callback: {callback.__name__}")
    
    def shutdown(self) -> None:
        """Perform graceful shutdown with timeout"""
        with self.shutdown_lock:
            if self.is_shutting_down:
                return
            self.is_shutting_down = True
        
        logger.info("🚨 Initiating application shutdown...")
        
        # Start emergency exit timer
        emergency_timer = threading.Timer(5.0, self._emergency_exit)
        emergency_timer.daemon = True
        emergency_timer.start()
        
        try:
            # Execute shutdown callbacks with timeout
            for i, callback in enumerate(self.shutdown_callbacks):
                try:
                    logger.debug(f"Executing shutdown callback {i+1}/{len(self.shutdown_callbacks)}")
                    callback()
                except Exception as e:
                    logger.error(f"Error in shutdown callback {callback.__name__}: {e}")
            
            logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            emergency_timer.cancel()
            self._force_exit()
    
    def _emergency_exit(self):
        """Emergency exit if graceful shutdown takes too long"""
        logger.warning("⚠️ Graceful shutdown timeout, forcing exit...")
        self._force_exit()
    
    def _force_exit(self):
        """Force immediate exit"""
        try:
            logger.info("💥 Forcing immediate application exit")
            os._exit(0)
        except:
            # Ultimate fallback
            sys.exit(1)


# Global shutdown manager instance
shutdown_manager = ShutdownManager()


def get_shutdown_manager() -> ShutdownManager:
    """Get the global shutdown manager instance"""
    return shutdown_manager


def register_shutdown_callback(callback: Callable) -> None:
    """Register a shutdown callback (convenience function)"""
    shutdown_manager.register_shutdown_callback(callback)


def shutdown_application() -> None:
    """Shutdown the application (convenience function)"""
    shutdown_manager.shutdown()