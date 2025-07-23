"""
🔧 Shutdown Manager

This module provides centralized shutdown coordination for all application components,
eliminating orphaned processes and ensuring clean application termination.

CRITICAL FIX: Replaces the incomplete shutdown procedure with:
- Centralized shutdown orchestration
- Ordered component termination
- Graceful cleanup without forced termination
- Comprehensive resource cleanup
"""

import logging
import threading
import time
import atexit
from typing import List, Callable, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ShutdownPriority(Enum):
    """Shutdown priority levels (higher numbers shut down first)"""
    CRITICAL = 100  # Session data, user settings
    HIGH = 80      # Database connections, file handles
    MEDIUM = 60    # Background threads, workers
    LOW = 40       # UI components, monitors
    CLEANUP = 20   # Temporary files, caches

@dataclass
class ShutdownTask:
    """A registered shutdown task"""
    name: str
    callback: Callable[[], None]
    priority: ShutdownPriority
    timeout: float = 5.0
    component: str = "unknown"

class ShutdownManager:
    """
    🔧 FIX: Centralized shutdown management
    
    This manager ensures all components are shut down gracefully in the correct order,
    eliminating the need for emergency os._exit() calls.
    """
    
    def __init__(self):
        self.shutdown_tasks: List[ShutdownTask] = []
        self.shutdown_lock = threading.RLock()
        self.is_shutting_down = False
        self.shutdown_complete = False
        
        # Register emergency shutdown handler
        atexit.register(self._emergency_shutdown)
        
        logger.info("🔧 ShutdownManager initialized")
    
    def register_shutdown_task(self, name: str, callback: Callable[[], None], 
                             priority: ShutdownPriority = ShutdownPriority.MEDIUM,
                             timeout: float = 5.0, component: str = "unknown"):
        """
        Register a shutdown task
        
        Args:
            name: Descriptive name for the task
            callback: Function to call during shutdown
            priority: Shutdown priority level
            timeout: Maximum time to wait for task completion
            component: Component name for logging
        """
        try:
            with self.shutdown_lock:
                if self.is_shutting_down:
                    logger.warning(f"⚠️ Cannot register shutdown task '{name}' - shutdown in progress")
                    return
                
                task = ShutdownTask(
                    name=name,
                    callback=callback,
                    priority=priority,
                    timeout=timeout,
                    component=component
                )
                
                self.shutdown_tasks.append(task)
                logger.debug(f"✅ Registered shutdown task: {name} ({component})")
                
        except Exception as e:
            logger.error(f"❌ Failed to register shutdown task '{name}': {e}")
    
    def unregister_shutdown_task(self, name: str):
        """Unregister a shutdown task"""
        try:
            with self.shutdown_lock:
                self.shutdown_tasks = [task for task in self.shutdown_tasks if task.name != name]
                logger.debug(f"🗑️ Unregistered shutdown task: {name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to unregister shutdown task '{name}': {e}")
    
    def shutdown_all(self, timeout: float = 30.0) -> bool:
        """
        Execute all registered shutdown tasks in priority order
        
        Args:
            timeout: Maximum total time to wait for all shutdowns
            
        Returns:
            bool: True if all tasks completed successfully
        """
        try:
            with self.shutdown_lock:
                if self.is_shutting_down:
                    logger.warning("⚠️ Shutdown already in progress")
                    return self.shutdown_complete
                
                self.is_shutting_down = True
                logger.info("🔧 Starting coordinated application shutdown")
                
                start_time = time.time()
                success_count = 0
                total_tasks = len(self.shutdown_tasks)
                
                # Sort tasks by priority (highest first)
                sorted_tasks = sorted(self.shutdown_tasks, key=lambda t: t.priority.value, reverse=True)
                
                # Group tasks by priority for parallel execution within same priority
                priority_groups = {}
                for task in sorted_tasks:
                    if task.priority not in priority_groups:
                        priority_groups[task.priority] = []
                    priority_groups[task.priority].append(task)
                
                # Execute each priority group
                for priority in sorted(priority_groups.keys(), key=lambda p: p.value, reverse=True):
                    tasks = priority_groups[priority]
                    logger.info(f"🔧 Shutting down {len(tasks)} {priority.name} priority tasks")
                    
                    # Execute tasks in this priority group (can be parallel)
                    group_success = self._execute_task_group(tasks)
                    success_count += sum(group_success)
                    
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        logger.warning(f"⚠️ Shutdown timeout exceeded ({elapsed:.1f}s > {timeout}s)")
                        break
                
                elapsed = time.time() - start_time
                self.shutdown_complete = (success_count == total_tasks)
                
                if self.shutdown_complete:
                    logger.info(f"✅ Coordinated shutdown completed successfully in {elapsed:.1f}s")
                    logger.info(f"✅ All {total_tasks} shutdown tasks executed")
                else:
                    logger.warning(f"⚠️ Shutdown completed with issues: {success_count}/{total_tasks} tasks successful")
                
                return self.shutdown_complete
                
        except Exception as e:
            logger.error(f"❌ Shutdown coordination failed: {e}")
            return False
    
    def _execute_task_group(self, tasks: List[ShutdownTask]) -> List[bool]:
        """Execute a group of tasks with the same priority"""
        results = []
        threads = []
        
        # Create threads for each task
        for task in tasks:
            def task_wrapper(t=task):
                success = self._execute_single_task(t)
                results.append(success)
            
            thread = threading.Thread(target=task_wrapper, name=f"shutdown-{task.name}")
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads with timeout
        for i, thread in enumerate(threads):
            try:
                thread.join(timeout=tasks[i].timeout)
                if thread.is_alive():
                    logger.warning(f"⚠️ Shutdown task '{tasks[i].name}' timed out")
                    results.append(False)
            except Exception as e:
                logger.error(f"❌ Error waiting for shutdown task '{tasks[i].name}': {e}")
                results.append(False)
        
        return results
    
    def _execute_single_task(self, task: ShutdownTask) -> bool:
        """Execute a single shutdown task"""
        try:
            logger.debug(f"🔧 Executing shutdown task: {task.name} ({task.component})")
            start_time = time.time()
            
            task.callback()
            
            elapsed = time.time() - start_time
            logger.debug(f"✅ Shutdown task '{task.name}' completed in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Shutdown task '{task.name}' failed: {e}")
            return False
    
    def _emergency_shutdown(self):
        """Emergency shutdown handler called by atexit"""
        if not self.is_shutting_down:
            logger.warning("🚨 Emergency shutdown triggered - executing critical tasks only")
            
            # Execute only critical priority tasks quickly
            critical_tasks = [task for task in self.shutdown_tasks 
                            if task.priority == ShutdownPriority.CRITICAL]
            
            for task in critical_tasks:
                try:
                    logger.warning(f"🚨 Emergency execution: {task.name}")
                    task.callback()
                except Exception as e:
                    logger.error(f"❌ Emergency task '{task.name}' failed: {e}")
    
    def get_shutdown_status(self) -> Dict[str, Any]:
        """Get current shutdown status"""
        with self.shutdown_lock:
            return {
                "is_shutting_down": self.is_shutting_down,
                "shutdown_complete": self.shutdown_complete,
                "registered_tasks": len(self.shutdown_tasks),
                "tasks_by_priority": {
                    priority.name: len([t for t in self.shutdown_tasks if t.priority == priority])
                    for priority in ShutdownPriority
                }
            }
    
    def force_shutdown(self):
        """🔧 DEPRECATED: Force shutdown - should not be needed with proper coordination"""
        logger.warning("🚨 DEPRECATED: force_shutdown() called - this indicates incomplete shutdown registration")
        logger.warning("🚨 All components should register proper shutdown tasks instead")
        
        # Still provide emergency functionality
        import os
        logger.error("🚨 EMERGENCY: Forcing application termination")
        os._exit(1)

# Global instance
_shutdown_manager: Optional[ShutdownManager] = None
_manager_lock = threading.RLock()

def get_shutdown_manager() -> ShutdownManager:
    """Get the global shutdown manager instance"""
    global _shutdown_manager
    with _manager_lock:
        if _shutdown_manager is None:
            _shutdown_manager = ShutdownManager()
        return _shutdown_manager

# Convenience functions
def register_shutdown_task(name: str, callback: Callable[[], None], 
                         priority: ShutdownPriority = ShutdownPriority.MEDIUM,
                         timeout: float = 5.0, component: str = "unknown"):
    """Register a shutdown task with the global manager"""
    manager = get_shutdown_manager()
    manager.register_shutdown_task(name, callback, priority, timeout, component)

def shutdown_application(timeout: float = 30.0) -> bool:
    """Shutdown the entire application gracefully"""
    manager = get_shutdown_manager()
    return manager.shutdown_all(timeout)
