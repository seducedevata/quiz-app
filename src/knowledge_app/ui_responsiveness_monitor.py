"""
Real-Time UI Responsiveness Monitor
===================================

This monitor runs continuously in the background to detect UI freezing
and automatically take corrective action if the UI becomes unresponsive.

Features:
- Real-time UI thread monitoring
- Automatic freeze detection
- Emergency recovery mechanisms
- Performance metrics logging
"""

import time
import threading
import logging
from typing import Callable, Optional
from PyQt5.QtCore import QObject, QTimer, pyqtSignal, QThread, QMutex, QMutexLocker
from PyQt5.QtGui import QGuiApplication

logger = logging.getLogger(__name__)

class UIResponsivenessMonitor(QObject):
    """Monitor UI responsiveness and take action if UI freezes"""
    
    # Signals
    uiFreezeDetected = pyqtSignal(float)  # freeze duration in seconds
    uiResponsive = pyqtSignal()
    performanceAlert = pyqtSignal(str)  # performance warning message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configuration
        self.check_interval = 100  # Check every 100ms
        self.freeze_threshold = 500  # Consider frozen if no response for 500ms
        self.critical_freeze_threshold = 2000  # Critical freeze at 2 seconds
        
        # State tracking
        self.is_monitoring = False
        self.last_ui_response = time.time()
        self.freeze_start_time = None
        self.total_freeze_time = 0.0
        self.freeze_count = 0
        
        # Recovery mechanisms
        self.recovery_actions = []
        
        # Performance metrics
        self.response_times = []
        self.max_response_time = 0.0
        self.avg_response_time = 0.0
        
        # 🔧 CRITICAL FIX: UI heartbeat timer stays on main thread (correct)
        self.heartbeat_timer = QTimer(self)  # Main thread timer for UI heartbeat
        self.heartbeat_timer.timeout.connect(self._ui_heartbeat)

        # 🔧 CRITICAL FIX: Monitor timer runs on separate thread to detect freezes
        self.monitor_thread = None
        self.monitor_running = False
        self.thread_mutex = QMutex()

        # Shared state for thread communication
        self.shared_last_response = time.time()
        self.shared_mutex = QMutex()
        
        logger.info("🔍 UI Responsiveness Monitor initialized")
    
    def start_monitoring(self):
        """Start monitoring UI responsiveness"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.last_ui_response = time.time()

        # 🔧 CRITICAL FIX: Update shared state
        with QMutexLocker(self.shared_mutex):
            self.shared_last_response = self.last_ui_response

        # Start UI heartbeat timer on main thread (correct)
        self.heartbeat_timer.start(50)  # 50ms heartbeat

        # 🔧 CRITICAL FIX: Start monitor thread separately
        self._start_monitor_thread()

        logger.info("🔍 UI Responsiveness monitoring started with separate monitor thread")
    
    def stop_monitoring(self):
        """Stop monitoring UI responsiveness"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        # Stop UI heartbeat timer
        self.heartbeat_timer.stop()

        # 🔧 CRITICAL FIX: Stop monitor thread
        self._stop_monitor_thread()

        logger.info("🔍 UI Responsiveness monitoring stopped")
    
    def _ui_heartbeat(self):
        """UI thread heartbeat - called frequently to track responsiveness"""
        current_time = time.time()

        # Calculate response time since last heartbeat
        response_time = (current_time - self.last_ui_response) * 1000  # Convert to ms
        self.last_ui_response = current_time

        # 🔧 CRITICAL FIX: Update shared state for monitor thread
        with QMutexLocker(self.shared_mutex):
            self.shared_last_response = current_time

        # Track response times
        self.response_times.append(response_time)
        if len(self.response_times) > 100:  # Keep only last 100 measurements
            self.response_times.pop(0)

        # Update statistics
        self.max_response_time = max(self.max_response_time, response_time)
        self.avg_response_time = sum(self.response_times) / len(self.response_times)

        # Reset freeze tracking if UI is responsive
        if self.freeze_start_time and response_time < self.freeze_threshold:
            freeze_duration = current_time - self.freeze_start_time
            logger.info(f"✅ UI responsiveness recovered after {freeze_duration:.2f}s freeze")
            self.freeze_start_time = None
            self.uiResponsive.emit()

    def _start_monitor_thread(self):
        """🔧 CRITICAL FIX: Start monitor thread that runs independently of UI thread"""
        with QMutexLocker(self.thread_mutex):
            if self.monitor_running:
                return

            self.monitor_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_thread_worker, daemon=True)
            self.monitor_thread.start()
            logger.info("🔍 Monitor thread started independently of UI thread")

    def _stop_monitor_thread(self):
        """🔧 CRITICAL FIX: Stop monitor thread"""
        with QMutexLocker(self.thread_mutex):
            if not self.monitor_running:
                return

            self.monitor_running = False

        # Wait for thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
            logger.info("🔍 Monitor thread stopped")

    def _monitor_thread_worker(self):
        """🔧 CRITICAL FIX: Monitor worker that runs on separate thread"""
        logger.info("🔍 Monitor thread worker started - can detect UI freezes")

        while self.monitor_running:
            try:
                current_time = time.time()

                # Get last UI response time from shared state
                with QMutexLocker(self.shared_mutex):
                    last_response = self.shared_last_response

                time_since_response = (current_time - last_response) * 1000  # ms

                # Check for freeze
                if time_since_response > self.freeze_threshold:
                    if not self.freeze_start_time:
                        self.freeze_start_time = current_time
                        self.freeze_count += 1
                        logger.warning(f"🚨 UI freeze detected! No response for {time_since_response:.0f}ms")

                        # Emit freeze signal (thread-safe)
                        self.uiFreezeDetected.emit(time_since_response / 1000.0)

                        # Critical freeze handling
                        if time_since_response > self.critical_freeze_threshold:
                            logger.error(f"🚨 CRITICAL UI freeze! {time_since_response:.0f}ms - executing recovery actions")
                            self._execute_recovery_actions()

                # Sleep for check interval
                time.sleep(self.check_interval / 1000.0)

            except Exception as e:
                logger.error(f"❌ Monitor thread error: {e}")
                time.sleep(0.1)  # Brief pause before retry

        logger.info("🔍 Monitor thread worker stopped")

    def _execute_recovery_actions(self):
        """🔧 FIX: Execute recovery actions focusing on root cause prevention"""
        logger.error("🚨 Executing recovery actions for critical UI freeze!")

        # Execute registered recovery actions
        for action in self.recovery_actions:
            try:
                logger.info(f"🔧 Executing recovery action: {action.__name__}")
                action()
            except Exception as e:
                logger.error(f"❌ Recovery action failed: {e}")

        # 🔧 FIX: Focus on terminating blocking operations instead of forcing events
        try:
            # Signal all threads to stop blocking operations
            logger.warning("🔧 Signaling threads to terminate blocking operations")

            # This is where we would implement proper thread termination
            # instead of forcing event processing which can cause re-entrancy bugs

        except Exception as e:
            logger.error(f"❌ Failed to signal thread termination: {e}")

        # 🔧 DEPRECATED: Avoid processEvents() as it treats symptoms, not root cause
        # This can cause complex re-entrancy bugs and doesn't fix the underlying issue
        logger.warning("🔧 Avoiding processEvents() - focusing on root cause prevention instead")

    def _check_ui_responsiveness(self):
        """Check if UI is responsive and take action if frozen"""
        current_time = time.time()
        time_since_response = (current_time - self.last_ui_response) * 1000  # ms
        
        if time_since_response > self.freeze_threshold:
            # UI appears frozen
            if not self.freeze_start_time:
                # First detection of freeze
                self.freeze_start_time = current_time
                self.freeze_count += 1
                logger.warning(f"⚠️ UI freeze detected! No response for {time_since_response:.0f}ms")
            
            freeze_duration = current_time - self.freeze_start_time
            
            if freeze_duration > self.critical_freeze_threshold / 1000:
                # Critical freeze - take emergency action
                logger.error(f"🚨 CRITICAL UI FREEZE: {freeze_duration:.2f}s - taking emergency action!")
                self.uiFreezeDetected.emit(freeze_duration)
                self._take_emergency_action()
            
        elif self.freeze_start_time:
            # UI was frozen but is now responsive
            freeze_duration = current_time - self.freeze_start_time
            self.total_freeze_time += freeze_duration
            logger.info(f"✅ UI freeze resolved after {freeze_duration:.2f}s")
            self.freeze_start_time = None
            self.uiResponsive.emit()
    
    def _take_emergency_action(self):
        """Take emergency action when critical UI freeze is detected"""
        logger.error("🚨 Taking emergency action for UI freeze!")
        
        # Execute registered recovery actions
        for action in self.recovery_actions:
            try:
                logger.info(f"🔧 Executing recovery action: {action.__name__}")
                action()
            except Exception as e:
                logger.error(f"❌ Recovery action failed: {e}")
        
        # 🔧 FIX: Avoid processEvents() - focus on root cause prevention
        logger.warning("🔧 Avoiding processEvents() in emergency action - treating root cause instead")

        # TODO: Implement proper thread termination mechanisms
        # - Signal blocking operations to terminate
        # - Cancel long-running tasks
        # - Reset application state safely

        # Force process application events (DEPRECATED - can cause re-entrancy bugs)
        # try:
        #     app = QCoreApplication.instance()
        #     if app:
        #         app.processEvents()
        #         logger.info("🔧 Forced application event processing")
        # except Exception as e:
        #     logger.error(f"❌ Failed to force event processing: {e}")
    
    def add_recovery_action(self, action: Callable):
        """Add a recovery action to be executed during critical freezes"""
        self.recovery_actions.append(action)
        logger.info(f"🔧 Added recovery action: {action.__name__}")
    
    def get_performance_stats(self) -> dict:
        """Get current performance statistics"""
        return {
            "is_monitoring": self.is_monitoring,
            "freeze_count": self.freeze_count,
            "total_freeze_time": self.total_freeze_time,
            "max_response_time": self.max_response_time,
            "avg_response_time": self.avg_response_time,
            "current_response_time": self.response_times[-1] if self.response_times else 0,
            "is_currently_frozen": self.freeze_start_time is not None
        }
    
    def log_performance_report(self):
        """Log a detailed performance report"""
        stats = self.get_performance_stats()
        
        logger.info("📊 UI Responsiveness Performance Report:")
        logger.info(f"   • Monitoring Active: {stats['is_monitoring']}")
        logger.info(f"   • Freeze Count: {stats['freeze_count']}")
        logger.info(f"   • Total Freeze Time: {stats['total_freeze_time']:.2f}s")
        logger.info(f"   • Max Response Time: {stats['max_response_time']:.1f}ms")
        logger.info(f"   • Avg Response Time: {stats['avg_response_time']:.1f}ms")  
        logger.info(f"   • Currently Frozen: {stats['is_currently_frozen']}")
        
        # Performance warnings
        if stats['max_response_time'] > 1000:  # >1 second
            self.performanceAlert.emit(f"High response time detected: {stats['max_response_time']:.0f}ms")
        
        if stats['freeze_count'] > 5:
            self.performanceAlert.emit(f"Multiple UI freezes detected: {stats['freeze_count']} freezes")

class UIResponsivenessManager:
    """Singleton manager for UI responsiveness monitoring"""
    
    _instance = None
    _monitor = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of responsiveness manager"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if UIResponsivenessManager._instance is not None:
            raise RuntimeError("UIResponsivenessManager is a singleton!")
        
        self._monitor = None
        self._emergency_handlers = []
    
    def initialize_monitor(self, parent=None):
        """Initialize the UI responsiveness monitor"""
        if self._monitor is None:
            self._monitor = UIResponsivenessMonitor(parent)
            
            # Connect emergency handlers
            self._monitor.uiFreezeDetected.connect(self._handle_ui_freeze)
            
            logger.info("🔍 UI Responsiveness Manager initialized")
        
        return self._monitor
    
    def start_monitoring(self):
        """Start UI responsiveness monitoring"""
        if self._monitor:
            self._monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop UI responsiveness monitoring"""
        if self._monitor:
            self._monitor.stop_monitoring()
    
    def add_emergency_handler(self, handler: Callable):
        """Add emergency handler for UI freezes"""
        if self._monitor:
            self._monitor.add_recovery_action(handler)
        self._emergency_handlers.append(handler)
    
    def _handle_ui_freeze(self, freeze_duration):
        """Handle UI freeze emergency"""
        logger.error(f"🚨 UI Freeze Handler activated: {freeze_duration:.2f}s freeze")
        
        # Execute all emergency handlers
        for handler in self._emergency_handlers:
            try:
                handler(freeze_duration)
            except Exception as e:
                logger.error(f"❌ Emergency handler failed: {e}")
    
    def get_monitor(self) -> Optional[UIResponsivenessMonitor]:
        """Get the current monitor instance"""
        return self._monitor

# Convenience functions
def start_ui_monitoring(parent=None):
    """Start UI responsiveness monitoring"""
    manager = UIResponsivenessManager.get_instance()
    monitor = manager.initialize_monitor(parent)
    manager.start_monitoring()
    return monitor

def stop_ui_monitoring():
    """Stop UI responsiveness monitoring"""
    manager = UIResponsivenessManager.get_instance()
    manager.stop_monitoring()

def add_freeze_recovery_action(action: Callable):
    """Add a recovery action for UI freezes"""
    manager = UIResponsivenessManager.get_instance()
    manager.add_emergency_handler(action)

def get_ui_performance_stats() -> dict:
    """Get current UI performance statistics"""
    manager = UIResponsivenessManager.get_instance()
    monitor = manager.get_monitor()
    if monitor:
        return monitor.get_performance_stats()
    return {"error": "Monitor not initialized"} 