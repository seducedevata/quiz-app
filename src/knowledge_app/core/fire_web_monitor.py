"""
🔥 FIRE Web Monitor - QtWebEngine Integration
Real-time training monitoring with beautiful web-based visualizations
"""

import time
import json
import logging
from typing import Dict, Any, Optional
from PyQt5.QtCore import QObject, pyqtSignal
from .fire_v21_estimator import FIREv21Estimator, TruthBasedPrediction

logger = logging.getLogger(__name__)


class FIREWebProgressWidget(QObject):
    """
    🔥 FIRE Web Progress Widget for QtWebEngine
    
    Provides beautiful real-time training visualizations via JavaScript
    """
    
    # Signals for web UI updates
    trainingStarted = pyqtSignal('QVariant')
    initialEstimate = pyqtSignal('QVariant') 
    realtimeUpdate = pyqtSignal('QVariant')
    trainingCompleted = pyqtSignal('QVariant')
    
    def __init__(self, bridge_parent):
        super().__init__()
        self.bridge = bridge_parent  # Reference to PythonBridge
        self.start_time = None
        self.total_steps = 0
        self.current_step = 0
        
        # Connect to bridge signals for web updates
        self.trainingStarted.connect(self._emit_to_web)
        self.initialEstimate.connect(self._emit_to_web)
        self.realtimeUpdate.connect(self._emit_to_web)
        self.trainingCompleted.connect(self._emit_to_web)
        
        logger.info("🔥 FIRE Web Progress Widget initialized")
    
    def _emit_to_web(self, data):
        """Emit data to web UI via the bridge"""
        if self.bridge:
            self.bridge.updateStatus.emit(f"🔥 FIRE: {data.get('message', 'Training update')}")
    
    def start_training(self, total_steps: int):
        """Called when training starts"""
        self.start_time = time.time()
        self.total_steps = total_steps
        self.current_step = 0
        
        data = {
            'type': 'training_started',
            'message': 'Training started with FIRE monitoring',
            'total_steps': total_steps,
            'timestamp': self.start_time
        }
        
        self.trainingStarted.emit(data)
        logger.info(f"🔥 FIRE: Training started with {total_steps} total steps")
    
    def update_initial_estimate(self, estimated_hours: float):
        """Update initial time estimate"""
        data = {
            'type': 'initial_estimate',
            'message': f'Initial estimate: {estimated_hours:.1f} hours',
            'estimated_hours': estimated_hours,
            'estimated_minutes': estimated_hours * 60,
            'confidence': 'High (Truth-based calculation)'
        }
        
        self.initialEstimate.emit(data)
        logger.info(f"🔥 FIRE: Initial estimate {estimated_hours:.1f} hours")
    
    def update_realtime_eta(self, eta_hours: float):
        """Update real-time ETA"""
        if not self.start_time:
            return
            
        elapsed_seconds = time.time() - self.start_time
        elapsed_hours = elapsed_seconds / 3600
        
        # Calculate progress percentage
        progress_percent = 0
        if self.total_steps > 0 and self.current_step > 0:
            progress_percent = (self.current_step / self.total_steps) * 100
        
        data = {
            'type': 'realtime_update',
            'message': f'ETA: {eta_hours:.1f}h | {progress_percent:.1f}% complete',
            'eta_hours': eta_hours,
            'eta_minutes': eta_hours * 60,
            'elapsed_hours': elapsed_hours,
            'elapsed_seconds': elapsed_seconds,
            'progress_percent': progress_percent,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'steps_per_hour': (self.current_step / elapsed_hours) if elapsed_hours > 0 else 0
        }
        
        self.realtimeUpdate.emit(data)
    
    def update_step_progress(self, current_step: int, loss: float = 0.0, accuracy: float = 0.0):
        """Update current training step"""
        self.current_step = current_step
        
        if current_step % 50 == 0:  # Update every 50 steps to avoid spam
            elapsed_seconds = time.time() - self.start_time if self.start_time else 0
            
            data = {
                'type': 'step_progress',
                'message': f'Step {current_step}/{self.total_steps} | Loss: {loss:.4f}',
                'current_step': current_step,
                'total_steps': self.total_steps,
                'loss': loss,
                'accuracy': accuracy,
                'elapsed_seconds': elapsed_seconds
            }
            
            self.realtimeUpdate.emit(data)
    
    def training_completed(self, success: bool, final_message: str):
        """Called when training completes"""
        elapsed_seconds = time.time() - self.start_time if self.start_time else 0
        elapsed_hours = elapsed_seconds / 3600
        
        data = {
            'type': 'training_completed',
            'message': final_message,
            'success': success,
            'total_time_hours': elapsed_hours,
            'total_time_seconds': elapsed_seconds,
            'final_step': self.current_step
        }
        
        self.trainingCompleted.emit(data)
        logger.info(f"🔥 FIRE: Training completed in {elapsed_hours:.2f} hours")


class FIREWebIntegration:
    """
    🔥 FIRE Web Integration Manager
    
    Connects FIRE v2.1 estimator with QtWebEngine UI
    """
    
    def __init__(self, bridge_parent):
        self.bridge = bridge_parent
        self.web_widget = FIREWebProgressWidget(bridge_parent)
        self.fire_estimator = None
        
        logger.info("🔥 FIRE Web Integration initialized")
    
    def create_fire_estimator(self) -> FIREv21Estimator:
        """Create FIRE estimator with web widget integration"""
        self.fire_estimator = FIREv21Estimator(ui_progress_widget=self.web_widget)
        return self.fire_estimator
    
    def start_training_session(self, trainer, train_dataset):
        """Start FIRE monitoring for training session"""
        if not self.fire_estimator:
            self.create_fire_estimator()
        
        # Initialize FIRE with trainer
        prediction = self.fire_estimator.initialize_with_trainer(trainer, train_dataset)
        
        # Start web monitoring
        total_steps = trainer.state.max_steps if trainer.state else 1000
        self.web_widget.start_training(total_steps)
        
        # Send initial estimate to web
        if prediction:
            self.web_widget.update_initial_estimate(prediction.estimated_hours)
        
        logger.info("🔥 FIRE: Training session started with web monitoring")
        return self.fire_estimator
    
    def get_web_widget(self) -> FIREWebProgressWidget:
        """Get the web progress widget for signal connections"""
        return self.web_widget