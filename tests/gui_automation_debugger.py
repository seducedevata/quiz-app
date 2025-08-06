"""
GUI Automation Debugger for Knowledge App

This script automatically navigates the GUI, clicks through training workflows,
detects errors in real-time, and provides comprehensive debugging information.
"""

import sys
import os
import time
import logging
import traceback
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QDialog, QMessageBox,
    QLabel, QProgressBar, QTabWidget, QComboBox
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtTest import QTest

# Import your app components
from main import ApplicationManager
from src.knowledge_app.ui.main_window import MainWindow
from src.knowledge_app.ui.training_dialog import AITrainingDialog
from src.knowledge_app.core.config_manager import ConfigManager

# Setup logging - suppress spam, only show important messages
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('gui_automation_debug.log'),
        logging.StreamHandler()
    ]
)

# Suppress specific noisy loggers
logging.getLogger('accelerate').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('bitsandbytes').setLevel(logging.ERROR)
logging.getLogger('peft').setLevel(logging.ERROR)
logging.getLogger('datasets').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class AutomationState(Enum):
    """States of the automation process"""
    STARTING = "starting"
    APP_LOADING = "app_loading"
    MAIN_MENU = "main_menu"
    CLICKING_TRAIN = "clicking_train"
    TRAINING_DIALOG = "training_dialog"
    SELECTING_MODEL = "selecting_model"
    TRAINING_STARTED = "training_started"
    MONITORING_TRAINING = "monitoring_training"
    ERROR_DETECTED = "error_detected"
    DEBUGGING = "debugging"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ErrorInfo:
    """Information about detected errors"""
    timestamp: datetime
    error_type: str
    message: str
    stack_trace: str
    screenshot_path: Optional[str] = None
    ui_state: Dict[str, Any] = None
    suggested_fix: Optional[str] = None

class GUIAutomationDebugger:
    """Main automation and debugging class"""
    
    def __init__(self):
        self.app: Optional[QApplication] = None
        self.main_window: Optional[MainWindow] = None
        self.training_dialog: Optional[AITrainingDialog] = None
        self.state = AutomationState.STARTING
        self.errors: List[ErrorInfo] = []
        self.screenshots_dir = Path("automation_screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        self.retry_count = 0
        self.max_retries = 3
        self.monitoring_active = False
        
    def run_automation(self) -> bool:
        """Run the complete automation workflow"""
        try:
            logger.info("🚀 Starting GUI Automation Debugger")
            
            # Step 1: Launch application
            if not self._launch_application():
                return False
                
            # Step 2: Navigate to training
            if not self._navigate_to_training():
                return False
                
            # Step 3: Select Mistral model
            if not self._select_mistral_model():
                return False
                
            # Step 4: Monitor training and errors
            if not self._monitor_training():
                return False
                
            logger.info("✅ Automation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Automation failed: {e}")
            self._handle_error("automation_failure", str(e), traceback.format_exc())
            return False
        finally:
            self._cleanup()
            
    def _launch_application(self) -> bool:
        """Launch the application and wait for it to load"""
        try:
            self.state = AutomationState.APP_LOADING
            logger.info("📱 Launching application...")
            
            # Create application manager
            app_manager = ApplicationManager()
            
            # Initialize and run app
            app_manager.initialize_logging()
            app_manager.initialize_managers()
            app_manager.create_application()
            
            self.app = app_manager.app
            
            # Create main window
            config = ConfigManager()
            self.main_window = MainWindow(config)
            
            # Show window and wait for it to be ready
            self.main_window.show()
            self._wait_for_window_ready()
            
            self.state = AutomationState.MAIN_MENU
            logger.info("✅ Application launched successfully")
            self._take_screenshot("app_launched")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to launch application: {e}")
            self._handle_error("launch_failure", str(e), traceback.format_exc())
            return False
            
    def _navigate_to_training(self) -> bool:
        """Navigate to the training dialog"""
        try:
            self.state = AutomationState.CLICKING_TRAIN
            logger.info("🎯 Looking for 'Train AI Model' button...")
            
            # Find the train button
            train_button = self._find_train_button()
            if not train_button:
                self._handle_error("button_not_found", "Train AI Model button not found", "")
                return False
                
            logger.info("🖱️ Clicking 'Train AI Model' button...")
            self._take_screenshot("before_train_click")
            
            # Click the button
            train_button.click()
            
            # Wait for training dialog to appear
            self._wait_for_training_dialog()
            
            self.state = AutomationState.TRAINING_DIALOG
            logger.info("✅ Training dialog opened")
            self._take_screenshot("training_dialog_opened")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to navigate to training: {e}")
            self._handle_error("navigation_failure", str(e), traceback.format_exc())
            return False
            
    def _select_mistral_model(self) -> bool:
        """Select the Mistral 7B model for training"""
        try:
            self.state = AutomationState.SELECTING_MODEL
            logger.info("🔥 Looking for Mistral-7B button...")

            # Debug: Check training data before clicking
            self._debug_training_setup()

            # Find the Mistral button
            mistral_button = self._find_mistral_button()
            if not mistral_button:
                self._handle_error("mistral_button_not_found", "Mistral-7B button not found", "")
                return False

            logger.info("🖱️ Clicking Mistral-7B QLoRA button...")
            self._take_screenshot("before_mistral_click")

            # Click the Mistral button
            mistral_button.click()

            # Wait a moment for processing and check for immediate errors
            QTest.qWait(3000)

            # Check if any error dialogs appeared immediately
            immediate_errors = self._find_error_dialogs()
            if immediate_errors:
                logger.error("🚨 Immediate error detected after clicking Mistral button")
                self._handle_training_errors(immediate_errors)
                return False

            self.state = AutomationState.TRAINING_STARTED
            logger.info("✅ Mistral model selected and training initiated")
            self._take_screenshot("mistral_selected")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to select Mistral model: {e}")
            self._handle_error("model_selection_failure", str(e), traceback.format_exc())
            return False

    def _debug_training_setup(self):
        """Debug training setup and configuration"""
        try:
            import os
            from pathlib import Path

            logger.info("🔍 Debugging training setup...")

            # Check current working directory
            cwd = os.getcwd()
            logger.info(f"📁 Current working directory: {cwd}")

            # Check if training data exists
            training_data_paths = [
                "lora_adapters_mistral/default/training_data_augmented_default.txt",
                Path(cwd) / "lora_adapters_mistral/default/training_data_augmented_default.txt",
                "data/training_data.txt",
                "training_data.txt"
            ]

            for path in training_data_paths:
                if os.path.exists(path):
                    logger.info(f"✅ Found training data: {path}")
                    # Check file size
                    size = os.path.getsize(path)
                    logger.info(f"📊 Training data size: {size} bytes")
                    break
                else:
                    logger.warning(f"❌ Training data not found: {path}")

            # Check GPU availability
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                else:
                    logger.warning("❌ CUDA not available")
            except Exception as e:
                logger.error(f"❌ Error checking GPU: {e}")

            # Check required directories
            required_dirs = [
                "data",
                "lora_adapters_mistral",
                "lora_adapters_mistral/default"
            ]

            for dir_path in required_dirs:
                if os.path.exists(dir_path):
                    logger.info(f"✅ Directory exists: {dir_path}")
                else:
                    logger.warning(f"❌ Directory missing: {dir_path}")

        except Exception as e:
            logger.error(f"Error in debug setup: {e}")
            
    def _monitor_training(self) -> bool:
        """Monitor training progress and detect errors"""
        try:
            self.state = AutomationState.MONITORING_TRAINING
            logger.info("👁️ Starting training monitoring...")
            
            self.monitoring_active = True
            start_time = time.time()
            max_monitoring_time = 300  # 5 minutes max
            
            while self.monitoring_active and (time.time() - start_time) < max_monitoring_time:
                # Check for error dialogs
                error_dialogs = self._find_error_dialogs()
                if error_dialogs:
                    self._handle_training_errors(error_dialogs)
                    return False

                # Check training progress (less frequently to reduce spam)
                if int(time.time() - start_time) % 10 == 0:  # Every 10 seconds
                    self._check_training_progress()

                # Take periodic screenshots
                if int(time.time() - start_time) % 30 == 0:  # Every 30 seconds
                    self._take_screenshot(f"monitoring_{int(time.time() - start_time)}")

                QTest.qWait(5000)  # Wait 5 seconds between checks (less spam)
                
            self.state = AutomationState.COMPLETED
            logger.info("✅ Training monitoring completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed during training monitoring: {e}")
            self._handle_error("monitoring_failure", str(e), traceback.format_exc())
            return False

    def _find_train_button(self) -> Optional[QPushButton]:
        """Find the Train AI Model button"""
        try:
            # Look for buttons with "Train" in the text
            buttons = self.main_window.findChildren(QPushButton)
            for button in buttons:
                if "Train" in button.text() and "AI" in button.text():
                    logger.info(f"Found train button: {button.text()}")
                    return button
            return None
        except Exception as e:
            logger.error(f"Error finding train button: {e}")
            return None

    def _find_mistral_button(self) -> Optional[QPushButton]:
        """Find the Mistral-7B training button"""
        try:
            if not self.training_dialog:
                return None

            buttons = self.training_dialog.findChildren(QPushButton)
            for button in buttons:
                if "Mistral" in button.text() and "7B" in button.text():
                    logger.info(f"Found Mistral button: {button.text()}")
                    return button
            return None
        except Exception as e:
            logger.error(f"Error finding Mistral button: {e}")
            return None

    def _find_error_dialogs(self) -> List[QMessageBox]:
        """Find any error dialogs currently displayed"""
        try:
            error_dialogs = []
            if self.app:
                for widget in self.app.allWidgets():
                    if isinstance(widget, QMessageBox) and widget.isVisible():
                        if "error" in widget.windowTitle().lower() or "Error" in widget.text():
                            error_dialogs.append(widget)
            return error_dialogs
        except Exception as e:
            logger.error(f"Error finding error dialogs: {e}")
            return []

    def _wait_for_window_ready(self, timeout: int = 10):
        """Wait for the main window to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.main_window and self.main_window.isVisible():
                QTest.qWait(1000)  # Additional wait for full initialization
                return
            QTest.qWait(100)
        raise TimeoutError("Main window did not become ready in time")

    def _wait_for_training_dialog(self, timeout: int = 10):
        """Wait for the training dialog to appear"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Look for training dialog
            dialogs = self.app.allWidgets()
            for widget in dialogs:
                if isinstance(widget, AITrainingDialog) and widget.isVisible():
                    self.training_dialog = widget
                    QTest.qWait(1000)  # Wait for dialog to fully load
                    return
            QTest.qWait(100)
        raise TimeoutError("Training dialog did not appear in time")

    def _take_screenshot(self, name: str):
        """Take a screenshot of the current state"""
        try:
            if not self.main_window:
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{name}.png"
            filepath = self.screenshots_dir / filename

            pixmap = self.main_window.grab()
            pixmap.save(str(filepath))

            logger.info(f"📸 Screenshot saved: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None

    def _handle_error(self, error_type: str, message: str, stack_trace: str, suggested_fix: str = None):
        """Handle and log errors with debugging information"""
        self.state = AutomationState.ERROR_DETECTED

        error_info = ErrorInfo(
            timestamp=datetime.now(),
            error_type=error_type,
            message=message,
            stack_trace=stack_trace,
            screenshot_path=self._take_screenshot(f"error_{error_type}"),
            ui_state=self._capture_ui_state(),
            suggested_fix=suggested_fix or self._suggest_fix(error_type)
        )

        self.errors.append(error_info)

        logger.error(f"🚨 ERROR DETECTED: {error_type}")
        logger.error(f"Message: {message}")
        logger.error(f"Stack trace: {stack_trace}")
        logger.error(f"Suggested fix: {error_info.suggested_fix}")

        # Save detailed error report
        self._save_error_report(error_info)

    def _handle_training_errors(self, error_dialogs: List[QMessageBox]):
        """Handle training-specific errors"""
        for dialog in error_dialogs:
            error_text = dialog.text()
            logger.error(f"🔥 Training error detected: {error_text}")

            # Extract specific error information and provide targeted fixes
            if "args.eval_strategy" in error_text:
                suggested_fix = "Add 'eval_dataset' parameter to training configuration or set 'eval_strategy' to 'no'"
            elif "REAL 7B training failed" in error_text:
                suggested_fix = "Check model availability and GPU memory. Ensure proper environment setup."
            elif "Training data not found" in error_text:
                suggested_fix = "Check if training data file exists. Path should be relative to project root. Verify 'lora_adapters_mistral/default/training_data_augmented_default.txt' exists."
            elif "Configuration issues found" in error_text:
                suggested_fix = "Check training data path, GPU memory (min 6GB), and output directory permissions."
            elif "GPU memory too low" in error_text:
                suggested_fix = "Reduce batch size, enable gradient checkpointing, or use CPU training mode."
            elif "Cannot create output directory" in error_text:
                suggested_fix = "Check write permissions for the output directory or change output path."
            else:
                suggested_fix = "Check training configuration and model compatibility. Enable debug logging for more details."

            self._handle_error(
                "training_error",
                error_text,
                "Training error dialog appeared",
                suggested_fix
            )

            # Take screenshot of error dialog
            self._take_screenshot("training_error_dialog")

            # Click OK to dismiss dialog
            dialog.accept()

    def _check_training_progress(self):
        """Check training progress and log status"""
        try:
            if not self.training_dialog:
                return

            # Look for progress indicators
            progress_bars = self.training_dialog.findChildren(QProgressBar)
            labels = self.training_dialog.findChildren(QLabel)

            # Track progress values to detect simulation vs real training
            progress_values = []
            status_messages = []

            for progress_bar in progress_bars:
                if progress_bar.isVisible():
                    value = progress_bar.value()
                    progress_values.append(value)

            # Check for status labels
            for label in labels:
                if label.isVisible() and ("Status:" in label.text() or "Epoch:" in label.text()):
                    status_text = label.text()
                    status_messages.append(status_text)

            # Detect if this is simulation vs real training
            self._detect_training_type(progress_values, status_messages)

        except Exception as e:
            logger.error(f"Error checking training progress: {e}")

    def _detect_training_type(self, progress_values: list, status_messages: list):
        """Detect if training is simulation or real 7B model training"""
        try:
            # Check for signs of real training first
            real_training_indicators = [
                any("Loading checkpoint shards" in msg for msg in status_messages),
                any("Loading mistralai/Mistral" in msg for msg in status_messages),
                any("REAL 7B PROGRESS" in msg for msg in status_messages),
                any("Setting up 4-bit quantization" in msg for msg in status_messages),
                any("Prepared training examples" in msg for msg in status_messages),
                any("Setting up trainer" in msg for msg in status_messages)
            ]

            # Check for signs of simulation (but only if no real training indicators)
            simulation_indicators = [
                any("Training Completed ✅" in msg for msg in status_messages),
                any("Epoch: 0/0" in msg for msg in status_messages) and not any(real_training_indicators),
                len([p for p in progress_values if p == -1]) > 3  # Multiple -1% readings without real training
            ]

            # Check for model loading indicators
            loading_indicators = [
                any("Loading checkpoint shards" in msg for msg in status_messages),
                any("Loading mistralai" in msg for msg in status_messages),
                any("Setting up 4-bit quantization" in msg for msg in status_messages)
            ]

            # Only report once to avoid spam
            if not hasattr(self, '_detection_reported'):
                self._detection_reported = {}

            if any(real_training_indicators) and not self._detection_reported.get('real_training'):
                print("🔥 REAL 7B MODEL TRAINING DETECTED!")
                print("✅ Mistral-7B is loading and will start training")
                self._detection_reported['real_training'] = True

            elif any(simulation_indicators) and not any(loading_indicators) and not self._detection_reported.get('simulation'):
                print("🎭 WARNING: Simulation mode detected instead of real training")
                self._detection_reported['simulation'] = True

            elif any(loading_indicators) and not self._detection_reported.get('loading'):
                print("📥 Model loading in progress... (this takes a few minutes)")
                self._detection_reported['loading'] = True

        except Exception as e:
            logger.error(f"Error detecting training type: {e}")

    def _capture_ui_state(self) -> Dict[str, Any]:
        """Capture current UI state for debugging"""
        try:
            state = {
                "automation_state": self.state.value,
                "main_window_visible": self.main_window.isVisible() if self.main_window else False,
                "training_dialog_visible": self.training_dialog.isVisible() if self.training_dialog else False,
                "active_widgets": [],
                "visible_buttons": []
            }

            if self.app:
                for widget in self.app.allWidgets():
                    if widget.isVisible():
                        widget_info = {
                            "type": type(widget).__name__,
                            "text": getattr(widget, 'text', lambda: '')() if hasattr(widget, 'text') else '',
                            "enabled": widget.isEnabled()
                        }
                        state["active_widgets"].append(widget_info)

                        if isinstance(widget, QPushButton):
                            state["visible_buttons"].append(widget_info)

            return state
        except Exception as e:
            logger.error(f"Error capturing UI state: {e}")
            return {"error": str(e)}

    def _suggest_fix(self, error_type: str) -> str:
        """Suggest fixes based on error type"""
        fixes = {
            "launch_failure": "Check if all dependencies are installed. Run 'pip install -r requirements.txt'",
            "button_not_found": "Verify UI layout. Check if ML features are enabled in config.",
            "mistral_button_not_found": "Ensure training dialog loaded properly. Check model availability.",
            "navigation_failure": "Check window focus and visibility. Retry with longer wait times.",
            "model_selection_failure": "Verify model configuration and GPU availability.",
            "training_error": "Check training configuration, model compatibility, and system resources.",
            "monitoring_failure": "Check system stability and memory usage."
        }
        return fixes.get(error_type, "Check logs for more details and retry the operation.")

    def _save_error_report(self, error_info: ErrorInfo):
        """Save detailed error report to file"""
        try:
            report_file = self.screenshots_dir / f"error_report_{error_info.timestamp.strftime('%Y%m%d_%H%M%S')}.json"

            report_data = {
                "timestamp": error_info.timestamp.isoformat(),
                "error_type": error_info.error_type,
                "message": error_info.message,
                "stack_trace": error_info.stack_trace,
                "screenshot_path": error_info.screenshot_path,
                "ui_state": error_info.ui_state,
                "suggested_fix": error_info.suggested_fix,
                "automation_state": self.state.value,
                "retry_count": self.retry_count
            }

            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"📋 Error report saved: {report_file}")

        except Exception as e:
            logger.error(f"Failed to save error report: {e}")

    def _cleanup(self):
        """Clean up resources"""
        try:
            self.monitoring_active = False

            if self.training_dialog:
                self.training_dialog.close()

            if self.main_window:
                self.main_window.close()

            if self.app:
                self.app.quit()

            logger.info("🧹 Cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def retry_automation(self) -> bool:
        """Retry the automation with increased timeouts"""
        if self.retry_count >= self.max_retries:
            logger.error(f"❌ Max retries ({self.max_retries}) reached. Automation failed.")
            return False

        self.retry_count += 1
        logger.info(f"🔄 Retrying automation (attempt {self.retry_count}/{self.max_retries})")

        # Clean up previous attempt
        self._cleanup()
        time.sleep(5)  # Wait before retry

        # Reset state
        self.state = AutomationState.STARTING
        self.app = None
        self.main_window = None
        self.training_dialog = None

        return self.run_automation()

    def generate_summary_report(self) -> str:
        """Generate a summary report of the automation run"""
        report = f"""
🤖 GUI Automation Debugger Summary Report
==========================================

Automation State: {self.state.value}
Total Errors: {len(self.errors)}
Retry Count: {self.retry_count}/{self.max_retries}
Screenshots Directory: {self.screenshots_dir}

"""

        if self.errors:
            report += "🚨 ERRORS DETECTED:\n"
            for i, error in enumerate(self.errors, 1):
                report += f"""
Error #{i}:
  Type: {error.error_type}
  Time: {error.timestamp}
  Message: {error.message}
  Suggested Fix: {error.suggested_fix}
  Screenshot: {error.screenshot_path}
"""
        else:
            report += "✅ No errors detected during automation.\n"

        return report


def main():
    """Main function to run the GUI automation debugger"""
    print("🔥 GUI Automation Debugger")
    print("=" * 30)

    debugger = GUIAutomationDebugger()

    try:
        success = debugger.run_automation()

        if not success and debugger.retry_count < debugger.max_retries:
            print("🔄 Retrying...")
            success = debugger.retry_automation()

        # Show final result
        if success:
            print("\n✅ SUCCESS: Training workflow is working!")
        else:
            print("\n❌ FAILED: Check automation_screenshots/ for details")

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
        debugger._cleanup()
        return 1
    except Exception as e:
        print(f"\n💥 Error: {e}")
        debugger._cleanup()
        return 1


if __name__ == "__main__":
    sys.exit(main())
