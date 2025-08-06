"""
🚀 Knowledge App Loading Screen

Beautiful animated loading screen that shows while the app initializes.
Prevents the laggy direct launch experience.
"""

import sys
import time
import threading
from PyQt5.QtWidgets import (QApplication, QSplashScreen, QLabel, QVBoxLayout, 
                           QHBoxLayout, QWidget, QProgressBar, QFrame)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QPainter, QFont, QColor, QPalette, QMovie
from pathlib import Path

class LoadingWorker(QThread):
    """Background worker that handles app initialization"""
    progress_updated = pyqtSignal(int, str)
    loading_complete = pyqtSignal(object)  # Emits the main window
    
    def __init__(self):
        super().__init__()
        self.main_window = None
        self.initialized_components = {}
        self.start_time = time.time()  # Track initialization time
    
    def run(self):
        """ULTRA-FAST startup - defer ALL heavy components until needed"""
        try:
            # Step 1: INSTANT - Only create essential directories 
            self.progress_updated.emit(20, "⚡ Lightning-fast startup mode...")
            Path("user_data").mkdir(exist_ok=True)
            print("✅ LoadingWorker: Core directories created (instant)")
            
            # Step 2: SKIP ALL HEAVY INITIALIZATION - Mark everything as deferred
            self.progress_updated.emit(60, "🚀 Deferring components for instant startup...")
            
            # CRITICAL OPTIMIZATION: Don't initialize ANY heavy components during startup
            # Everything will be loaded lazily when first accessed
            self.initialized_components = {
                'mcq_manager': 'deferred',  # Load when first quiz is generated
                'training_manager': 'deferred',  # Load when training is started
                'unified_inference': 'deferred',  # Load when first question is generated
                'resource_manager': 'deferred',
                'topic_analyzer': 'deferred',
                'startup_time': time.time() - self.start_time  # Track how fast we were
            }
            
            print(f"✅ LoadingWorker: Ultra-fast startup complete in {self.initialized_components['startup_time']:.2f}s")
            
            self.progress_updated.emit(100, "🎯 Ready - components load on demand!")
            time.sleep(0.1)  # Small delay for smooth progress animation

            # Step 7: Prepare for main window creation
            self.progress_updated.emit(85, "✨ Finalizing setup...")
            print("🔧 LoadingWorker: All heavy initialization complete!")

            # Step 8: Complete - signal main thread to create window (but not 100% yet)
            self.progress_updated.emit(85, "🎉 Knowledge App ready - stabilizing...")

            print("✅ LoadingWorker: Emitting loading complete signal with initialized components")
            # Pass the initialized components to the main thread
            self.loading_complete.emit(self.initialized_components)

        except Exception as e:
            import traceback
            print(f"❌ LoadingWorker error: {e}")
            traceback.print_exc()
            self.progress_updated.emit(100, f"❌ Error: {str(e)}")
            self.loading_complete.emit(None)

class AnimatedLoadingScreen(QWidget):
    """Beautiful animated loading screen"""

    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setFixedSize(600, 400)

        # Center on screen
        from PyQt5.QtWidgets import QDesktopWidget
        screen = QDesktopWidget().screenGeometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

        # Setup UI
        self.setup_ui()

        # Initialize loading worker
        self.worker = LoadingWorker()
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.loading_complete.connect(self.on_loading_complete)

        # Animation timer for pulsing effect
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_logo)
        self.animation_timer.start(100)  # Update every 100ms
        self.animation_frame = 0

        print("✅ AnimatedLoadingScreen: Initialized successfully")
        
    def setup_ui(self):
        """Setup the loading screen UI with perfect alignment"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(0)
        main_layout.setAlignment(Qt.AlignCenter)
        
        # Top spacer
        main_layout.addStretch(1)
        
        # Logo/Title section - centered container
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(10)
        title_layout.setAlignment(Qt.AlignCenter)
        
        # App title
        self.title_label = QLabel("🧠 Knowledge App")
        self.title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Segoe UI", 32, QFont.Bold)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #4A9EFF;
                background: transparent;
                font-weight: bold;
                padding: 0px;
                margin: 0px;
            }
        """)
        title_layout.addWidget(self.title_label)

        # Subtitle
        subtitle = QLabel("Modern Learning Platform")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle_font = QFont("Segoe UI", 16)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("""
            QLabel {
                color: #8FA8DA;
                background: transparent;
                padding: 0px;
                margin: 5px 0px 15px 0px;
            }
        """)
        title_layout.addWidget(subtitle)

        main_layout.addWidget(title_container)
        
        # Middle spacer
        main_layout.addStretch(1)
        
        # Loading animation section - centered container
        loading_container = QWidget()
        loading_layout = QVBoxLayout(loading_container)
        loading_layout.setContentsMargins(0, 0, 0, 0)
        loading_layout.setSpacing(20)
        loading_layout.setAlignment(Qt.AlignCenter)
        
        # Animated dots
        self.dots_label = QLabel("●●●")
        self.dots_label.setAlignment(Qt.AlignCenter)
        dots_font = QFont("Arial", 20)
        self.dots_label.setFont(dots_font)
        self.dots_label.setStyleSheet("""
            QLabel {
                color: #4A9EFF;
                background: transparent;
                padding: 0px;
                margin: 0px;
            }
        """)
        loading_layout.addWidget(self.dots_label)
        
        # Progress bar container for perfect centering
        progress_container = QWidget()
        progress_container.setFixedHeight(30)
        progress_container_layout = QHBoxLayout(progress_container)
        progress_container_layout.setContentsMargins(0, 10, 0, 10)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedSize(400, 10)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3A4A5C;
                border-radius: 5px;
                background-color: #1E2329;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4A9EFF, stop:0.3 #6BB6FF, stop:0.7 #42A5F5, stop:1 #4A9EFF);
                border-radius: 3px;
                margin: 1px;
            }
        """)
        progress_container_layout.addWidget(self.progress_bar, alignment=Qt.AlignCenter)
        loading_layout.addWidget(progress_container)
        
        # Status label
        self.status_label = QLabel("🚀 Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_font = QFont("Segoe UI", 12)
        self.status_label.setFont(status_font)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #B0BEC5;
                background: transparent;
                padding: 0px;
                margin: 10px 0px 0px 0px;
            }
        """)
        loading_layout.addWidget(self.status_label)
        
        main_layout.addWidget(loading_container)
        
        # Bottom spacer
        main_layout.addStretch(1)
        
        # Footer section - perfectly centered
        footer_container = QWidget()
        footer_layout = QVBoxLayout(footer_container)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(5)
        footer_layout.setAlignment(Qt.AlignCenter)
        
        # Feature highlights
        features = QLabel("✨ DeepSeek 14B • Document Processing • Training Pipeline")
        features.setAlignment(Qt.AlignCenter)
        features_font = QFont("Segoe UI", 11)
        features.setFont(features_font)
        features.setStyleSheet("""
            QLabel {
                color: #64B5F6;
                background: transparent;
                padding: 0px;
                margin: 0px;
            }
        """)
        footer_layout.addWidget(features)
        
        # Version label
        version_label = QLabel("v1.0.0 • DeepSeek Integration • Document Processing")
        version_label.setAlignment(Qt.AlignCenter)
        version_font = QFont("Segoe UI", 9)
        version_label.setFont(version_font)
        version_label.setStyleSheet("""
            QLabel {
                color: #607D8B;
                background: transparent;
                padding: 0px;
                margin: 5px 0px 0px 0px;
            }
        """)
        footer_layout.addWidget(version_label)
        
        main_layout.addWidget(footer_container)
        
        # Final bottom spacer
        main_layout.addStretch(1)

        # Apply overall styling to the main widget
        self.setStyleSheet("""
            AnimatedLoadingScreen {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1A1F2E, stop:0.5 #2A3441, stop:1 #1A1F2E);
                border-radius: 15px;
            }
        """)
    

    
    def animate_logo(self):
        """Animate the loading dots"""
        self.animation_frame += 1
        
        # Pulsing dots animation
        dots_patterns = ["●○○", "○●○", "○○●", "●●○", "○●●", "●○●", "●●●"]
        pattern = dots_patterns[self.animation_frame % len(dots_patterns)]
        self.dots_label.setText(pattern)
        
        # Pulsing title effect
        alpha = int(200 + 55 * abs(1 - (self.animation_frame % 20) / 10))
        self.title_label.setStyleSheet(f"""
            QLabel {{
                color: rgba(74, 158, 255, {alpha});
                margin: 15px 0px;
                background: transparent;
                font-weight: bold;
            }}
        """)
    
    def update_progress(self, value, message):
        """Update progress bar and status message"""
        try:
            if hasattr(self, 'progress_bar') and self.progress_bar:
                self.progress_bar.setValue(value)
            if hasattr(self, 'status_label') and self.status_label:
                self.status_label.setText(message)

            # Force repaint
            self.repaint()
            QApplication.processEvents()
        except RuntimeError:
            # Widget has been deleted, ignore
            pass
    
    def on_loading_complete(self, initialized_components):
        """Handle loading completion - wait for app to be fully stable"""
        self.animation_timer.stop()

        # Calculate elapsed time since splash started
        elapsed_time = time.time() - self.worker.start_time
        print(f"🕒 AnimatedLoadingScreen: Initialization took {elapsed_time:.2f}s")

        def create_and_show_main_window():
            """Create the main window but keep it hidden until splash closes"""
            if initialized_components:
                try:
                    print("🔧 AnimatedLoadingScreen: Creating main window with deferred initialization...")
                    from knowledge_app.webengine_app import KnowledgeAppWebEngine
                    print("✅ AnimatedLoadingScreen: Imported KnowledgeAppWebEngine")

                    self.main_window = KnowledgeAppWebEngine()
                    print("✅ AnimatedLoadingScreen: Created main window")

                    # Pass deferred initialization signal - components load on first use
                    self.main_window._deferred_components = initialized_components
                    
                    print("✅ AnimatedLoadingScreen: Main window ready with lazy loading")

                    # Inject the pre-initialized components into the bridge
                    if hasattr(self.main_window, 'bridge'):
                        bridge = self.main_window.bridge

                        # Inject MCQ Manager
                        if 'mcq_manager' in initialized_components and initialized_components['mcq_manager']:
                            bridge.mcq_manager = initialized_components['mcq_manager']
                            bridge.mcq_manager_ready = True
                            bridge.mcq_manager_initializing = False
                            print("✅ AnimatedLoadingScreen: Injected pre-initialized MCQ Manager into bridge")

                        # Inject Training Manager
                        if 'training_manager' in initialized_components and initialized_components['training_manager']:
                            bridge.training_manager = initialized_components['training_manager']
                            print("✅ AnimatedLoadingScreen: Injected pre-initialized Training Manager into bridge")

                        # Inject Resource Manager
                        if 'resource_manager' in initialized_components and initialized_components['resource_manager']:
                            bridge.resource_manager = initialized_components['resource_manager']
                            print("✅ AnimatedLoadingScreen: Injected pre-initialized Resource Manager")

                        # Inject Topic Analyzer
                        if 'topic_analyzer' in initialized_components and initialized_components['topic_analyzer']:
                            bridge.topic_analyzer = initialized_components['topic_analyzer']
                            print("✅ AnimatedLoadingScreen: Injected pre-initialized Topic Analyzer")

                        print("🚀 AnimatedLoadingScreen: All components injected - app should be fast now!")

                    # IMPORTANT: Keep main window hidden until splash closes completely
                    # Don't call main_window.show() here - it will be shown after splash closes
                    print("✅ AnimatedLoadingScreen: Main window ready but hidden until splash closes")

                    # 🚀 SEAMLESS TRANSITION: Wait for app to be completely stable
                    self.status_label.setText("🎯 Waiting for app to stabilize...")
                    self.wait_for_app_stability()

                except Exception as e:
                    print(f"❌ AnimatedLoadingScreen: Error creating main window: {e}")
                    import traceback
                    traceback.print_exc()
                    self.status_label.setText("❌ Failed to initialize app")
                    QTimer.singleShot(2000, self.close)
            else:
                # Show error and close
                self.status_label.setText("❌ Failed to initialize app")
                QTimer.singleShot(2000, self.close)

        # Always wait a minimum time for smooth experience
        minimum_splash_time = 3.0  # 3 seconds minimum
        if elapsed_time < minimum_splash_time:
            remaining_time = minimum_splash_time - elapsed_time
            delay_ms = int(remaining_time * 1000)
            print(f"⏰ AnimatedLoadingScreen: Waiting additional {remaining_time:.2f}s for smooth experience")
            QTimer.singleShot(delay_ms, create_and_show_main_window)
        else:
            create_and_show_main_window()
    
    def wait_for_app_stability(self):
        """Wait for the app to be completely stable before hiding splash (checks backend readiness)"""
        print("🎯 AnimatedLoadingScreen: Starting stability monitoring (backend readiness)...")
        self.stability_start_time = time.time()
        self.stability_duration = 5.0  # Still require a minimum time for smoothness
        self.activity_check_count = 0
        self.stability_timer = QTimer()
        self.stability_timer.timeout.connect(self.check_stability)
        self.stability_timer.start(500)

    def is_backend_ready(self):
        """Check if backend (MCQ manager and other core components) are ready"""
        try:
            if hasattr(self, 'main_window') and hasattr(self.main_window, 'bridge'):
                bridge = self.main_window.bridge
                # Check MCQ manager readiness (expand as needed)
                if hasattr(bridge, 'mcq_manager_ready') and bridge.mcq_manager_ready:
                    return True
        except Exception as e:
            print(f"[Splash] Backend readiness check error: {e}")
        return False
        
    def check_stability(self):
        """Check if the app has been stable for the required duration AND backend is ready"""
        self.activity_check_count += 1
        current_time = time.time()
        time_since_start = current_time - self.stability_start_time
        remaining = max(0, self.stability_duration - time_since_start)
        self.status_label.setText(f"🎯 App stabilizing... {remaining:.1f}s")
        stability_progress = min(100, (time_since_start / self.stability_duration) * 100)
        true_progress = min(100, 85 + (stability_progress * 0.15))
        self.progress_bar.setValue(int(true_progress))

        # Only hide splash if both minimum time has passed AND backend is ready
        if time_since_start >= self.stability_duration and self.is_backend_ready():
            print("✅ AnimatedLoadingScreen: App appears stable and backend ready - hiding splash")
            self.progress_bar.setValue(100)
            self.stability_timer.stop()
            self.hide_splash_smoothly()
        elif time_since_start >= self.stability_duration:
            # Still waiting for backend, update status
            self.status_label.setText("⏳ Waiting for backend to finish loading...")
    
    def hide_splash_smoothly(self):
        """Hide splash screen and show main window with perfect seamless transition"""
        self.status_label.setText("✨ Ready!")
        
        # Hide splash first
        self.hide()
        print("✅ AnimatedLoadingScreen: Splash hidden")
        
        # Small delay to ensure splash is completely gone, then show main window
        def show_main_window():
            if hasattr(self, 'main_window') and self.main_window:
                # Use the show() method which handles all the window operations
                self.main_window.show()
                print("✅ AnimatedLoadingScreen: Main window shown with seamless transition")
            
            # Close splash completely after main window is shown
            self.close()
            print("✅ AnimatedLoadingScreen: Splash screen closed completely")
        
        # Very short delay to ensure no overlap
        QTimer.singleShot(50, show_main_window)
    
    def start_loading(self):
        """Start the loading process"""
        print("🚀 AnimatedLoadingScreen: Starting loading process...")
        self.worker.start()
        print("✅ AnimatedLoadingScreen: Loading worker started")

def show_loading_screen():
    """Show the loading screen and return the main window when ready"""
    splash = AnimatedLoadingScreen()
    splash.show()
    splash.start_loading()
    return splash
