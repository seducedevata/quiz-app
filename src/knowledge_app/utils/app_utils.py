"""
Application utilities for Knowledge App
Pure utility functions - no QtWidgets bloatware
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_app_info() -> Dict[str, Any]:
    """Get application information"""
    return {
        "name": "Knowledge App",
        "version": "1.0.0",
        "description": "AI-powered quiz application with QtWebEngine",
        "platform": sys.platform,
        "python_version": sys.version,
    }


def check_system_requirements() -> bool:
    """Check if system meets minimum requirements"""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            return False
            
        # Check if running in correct directory
        current_dir = Path.cwd()
        if not (current_dir / "src" / "knowledge_app").exists():
            logger.error("Application must be run from project root directory")
            return False
            
        return True
    except Exception as e:
        logger.error(f"System requirements check failed: {e}")
        return False


def setup_logging(log_dir: Optional[Path] = None) -> None:
    """Setup application logging"""
    try:
        if log_dir is None:
            log_dir = Path("user_data")
        
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "app.log"
        
        # Only configure file logging - console logging handled by main.py
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        logger.info("Logging initialized successfully")
    except Exception as e:
        print(f"Failed to setup logging: {e}", file=sys.stderr)


def create_user_directories() -> None:
    """Create necessary user directories"""
    try:
        directories = [
            "user_data",
            "data/cache",
            "data/processed_docs",
            "data/uploaded_books"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        logger.info("User directories created successfully")
    except Exception as e:
        logger.error(f"Failed to create user directories: {e}")


def show_message(message: str, title: str = "Knowledge App") -> None:
    """Show message to user via logging (no GUI dialogs)"""
    logger.info(f"{title}: {message}")


def show_error(message: str, title: str = "Error") -> None:
    """Show error message to user via logging (no GUI dialogs)"""
    logger.error(f"{title}: {message}")


def get_resource_path(relative_path: str) -> Path:
    """Get path to resource file"""
    return Path(__file__).parent.parent / relative_path


def ensure_file_exists(file_path: Path) -> bool:
    """Ensure a file exists, create if necessary"""
    try:
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
        return True
    except Exception as e:
        logger.error(f"Failed to ensure file exists {file_path}: {e}")
        return False