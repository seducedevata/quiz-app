#!/usr/bin/env python3
"""
Dependency Manager for Knowledge App

This module handles checking and managing application dependencies.
"""

import os
import sys
import logging
import subprocess
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are available without importing heavy libraries"""
    try:
        # CRITICAL MEMORY FIX: Check only lightweight dependencies during startup
        import PyQt5
        import PIL
        import numpy
        import requests

        logger.info("Core lightweight dependencies are available")
        return True

    except ImportError as e:
        logger.error(f"Missing core dependency: {e}")
        return False


def check_ml_dependencies():
    """Check ML dependencies only when needed (lazy check)"""
    try:
        import torch
        import transformers

        logger.info("ML dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"Missing ML dependency: {e}")
        return False


def install_dependencies():
    """Install required dependencies"""
    try:
        # Install core dependencies
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "PyQt5",
                "Pillow",
                "numpy",
                "transformers",
                "requests",
            ]
        )

        logger.info("Dependencies installed successfully")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        raise


def check_gpu_support():
    """Check if GPU support is available"""
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            logger.info(f"Found {device_count} GPU(s): {', '.join(device_names)}")
            return True
        else:
            logger.info("No GPU support available")
            return False

    except Exception as e:
        logger.error(f"Error checking GPU support: {e}")
        return False


def check_disk_space(path: str = None) -> Optional[float]:
    """Check available disk space

    Args:
        path: Path to check disk space for. Defaults to current directory.

    Returns:
        Available disk space in GB, or None if check fails
    """
    try:
        if path is None:
            path = os.getcwd()

        total, used, free = os.statvfs(path) if os.name == "posix" else (0, 0, 0)

        if os.name == "posix":
            # Unix-like systems
            available_gb = (free * total) / (1024 * 1024 * 1024)
        else:
            # Windows
            import ctypes

            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(path), None, None, ctypes.pointer(free_bytes)
            )
            available_gb = free_bytes.value / (1024 * 1024 * 1024)

        logger.info(f"Available disk space: {available_gb:.1f}GB on {path}")
        return available_gb

    except Exception as e:
        logger.error(f"Error checking disk space: {e}")
        return None


def check_memory():
    """Check available system memory"""
    try:
        import psutil

        # Get memory info
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024 * 1024 * 1024)
        available_gb = memory.available / (1024 * 1024 * 1024)

        logger.info(f"Memory: {available_gb:.1f}GB available out of {total_gb:.1f}GB total")

        return {"total": total_gb, "available": available_gb, "percent": memory.percent}

    except Exception as e:
        logger.error(f"Error checking memory: {e}")
        return None


def check_system_requirements():
    """Check if system meets minimum requirements"""
    try:
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            logger.warning(
                f"Python version {python_version.major}.{python_version.minor} is below recommended 3.8"
            )

        # Check memory
        memory_info = check_memory()
        if memory_info and memory_info["available"] < 4:  # Less than 4GB available
            logger.warning("Less than 4GB of available memory")

        # Check disk space
        available_space = check_disk_space()
        if available_space and available_space < 10:  # Less than 10GB available
            logger.warning("Less than 10GB of available disk space")

        # Check GPU
        if not check_gpu_support():
            logger.warning("No GPU support available - application will run in CPU-only mode")

    except Exception as e:
        logger.error(f"Error checking system requirements: {e}")
        raise