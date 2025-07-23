"""
Hardware Utilities

This module provides functionality for detecting and analyzing hardware capabilities
relevant for machine learning training.
"""

import os
import platform
import psutil
import logging
from typing import Dict, Any, Optional
import torch
import torch.cuda as cuda
import numpy as np

logger = logging.getLogger(__name__)


def get_gpu_info() -> Dict[str, Any]:
    """Get information about available GPU(s)"""
    gpu_info = {
        "available": False,
        "name": None,
        "vram_gb": 0,
        "compute_capability": 0,
        "cuda_version": None,
    }

    try:
        if not torch.cuda.is_available():
            return gpu_info

        gpu_info["available"] = True
        gpu_info["name"] = torch.cuda.get_device_name(0)
        gpu_info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info["compute_capability"] = float(
            f"{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}"
        )
        gpu_info["cuda_version"] = torch.version.cuda

    except Exception as e:
        logger.warning(f"Error getting GPU info: {e}")

    return gpu_info


def get_cpu_info() -> Dict[str, Any]:
    """Get information about the CPU"""
    try:
        return {
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "ram_gb": psutil.virtual_memory().total / (1024**3),
        }
    except Exception as e:
        logger.warning(f"Error getting CPU info: {e}")
        return {"cores": None, "threads": None, "frequency": None, "ram_gb": None}


def get_hardware_specs() -> Dict[str, Any]:
    """Get comprehensive hardware specifications"""
    gpu_info = get_gpu_info()
    cpu_info = get_cpu_info()

    specs = {
        "gpu_available": gpu_info["available"],
        "gpu_name": gpu_info["name"],
        "gpu_vram_gb": gpu_info["vram_gb"],
        "gpu_compute_capability": gpu_info["compute_capability"],
        "cuda_version": gpu_info["cuda_version"],
        "cpu_cores": cpu_info["cores"],
        "cpu_threads": cpu_info["threads"],
        "cpu_frequency": cpu_info["frequency"],
        "ram_gb": cpu_info["ram_gb"],
        "os_system": platform.system(),
        "os_release": platform.release(),
    }

    # Add derived metrics
    specs["compute_score"] = _calculate_compute_score(specs)

    return specs


def _calculate_compute_score(specs: Dict[str, Any]) -> float:
    """Calculate a normalized compute capability score (0-10)"""
    score = 0.0

    # GPU Score (up to 7 points)
    if specs["gpu_available"]:
        # VRAM score (up to 3 points)
        vram_score = min(3.0, specs["gpu_vram_gb"] / 8.0)

        # Compute capability score (up to 4 points)
        compute_score = min(4.0, specs["gpu_compute_capability"] / 2.0)

        score += vram_score + compute_score

    # CPU Score (up to 3 points)
    if specs["cpu_cores"]:
        # Core count score (up to 2 points)
        core_score = min(2.0, specs["cpu_cores"] / 8.0)

        # RAM score (up to 1 point)
        ram_score = min(1.0, specs["ram_gb"] / 32.0)

        score += core_score + ram_score

    return score


def estimate_optimal_batch_size(specs: Dict[str, Any]) -> int:
    """Estimate optimal batch size based on hardware"""
    if not specs["gpu_available"]:
        # CPU-only mode - conservative batch size
        return 32

    # Start with base batch size
    base_size = 64

    # Adjust for VRAM
    vram_factor = min(4, max(1, specs["gpu_vram_gb"] / 8))

    # Adjust for compute capability
    compute_factor = min(2, max(1, specs["gpu_compute_capability"] / 7.0))

    # Calculate final size
    batch_size = int(base_size * vram_factor * compute_factor)

    # Round to nearest power of 2
    return 2 ** int(np.log2(batch_size) + 0.5)

def get_real_time_gpu_utilization() -> Dict[str, Any]:
    """Get real-time GPU utilization metrics"""
    try:
        import pynvml
        pynvml.nvmlInit()

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Get utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        # Get memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # Get temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        # Get power usage
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts

        return {
            "gpu_utilization": util.gpu,
            "memory_utilization": util.memory,
            "memory_used_mb": mem_info.used // 1024 // 1024,
            "memory_total_mb": mem_info.total // 1024 // 1024,
            "memory_free_mb": mem_info.free // 1024 // 1024,
            "temperature_c": temp,
            "power_usage_w": power,
            "is_optimal": util.gpu > 80  # Consider >80% as optimal utilization
        }

    except Exception as e:
        print(f"⚠️ Could not get GPU utilization: {e}")
        return {
            "gpu_utilization": 0,
            "memory_utilization": 0,
            "memory_used_mb": 0,
            "memory_total_mb": 0,
            "memory_free_mb": 0,
            "temperature_c": 0,
            "power_usage_w": 0,
            "is_optimal": False
        }


def can_use_mixed_precision(specs: Dict[str, Any]) -> bool:
    """Check if mixed precision training is supported"""
    if not specs["gpu_available"]:
        return False

    # Mixed precision requires Volta, Turing, or Ampere GPU
    return specs["gpu_compute_capability"] >= 7.0


def get_gpu_utilization() -> Dict[str, Any]:
    """Get current GPU utilization stats with comprehensive monitoring and memory availability"""
    try:
        if not torch.cuda.is_available():
            return {"available": False, "utilization": 0, "memory_used": 0, "memory_total": 0}

        device = torch.cuda.current_device()
        device_props = torch.cuda.get_device_properties(device)
        device_name = device_props.name

        # Memory statistics from PyTorch
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        memory_total = device_props.total_memory
        memory_cached = torch.cuda.memory_cached(device) if hasattr(torch.cuda, 'memory_cached') else memory_reserved

        # Memory utilization percentages
        memory_allocated_percent = (memory_allocated / memory_total) * 100
        memory_reserved_percent = (memory_reserved / memory_total) * 100
        
        # Try to get detailed GPU utilization using official nvidia-ml-py
        gpu_utilization_percent = None
        gpu_temperature = None
        gpu_power_usage = None
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            
            # Get GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization_percent = utilization.gpu
            
            # Get temperature
            try:
                gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                pass
                
            # Get power usage
            try:
                gpu_power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
            except:
                pass
                
            pynvml.nvmlShutdown()
            
        except ImportError:
            logger.error("🚫 GPU UTILIZATION ESTIMATION FALLBACK DISABLED")
            logger.error("❌ nvidia-ml-py not available - cannot estimate GPU utilization")
            logger.error("🚨 APPLICATION MUST USE DIRECT GPU MONITORING ONLY - NO ESTIMATION FALLBACKS")
            raise Exception("GPU utilization estimation disabled - direct monitoring required")

        except Exception as e:
            logger.error(f"🚫 GPU STATS FALLBACK DISABLED: {e}")
            logger.error("❌ Cannot get detailed GPU stats - no fallback estimation allowed")
            logger.error("🚨 APPLICATION MUST USE DIRECT GPU QUERIES ONLY - NO FALLBACK ESTIMATION")
            raise Exception(f"GPU stats fallback disabled - direct query required: {e}")
        
        # Create comprehensive GPU stats with safe values
        gpu_stats = {
            "available": True,
            "device_name": str(device_name),
            "device_index": int(device),
            
            # Memory statistics
            "memory_utilization": float(memory_allocated_percent),
            "memory_reserved_percent": float(memory_reserved_percent),
            "memory_allocated_mb": float(memory_allocated / (1024 * 1024)),
            "memory_reserved_mb": float(memory_reserved / (1024 * 1024)),
            "memory_total_mb": float(memory_total / (1024 * 1024)),
            "memory_cached_mb": float(memory_cached / (1024 * 1024)),
            
            # Compute utilization
            "gpu_utilization": float(gpu_utilization_percent or memory_allocated_percent),
            
            # Additional stats if available (ensure they're JSON serializable)
            "temperature_c": int(gpu_temperature) if gpu_temperature is not None else None,
            "power_usage_w": float(gpu_power_usage) if gpu_power_usage is not None else None,
            
            # Status indicators
            "is_active": bool(memory_allocated > 1024 * 1024),  # >1MB allocated = active
            "utilization_status": "high" if (gpu_utilization_percent or 0) > 80 else "medium" if (gpu_utilization_percent or 0) > 40 else "low"
        }
        
        return gpu_stats

    except Exception as e:
        logger.warning(f"Error getting GPU utilization: {e}")
        return {
            "available": False,
            "utilization": 0,
            "memory_used": 0,
            "memory_total": 0,
            "error": str(e),
            "gpu_utilization": 0,
            "memory_utilization": 0
        }


def get_gpu_memory_availability() -> Dict[str, Any]:
    """
    Check GPU memory availability for model loading, accounting for other applications.
    Returns detailed memory info and recommendations for GPU layer allocation.
    """
    try:
        if not torch.cuda.is_available():
            return {
                "available": False,
                "total_memory_gb": 0,
                "free_memory_gb": 0,
                "used_memory_gb": 0,
                "can_load_models": False,
                "recommended_gpu_layers": 0,
                "sharing_mode": False,
                "other_apps_using_gpu": False
            }

        device = torch.cuda.current_device()
        device_props = torch.cuda.get_device_properties(device)
        total_memory = device_props.total_memory

        # Get current memory usage
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)

        # Try to get more detailed memory info using nvidia-ml-py
        free_memory = total_memory - reserved_memory
        used_by_others = 0

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)

            # Get memory info from NVML (more accurate for detecting other applications)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = mem_info.total
            used_memory = mem_info.used
            free_memory = mem_info.free

            # Estimate memory used by other applications (not PyTorch)
            pytorch_memory = max(allocated_memory, reserved_memory)
            used_by_others = max(0, used_memory - pytorch_memory)

            pynvml.nvmlShutdown()

        except ImportError:
            logger.warning("nvidia-ml-py not available, using PyTorch memory estimates")
        except Exception as e:
            logger.warning(f"Error getting detailed GPU memory info: {e}")

        # Convert to GB for easier handling
        total_gb = total_memory / (1024**3)
        free_gb = free_memory / (1024**3)
        used_gb = (total_memory - free_memory) / (1024**3)
        used_by_others_gb = used_by_others / (1024**3)

        # Determine if other applications are heavily using GPU
        other_apps_using_gpu = used_by_others_gb > 1.0  # More than 1GB used by other apps
        sharing_mode = other_apps_using_gpu or (used_gb / total_gb) > 0.6  # More than 60% total usage

        # Calculate recommended GPU layers based on available memory
        # Rough estimate: each GPU layer needs ~200-500MB depending on model size
        # For large models like deepseek-r1:14b, be more conservative
        available_for_models = free_gb - 1.0  # Reserve 1GB for system/other processes

        if available_for_models < 2.0:
            recommended_gpu_layers = 0  # Not enough memory for GPU inference
            can_load_models = False
        elif available_for_models < 4.0:
            recommended_gpu_layers = 10  # Limited GPU layers
            can_load_models = True
        elif available_for_models < 6.0:
            recommended_gpu_layers = 20  # Moderate GPU layers
            can_load_models = True
        elif available_for_models < 8.0:
            recommended_gpu_layers = 35  # Most layers on GPU
            can_load_models = True
        else:
            recommended_gpu_layers = -1  # All layers on GPU
            can_load_models = True

        return {
            "available": True,
            "total_memory_gb": round(total_gb, 2),
            "free_memory_gb": round(free_gb, 2),
            "used_memory_gb": round(used_gb, 2),
            "used_by_others_gb": round(used_by_others_gb, 2),
            "can_load_models": can_load_models,
            "recommended_gpu_layers": recommended_gpu_layers,
            "sharing_mode": sharing_mode,
            "other_apps_using_gpu": other_apps_using_gpu,
            "memory_usage_percent": round((used_gb / total_gb) * 100, 1)
        }

    except Exception as e:
        logger.error(f"Error checking GPU memory availability: {e}")
        return {
            "available": False,
            "total_memory_gb": 0,
            "free_memory_gb": 0,
            "used_memory_gb": 0,
            "can_load_models": False,
            "recommended_gpu_layers": 0,
            "sharing_mode": False,
            "other_apps_using_gpu": False,
            "error": str(e)
        }