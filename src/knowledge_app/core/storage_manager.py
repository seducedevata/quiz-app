"""
Storage management functionality for the Knowledge App
"""

from .async_converter import async_subprocess


from .async_converter import async_subprocess


import os
import shutil
import time
import logging
import mmap
import asyncio
import aiofiles
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from .cache_manager import BaseCacheManager
import threading
from ..utils.storage import get_storage_usage as get_storage_usage_util

logger = logging.getLogger(__name__)


class StorageManager(BaseCacheManager):
    """Manages storage for the application"""

    def __init__(self, config: Union[Dict[str, Any], "AppConfig"]):
        """
        Initialize storage manager

        Args:
            config: Configuration dictionary or AppConfig object with storage settings
        """
        if isinstance(config, dict):
            storage_config = config
        else:
            # Convert AppConfig to storage config dict
            storage_config = {
                "data_path": config.get_setting("paths.data"),
                "max_cache_size": config.get_setting("storage_config.max_cache_size"),
                "cleanup_threshold": config.get_setting("storage_config.cleanup_threshold"),
                "cache_expiry": config.get_setting("storage_settings.cache_ttl"),
                "use_mmap": True,
            }

        # Extract storage paths
        self.base_path = Path(storage_config.get("data_path", "data")).parent
        self.data_path = Path(storage_config.get("data_path", "data"))
        self.models_path = self.data_path / "models"
        self.user_data_path = self.data_path / "user_data"
        self.uploaded_books_path = self.data_path / "uploaded_books"

        # Create required directories
        for path in [
            self.data_path,
            self.models_path,
            self.user_data_path,
            self.uploaded_books_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # Initialize base cache manager with optimized settings
        cache_config = {
            "base_path": self.data_path / "cache",
            "max_size": storage_config.get("max_cache_size", 4 * 1024 * 1024 * 1024),  # 4GB cache
            "cleanup_threshold": storage_config.get("cleanup_threshold", 0.85),
            "cache_expiry": storage_config.get("cache_expiry", 7200),  # 2 hours
        }
        super().__init__(cache_config)

        # Initialize thread pool for parallel I/O
        self._io_pool = ThreadPoolExecutor(max_workers=4)

        # Track open file handles
        self._open_handles: Dict[str, Any] = {}
        self._handles_lock = threading.Lock()

        # Set up memory mapping for large files
        self.use_mmap = storage_config.get("use_mmap", True)
        self.mmap_threshold = 100 * 1024 * 1024

        # Configure read-ahead for sequential reads
        self._configure_readahead()

    def _configure_readahead(self):
        """Configure system read-ahead for better sequential read performance"""
        try:
            # Increase read-ahead on Linux
            if os.name == "posix":
                import subprocess

                for path in [self.models_path, self.data_path / "cache"]:
                    await async_subprocess(
                        ["blockdev", "--setra", "16384", str(path)],
                        check=False,
                        capture_output=True,
                    )

            # On Windows, use large file buffers
            elif os.name == "nt":
                import win32file

                self.buffer_size = 1024 * 1024  # 1MB buffer
        except Exception as e:
            logger.warning(f"Could not configure read-ahead: {e}")

    async def _read_file_async(self, path: Path) -> bytes:
        """Read file asynchronously"""
        async with aiofiles.open(path, "rb") as f:
            return await f.read()

    def _read_file_mmap(self, path: Path) -> bytes:
        """Read file using memory mapping"""
        with open(path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return mm.read()

    async def read_file_optimized(self, path: Path) -> bytes:
        """Read file with optimized strategy based on size"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_size = path.stat().st_size

        if file_size > self.mmap_threshold and self.use_mmap:
            # Use memory mapping for large files
            return await asyncio.get_event_loop().run_in_executor(
                self._io_pool, self._read_file_mmap, path
            )
        else:
            # Use async I/O for smaller files
            return await self._read_file_async(path)

    async def write_file_optimized(self, path: Path, data: bytes) -> None:
        """Write file with optimized buffering"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Use large buffer for writing
        buffer_size = 1024 * 1024  # 1MB buffer

        async with aiofiles.open(path, "wb") as f:
            # Write in chunks for large data
            if len(data) > buffer_size:
                for i in range(0, len(data), buffer_size):
                    chunk = data[i : i + buffer_size]
                    await f.write(chunk)
            else:
                await f.write(data)

            # Force flush to disk
            await f.flush()
            os.fsync(f.fileno())

    def optimize_file_access(self, path: Path) -> None:
        """Optimize file access patterns for a given path"""
        try:
            if os.name == "posix":
                # Set read-ahead
                import subprocess

                # Use async subprocess for non-blocking operation
                from .async_converter import async_subprocess
                await async_subprocess(
                    ["blockdev", "--setra", "16384", str(path)], check=False, capture_output=True
                )

                # Advise sequential access
                with open(path, "rb") as f:
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)

            elif os.name == "nt":
                import win32file

                handle = win32file.CreateFile(
                    str(path),
                    win32file.GENERIC_READ,
                    win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE,
                    None,
                    win32file.OPEN_EXISTING,
                    win32file.FILE_FLAG_SEQUENTIAL_SCAN,
                    0,
                )
                win32file.CloseHandle(handle)

        except Exception as e:
            logger.warning(f"Could not optimize file access for {path}: {e}")

    def prefetch_files(self, paths: List[Path]) -> None:
        """Prefetch files into system cache"""
        try:
            for path in paths:
                if not path.exists():
                    continue

                # Read file in large chunks to prime the cache
                with open(path, "rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break

        except Exception as e:
            logger.warning(f"Error during file prefetch: {e}")

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage usage information for all managed directories

        Returns:
            Dict with storage statistics
        """
        try:
            info = super().get_cache_info()
            info.update(
                {
                    "models_size": self._calculate_size(self.models_path),
                    "user_data_size": self._calculate_size(self.user_data_path),
                    "uploaded_books_size": self._calculate_size(self.uploaded_books_path),
                    "total_data_size": self._calculate_size(self.data_path),
                }
            )
            return info
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            return {}

    def cleanup_storage(self) -> bool:
        """
        Clean up storage by removing expired cache items

        Returns:
            bool: True if cleanup was successful
        """
        try:
            self._cleanup_expired()
            return True
        except Exception as e:
            logger.error(f"Error during storage cleanup: {e}")
            return False

    def add_user_data(
        self, source_path: Union[str, Path], category: str = "general"
    ) -> Optional[Path]:
        """
        Add user data file to storage

        Args:
            source_path: Path to source file
            category: Data category (subdirectory)

        Returns:
            Optional[Path]: Path to stored file if successful
        """
        try:
            source_path = Path(source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")

            target_dir = self.user_data_path / category
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / source_path.name

            shutil.copy2(source_path, target_path)
            return target_path

        except Exception as e:
            logger.error(f"Error adding user data: {e}")
            return None

    def add_model(self, source_path: Union[str, Path], model_name: str) -> Optional[Path]:
        """
        Add model file to storage

        Args:
            source_path: Path to source model file
            model_name: Name of the model

        Returns:
            Optional[Path]: Path to stored model if successful
        """
        try:
            source_path = Path(source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Model file not found: {source_path}")

            target_path = self.models_path / f"{model_name}.pt"
            shutil.copy2(source_path, target_path)
            return target_path

        except Exception as e:
            logger.error(f"Error adding model: {e}")
            return None

    def add_book(self, source_path: Union[str, Path]) -> Optional[Path]:
        """
        Add book file to storage

        Args:
            source_path: Path to source book file

        Returns:
            Optional[Path]: Path to stored book if successful
        """
        try:
            source_path = Path(source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Book file not found: {source_path}")

            target_path = self.uploaded_books_path / source_path.name
            shutil.copy2(source_path, target_path)
            return target_path

        except Exception as e:
            logger.error(f"Error adding book: {e}")
            return None

    def cleanup(self):
        """Clean up resources"""
        try:
            # Close all open file handles
            with self._handles_lock:
                for handle in self._open_handles.values():
                    try:
                        handle.close()
                    except Exception as e:
                        logger.warning(f"Error closing handle: {e}")
                self._open_handles.clear()

            # Shutdown thread pool
            if self._io_pool:
                self._io_pool.shutdown(wait=True)

            # Call parent cleanup
            super().cleanup()

        except Exception as e:
            logger.error(f"Error during storage cleanup: {e}")

    def _track_handle(self, path: str, handle: Any):
        """Track an open file handle"""
        with self._handles_lock:
            if path in self._open_handles:
                try:
                    self._open_handles[path].close()
                except Exception:
                    pass
            self._open_handles[path] = handle

    def _release_handle(self, path: str):
        """Release a tracked file handle"""
        with self._handles_lock:
            if path in self._open_handles:
                try:
                    self._open_handles[path].close()
                except Exception:
                    pass
                del self._open_handles[path]

    def get_storage_usage(self) -> Dict[str, int]:
        """Get total storage usage in bytes"""
        return get_storage_usage_util(self.base_path)

    def ensure_space_available(self, required_space: int) -> bool:
        """
        Ensure enough space is available for the required amount

        Args:
            required_space: Required space in bytes

        Returns:
            bool: True if space was made available, False otherwise

        Raises:
            OSError: If unable to free enough space
        """
        try:
            # Check current directory size first
            current_size = self.calculate_directory_size(self.base_path)
            if current_size > self.max_size:
                raise OSError(
                    f"Current cache size ({current_size}) exceeds maximum ({self.max_size})"
                )

            # Use the inherited _ensure_space method from the base cache manager
            if self._ensure_space(required_space):
                return True
            else:
                raise OSError(f"Unable to free {required_space} bytes of space")
        except OSError:
            # Re-raise OSError as-is
            raise
        except Exception as e:
            logger.error(f"Error ensuring space availability: {e}")
            raise OSError(f"Unable to ensure space availability: {e}")

    def calculate_directory_size(self, directory: Union[str, Path]) -> int:
        """
        Calculate the total size of a directory

        Args:
            directory: Path to the directory

        Returns:
            int: Total size in bytes
        """
        return self._calculate_size(Path(directory))

    def add_file_to_cache(self, source_path: Union[str, Path], category: str) -> Optional[Path]:
        """
        Add a file to the cache

        Args:
            source_path: Path to the source file
            category: Category/subdirectory for the cached file

        Returns:
            Optional[Path]: Path to the cached file if successful

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If category is empty
        """
        if not category or not category.strip():
            raise ValueError("Category cannot be empty")

        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Use the inherited add_to_cache method
        return self.add_to_cache(source_path, category)

    def get_cached_file(self, filename: str, category: str) -> Optional[Path]:
        """
        Get a cached file by filename and category

        Args:
            filename: Name of the file
            category: Category/subdirectory of the cached file

        Returns:
            Optional[Path]: Path to the cached file if found
        """
        cache_path = self.base_path / category / filename
        if cache_path.exists():
            return cache_path
        return None

    def clear_cache(self) -> bool:
        """
        Clear all cached files

        Returns:
            bool: True if successful
        """
        try:
            # Remove all files from the cache directory
            if self.base_path.exists():
                for item in self.base_path.rglob("*"):
                    if item.is_file():
                        try:
                            item.unlink()
                        except Exception as e:
                            logger.warning(f"Could not remove file {item}: {e}")

                # Remove empty directories
                for item in self.base_path.rglob("*"):
                    if item.is_dir() and not any(item.iterdir()):
                        try:
                            item.rmdir()
                        except Exception as e:
                            logger.warning(f"Could not remove directory {item}: {e}")

            # Clear the cache tracking data
            with self._timed_lock():
                self._cache_items.clear()
                self._total_cache_size = 0

            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False