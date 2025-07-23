"""
Unified image management for Knowledge App
"""

from .async_converter import async_requests_post, async_requests_get


from .async_converter import async_requests_post, async_requests_get


import os
import logging
import requests
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, TiffImagePlugin, ExifTags
from typing import Optional, Dict, Any, Union, Tuple
from datetime import datetime
from PIL.ExifTags import TAGS
from .cache_manager import BaseCacheManager
from .proper_config_manager import ProperConfigManager as get_config
from .storage_manager import StorageManager

logger = logging.getLogger(__name__)


class ImageManager(BaseCacheManager):
    """Unified image management system"""

    _instance = None
    _lock = None  # Will use BaseCacheManager's lock

    def __new__(
        cls,
        config: Optional[Dict[str, Any]] = None,
        storage_manager: Optional[StorageManager] = None,
    ):
        if cls._instance is None:
            cls._instance = super(ImageManager, cls).__new__(cls)
            cls._instance._initialized = False
        if config is not None:
            cls._instance._config = config
        if storage_manager is not None:
            cls._instance.storage_manager = storage_manager
        return cls._instance

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        storage_manager: Optional[StorageManager] = None,
    ):
        if self._initialized:
            if config:
                self.update_config(config)
            return

        if config is None:
            config = get_config()._config

        # Initialize base cache manager
        cache_config = {
            "base_path": Path(config.get("paths", {}).get("image_cache", "data/image_cache")),
            "max_size": config.get("storage_config", {}).get(
                "image_cache_limit", 500 * 1024 * 1024
            ),  # 500MB
            "cleanup_threshold": config.get("storage_settings", {}).get(
                "cleanup_threshold_mb", 0.9
            ),
            "cache_expiry": config.get("storage_settings", {}).get("cache_ttl", 3600),  # 1 hour
        }
        super().__init__(cache_config)

        self._initialized = True
        self._config = config
        self.storage_manager = storage_manager

        # Image processing settings
        app_settings = config.get("app_settings", {})
        self.max_size = app_settings.get("max_image_size", 1920)  # Max dimension
        self.quality = app_settings.get("image_quality", 85)
        self.supported_formats = ["jpg", "jpeg", "png", "webp"]  # Fixed list of supported formats
        self.unsplash_key = config.get("api_keys", {}).get("unsplash")

    def add_image(self, image_path: Union[str, Path]) -> bool:
        """Add an image to the application's image cache"""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return False

            # Check if the file is a supported image format
            if image_path.suffix.lower().lstrip(".") not in self.supported_formats:
                logger.error(f"Unsupported image format: {image_path.suffix}")
                return False

            # Load and validate the image
            try:
                image = self.load_image(image_path)
            except Exception as e:
                logger.error(f"Failed to load image {image_path}: {e}")
                return False

            # Process the image (resize if needed, optimize quality)
            try:
                # Calculate new dimensions while maintaining aspect ratio
                width, height = image.size
                if width > self.max_size or height > self.max_size:
                    if width > height:
                        new_width = self.max_size
                        new_height = int(height * (self.max_size / width))
                    else:
                        new_height = self.max_size
                        new_width = int(width * (self.max_size / height))
                    image = self.resize_image(image, new_width, new_height)

                # Generate a unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = f"{timestamp}_{image_path.stem}.jpg"
                cache_path = self.base_path / new_filename

                # Ensure cache directory exists
                self.base_path.mkdir(parents=True, exist_ok=True)

                # Save the processed image
                self.save_image(image, cache_path, format="JPEG", optimize=True)

                # Clean up
                image.close()

                logger.info(f"Successfully added image: {new_filename}")
                return True

            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
                if "image" in locals():
                    try:
                        image.close()
                    except:
                        pass
                return False

        except Exception as e:
            logger.error(f"Error adding image {image_path}: {e}")
            return False

    def safe_open_image(self, image_path: Union[str, Path]) -> Optional[Image.Image]:
        """Safely open an image file"""
        try:
            img = Image.open(image_path)
            img.load()  # Force load image data into memory
            return img
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            return None

    def safe_close_image(self, img: Optional[Image.Image]) -> None:
        """Safely close an image"""
        try:
            if img is not None:
                img.close()
        except Exception as e:
            logger.error(f"Failed to close image: {e}")

    def get_image_for_topic(self, topic_name: str) -> Optional[Image.Image]:
        """Get an image for a topic from Unsplash"""
        if not self.unsplash_key:
            logger.error("Unsplash API key not configured")
            return None

        cache_filepath = self.base_path / f"{topic_name.lower().replace(' ', '_')}.jpg"

        # Check cache first
        if cache_filepath.exists():
            return self.safe_open_image(cache_filepath)

        try:
            response = await async_requests_get(
                "https://api.unsplash.com/photos/random",
                params={
                    "query": topic_name,
                    "client_id": self.unsplash_key,
                    "orientation": "landscape",
                    "count": 1,
                },
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            if not data or not isinstance(data, dict) and not (isinstance(data, list) and data):
                logger.error(f"Unsplash API response format unexpected for '{topic_name}'")
                return None

            image_url = (
                data[0]["urls"]["regular"] if isinstance(data, list) else data["urls"]["regular"]
            )
            image_response = await async_requests_get(image_url, timeout=15)
            image_response.raise_for_status()

            self.base_path.mkdir(parents=True, exist_ok=True)
            with open(cache_filepath, "wb") as f:
                f.write(image_response.content)

            return self.safe_open_image(cache_filepath)

        except requests.exceptions.RequestException as e:
            logger.error(f"Unsplash API request failed for '{topic_name}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching/processing image for '{topic_name}': {e}")

        return None

    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load an image from file"""
        if not isinstance(image_path, (str, Path)):
            raise ValueError("Invalid image path")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path)
            image.load()  # Load image data
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

    def resize_image(
        self, image: Image.Image, width: int, height: Optional[int] = None
    ) -> Image.Image:
        """Resize image to target dimensions"""
        if not isinstance(image, Image.Image):
            raise ValueError("Invalid image object")

        if width <= 0:
            raise ValueError("Width must be positive")

        if height is None:
            # Calculate height to maintain aspect ratio
            ratio = width / image.size[0]
            height = int(image.size[1] * ratio)
        elif height <= 0:
            raise ValueError("Height must be positive")

        return image.resize((width, height), Image.Resampling.LANCZOS)

    def _get_exif_data(self, image: Image.Image) -> Optional[bytes]:
        """Extract EXIF data from image"""
        try:
            # First try to get EXIF bytes from image.info
            if "exif" in image.info:
                return image.info["exif"]

            # If that fails, try to get it via _getexif
            if hasattr(image, "_getexif") and image._getexif():
                # Convert to bytes format that PIL can use
                exif = image.getexif()
                if exif:
                    return exif.tobytes()
        except Exception as e:
            logger.warning(f"Failed to extract EXIF data: {e}")

        return None

    def save_image(
        self,
        image: Image.Image,
        output_path: Union[str, Path],
        format: Optional[str] = None,
        optimize: bool = False,
    ) -> None:
        """Save image to file"""
        if not isinstance(image, Image.Image):
            raise ValueError("Invalid image object")

        output_path = Path(output_path)

        # Determine format
        if format is None:
            format = output_path.suffix[1:].upper()
        else:
            format = format.upper()

        # Normalize format names
        format_map = {"JPG": "JPEG", "JPEG": "JPEG", "PNG": "PNG", "WEBP": "WEBP"}

        format = format_map.get(format)
        if format not in format_map.values():
            raise ValueError(f"Unsupported format: {format}")

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Create a copy of the image to avoid modifying the original
            img_copy = image.copy()

            # Get EXIF data if available
            exif = self._get_exif_data(image)

            # Convert to RGB/RGBA if needed
            if format == "JPEG" and img_copy.mode != "RGB":
                img_copy = img_copy.convert("RGB")
            elif format in ["PNG", "WEBP"] and img_copy.mode not in ["RGB", "RGBA"]:
                img_copy = img_copy.convert("RGBA")

            save_args = {"format": format}

            # Set format-specific parameters
            if format == "JPEG":
                if optimize:
                    save_args["quality"] = 30  # More aggressive compression
                    save_args["optimize"] = True
                    save_args["progressive"] = True
                else:
                    save_args["quality"] = self.quality
                if exif:
                    save_args["exif"] = exif
            elif format == "PNG":
                if optimize:
                    save_args["optimize"] = True
                    save_args["compress_level"] = 9
            elif format == "WEBP":
                if optimize:
                    save_args["quality"] = 30  # More aggressive compression
                    save_args["method"] = 6
                    save_args["lossless"] = False
                else:
                    save_args["quality"] = self.quality
                    save_args["method"] = 4
                if exif:
                    save_args["exif"] = exif

            # Save the image
            img_copy.save(output_path, **save_args)

            # Clean up
            img_copy.close()

        except Exception as e:
            # Clean up in case of error
            if "img_copy" in locals():
                try:
                    img_copy.close()
                except:
                    pass
            raise ValueError(f"Failed to save image: {e}")

    def process_image(
        self,
        image: Image.Image,
        resize: Optional[Union[int, Tuple[int, int]]] = None,
        format: Optional[str] = None,
        optimize: bool = False,
        preserve_metadata: bool = False,
    ) -> Image.Image:
        """Process image with specified operations"""
        if not isinstance(image, Image.Image):
            raise ValueError("Invalid image object")

        # Store original metadata if needed
        metadata = None
        if preserve_metadata:
            metadata = self._get_exif_data(image)

        # Create a copy to avoid modifying original
        processed = image.copy()

        # Resize if specified
        if resize:
            if isinstance(resize, (tuple, list)) and len(resize) == 2:
                width, height = resize
            else:
                width = resize
                height = None
            processed = self.resize_image(processed, width, height)

        # Convert format if needed
        if format:
            format = format.upper()
            if format not in ["JPEG", "PNG", "WEBP"]:
                raise ValueError(f"Unsupported format: {format}")

            # Convert to RGB if saving as JPEG
            if format == "JPEG" and processed.mode != "RGB":
                processed = processed.convert("RGB")
            elif format == "PNG" and processed.mode not in ["RGB", "RGBA"]:
                processed = processed.convert("RGBA")
            elif format == "WEBP" and processed.mode not in ["RGB", "RGBA"]:
                processed = processed.convert("RGBA")

        # Restore metadata if preserved
        if metadata:
            processed.info["exif"] = metadata

        return processed

    def enhance_image(
        self,
        image: Image.Image,
        contrast: float = 1.0,
        brightness: float = 1.0,
        sharpness: float = 1.0,
    ) -> Image.Image:
        """Enhance image with specified parameters"""
        if not isinstance(image, Image.Image):
            raise ValueError("Invalid image object")

        # Validate enhancement values
        for name, value in [
            ("contrast", contrast),
            ("brightness", brightness),
            ("sharpness", sharpness),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        # Create a copy to avoid modifying original
        enhanced = image.copy()

        # Apply enhancements in sequence
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast)

        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness)

        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness)

        return enhanced

    def update_config(self, config: Union[Dict[str, Any], "AppConfig"]) -> None:
        """Update configuration with new values"""
        if isinstance(config, dict):
            self._config.update(config)
        else:
            # Convert AppConfig to dict
            self._config = {
                "paths": {
                    "image_cache": config.get_setting("paths.image_cache", "data/image_cache")
                },
                "storage_config": {
                    "image_cache_limit": config.get_setting(
                        "storage_config.image_cache_limit", 500 * 1024 * 1024
                    )
                },
                "storage_settings": {
                    "cleanup_threshold_mb": config.get_setting(
                        "storage_settings.cleanup_threshold_mb", 0.9
                    ),
                    "cache_ttl": config.get_setting("storage_settings.cache_ttl", 3600),
                },
                "app_settings": {
                    "max_image_size": config.get_setting("app_settings.max_image_size", 1920),
                    "image_quality": config.get_setting("app_settings.image_quality", 85),
                },
                "api_keys": {"unsplash": config.get_setting("api_keys.unsplash", "")},
            }

    def cleanup(self) -> None:
        """Clean up resources"""
        super().cleanup()

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global image manager instance
_image_manager = None


def get_image_manager(
    config: Optional[Dict[str, Any]] = None, storage_manager: Optional[StorageManager] = None
) -> ImageManager:
    """Get the singleton instance of ImageManager"""
    global _image_manager
    if _image_manager is None:
        _image_manager = ImageManager(config, storage_manager)
    return _image_manager