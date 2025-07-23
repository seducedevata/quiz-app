"""
🔧 Local Model Cache Service

This module provides centralized caching for local AI model discovery,
eliminating repetitive network requests to localhost AI servers.

CRITICAL FIX: Consolidates model discovery across:
- DynamicModelDetector
- OfflineMCQGenerator  
- LMStudioMCQGenerator
- UI components
"""

import time
import logging
import asyncio
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about a local AI model"""
    name: str
    size: Optional[str] = None
    modified_at: Optional[str] = None
    digest: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class CacheEntry:
    """Cache entry with timestamp and data"""
    timestamp: float
    models: List[ModelInfo]
    server_type: str  # 'ollama', 'lm_studio', etc.
    error: Optional[str] = None

class LocalModelCache:
    """
    🔧 FIX: Centralized cache for local AI model discovery
    
    This service eliminates repetitive network requests by caching
    model lists for a configurable duration.
    """
    
    def __init__(self, cache_duration: int = 30):
        """
        Initialize the model cache
        
        Args:
            cache_duration: Cache duration in seconds (default: 30)
        """
        self.cache_duration = cache_duration
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.RLock()
        self.last_cleanup = time.time()
        
        # Server configurations
        self.server_configs = {
            'ollama': {
                'url': 'http://localhost:11434/api/tags',
                'timeout': 2,
                'parser': self._parse_ollama_response
            },
            'lm_studio': {
                'url': 'http://localhost:1234/v1/models',
                'timeout': 2,
                'parser': self._parse_lm_studio_response
            }
        }
        
        logger.info("🔧 LocalModelCache initialized with {cache_duration}s cache duration")
    
    async def get_models(self, server_type: str = 'ollama') -> List[ModelInfo]:
        """
        Get cached models or fetch fresh data if cache is stale
        
        Args:
            server_type: Type of server ('ollama', 'lm_studio')
            
        Returns:
            List of available models
        """
        with self.cache_lock:
            # Check if we have valid cached data
            if server_type in self.cache:
                entry = self.cache[server_type]
                age = time.time() - entry.timestamp
                
                if age < self.cache_duration:
                    logger.debug(f"✅ Using cached {server_type} models (age: {age:.1f}s)")
                    return entry.models
                else:
                    logger.debug(f"🔄 Cache expired for {server_type} (age: {age:.1f}s)")
            
            # Cache miss or expired - fetch fresh data
            return await self._fetch_and_cache_models(server_type)
    
    async def _fetch_and_cache_models(self, server_type: str) -> List[ModelInfo]:
        """Fetch models from server and update cache"""
        try:
            config = self.server_configs.get(server_type)
            if not config:
                logger.error(f"❌ Unknown server type: {server_type}")
                return []
            
            logger.debug(f"🔄 Fetching {server_type} models from {config['url']}")
            
            # Make HTTP request
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config['timeout'])) as session:
                async with session.get(config['url']) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = config['parser'](data)
                        
                        # Update cache
                        with self.cache_lock:
                            self.cache[server_type] = CacheEntry(
                                timestamp=time.time(),
                                models=models,
                                server_type=server_type
                            )
                        
                        logger.info(f"✅ Cached {len(models)} {server_type} models")
                        return models
                    else:
                        logger.warning(f"⚠️ {server_type} server returned status {response.status}")
                        return []
                        
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Timeout connecting to {server_type} server")
            return []
        except Exception as e:
            logger.error(f"❌ Error fetching {server_type} models: {e}")
            
            # Cache the error to prevent repeated failed requests
            with self.cache_lock:
                self.cache[server_type] = CacheEntry(
                    timestamp=time.time(),
                    models=[],
                    server_type=server_type,
                    error=str(e)
                )
            
            return []
    
    def _parse_ollama_response(self, data: Dict[str, Any]) -> List[ModelInfo]:
        """Parse Ollama API response"""
        models = []
        try:
            for model_data in data.get('models', []):
                model = ModelInfo(
                    name=model_data.get('name', 'unknown'),
                    size=model_data.get('size'),
                    modified_at=model_data.get('modified_at'),
                    digest=model_data.get('digest'),
                    details=model_data.get('details', {})
                )
                models.append(model)
        except Exception as e:
            logger.error(f"❌ Error parsing Ollama response: {e}")
        
        return models
    
    def _parse_lm_studio_response(self, data: Dict[str, Any]) -> List[ModelInfo]:
        """Parse LM Studio API response"""
        models = []
        try:
            for model_data in data.get('data', []):
                model = ModelInfo(
                    name=model_data.get('id', 'unknown'),
                    details=model_data
                )
                models.append(model)
        except Exception as e:
            logger.error(f"❌ Error parsing LM Studio response: {e}")
        
        return models
    
    def invalidate_cache(self, server_type: Optional[str] = None):
        """
        Invalidate cache entries
        
        Args:
            server_type: Specific server to invalidate, or None for all
        """
        with self.cache_lock:
            if server_type:
                if server_type in self.cache:
                    del self.cache[server_type]
                    logger.info(f"🔄 Invalidated {server_type} cache")
            else:
                self.cache.clear()
                logger.info("🔄 Invalidated all model caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.cache_lock:
            stats = {}
            for server_type, entry in self.cache.items():
                age = time.time() - entry.timestamp
                stats[server_type] = {
                    'model_count': len(entry.models),
                    'age_seconds': age,
                    'is_fresh': age < self.cache_duration,
                    'has_error': entry.error is not None,
                    'error': entry.error
                }
            return stats
    
    def cleanup_expired_entries(self):
        """Remove expired cache entries"""
        current_time = time.time()
        
        # Only cleanup every 60 seconds to avoid overhead
        if current_time - self.last_cleanup < 60:
            return
        
        with self.cache_lock:
            expired_keys = []
            for server_type, entry in self.cache.items():
                age = current_time - entry.timestamp
                if age > self.cache_duration * 2:  # Keep for 2x cache duration
                    expired_keys.append(server_type)
            
            for key in expired_keys:
                del self.cache[key]
                logger.debug(f"🧹 Cleaned up expired cache entry: {key}")
            
            self.last_cleanup = current_time

# Global cache instance
_local_model_cache: Optional[LocalModelCache] = None
_cache_lock = threading.RLock()

def get_local_model_cache(cache_duration: int = 30) -> LocalModelCache:
    """Get the global local model cache instance"""
    global _local_model_cache
    with _cache_lock:
        if _local_model_cache is None:
            _local_model_cache = LocalModelCache(cache_duration)
        return _local_model_cache

# Convenience functions for easy migration
async def get_ollama_models() -> List[ModelInfo]:
    """Get cached Ollama models"""
    cache = get_local_model_cache()
    return await cache.get_models('ollama')

async def get_lm_studio_models() -> List[ModelInfo]:
    """Get cached LM Studio models"""
    cache = get_local_model_cache()
    return await cache.get_models('lm_studio')

def invalidate_model_cache(server_type: Optional[str] = None):
    """Invalidate model cache"""
    cache = get_local_model_cache()
    cache.invalidate_cache(server_type)
