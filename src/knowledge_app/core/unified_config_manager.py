"""
🔧 UNIFIED CONFIGURATION MANAGER
Consolidates all configuration access patterns to eliminate inconsistencies.
Replaces fragmented config access with a single, thread-safe source of truth.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union
import time

logger = logging.getLogger(__name__)


class UnifiedConfigManager:
    """
    🔧 SINGLE SOURCE OF TRUTH for all application configuration
    
    CRITICAL FIX: Consolidates fragmented configuration access patterns:
    - Proper config manager
    - App config 
    - User settings
    - Training presets
    - API configurations
    
    Features:
    - Thread-safe access
    - Automatic backup and recovery
    - Schema validation
    - Default value fallbacks
    - Real-time persistence
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
            
        with self._lock:
            if self._initialized:
                return
                
            # Configuration storage
            self._config_data = {}
            self._config_lock = threading.RLock()
            self._last_save_time = 0
            self._auto_save_interval = 5.0  # Auto-save every 5 seconds
            
            # Configuration files
            self.config_dir = Path("config")
            self.config_dir.mkdir(exist_ok=True)
            
            self.unified_config_file = self.config_dir / "unified_config.json"
            self.backup_config_file = self.config_dir / "unified_config_backup.json"
            self.user_settings_file = Path("user_data/user_settings.json")
            
            # Initialize configuration schema
            self._init_config_schema()
            
            # Load existing configuration
            self._load_configuration()
            
            self._initialized = True
            logger.info("✅ UnifiedConfigManager initialized with consolidated configuration")
    
    def _init_config_schema(self):
        """Initialize the default configuration schema"""
        self._default_schema = {
            "app": {
                "version": "2.0.0",
                "name": "Knowledge App",
                "debug_mode": False,
                "log_level": "INFO",
                "ui_theme": "dark"
            },
            "api_keys": {
                "openai": "",
                "anthropic": "",
                "google": "",
                "azure": ""
            },
            "training": {
                "presets": {
                    "quick_training": {
                        "base_model": "microsoft/DialoGPT-small",
                        "lora": {
                            "r": 8,
                            "alpha": 16,
                            "dropout": 0.1,
                            "target_modules": ["c_attn"]
                        },
                        "training": {
                            "epochs": 1,
                            "batch_size": 2,
                            "learning_rate": 0.0001,
                            "max_steps": 100,
                            "gradient_accumulation_steps": 4,
                            "warmup_steps": 20
                        }
                    },
                    "standard_training": {
                        "base_model": "microsoft/DialoGPT-small",
                        "lora": {
                            "r": 16,
                            "alpha": 32,
                            "dropout": 0.1,
                            "target_modules": ["c_attn", "c_proj"]
                        },
                        "training": {
                            "epochs": 2,
                            "batch_size": 4,
                            "learning_rate": 0.0002,
                            "max_steps": 500,
                            "gradient_accumulation_steps": 2,
                            "warmup_steps": 50
                        }
                    },
                    "intensive_training": {
                        "base_model": "microsoft/DialoGPT-small",
                        "lora": {
                            "r": 32,
                            "alpha": 64,
                            "dropout": 0.1,
                            "target_modules": ["c_attn", "c_proj", "c_fc"]
                        },
                        "training": {
                            "epochs": 5,
                            "batch_size": 8,
                            "learning_rate": 0.0003,
                            "max_steps": 1000,
                            "gradient_accumulation_steps": 1,
                            "warmup_steps": 100
                        }
                    }
                },
                "default_base_models": [
                    "microsoft/DialoGPT-small",
                    "microsoft/DialoGPT-medium",
                    "gpt2",
                    "gpt2-medium"
                ],
                "output_base_dir": "lora_adapters_mistral"
            },
            "mcq_generation": {
                "default_difficulty": "medium",
                "default_question_count": 10,
                "cache_questions": True,
                "max_cache_size": 100,
                # 🕒 DYNAMIC TIMEOUT: No hardcoded timeouts - using completion detection
                "use_dynamic_timeout": True,
                "max_stagnation_time": 10.0,  # Max seconds without progress before timeout
                "min_response_time": 2.0,     # Min seconds before considering complete
                "fallback_enabled": False
            },
            "system": {
                "max_memory_usage_mb": 4096,
                "gpu_memory_threshold": 0.8,
                "auto_cleanup_enabled": True,
                "performance_monitoring": True,
                "ui_responsiveness_threshold": 1000
            },
            "user_preferences": {
                "quiz_history_limit": 1000,
                "auto_save_progress": True,
                "show_explanations": True,
                "theme": "dark",
                "sound_enabled": False
            }
        }
    
    def _load_configuration(self):
        """Load configuration from all sources and merge them"""
        try:
            # Start with default schema
            self._config_data = self._deep_copy_dict(self._default_schema)
            
            # Load main unified config
            if self.unified_config_file.exists():
                try:
                    with open(self.unified_config_file, 'r', encoding='utf-8') as f:
                        unified_config = json.load(f)
                    self._merge_config(self._config_data, unified_config)
                    logger.info("✅ Loaded unified configuration file")
                except Exception as e:
                    logger.error(f"❌ Failed to load unified config: {e}")
                    self._try_load_backup()
            
            # Load user settings and merge
            if self.user_settings_file.exists():
                try:
                    with open(self.user_settings_file, 'r', encoding='utf-8') as f:
                        user_settings = json.load(f)
                    
                    # Merge user settings into appropriate sections
                    if 'api_keys' in user_settings:
                        self._config_data['api_keys'].update(user_settings['api_keys'])
                    
                    if 'preferences' in user_settings:
                        self._config_data['user_preferences'].update(user_settings['preferences'])
                    
                    # Store other user settings
                    self._config_data['user_data'] = user_settings
                    
                    logger.info("✅ Merged user settings into unified configuration")
                except Exception as e:
                    logger.error(f"❌ Failed to load user settings: {e}")
            
            logger.info("✅ Configuration loaded and consolidated successfully")
            
        except Exception as e:
            logger.error(f"❌ Critical error loading configuration: {e}")
            logger.warning("⚠️ Using default configuration schema")
            self._config_data = self._deep_copy_dict(self._default_schema)
    
    def _try_load_backup(self):
        """Try to load from backup configuration"""
        try:
            if self.backup_config_file.exists():
                with open(self.backup_config_file, 'r', encoding='utf-8') as f:
                    backup_config = json.load(f)
                self._merge_config(self._config_data, backup_config)
                logger.warning("⚠️ Loaded configuration from backup file")
            else:
                logger.warning("⚠️ No backup configuration available")
        except Exception as e:
            logger.error(f"❌ Failed to load backup configuration: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation path
        
        Args:
            key_path: Dot-separated path (e.g., "training.presets.standard_training")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        with self._config_lock:
            try:
                # Type safety: ensure key_path is a string
                if not isinstance(key_path, str):
                    logger.warning(f"⚠️ Config key expected string, got {type(key_path)}: {key_path}")
                    key_path = str(key_path)
                
                keys = key_path.split('.')
                value = self._config_data
                
                for key in keys:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        return default
                
                return value
                
            except Exception as e:
                logger.error(f"❌ Error getting config value '{key_path}': {e}")
                return default
    
    def set(self, key_path: str, value: Any, save_immediately: bool = False) -> bool:
        """
        Set configuration value using dot notation path
        
        Args:
            key_path: Dot-separated path
            value: Value to set
            save_immediately: Whether to save to disk immediately
            
        Returns:
            True if successful, False otherwise
        """
        with self._config_lock:
            try:
                # Type safety: ensure key_path is a string
                if not isinstance(key_path, str):
                    logger.warning(f"⚠️ Config key expected string, got {type(key_path)}: {key_path}")
                    key_path = str(key_path)
                
                keys = key_path.split('.')
                current = self._config_data
                
                # Navigate to parent of target key
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the value
                current[keys[-1]] = value
                
                logger.debug(f"✅ Config set: {key_path} = {value}")
                
                if save_immediately:
                    self.save()
                else:
                    # Schedule auto-save
                    self._schedule_auto_save()
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Error setting config value '{key_path}': {e}")
                return False
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section"""
        return self.get(section, {})
    
    def get_training_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get a specific training preset configuration"""
        return self.get(f"training.presets.{preset_name}", {})
    
    def get_api_key(self, provider: str) -> str:
        """Get API key for a specific provider"""
        return self.get(f"api_keys.{provider}", "")
    
    def set_api_key(self, provider: str, api_key: str, save_immediately: bool = True) -> bool:
        """Set API key for a specific provider"""
        return self.set(f"api_keys.{provider}", api_key, save_immediately)
    
    def get_user_preference(self, preference: str, default: Any = None) -> Any:
        """Get user preference value"""
        return self.get(f"user_preferences.{preference}", default)
    
    def set_user_preference(self, preference: str, value: Any, save_immediately: bool = True) -> bool:
        """Set user preference value"""
        return self.set(f"user_preferences.{preference}", value, save_immediately)
    
    def save(self) -> bool:
        """Save configuration to disk"""
        with self._config_lock:
            try:
                # Create backup first
                if self.unified_config_file.exists():
                    try:
                        import shutil
                        shutil.copy2(self.unified_config_file, self.backup_config_file)
                    except Exception as backup_error:
                        logger.warning(f"⚠️ Failed to create config backup: {backup_error}")
                
                # 🛡️ SECURITY FIX: Save unified configuration with atomic write to prevent race conditions
                temp_config_file = self.unified_config_file.with_suffix('.tmp')
                try:
                    with open(temp_config_file, 'w', encoding='utf-8') as f:
                        json.dump(self._config_data, f, indent=2, ensure_ascii=False)

                    # Atomic move to prevent corruption
                    if os.name == 'nt':  # Windows
                        if self.unified_config_file.exists():
                            self.unified_config_file.unlink()
                        temp_config_file.rename(self.unified_config_file)
                    else:  # Unix-like systems
                        temp_config_file.rename(self.unified_config_file)

                except Exception as e:
                    # Clean up temp file on error
                    if temp_config_file.exists():
                        temp_config_file.unlink()
                    raise e
                
                # Also save user-specific data to user settings file
                self._save_user_settings()
                
                self._last_save_time = time.time()
                logger.debug("✅ Configuration saved successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to save configuration: {e}")
                return False
    
    def _save_user_settings(self):
        """Save user-specific settings to separate file"""
        try:
            # Ensure user_data directory exists
            self.user_settings_file.parent.mkdir(exist_ok=True)
            
            # Extract user-specific data
            user_data = {
                "api_keys": self._config_data.get("api_keys", {}),
                "preferences": self._config_data.get("user_preferences", {}),
                **self._config_data.get("user_data", {})
            }
            
            with open(self.user_settings_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, indent=2, ensure_ascii=False)
            
            logger.debug("✅ User settings saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to save user settings: {e}")
    
    def _schedule_auto_save(self):
        """Schedule automatic save after interval"""
        current_time = time.time()
        if current_time - self._last_save_time > self._auto_save_interval:
            self.save()
    
    def _merge_config(self, base: Dict, update: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _deep_copy_dict(self, d: Dict) -> Dict:
        """Create a deep copy of a dictionary"""
        import copy
        return copy.deepcopy(d)
    
    def validate_config(self) -> bool:
        """Validate the current configuration"""
        try:
            # Basic validation checks
            required_sections = ["app", "training", "mcq_generation", "system"]
            for section in required_sections:
                if section not in self._config_data:
                    logger.error(f"❌ Missing required config section: {section}")
                    return False
            
            # Validate training presets
            presets = self.get("training.presets", {})
            if not presets:
                logger.error("❌ No training presets found")
                return False
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values"""
        try:
            with self._config_lock:
                self._config_data = self._deep_copy_dict(self._default_schema)
                self.save()
                logger.info("✅ Configuration reset to defaults")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to reset configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration for debugging"""
        return {
            "app_version": self.get("app.version"),
            "training_presets_count": len(self.get("training.presets", {})),
            "api_keys_configured": [k for k, v in self.get("api_keys", {}).items() if v],
            "user_preferences_count": len(self.get("user_preferences", {})),
            "config_file_exists": self.unified_config_file.exists(),
            "last_save_time": self._last_save_time
        }


# Global instance accessor
_unified_config_manager = None
_config_lock = threading.RLock()

def get_unified_config_manager() -> UnifiedConfigManager:
    """Get the global unified configuration manager instance"""
    global _unified_config_manager
    with _config_lock:
        if _unified_config_manager is None:
            _unified_config_manager = UnifiedConfigManager()
        return _unified_config_manager


# Convenience functions for common configuration access patterns
def get_config(key_path: str, default: Any = None) -> Any:
    """Get configuration value using unified manager"""
    return get_unified_config_manager().get(key_path, default)


def set_config(key_path: str, value: Any, save_immediately: bool = False) -> bool:
    """Set configuration value using unified manager"""
    return get_unified_config_manager().set(key_path, value, save_immediately)


def get_training_preset(preset_name: str) -> Dict[str, Any]:
    """Get training preset configuration"""
    return get_unified_config_manager().get_training_preset(preset_name)


def get_api_key(provider: str) -> str:
    """Get API key for provider"""
    return get_unified_config_manager().get_api_key(provider)


def set_api_key(provider: str, api_key: str) -> bool:
    """Set API key for provider"""
    return get_unified_config_manager().set_api_key(provider, api_key)
