"""
🔧 Settings Service

This service handles all user settings operations, extracted from the PythonBridge
to follow the Single Responsibility Principle.

CRITICAL FIX: Eliminates the "God Object" anti-pattern by separating concerns:
- Loading and saving user settings
- API key management
- Settings validation and backup
- Settings corruption recovery
"""

import json
import logging
import threading
from typing import Dict, Any, Optional
from pathlib import Path
from ..utils.crypto_utils import CryptoManager

logger = logging.getLogger(__name__)

class SettingsService:
    """
    🔧 FIX: Dedicated service for settings management
    
    This service handles all settings-related operations that were previously
    mixed into the PythonBridge "God Object".
    """
    
    def __init__(self, settings_dir: str = "data"):
        self.settings_dir = Path(settings_dir)
        self.settings_file = self.settings_dir / "user_settings.json"
        self.backup_file = self.settings_file.with_suffix('.json.bak')
        
        # Thread safety
        self.settings_lock = threading.RLock()
        
        # Initialize crypto manager
        self.crypto_manager = CryptoManager(str(self.settings_dir / ".key"))

        # Default settings
        self.default_settings = {
            "theme": "light",
            "font_size": 10,
            "storage_limit": 1073741824,  # 1GB
            "auto_switch_images": False,
            "offline_mode": True,
            "answered_questions_history": [],
            "default_timer": 30,
            "show_answers": True,
            "api_keys": {
                "openai": "",
                "anthropic": "",
                "gemini": "",
                "groq": "",
                "openrouter": ""
            },
            "default_game_mode": "casual",
            "default_difficulty": "medium",
            "default_submode": "mixed",
            "default_quiz_mode": "offline"
        }
        
        # Ensure settings directory exists
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🔧 SettingsService initialized")
    
    def _encrypt_api_keys(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt API keys before saving"""
        encrypted_settings = settings.copy()
        api_keys = encrypted_settings.get("api_keys", {})

        encrypted_keys = {}
        for provider, key in api_keys.items():
            encrypted_keys[provider] = self.crypto_manager.encrypt(key)

        encrypted_settings["api_keys"] = encrypted_keys
        return encrypted_settings

    def _decrypt_api_keys(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt API keys after loading"""
        decrypted_settings = settings.copy()
        api_keys = decrypted_settings.get("api_keys", {})

        decrypted_keys = {}
        for provider, encrypted_key in api_keys.items():
            decrypted_keys[provider] = self.crypto_manager.decrypt(encrypted_key)

        decrypted_settings["api_keys"] = decrypted_keys
        return decrypted_settings

    def load_settings(self) -> Dict[str, Any]:
        """
        Load user settings with corruption recovery
        
        Returns:
            Dict containing user settings
        """
        try:
            with self.settings_lock:
                if not self.settings_file.exists():
                    logger.info("📋 No settings file found, using defaults")
                    return self.default_settings.copy()
                
                try:
                    with open(self.settings_file, 'r', encoding='utf-8') as f:
                        settings = json.load(f)
                    
                    # Merge with defaults to ensure all keys exist
                    merged_settings = self.default_settings.copy()
                    merged_settings.update(settings)
                    
                    # Decrypt API keys
                    decrypted_settings = self._decrypt_api_keys(merged_settings)

                    logger.info("✅ Settings loaded successfully")
                    return decrypted_settings

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    # 🔧 FIX: File exists but is corrupted - attempt recovery
                    logger.warning(f"⚠️ Settings file corrupted: {e}")
                    return self._recover_corrupted_settings(e)
                    
        except Exception as e:
            logger.error(f"❌ Failed to load settings: {e}")
            return self.default_settings.copy()
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save user settings with encryption and backup"""
        try:
            with self.settings_lock:
                # Create backup of current settings if they exist
                if self.settings_file.exists():
                    with open(self.settings_file, 'r', encoding='utf-8') as f:
                        current = f.read()
                    with open(self.backup_file, 'w', encoding='utf-8') as f:
                        f.write(current)

                # Encrypt API keys before saving
                encrypted_settings = self._encrypt_api_keys(settings)

                # Save new settings
                with open(self.settings_file, 'w', encoding='utf-8') as f:
                    json.dump(encrypted_settings, f, indent=4)

                logger.info("✅ Settings saved successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save settings: {e}")
            return False
    
    def update_setting(self, key: str, value: Any) -> bool:
        """
        Update a single setting
        
        Args:
            key: Setting key (supports dot notation for nested keys)
            value: New value
            
        Returns:
            bool: True if successful
        """
        try:
            settings = self.load_settings()
            
            # Type safety: ensure key is a string
            if not isinstance(key, str):
                logger.warning(f"⚠️ Settings key expected string, got {type(key)}: {key}")
                key = str(key)
            
            # Handle nested keys (e.g., "api_keys.openai")
            if '.' in key:
                keys = key.split('.')
                current = settings
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                settings[key] = value
            
            return self.save_settings(settings)
            
        except Exception as e:
            logger.error(f"❌ Failed to update setting {key}: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a single setting value
        
        Args:
            key: Setting key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        try:
            settings = self.load_settings()
            
            # Type safety: ensure key is a string
            if not isinstance(key, str):
                logger.warning(f"⚠️ Settings key expected string, got {type(key)}: {key}")
                key = str(key)
            
            # Handle nested keys (e.g., "api_keys.openai")
            if '.' in key:
                keys = key.split('.')
                current = settings
                for k in keys:
                    if isinstance(current, dict) and k in current:
                        current = current[k]
                    else:
                        return default
                return current
            else:
                return settings.get(key, default)
                
        except Exception as e:
            logger.error(f"❌ Failed to get setting {key}: {e}")
            return default
    
    def test_api_key(self, provider: str, api_key: str) -> Dict[str, Any]:
        """
        Test an API key for a specific provider
        
        Args:
            provider: API provider name
            api_key: API key to test
            
        Returns:
            Dict with test result
        """
        try:
            # This would implement actual API key testing
            # For now, just validate format
            if not api_key or len(api_key) < 10:
                return {
                    'valid': False,
                    'error': 'API key too short or empty'
                }
            
            # Provider-specific validation
            if provider == 'openai' and not api_key.startswith('sk-'):
                return {
                    'valid': False,
                    'error': 'OpenAI API keys should start with "sk-"'
                }
            
            # TODO: Implement actual API testing
            logger.info(f"🔑 API key format validated for {provider}")
            return {
                'valid': True,
                'message': f'{provider} API key format is valid'
            }
            
        except Exception as e:
            logger.error(f"❌ API key test failed for {provider}: {e}")
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _recover_corrupted_settings(self, error: Exception) -> Dict[str, Any]:
        """🔧 FIX: Attempt to recover data from corrupted settings file"""
        logger.warning(f"🔧 Attempting to recover corrupted settings file")
        
        # Step 1: Try to load backup file
        if self.backup_file.exists():
            try:
                with open(self.backup_file, 'r', encoding='utf-8') as f:
                    backup_settings = json.load(f)
                logger.info("✅ Successfully recovered settings from backup file")
                
                # Create new backup from recovered data
                self._create_settings_backup(backup_settings)
                return {**self.default_settings, **backup_settings}
                
            except Exception as backup_error:
                logger.warning(f"⚠️ Backup file also corrupted: {backup_error}")
        
        # Step 2: Try to salvage API keys using regex
        salvaged_data = {}
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                corrupted_content = f.read()
            
            # Try to extract API keys using regex
            import re
            api_key_patterns = {
                'openai': r'"openai"\s*:\s*"([^"]*)"',
                'anthropic': r'"anthropic"\s*:\s*"([^"]*)"',
                'gemini': r'"gemini"\s*:\s*"([^"]*)"',
                'groq': r'"groq"\s*:\s*"([^"]*)"',
                'openrouter': r'"openrouter"\s*:\s*"([^"]*)"'
            }
            
            salvaged_api_keys = {}
            for provider, pattern in api_key_patterns.items():
                match = re.search(pattern, corrupted_content)
                if match and match.group(1):
                    salvaged_api_keys[provider] = match.group(1)
                    logger.info(f"✅ Salvaged {provider} API key")
            
            if salvaged_api_keys:
                salvaged_data['api_keys'] = {**self.default_settings['api_keys'], **salvaged_api_keys}
                logger.info(f"✅ Salvaged {len(salvaged_api_keys)} API keys from corrupted file")
            
            # Try to salvage other important settings
            other_patterns = {
                'default_quiz_mode': r'"default_quiz_mode"\s*:\s*"([^"]*)"',
                'theme': r'"theme"\s*:\s*"([^"]*)"',
                'default_difficulty': r'"default_difficulty"\s*:\s*"([^"]*)"'
            }
            
            for key, pattern in other_patterns.items():
                match = re.search(pattern, corrupted_content)
                if match and match.group(1):
                    salvaged_data[key] = match.group(1)
                    logger.info(f"✅ Salvaged {key}: {match.group(1)}")
                    
        except Exception as salvage_error:
            logger.error(f"❌ Failed to salvage data from corrupted file: {salvage_error}")
        
        # Step 3: Merge salvaged data with defaults
        recovered_settings = {**self.default_settings, **salvaged_data}
        
        # Step 4: Save recovered settings
        if salvaged_data:
            logger.warning("🚨 SETTINGS RECOVERY: Your settings file was corrupted but we recovered some data.")
            logger.warning("🚨 Please verify your API keys and other settings in the Settings menu.")
            
            try:
                self._create_settings_backup(recovered_settings)
                with open(self.settings_file, 'w', encoding='utf-8') as f:
                    json.dump(recovered_settings, f, indent=2, ensure_ascii=False)
                logger.info("✅ Saved recovered settings")
            except Exception as save_error:
                logger.error(f"❌ Failed to save recovered settings: {save_error}")
        else:
            logger.warning("🚨 SETTINGS CORRUPTION: Could not recover any data. Using defaults.")
        
        return recovered_settings
    
    def _create_settings_backup(self, settings: Dict[str, Any]):
        """Create backup of settings file"""
        try:
            with open(self.backup_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            logger.debug(f"✅ Created settings backup: {self.backup_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create settings backup: {e}")
    
    def _validate_settings(self, settings: Dict[str, Any]) -> bool:
        """Validate settings structure and values"""
        try:
            # Check required keys exist
            required_keys = ['theme', 'api_keys', 'default_quiz_mode']
            for key in required_keys:
                if key not in settings:
                    logger.error(f"❌ Missing required setting: {key}")
                    return False
            
            # Validate API keys structure
            if not isinstance(settings.get('api_keys'), dict):
                logger.error("❌ api_keys must be a dictionary")
                return False
            
            # Validate theme
            valid_themes = ['light', 'dark']
            if settings.get('theme') not in valid_themes:
                logger.error(f"❌ Invalid theme: {settings.get('theme')}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Settings validation error: {e}")
            return False
