"""
OpenRouter Configuration Manager
Centralizes OpenRouter model configuration and makes it dynamic/configurable
"""

import json
import logging
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class OpenRouterConfig:
    """Manages OpenRouter model configurations dynamically"""
    
    def __init__(self):
        self.config_file = Path("user_data/openrouter_config.json")
        self.default_config = {
            "preferred_model": "qwen/qwq-32b-preview",
            "free_models": [
                # Updated list of OpenRouter free models (as of 2025)
                "qwen/qwq-32b-preview",       # 32B reasoning specialist with thinking tokens
                "meta-llama/llama-3.1-8b-instruct",
                "meta-llama/llama-3.1-70b-instruct", 
                "meta-llama/llama-3.2-1b-instruct",
                "meta-llama/llama-3.2-3b-instruct",
                "meta-llama/llama-3.2-11b-vision-instruct",
                "meta-llama/llama-3.2-90b-vision-instruct",
                "microsoft/phi-3-mini-128k-instruct",
                "microsoft/phi-3-medium-128k-instruct",
                "mistralai/mistral-7b-instruct",
                "huggingface/zephyr-7b-beta",
                "openchat/openchat-7b",
                "undi95/toppy-m-7b",
                "gryphe/mythomist-7b",
                "nousresearch/nous-capybara-7b",
                "teknium/openhermes-2-mistral-7b",
                "togethercomputer/redpajama-incite-7b-chat",
                "psyfighter2/psyfighter-13b-2",
                "koboldai/psyfighter-13b-2",
                "intel/neural-chat-7b-v3-1",
                "pygmalionai/mythalion-13b",
                "jondurbin/airoboros-l2-70b-gpt4-1.4.1",
                "austism/chronos-hermes-13b"
            ],
            "paid_models": [
                "openai/gpt-4-turbo-preview",
                "anthropic/claude-3-opus-20240229",
                "anthropic/claude-3-sonnet-20240229",
                "google/gemini-pro-1.5",
                "meta-llama/llama-3.1-405b-instruct"
            ],
            "model_preferences": {
                "reasoning": "qwen/qwq-32b-preview",
                "general": "meta-llama/llama-3.1-70b-instruct",
                "coding": "meta-llama/llama-3.1-70b-instruct",
                "creative": "gryphe/mythomist-7b"
            }
        }
        
    def load_config(self) -> Dict[str, Any]:
        """Load OpenRouter configuration from file or create default"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"✅ Loaded OpenRouter config from {self.config_file}")
                return config
            else:
                logger.info("📝 Creating default OpenRouter configuration")
                self.save_config(self.default_config)
                return self.default_config.copy()
        except Exception as e:
            logger.error(f"❌ Failed to load OpenRouter config: {e}")
            return self.default_config.copy()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save OpenRouter configuration to file"""
        try:
            self.config_file.parent.mkdir(exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Saved OpenRouter config to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save OpenRouter config: {e}")
            return False
    
    def get_free_models(self) -> List[str]:
        """Get list of free models"""
        config = self.load_config()
        return config.get("free_models", self.default_config["free_models"])
    
    def get_preferred_model(self, task_type: str = "general") -> str:
        """Get preferred model for a specific task type"""
        config = self.load_config()
        preferences = config.get("model_preferences", {})
        return preferences.get(task_type, config.get("preferred_model", self.default_config["preferred_model"]))
    
    def add_model(self, model_id: str, is_free: bool = True) -> bool:
        """Add a new model to the configuration"""
        try:
            config = self.load_config()
            target_list = "free_models" if is_free else "paid_models"
            
            if model_id not in config[target_list]:
                config[target_list].append(model_id)
                self.save_config(config)
                logger.info(f"✅ Added {model_id} to {target_list}")
                return True
            else:
                logger.info(f"ℹ️ Model {model_id} already in {target_list}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to add model {model_id}: {e}")
            return False
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from both free and paid lists"""
        try:
            config = self.load_config()
            removed = False
            
            if model_id in config["free_models"]:
                config["free_models"].remove(model_id)
                removed = True
                
            if model_id in config["paid_models"]:
                config["paid_models"].remove(model_id)
                removed = True
            
            if removed:
                self.save_config(config)
                logger.info(f"✅ Removed {model_id} from configuration")
                
            return removed
        except Exception as e:
            logger.error(f"❌ Failed to remove model {model_id}: {e}")
            return False
    
    def update_model_preference(self, task_type: str, model_id: str) -> bool:
        """Update preferred model for a specific task type"""
        try:
            config = self.load_config()
            if "model_preferences" not in config:
                config["model_preferences"] = {}
                
            config["model_preferences"][task_type] = model_id
            self.save_config(config)
            logger.info(f"✅ Set preferred {task_type} model to {model_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update preference for {task_type}: {e}")
            return False


# Global instance
openrouter_config = OpenRouterConfig()


def get_openrouter_models(use_free_only: bool = True) -> List[str]:
    """
    🔧 DYNAMIC MODEL LOADING: Get OpenRouter models from configuration
    
    This replaces hardcoded model lists in the online generator
    """
    if use_free_only:
        return openrouter_config.get_free_models()
    else:
        config = openrouter_config.load_config()
        return config.get("free_models", []) + config.get("paid_models", [])


def get_preferred_openrouter_model(task_type: str = "general") -> str:
    """Get the preferred model for a specific task"""
    return openrouter_config.get_preferred_model(task_type)
