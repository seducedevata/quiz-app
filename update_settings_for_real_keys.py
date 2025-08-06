#!/usr/bin/env python3
"""
Update settings with instructions for real API keys
"""

import json
from pathlib import Path

def update_settings_with_instructions():
    """Update settings file with real API key instructions"""
    
    # Create settings with proper structure
    settings = {
        "api_keys": {
            "openrouter": "",  # Empty = disabled
            "groq": "",  # Replace with real Groq API key from https://console.groq.com/keys
            "tavily": "",  # Replace with real Tavily API key from https://app.tavily.com
            "anthropic": "",  # Empty = disabled
            "openai": "",  # Empty = disabled  
            "gemini": ""  # Empty = disabled
        },
        "api_providers_enabled": {
            "openrouter": False,  # User disabled
            "groq": True,  # User enabled - needs real API key
            "tavily": True,  # User enabled - needs real API key
            "anthropic": False,  # User disabled
            "openai": False,  # User disabled
            "gemini": False  # User disabled
        },
        "_instructions": {
            "api_keys_note": "Replace empty strings with real API keys from the respective providers",
            "groq_key_url": "https://console.groq.com/keys",
            "tavily_key_url": "https://app.tavily.com",
            "openrouter_key_url": "https://openrouter.ai/keys"
        }
    }
    
    # Save settings
    settings_dir = Path("user_data")
    settings_dir.mkdir(exist_ok=True)
    
    settings_path = settings_dir / "user_settings.json"
    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2)
    
    print(f"✅ Updated settings file: {settings_path}")
    print(f"")
    print(f"📝 IMPORTANT: Replace the empty API keys with real ones:")
    print(f"   • Groq API key: https://console.groq.com/keys")
    print(f"   • Tavily API key: https://app.tavily.com")
    print(f"")
    print(f"🎯 Current setup:")
    print(f"   ✅ OpenRouter: DISABLED (as requested)")
    print(f"   ✅ Groq: ENABLED (needs real API key)")
    print(f"   ✅ Tavily: ENABLED (needs real API key)")
    print(f"   ✅ Auto-restoration of OpenRouter key: DISABLED")

if __name__ == "__main__":
    update_settings_with_instructions()
