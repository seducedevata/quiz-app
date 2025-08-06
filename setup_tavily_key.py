#!/usr/bin/env python3
"""
Setup script to store Tavily API key in secure storage for testing
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_app.core.secure_api_key_manager import SecureApiKeyManager

def setup_tavily_key():
    """Store the Tavily API key in secure storage"""
    try:
        # Initialize secure key manager
        key_manager = SecureApiKeyManager()
        
        # Store the Tavily API key
        tavily_key = "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"
        success = key_manager.store_api_key('tavily', tavily_key)
        
        if success:
            print("[SUCCESS] Tavily API key stored successfully in secure storage")
            
            # Verify it was stored
            retrieved_key = key_manager.get_api_key('tavily')
            if retrieved_key:
                print("[SUCCESS] Tavily API key retrieved successfully from secure storage")
                print(f"[INFO] Key length: {len(retrieved_key)} characters")
            else:
                print("[ERROR] Failed to retrieve Tavily API key from secure storage")
        else:
            print("[ERROR] Failed to store Tavily API key in secure storage")
            
    except Exception as e:
        print(f"[ERROR] Exception during setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    setup_tavily_key()
