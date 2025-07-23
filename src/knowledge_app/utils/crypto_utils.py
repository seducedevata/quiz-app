"""
🔐 Crypto Utils

Handles secure encryption/decryption of sensitive data like API keys
"""

import os
import base64
from cryptography.fernet import Fernet
from pathlib import Path

class CryptoManager:
    """Handles encryption/decryption of sensitive data"""

    def __init__(self, key_file: str = "data/.key"):
        self.key_path = Path(key_file)
        self.key = self._load_or_create_key()
        self.cipher = Fernet(self.key)

    def _load_or_create_key(self) -> bytes:
        """Load existing key or generate a new one"""
        if self.key_path.exists():
            with open(self.key_path, 'rb') as f:
                return f.read()

        # Generate new key
        key = Fernet.generate_key()
        self.key_path.parent.mkdir(parents=True, exist_ok=True)

        # Save key
        with open(self.key_path, 'wb') as f:
            f.write(key)

        return key

    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        if not data:
            return ""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt encrypted string data"""
        if not encrypted_data:
            return ""
        try:
            return self.cipher.decrypt(encrypted_data.encode()).decode()
        except Exception:
            return ""  # Return empty string on decryption failure
