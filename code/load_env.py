#!/usr/bin/env python3
"""
Utility to load environment variables from .env file
"""

import os
from pathlib import Path

def load_dotenv(env_file=".env"):
    """Load environment variables from .env file"""
    env_path = Path(env_file)
    
    if not env_path.exists():
        print(f"⚠️  .env file not found: {env_path}")
        return False
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
                    print(f"✅ Loaded: {key}")
        
        print(f"✅ Loaded environment variables from {env_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
        return False

def get_hf_token():
    """Get Hugging Face token from environment"""
    # First try to load from .env if not already loaded
    if "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        load_dotenv()
    
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        print(f"✅ HF Token found: {token[:10]}...")
        return token
    else:
        print("❌ HUGGING_FACE_HUB_TOKEN not found in environment")
        return None

if __name__ == "__main__":
    # Test loading .env file
    success = load_dotenv()
    if success:
        token = get_hf_token()
        if token:
            print("🎉 Environment setup successful!")
        else:
            print("❌ Token not found in .env file")
    else:
        print("❌ Failed to load .env file") 