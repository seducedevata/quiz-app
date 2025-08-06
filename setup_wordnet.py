#!/usr/bin/env python3
"""
NLTK WordNet downloader script for Knowledge App
Downloads WordNet data to the correct location
"""

import nltk
import ssl
import os
import sys

def download_wordnet():
    """Download WordNet data with SSL handling"""
    try:
        # Create unverified SSL context for downloads
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Set the NLTK data path
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Add to NLTK data path
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.append(nltk_data_dir)
        
        print(f"📂 NLTK data directory: {nltk_data_dir}")
        print("🔄 Downloading WordNet...")
        
        # Download required NLTK data
        resources = ['wordnet', 'omw-1.4', 'punkt', 'stopwords', 'averaged_perceptron_tagger']
        
        for resource in resources:
            try:
                print(f"📥 Downloading {resource}...")
                nltk.download(resource, download_dir=nltk_data_dir, quiet=False)
                print(f"✅ Successfully downloaded {resource}")
            except Exception as e:
                print(f"❌ Failed to download {resource}: {e}")
        
        # Test WordNet
        print("🧪 Testing WordNet...")
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets('computer')
        if synsets:
            print(f"✅ WordNet test successful! Found {len(synsets)} synsets for 'computer'")
            return True
        else:
            print("❌ WordNet test failed - no synsets found")
            return False
            
    except Exception as e:
        print(f"❌ WordNet setup failed: {e}")
        return False

if __name__ == "__main__":
    success = download_wordnet()
    if success:
        print("🎉 WordNet setup complete!")
        sys.exit(0)
    else:
        print("💥 WordNet setup failed!")
        sys.exit(1)
