#!/usr/bin/env python3
"""
Download pretrained VITS model for Vietnamese finetuning.
Downloads from Coqui TTS model hub.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path

# ---------- Paths ----------
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_PATH, "..", "models")
VITS_PRETRAINED_DIR = os.path.join(MODELS_DIR, "vits_pretrained")

# ---------- Model URLs ----------
VITS_MODELS = {
    "vits_ljspeech": {
        "url": "https://coqui.gateway.scarf.sh/tts/models/vits-ljspeech-16khz/model.pth",
        "config_url": "https://coqui.gateway.scarf.sh/tts/models/vits-ljspeech-16khz/config.json",
        "description": "VITS trained on LJSpeech 16kHz (English, single-speaker)"
    },
    "vits_ljspeech_22khz": {
        "url": "https://coqui.gateway.scarf.sh/tts/models/vits-ljspeech-22khz/model.pth",
        "config_url": "https://coqui.gateway.scarf.sh/tts/models/vits-ljspeech-22khz/config.json",
        "description": "VITS trained on LJSpeech 22kHz (English, single-speaker)"
    },
    "vits_vctk": {
        "url": "https://coqui.gateway.scarf.sh/tts/models/vits-vctk-16khz/model.pth",
        "config_url": "https://coqui.gateway.scarf.sh/tts/models/vits-vctk-16khz/config.json", 
        "description": "VITS trained on VCTK 16kHz (English, multi-speaker)"
    }
}

def download_file(url: str, filepath: str, description: str = ""):
    """Download a file with progress bar."""
    print(f"📥 Downloading {description or os.path.basename(filepath)}...")
    print(f"   URL: {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r   Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
    
    print(f"\n✅ Downloaded: {filepath}")

def main():
    print("🚀 VITS Pretrained Model Downloader")
    print("=" * 50)
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VITS_PRETRAINED_DIR, exist_ok=True)
    
    print(f"📁 Output directory: {VITS_PRETRAINED_DIR}")
    print()
    
    # Choose model
    print("Available models:")
    for i, (name, info) in enumerate(VITS_MODELS.items(), 1):
        print(f"  {i}. {name}: {info['description']}")
    
    print()
    choice = input("Choose model (1-2) or press Enter for vits_ljspeech: ").strip()
    
    if not choice:
        choice = "1"
    
    try:
        choice_idx = int(choice) - 1
        model_name = list(VITS_MODELS.keys())[choice_idx]
        model_info = VITS_MODELS[model_name]
    except (ValueError, IndexError):
        print("❌ Invalid choice, using vits_ljspeech")
        model_name = "vits_ljspeech"
        model_info = VITS_MODELS[model_name]
    
    print(f"\n🎯 Selected: {model_name}")
    print(f"📝 Description: {model_info['description']}")
    print()
    
    # Download model
    model_path = os.path.join(VITS_PRETRAINED_DIR, "model.pth")
    config_path = os.path.join(VITS_PRETRAINED_DIR, "config.json")
    
    try:
        # Download model file
        download_file(model_info['url'], model_path, f"{model_name} model")
        
        # Download config file
        download_file(model_info['config_url'], config_path, f"{model_name} config")
        
        print(f"\n🎉 Download completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📁 Config saved to: {config_path}")
        print(f"\n💡 Now you can run: python scripts/finetune_vits.py")
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("💡 Try alternative download methods:")
        print("   1. Manual download from Coqui TTS:")
        print("      - Model: https://coqui.gateway.scarf.sh/tts/models/vits-ljspeech-16khz/model.pth")
        print("      - Config: https://coqui.gateway.scarf.sh/tts/models/vits-ljspeech-16khz/config.json")
        print("   2. Alternative URLs (try these):")
        print("      - https://coqui.gateway.scarf.sh/tts/models/vits-ljspeech-22khz/model.pth")
        print("      - https://coqui.gateway.scarf.sh/tts/models/vits-vctk-16khz/model.pth")
        print("   3. Use wget/curl with retry:")
        print("      curl -L -o model.pth https://coqui.gateway.scarf.sh/tts/models/vits-ljspeech-16khz/model.pth")
        print("   4. Check network connection and firewall")
        print("   5. Try using VPN if behind corporate network")
        
        # Don't exit, let user try manual download
        print(f"\n📁 Please download manually to: {VITS_PRETRAINED_DIR}")
        print("   Then run: python scripts/finetune_vits.py")

if __name__ == "__main__":
    main()
