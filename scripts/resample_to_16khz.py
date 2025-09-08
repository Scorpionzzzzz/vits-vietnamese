#!/usr/bin/env python3
"""
Resample WAV files from 22.05kHz to 16kHz mono for MMS-TTS-VIE compatibility.
MMS-TTS-VIE expects 16kHz mono audio (sampling_rate=16000).
"""

import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

def resample_audio(input_dir: str, output_dir: str, force: bool = False):
    """Resample all WAV files to 16kHz mono using ffmpeg."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_path}")
        return False
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all WAV files
    wav_files = list(input_path.glob("*.wav"))
    if not wav_files:
        print(f"❌ No WAV files found in: {input_path}")
        return False
    
    print(f"📁 Found {len(wav_files)} WAV files")
    print(f"🔄 Resampling to 16kHz mono...")
    print(f"📁 Output: {output_path}")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg not found. Please install ffmpeg first.")
        print("   Windows: https://ffmpeg.org/download.html")
        print("   Or use: conda install ffmpeg")
        return False
    
    success_count = 0
    error_count = 0
    
    for wav_file in tqdm(wav_files, desc="Resampling"):
        output_file = output_path / wav_file.name
        
        # Skip if output exists and not forcing
        if output_file.exists() and not force:
            continue
            
        try:
            # ffmpeg command: convert to 16kHz mono
            cmd = [
                "ffmpeg", "-y",  # Overwrite output
                "-i", str(wav_file),  # Input file
                "-ac", "1",  # Mono (1 channel)
                "-ar", "16000",  # 16kHz sample rate
                "-loglevel", "error",  # Minimal logging
                str(output_file)  # Output file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                success_count += 1
            else:
                print(f"❌ Error processing {wav_file.name}: {result.stderr}")
                error_count += 1
                
        except Exception as e:
            print(f"❌ Exception processing {wav_file.name}: {e}")
            error_count += 1
    
    print(f"\n✅ Resampling completed!")
    print(f"   Success: {success_count}")
    print(f"   Errors: {error_count}")
    print(f"   Output directory: {output_path}")
    
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description="Resample WAV files to 16kHz mono")
    parser.add_argument("input_dir", help="Input directory containing WAV files")
    parser.add_argument("output_dir", help="Output directory for resampled files")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing files")
    
    args = parser.parse_args()
    
    if resample_audio(args.input_dir, args.output_dir, args.force):
        print(f"\n💡 Next steps:")
        print(f"   1. Create JSONL files: python scripts/create_jsonl_for_hf.py")
        print(f"   2. Clone finetune repo: git clone https://github.com/ylacombe/finetune-hf-vits")
        print(f"   3. Follow finetuning steps in the repo")
    else:
        print("❌ Resampling failed")

if __name__ == "__main__":
    main()
