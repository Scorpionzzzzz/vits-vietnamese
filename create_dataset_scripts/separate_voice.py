#!/usr/bin/env python3
"""
Voice Separation Tool
Separates voice from background music using Demucs
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List
import torch
import numpy as np
import librosa
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceSeparator:
    def __init__(self, input_dir: str = "../data/downloaded_audio", output_dir: str = "../data/separated_voice"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load Demucs model
        self.demucs_model = None
        self._load_model()
    
    def _load_model(self):
        """Load Demucs model"""
        logger.info("Loading Demucs model...")
        self.demucs_model = get_model("htdemucs")
        self.demucs_model.to(self.device)
        self.demucs_model.eval()
        logger.info(f"✅ Demucs model loaded on {self.device}")
    
    def separate_voice(self, audio_path: str) -> str:
        """Separate voice from background music"""
        logger.info(f"Separating voice from: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=44100, mono=False)
        # Ensure stereo (2 channels) for Demucs
        if audio.ndim == 1:
            # mono -> duplicate to stereo
            logger.info("Input is mono. Duplicating channel for Demucs.")
            audio = np.stack([audio, audio], axis=0)
        elif audio.ndim == 2 and audio.shape[0] == 1:
            logger.info("Input has 1 channel. Duplicating to 2 channels for Demucs.")
            audio = np.concatenate([audio, audio], axis=0)
        # skip too short files
        if audio.shape[-1] < sr:
            raise ValueError("Audio too short (<1s)")
        
        # Convert to tensor format (batch, channels, time)
        audio_tensor = torch.from_numpy(audio).float()  # (2, T)
        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(0)     # (1, 2, T)
        
        # Move to device
        audio_tensor = audio_tensor.to(self.device)
        
        # Apply Demucs separation with CUDA fallback to CPU on OOM
        with torch.no_grad():
            try:
                sources = apply_model(self.demucs_model, audio_tensor, device=self.device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device == "cuda":
                    logger.warning("CUDA OOM. Falling back to CPU for this file.")
                    torch.cuda.empty_cache()
                    sources = apply_model(self.demucs_model.cpu(), audio_tensor.cpu(), device="cpu")
                else:
                    raise
        
        # Extract vocals (4th source in htdemucs)
        sources = sources.squeeze(0)
        vocals = sources[3].cpu().numpy()
        
        # Save separated voice
        input_filename = Path(audio_path).stem
        output_filename = f"{input_filename}_voice.wav"
        output_path = self.output_dir / output_filename
        # shape handling for mono/stereo
        wav_to_save = vocals.T if vocals.ndim == 2 else vocals
        sf.write(output_path, wav_to_save, sr)
        
        # free GPU memory between files
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"✅ Voice separated: {output_filename}")
        return str(output_path)
    
    def separate_single_file(self, audio_path: str):
        """Separate voice from a single file"""
        if not os.path.exists(audio_path):
            logger.error(f"File not found: {audio_path}")
            return None
        
        return self.separate_voice(audio_path)
    
    def separate_all_files(self):
        """Separate voice from all WAV files in input directory"""
        wav_files = sorted(self.input_dir.glob("*.wav"))
        
        if not wav_files:
            logger.warning(f"No WAV files found in {self.input_dir}")
            return []
        
        logger.info(f"Found {len(wav_files)} WAV files to process")
        
        separated_files = []
        for i, wav_file in enumerate(wav_files, 1):
            try:
                logger.info(f"Progress: {i}/{len(wav_files)}")
                output_path = self.separate_voice(str(wav_file))
                separated_files.append(output_path)
            except Exception as e:
                logger.error(f"Failed to separate {wav_file}: {e}")
                continue
        
        logger.info(f"✅ Separation complete! {len(separated_files)}/{len(wav_files)} files processed")
        return separated_files

def main():
    parser = argparse.ArgumentParser(description="Voice Separation Tool")
    parser.add_argument("--input_file", help="Single audio file to process")
    parser.add_argument("--input_dir", default="../data/downloaded_audio", help="Input directory with WAV files")
    parser.add_argument("--output_dir", default="../data/separated_voice", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Process all files in input directory")
    
    args = parser.parse_args()
    
    separator = VoiceSeparator(args.input_dir, args.output_dir)
    
    if args.input_file:
        # Process single file
        separator.separate_single_file(args.input_file)
    elif args.all:
        # Process all files
        separator.separate_all_files()
    else:
        # Interactive mode
        print("Enter audio file path to separate voice:")
        audio_file = input().strip()
        if audio_file:
            separator.separate_single_file(audio_file)
        else:
            print("No file provided!")

if __name__ == "__main__":
    main() 