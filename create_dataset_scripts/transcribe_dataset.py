#!/usr/bin/env python3
"""
Transcription and Dataset Creator for finetune-hf-vits
Creates JSONL dataset compatible with MMS-TTS-VIE finetuning
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import List, Tuple
import torch
import numpy as np
import librosa
import soundfile as sf
from faster_whisper import WhisperModel
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptionProcessor:
    def __init__(self, input_dir: str = "data/separated_voice", output_dir: str = "data",
                 seg_min_len_sec: float = 3.0, seg_max_len_sec: float = 8.0,
                 seg_top_db: float = 30.0, seg_merge_silence_sec: float = 0.6,
                 append: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.wavs_dir = self.output_dir / "wavs_16khz"  # Changed to match finetune-hf-vits format
        self.train_jsonl = self.output_dir / "train.jsonl"
        self.val_jsonl = self.output_dir / "val.jsonl"
        self.test_jsonl = self.output_dir / "test.jsonl"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.wavs_dir.mkdir(exist_ok=True)
        
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Audio processing parameters - Changed to 16kHz for MMS-TTS-VIE
        self.target_sr = 16000  # Changed from 22050 to 16000
        self.target_channels = 1

        # Segmentation parameters
        self.seg_min_len_sec = seg_min_len_sec
        self.seg_max_len_sec = seg_max_len_sec
        self.seg_top_db = seg_top_db
        self.seg_merge_silence_sec = seg_merge_silence_sec
        
        # Append mode (add to existing metadata/wavs)
        self.append = append

        # Load Whisper model
        self.whisper_model = None
        self._load_whisper_model()
    
    def _load_whisper_model(self):
        """Load Whisper model"""
        logger.info("Loading Whisper model...")
        self.whisper_model = WhisperModel(
            "large-v3",
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "int8"
        )
        logger.info("✅ Whisper model loaded successfully")
    
    def split_audio_by_silence(self, audio_path: str) -> List[str]:
        """Segment audio to chunks with strict length constraints"""
        logger.info(f"Splitting audio: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # Detect non-silent intervals
        intervals = librosa.effects.split(y, top_db=self.seg_top_db)
        if len(intervals) == 0:
            return []

        # Merge intervals separated by short silences (<= seg_merge_silence_sec)
        merged: List[Tuple[int, int]] = []
        max_gap = int(self.seg_merge_silence_sec * sr)
        cur_start, cur_end = intervals[0]
        
        for start, end in intervals[1:]:
            if start - cur_end <= max_gap:
                cur_end = end
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((cur_start, cur_end))

        # Build chunks with strict length constraints
        chunks: List[Tuple[int, int]] = []
        min_len = int(self.seg_min_len_sec * sr)
        max_len = int(self.seg_max_len_sec * sr)
        
        for start, end in merged:
            chunk_len = end - start
            
            if chunk_len < min_len:
                # Too short, skip this interval
                logger.debug(f"Skipping short interval: {chunk_len/sr:.2f}s")
                continue
                
            elif chunk_len > max_len:
                # Too long, split into smaller chunks
                logger.debug(f"Splitting long interval: {chunk_len/sr:.2f}s")
                num_splits = max(1, int(chunk_len / max_len) + (1 if chunk_len % max_len > 0 else 0))
                split_len = chunk_len // num_splits
                
                for i in range(num_splits):
                    split_start = start + i * split_len
                    split_end = min(start + (i + 1) * split_len, end)
                    split_chunk_len = split_end - split_start
                    
                    # Ensure the split chunk meets minimum length
                    if split_chunk_len >= min_len:
                        chunks.append((split_start, split_end))
                        logger.debug(f"Added split chunk: {split_chunk_len/sr:.2f}s")
                    else:
                        # If this split is too short, merge with previous if possible
                        if chunks and (split_end - chunks[-1][0]) <= max_len:
                            # Merge with previous chunk
                            prev_start, prev_end = chunks.pop()
                            chunks.append((prev_start, split_end))
                            logger.debug(f"Merged with previous: {(split_end - prev_start)/sr:.2f}s")
                        else:
                            # Skip this short chunk
                            logger.debug(f"Skipping short split: {split_chunk_len/sr:.2f}s")
            else:
                # Perfect length, add directly
                chunks.append((start, end))
                logger.debug(f"Added perfect chunk: {chunk_len/sr:.2f}s")

        # Final validation: ensure all chunks meet length constraints
        validated_chunks: List[Tuple[int, int]] = []
        for start, end in chunks:
            chunk_len = end - start
            chunk_duration = chunk_len / sr
            
            if min_len <= chunk_len <= max_len:
                validated_chunks.append((start, end))
                logger.debug(f"Validated chunk: {chunk_duration:.2f}s")
            else:
                logger.warning(f"Invalid chunk length: {chunk_duration:.2f}s (should be {self.seg_min_len_sec}-{self.seg_max_len_sec}s)")
                # Try to fix by splitting if too long
                if chunk_len > max_len:
                    num_splits = max(1, int(chunk_len / max_len) + (1 if chunk_len % max_len > 0 else 0))
                    split_len = chunk_len // num_splits
                    
                    for i in range(num_splits):
                        split_start = start + i * split_len
                        split_end = min(start + (i + 1) * split_len, end)
                        split_chunk_len = split_end - split_start
                        
                        if min_len <= split_chunk_len <= max_len:
                            validated_chunks.append((split_start, split_end))
                            logger.debug(f"Fixed long chunk: {split_chunk_len/sr:.2f}s")

        # Extract and save chunks
        chunk_paths = []
        for i, (start, end) in enumerate(validated_chunks):
            chunk_audio = y[start:end]
            chunk_duration = len(chunk_audio) / sr
            
            # Final length check before saving
            if self.seg_min_len_sec <= chunk_duration <= self.seg_max_len_sec:
                chunk_path = f"{audio_path}_chunk_{i:03d}.wav"
                sf.write(chunk_path, chunk_audio, sr)
                chunk_paths.append(chunk_path)
                logger.info(f"Saved chunk {i}: {chunk_duration:.2f}s")
            else:
                logger.warning(f"Final validation failed for chunk {i}: {chunk_duration:.2f}s")

        logger.info(f"Created {len(chunk_paths)} validated audio chunks")
        return chunk_paths

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper"""
        try:
            segments, _ = self.whisper_model.transcribe(audio_path, language="vi")
            transcript = " ".join([segment.text.strip() for segment in segments])
            return transcript.strip()
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return "[SILENCE]"

    # Vietnamese number to words conversion methods
    _NUM_WORDS = {
        0: "không", 1: "một", 2: "hai", 3: "ba", 4: "bốn", 5: "năm",
        6: "sáu", 7: "bảy", 8: "tám", 9: "chín", 10: "mười"
    }

    def _read_tens_pair(self, n: int) -> str:
        if n < 11:
            return self._NUM_WORDS[n]
        if n < 20:
            return f"mười {self._NUM_WORDS[n % 10]}"
        if n % 10 == 0:
            return f"{self._NUM_WORDS[n // 10]} mươi"
        return f"{self._NUM_WORDS[n // 10]} mươi {self._NUM_WORDS[n % 10]}"

    def _read_hundreds_block(self, n: int, read_full: bool = False) -> str:
        if n == 0:
            return ""
        if n < 100:
            if read_full:
                return self._read_tens_pair(n)
            return self._read_tens_pair(n)
        
        hundreds = n // 100
        duoi = n % 100
        
        words = []
        if hundreds > 0:
            if hundreds == 1:
                words.append("một trăm")
            else:
                words.append(f"{self._NUM_WORDS[hundreds]} trăm")
        
        if duoi == 0:
            return " ".join(words)
        if duoi < 10:
            words.append("lẻ")
            words.append(self._NUM_WORDS[duoi])
        else:
            words.append(self._read_tens_pair(duoi))
        return " ".join(words)

    _UNITS = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ", "tỷ tỷ"]

    def vi_number_to_words(self, n: int) -> str:
        if n == 0:
            return self._NUM_WORDS[0]
        parts = []
        unit_idx = 0
        while n > 0:
            block = n % 1000
            n //= 1000
            if block:
                read_full = len(parts) > 0
                block_words = self._read_hundreds_block(block, read_full=read_full)
                unit = self._UNITS[unit_idx]
                if unit:
                    parts.append(f"{block_words} {unit}".strip())
                else:
                    parts.append(block_words)
            unit_idx += 1
        return " ".join(reversed([p for p in parts if p]))

    # Match only real currency tokens, not the Vietnamese letter "đ"
    # - words: USD, VND (word boundaries)
    # - symbols: $ and ₫
    _CURRENCY_PATTERN = re.compile(r"(?i)\b(?:usd|vnd)\b|[$₫]")

    def normalize_vi_text(self, text: str) -> str:
        s = text.strip()
        def _currency_repl(m):
            sym = m.group(0)
            low = sym.lower()
            if sym == "$" or low == "usd":
                return " đô"
            if sym == "₫" or low == "vnd":
                return " đồng"
            return sym
        s = self._CURRENCY_PATTERN.sub(_currency_repl, s)
        def _replace_number(m):
            num_str = m.group(0)
            try:
                return self.vi_number_to_words(int(num_str))
            except Exception:
                return num_str
        s = re.sub(r"\d+", _replace_number, s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    def normalize_audio(self, audio_path: str, output_path: str):
        """Normalize audio to 16kHz mono for MMS-TTS-VIE"""
        logger.info(f"Normalizing audio: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Resample to 16kHz for MMS-TTS-VIE
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize volume
        audio = librosa.util.normalize(audio)
        
        # Convert to 16-bit PCM
        audio = (audio * 32767).astype(np.int16)
        
        # Save as WAV
        sf.write(output_path, audio, self.target_sr, subtype='PCM_16')
        
        logger.info(f"Normalized audio saved: {output_path}")
    
    def process_single_file(self, audio_path: str, start_index: int = 1) -> Tuple[List[dict], int]:
        """Process a single audio file"""
        logger.info(f"Processing file: {audio_path}")
        
        try:
            # Split by controlled silence with target lengths 3-8s
            segments = self.split_audio_by_silence(audio_path)
            
            # Process each segment
            metadata_entries = []
            current_index = start_index
            
            for segment_path in segments:
                try:
                    # Transcribe
                    transcript = self.transcribe_audio(segment_path)
                    
                    # Skip if transcript is too short or just silence
                    if len(transcript.strip()) < 10 or transcript == "[SILENCE]":
                        logger.info(f"Skipping short/silent segment at index {current_index}")
                        current_index += 1  # Still increment to keep numbering consistent
                        continue
                    
                    # Normalize audio to 16kHz
                    output_filename = f"{current_index:03d}.wav"
                    output_path = self.wavs_dir / output_filename
                    self.normalize_audio(segment_path, str(output_path))
                    
                    # Add to metadata - format: filename|text|normalized_text
                    metadata_entries.append({
                        'file_name': output_filename,
                        'transcript': transcript,
                        'normalized_text': self.normalize_vi_text(transcript)
                    })
                    
                    current_index += 1
                    
                    # Clean up segment
                    os.remove(segment_path)
                    
                except Exception as e:
                    logger.error(f"Error processing segment {segment_path}: {e}")
                    current_index += 1  # Keep index consistent even on error
                    continue
            
            logger.info(f"Processed {len(metadata_entries)} segments from file")
            return metadata_entries, current_index - start_index
            
        except Exception as e:
            logger.error(f"Error processing file {audio_path}: {e}")
            return [], 0
    
    def create_jsonl_files(self, metadata_entries: List[dict]):
        """Create JSONL files compatible with finetune-hf-vits"""
        logger.info("Creating JSONL files for finetune-hf-vits...")
        
        # Convert to JSONL format
        jsonl_entries = []
        for entry in metadata_entries:
            if not entry.get('transcript') or not entry.get('file_name'):
                logger.warning(f"Skipping empty entry: {entry}")
                continue
            
            # Create absolute path to audio file
            audio_path = str(self.wavs_dir / entry['file_name']).replace('/', '\\')
            
            # JSONL format: {"audio": "path", "text": "transcript"}
            jsonl_entries.append({
                "audio": audio_path,
                "text": entry['transcript'].strip()
            })
        
        # Create train/val/test splits
        if jsonl_entries:
            self._create_dataset_splits(jsonl_entries)
        else:
            logger.warning("No valid entries to write!")
    
    def _create_dataset_splits(self, jsonl_entries: List[dict], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Create deterministic train/val/test splits"""
        import random
        
        # Shuffle with fixed seed for reproducibility
        rnd = random.Random(42)
        shuffled = jsonl_entries.copy()
        rnd.shuffle(shuffled)
        
        n = len(shuffled)
        train_n = max(1 if n >= 3 else 0, int(round(train_ratio * n)))
        val_n = max(1 if n >= 10 else 0, int(round(val_ratio * n)))
        
        # Adjust to total
        if train_n + val_n > n - 1:
            val_n = max(0, n - train_n - 1)
        test_n = max(0, n - train_n - val_n)
        
        train_data = shuffled[:train_n]
        val_data = shuffled[train_n:train_n + val_n]
        test_data = shuffled[train_n + val_n:]
        
        # Write JSONL files
        self._write_jsonl(self.train_jsonl, train_data)
        self._write_jsonl(self.val_jsonl, val_data)
        self._write_jsonl(self.test_jsonl, test_data)
        
        logger.info(f"✅ Created splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    def _write_jsonl(self, file_path: Path, data: List[dict]):
        """Write data to JSONL file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"JSONL saved to: {file_path}")
    
    def _next_index_from_existing(self) -> int:
        """Determine next index by scanning existing wavs (NNN.wav)."""
        if not self.wavs_dir.exists():
            return 1
        max_idx = 0
        for p in self.wavs_dir.glob("*.wav"):
            try:
                stem = Path(p).stem
                idx = int(stem)
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                continue
        return max_idx + 1 if max_idx > 0 else 1

    def process_all_files(self):
        """Process all voice files in input directory"""
        voice_files = list(self.input_dir.glob("*_voice.wav"))
        
        if not voice_files:
            logger.warning(f"No voice files found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(voice_files)} voice files to process")
        
        all_metadata = []
        total_segments = 0
        start_base = self._next_index_from_existing() if self.append else 1
        
        for i, voice_file in enumerate(voice_files, 1):
            try:
                logger.info(f"Progress: {i}/{len(voice_files)}")
                metadata_entries, segments_count = self.process_single_file(
                    str(voice_file), start_index=start_base + total_segments
                )
                all_metadata.extend(metadata_entries)
                total_segments += segments_count
            except Exception as e:
                logger.error(f"Failed to process {voice_file}: {e}")
                continue
        
        # Create final JSONL files
        if all_metadata:
            # Filter out any empty entries before writing
            valid_metadata = [entry for entry in all_metadata if entry.get('transcript') and entry.get('file_name')]
            logger.info(f"Writing {len(valid_metadata)} valid entries out of {len(all_metadata)} total")
            self.create_jsonl_files(valid_metadata)
        else:
            logger.warning("No metadata entries to write!")
        
        logger.info(f"✅ Processing complete! Total segments: {total_segments}")
        logger.info(f"Dataset saved to: {self.output_dir}")
        logger.info(f"Audio files: {self.wavs_dir}")
        logger.info(f"JSONL files: {self.train_jsonl}, {self.val_jsonl}, {self.test_jsonl}")

def main():
    parser = argparse.ArgumentParser(description="Transcription and Dataset Creator for finetune-hf-vits")
    parser.add_argument("--input_file", help="Single audio file to process")
    parser.add_argument("--input_dir", default="data/separated_voice", help="Input directory with voice files")
    parser.add_argument("--output_dir", default="data", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Process all files in input directory")
    
    args = parser.parse_args()
    
    processor = TranscriptionProcessor(args.input_dir, args.output_dir)
    
    if args.input_file:
        # Process single file
        metadata_entries, segments_count = processor.process_single_file(args.input_file)
        if metadata_entries:
            processor.create_jsonl_files(metadata_entries)
    elif args.all:
        # Process all files
        processor.process_all_files()
    else:
        # Interactive mode
        print("Enter audio file path to transcribe:")
        audio_file = input().strip()
        if audio_file:
            metadata_entries, segments_count = processor.process_single_file(audio_file)
            if metadata_entries:
                processor.create_jsonl_files(metadata_entries)
        else:
            print("No file provided!")

if __name__ == "__main__":
    main() 