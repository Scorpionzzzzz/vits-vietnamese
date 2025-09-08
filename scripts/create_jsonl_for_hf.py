#!/usr/bin/env python3
"""
Convert metadata CSV to JSONL format for HuggingFace Datasets.
Required format: {"audio": "path/to/audio.wav", "text": "text content"}
"""

import csv
import json
import argparse
import os
from pathlib import Path

def csv_to_jsonl(csv_file: str, jsonl_file: str, wav_dir: str, text_column: int = 2):
    """
    Convert CSV to JSONL format.
    
    Args:
        csv_file: Input CSV file (filename|text|normalized_text)
        jsonl_file: Output JSONL file
        wav_dir: Directory containing WAV files
        text_column: Which column to use for text (1=text, 2=normalized_text)
    """
    
    csv_path = Path(csv_file)
    jsonl_path = Path(jsonl_file)
    wav_path = Path(wav_dir)
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    if not wav_path.exists():
        print(f"❌ WAV directory not found: {wav_path}")
        return False
    
    # Create output directory
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Converting {csv_path} to {jsonl_path}")
    print(f"📁 WAV directory: {wav_path}")
    print(f"📝 Using text column: {text_column} (0-indexed)")
    
    converted_count = 0
    error_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as csv_f, \
         open(jsonl_path, 'w', encoding='utf-8') as jsonl_f:
        
        reader = csv.reader(csv_f, delimiter='|')
        
        for row_num, row in enumerate(reader, 1):
            if len(row) < 3:
                print(f"⚠️  Row {row_num}: Insufficient columns, skipping")
                continue
            
            filename = row[0].strip()
            text = row[text_column].strip()  # Use specified column
            
            if not filename or not text:
                print(f"⚠️  Row {row_num}: Empty filename or text, skipping")
                continue
            
            # Check if WAV file exists
            wav_file = wav_path / filename
            if not wav_file.exists():
                print(f"⚠️  Row {row_num}: WAV file not found: {wav_file}")
                error_count += 1
                continue
            
            # Create JSON object
            json_obj = {
                "audio": str(wav_file),
                "text": text
            }
            
            # Write to JSONL
            jsonl_f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
            converted_count += 1
    
    print(f"\n✅ Conversion completed!")
    print(f"   Converted: {converted_count}")
    print(f"   Errors: {error_count}")
    print(f"   Output: {jsonl_path}")
    
    return converted_count > 0

def main():
    parser = argparse.ArgumentParser(description="Convert CSV metadata to JSONL for HuggingFace")
    parser.add_argument("csv_file", help="Input CSV file (metadata_train.csv or metadata_val.csv)")
    parser.add_argument("jsonl_file", help="Output JSONL file")
    parser.add_argument("wav_dir", help="Directory containing WAV files")
    parser.add_argument("--text-column", "-t", type=int, default=2, 
                       help="Text column index (0=filename, 1=text, 2=normalized_text, default=2)")
    
    args = parser.parse_args()
    
    if csv_to_jsonl(args.csv_file, args.jsonl_file, args.wav_dir, args.text_column):
        print(f"\n💡 JSONL file created successfully!")
        print(f"   You can now use this for HuggingFace finetuning")
    else:
        print("❌ Conversion failed")

if __name__ == "__main__":
    main()
