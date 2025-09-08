#!/usr/bin/env python3
import argparse
from datasets import load_dataset, DatasetDict, Audio


def main():
    parser = argparse.ArgumentParser(description="Push local VI TTS JSONL dataset to HuggingFace Hub with proper Audio column")
    parser.add_argument("--train", required=True, help="Path to train.jsonl")
    parser.add_argument("--val", required=True, help="Path to val.jsonl")
    parser.add_argument("--repo", required=True, help="Target hub repo, e.g. your-username/vi-tts-vie")
    parser.add_argument("--private", action="store_true", help="Create a private repo")
    args = parser.parse_args()

    dataset = DatasetDict({
        "train": load_dataset("json", data_files=args.train, split="train"),
        "test": load_dataset("json", data_files=args.val, split="train"),
    })

    # Ensure audio column is cast to Audio with 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    dataset.push_to_hub(args.repo, private=args.private)


if __name__ == "__main__":
    main()


