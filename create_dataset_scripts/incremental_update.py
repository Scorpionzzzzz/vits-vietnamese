#!/usr/bin/env python3
"""
Incremental dataset updater

Workflow:
1) So sánh 2 thư mục: downloaded_audio (raw wav) và separated_voice (voice-only)
2) Với mỗi file .wav mới trong downloaded_audio chưa có file *_voice.wav tương ứng → tách giọng (Demucs)
3) Append các đoạn mới vào dataset: chuẩn hóa, phiên âm Whisper, ghi nối thêm vào metadata.csv
4) Tạo lại metadata_train.csv, metadata_val.csv, metadata_test.csv (deterministic split)

Mặc định đường dẫn:
- downloaded_audio: data/downloaded_audio
- separated_voice: data/separated_voice
- dataset_out: data/training_data/dataset
"""

import argparse
from pathlib import Path
import logging
from typing import List

# Import pipeline modules
from separate_voice import VoiceSeparator
from transcribe_dataset import TranscriptionProcessor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("incremental_update")


def find_new_downloads(download_dir: Path, separated_dir: Path) -> List[Path]:
    """Tìm các file .wav mới ở download_dir chưa có *_voice.wav tương ứng ở separated_dir."""
    new_files = []
    for wav in sorted(download_dir.glob("*.wav")):
        stem = wav.stem
        voice_file = separated_dir / f"{stem}_voice.wav"
        if not voice_file.exists():
            new_files.append(wav)
    return new_files


def separate_new(download_dir: Path, separated_dir: Path) -> List[Path]:
    """Tách giọng cho các file mới, trả về danh sách *_voice.wav được tạo ra."""
    separated_dir.mkdir(parents=True, exist_ok=True)
    new_wavs = find_new_downloads(download_dir, separated_dir)
    if not new_wavs:
        logger.info("Không có file mới để tách giọng.")
        return []

    logger.info(f"Có {len(new_wavs)} file mới sẽ tách giọng...")
    sep = VoiceSeparator(str(download_dir), str(separated_dir))
    created = []
    for i, src in enumerate(new_wavs, 1):
        try:
            logger.info(f"[{i}/{len(new_wavs)}] Tách giọng: {src}")
            out = sep.separate_single_file(str(src))
            if out:
                created.append(Path(out))
        except Exception as e:
            logger.error(f"Lỗi khi tách giọng {src}: {e}")
            continue
    logger.info(f"Tách giọng xong: {len(created)}/{len(new_wavs)}")
    return created


def append_transcribe_and_split(separated_dir: Path, dataset_out: Path, only_files: List[Path]):
    """Append các file *_voice.wav mới vào metadata và split lại."""
    proc = TranscriptionProcessor(str(separated_dir), str(dataset_out), append=True)

    total_segments = 0
    for i, vf in enumerate(only_files, 1):
        try:
            start_idx = proc._next_index_from_existing()  # tiếp tục numbering
            logger.info(f"[{i}/{len(only_files)}] Transcribe & append: {vf} (start={start_idx})")
            entries, seg_count = proc.process_single_file(str(vf), start_index=start_idx)
            if entries:
                proc.create_metadata_csv(entries)  # append mode
            total_segments += seg_count
        except Exception as e:
            logger.error(f"Lỗi khi xử lý {vf}: {e}")
            continue

    if total_segments == 0:
        logger.info("Không có phân đoạn hợp lệ để thêm vào metadata.")
    else:
        logger.info(f"Đã thêm {total_segments} phân đoạn vào metadata.csv")

    # Tạo lại split từ metadata.csv mới nhất
    proc.create_dataset_splits(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)


def main():
    ap = argparse.ArgumentParser(description="Incremental dataset updater")
    ap.add_argument("--download_dir", default="data/downloaded_audio", help="Thư mục audio đã tải về")
    ap.add_argument("--separated_dir", default="data/separated_voice", help="Thư mục chứa *_voice.wav")
    ap.add_argument("--dataset_out", default="data/training_data/dataset", help="Thư mục dataset đầu ra")
    args = ap.parse_args()

    download_dir = Path(args.download_dir)
    separated_dir = Path(args.separated_dir)
    dataset_out = Path(args.dataset_out)

    if not download_dir.exists():
        logger.error(f"Không tồn tại: {download_dir}")
        return

    new_separated = separate_new(download_dir, separated_dir)
    if not new_separated:
        logger.info("Không có gì để cập nhật vào dataset (metadata giữ nguyên).")
        return

    append_transcribe_and_split(separated_dir, dataset_out, new_separated)
    logger.info("✅ Hoàn tất cập nhật dataset (append + re-split)")


if __name__ == "__main__":
    main()


