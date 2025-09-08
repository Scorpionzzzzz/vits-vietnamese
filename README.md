
# Vietnamese TTS Finetuning Pipeline

This repository provides a complete pipeline for finetuning Vietnamese Text-to-Speech (TTS) models using Hugging Face Transformers and MMS-TTS-VIE. The workflow is cross-platform (Windows/Linux/Mac) and includes data preparation, training, inference, and troubleshooting.

## Features
- Download and preprocess audio data
- Transcribe audio to text (Vietnamese) using Whisper
- Segment, normalize, and resample audio
- Build JSONL datasets for Hugging Face
- Finetune MMS-TTS-VIE models
- Run inference and evaluate results

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n tts python=3.10 -y
conda activate tts
pip install -r requirements.txt
# Optional tools
pip install yt-dlp demucs faster-whisper
conda install -c conda-forge ffmpeg -y
```

### 2. Download Audio

```bash
yt-dlp -x --audio-format wav --audio-quality 0 -o "%(id)s.%(ext)s" <YOUTUBE_URL>
```
Optionally, separate vocals:
```bash
python -m demucs --two-stems=vocals <input.wav>
```

### 3. Transcribe Audio (Vietnamese)

```python
from faster_whisper import WhisperModel
model = WhisperModel("medium", device="cuda", compute_type="float16")
segments, _ = model.transcribe("input.wav", language="vi", beam_size=5)
for seg in segments:
    print(seg.text)
```

### 4. Segment & Normalize Audio

Use provided scripts to segment audio into 3–8s chunks, normalize, and resample to 16kHz. See `create_dataset_scripts/` for details.

### 5. Build JSONL Dataset

Use scripts to convert metadata to JSONL format for Hugging Face training. Example:
```bash
python scripts/create_jsonl_for_hf.py <metadata.csv> <output.jsonl> <wavs_dir> -t 2
```

### 6. Push Dataset to Hugging Face Hub

```bash
huggingface-cli login
python scripts/push_vi_dataset.py --train <train.jsonl> --val <val.jsonl> --repo <your-username>/vi-tts-vie --private
```

### 7. Prepare MMS Checkpoint

```bash
cd finetune-hf-vits/monotonic_align
python setup.py build_ext --inplace
cd ..
python convert_original_discriminator_checkpoint.py --language_code vie --pytorch_dump_folder_path checkpoints/mms-vie-train
```

### 8. Finetune Model

Edit `finetune-hf-vits/configs/finetune_vi_mms.json` as needed. Example command:
```bash
accelerate launch run_vits_finetuning.py configs/finetune_vi_mms.json
```
Or use CLI arguments (see script for details).

### 9. Inference

```python
from transformers import pipeline
tts = pipeline("text-to-speech", model="finetune-hf-vits/runs/mms_vie_ft_single_spk", device=0)
out = tts("Xin chào, đây là mô hình đã finetune.")
import scipy.io.wavfile as wav
wav.write("generated_audio/sample.wav", out["sampling_rate"], out["audio"][0])
```

---

## Troubleshooting

- Output directory exists: add `--overwrite_output_dir` or change `--output_dir`
- JSON BOM error: save config as UTF-8 (no BOM)
- `num_proc must be > 0`: set `preprocessing_num_workers=1`
- W&B prompt: use `--report_to tensorboard` or set `WANDB_DISABLED=true`
- Slow preprocessing: increase `preprocessing_num_workers` if stable
- VRAM issues: reduce batch size or enable `fp16`

---

## Project Structure

- `create_dataset_scripts/`: Data preparation scripts
- `finetune-hf-vits/`: Model code, configs, checkpoints
- `data/`: Audio and dataset files
- `generated_audio/`: Inference outputs
- `output/`: Evaluation results

---

## License

See LICENSE file for details.
