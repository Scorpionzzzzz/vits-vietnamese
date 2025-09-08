## Finetune Vietnamese TTS on Windows (end-to-end)

This project provides a Windows-friendly pipeline to:
- Download audio from YouTube → (optional) separate music/voice → transcribe (Whisper) → normalize → segment (3–8s) → resample 16k → JSONL → push dataset to Hugging Face Hub
- Finetune MMS-TTS-VIE with `accelerate`
- Run inference and debug common issues

All commands below use Windows PowerShell.

---

### 0) Prerequisites

```powershell
conda create -n tts python=3.10 -y
conda activate tts
pip install -r requirements.txt
# Extra tools
pip install yt-dlp demucs faster-whisper
conda install -c conda-forge ffmpeg -y
```

---

### 1) Download audio from YouTube

```powershell
mkdir D:\Workspace\NLP\TEXT-TO-SPEECH\data\raw
cd D:\Workspace\NLP\TEXT-TO-SPEECH\data\raw
yt-dlp -x --audio-format wav --audio-quality 0 -o "%(id)s.%(ext)s" <YOUTUBE_URL>
```
Optional (if heavy background music): separate vocals
```powershell
python -m demucs --two-stems=vocals "D:\Workspace\NLP\TEXT-TO-SPEECH\data\raw\YOUR.wav"
```
Use `vocals.wav` if created.

---

### 2) Transcribe with Faster-Whisper (Vietnamese)

```powershell
python - << 'PY'
from faster_whisper import WhisperModel
import os

inp = r"D:\Workspace\NLP\TEXT-TO-SPEECH\data\raw\YOUR.wav"  # or vocals.wav
out_dir = r"D:\Workspace\NLP\TEXT-TO-SPEECH\data\transcript"
os.makedirs(out_dir, exist_ok=True)

model = WhisperModel("medium", device="cuda", compute_type="float16")
segments, _ = model.transcribe(inp, language="vi", beam_size=5)

with open(os.path.join(out_dir, "transcript.tsv"), "w", encoding="utf-8") as f:
    f.write("start\tend\ttext\n")
    for seg in segments:
        f.write(f"{seg.start:.2f}\t{seg.end:.2f}\t{seg.text.strip()}\n")
print("Wrote transcript.tsv")
PY
```

---

### 3) Segment into 3–8s WAVs + metadata.csv

```powershell
python - << 'PY'
import os, csv, subprocess
from pathlib import Path

root = Path(r"D:\Workspace\NLP\TEXT-TO-SPEECH\data")
wav_in = root/"raw"/"YOUR.wav"  # or vocals.wav
transcript = root/"transcript"/"transcript.tsv"
wavs_dir = root/"dataset"/"wavs"
os.makedirs(wavs_dir, exist_ok=True)

rows = []
with open(transcript, encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f, 1):
        s, e, text = line.rstrip("\n").split("\t", 2)
        s, e = float(s), float(e)
        if not (3 <= e - s <= 8):
            continue
        out = wavs_dir/f"{i:05d}.wav"
        subprocess.run([
            "ffmpeg","-y","-loglevel","error","-i",str(wav_in),
            "-ss",str(s),"-to",str(e),"-ac","1","-ar","22050",str(out)
        ], check=True)
        rows.append((out.name, text.strip()))

meta = root/"dataset"/"metadata.csv"
with open(meta, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="|")
    for name, text in rows:
        w.writerow([name, text, text])
print("Wrote", meta, "n_samples=", len(rows))
PY
```

---

### 4) Resample to 16 kHz and build JSONL

```powershell
python scripts\resample_to_16khz.py D:\Workspace\NLP\TEXT-TO-SPEECH\data\dataset\wavs D:\Workspace\NLP\TEXT-TO-SPEECH\data\wavs_16khz
```
Split metadata into `metadata_train.csv` / `metadata_val.csv` (e.g., 90/10), then:
```powershell
python scripts\create_jsonl_for_hf.py D:\Workspace\NLP\TEXT-TO-SPEECH\data\dataset\metadata_train.csv D:\Workspace\NLP\TEXT-TO-SPEECH\data\train.jsonl D:\Workspace\NLP\TEXT-TO-SPEECH\data\wavs_16khz -t 2
python scripts\create_jsonl_for_hf.py D:\Workspace\NLP\TEXT-TO-SPEECH\data\dataset\metadata_val.csv   D:\Workspace\NLP\TEXT-TO-SPEECH\data\val.jsonl   D:\Workspace\NLP\TEXT-TO-SPEECH\data\wavs_16khz -t 2
```

---

### 5) Push dataset to Hugging Face Hub (Audio@16k)

```powershell
conda activate tts
huggingface-cli login
python scripts\push_vi_dataset.py --train "D:\Workspace\NLP\TEXT-TO-SPEECH\data\train.jsonl" --val "D:\Workspace\NLP\TEXT-TO-SPEECH\data\val.jsonl" --repo "<YOUR_USERNAME>/vi-tts-vie" --private
```
Verify: Hub dataset has splits `train`/`test` and columns `audio` (Audio 16k) and `text`.

---

### 6) Prepare MMS training checkpoint (once)

```powershell
conda activate tts
cd D:\Workspace\NLP\TEXT-TO-SPEECH\finetune-hf-vits\monotonic_align
if (-not (Test-Path .\monotonic_align)) { mkdir monotonic_align }
python setup.py build_ext --inplace
cd ..
python convert_original_discriminator_checkpoint.py --language_code vie --pytorch_dump_folder_path checkpoints\mms-vie-train
```

---

### 7) Finetune

Option A (config file): edit `finetune-hf-vits\configs\finetune_vi_mms.json`
- `dataset_name`: `<YOUR_USERNAME>/vi-tts-vie`
- `model_name_or_path`: `checkpoints/mms-vie-train`
- you can set `"fp16": true`

Run:
```powershell
conda activate tts
cd D:\Workspace\NLP\TEXT-TO-SPEECH\finetune-hf-vits
accelerate launch run_vits_finetuning.py .\configs\finetune_vi_mms.json
```

Option B (single command):
```powershell
conda activate tts
cd D:\Workspace\NLP\TEXT-TO-SPEECH\finetune-hf-vits
accelerate launch run_vits_finetuning.py `
  --model_name_or_path checkpoints/mms-vie-train `
  --dataset_name <YOUR_USERNAME>/vi-tts-vie `
  --audio_column_name audio `
  --text_column_name text `
  --train_split_name train `
  --eval_split_name test `
  --do_train --do_eval `
  --per_device_train_batch_size 8 `
  --per_device_eval_batch_size 8 `
  --learning_rate 2e-4 `
  --fp16 `
  --output_dir runs/mms_vie_ft_single_spk `
  --overwrite_output_dir
```

Outputs are saved to: `finetune-hf-vits\runs\mms_vie_ft_single_spk`.

---

### 8) Inference

```python
from transformers import pipeline
import scipy.io.wavfile as wav

model_dir = r"D:\Workspace\NLP\TEXT-TO-SPEECH\finetune-hf-vits\runs\mms_vie_ft_single_spk"
tts = pipeline("text-to-speech", model=model_dir, device=0)
out = tts("Xin chào, đây là mô hình đã finetune.")
wav.write(r"D:\Workspace\NLP\TEXT-TO-SPEECH\generated_audio\sample.wav", out["sampling_rate"], out["audio"][0])
```

---

### 9) Troubleshooting

- Output dir exists → add `--overwrite_output_dir` or change `--output_dir`
- JSON BOM error → save config as UTF-8 (no BOM)
- `num_proc must be > 0` → set `"preprocessing_num_workers": 1`
- W&B prompt → choose 3, or `--report_to tensorboard`, or `$env:WANDB_DISABLED="true"`
- Slow preprocessing → increase `preprocessing_num_workers` to 2 if stable
- VRAM issues → reduce batch size or enable `fp16`
