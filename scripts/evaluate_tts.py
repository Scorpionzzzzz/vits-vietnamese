import os
import argparse
import soundfile as sf
import numpy as np
from tqdm import tqdm
from transformers import VitsModel, VitsTokenizer
from pesq import pesq
from jiwer import wer
import torch
import json
from faster_whisper import WhisperModel

# Đường dẫn tới mô hình và tokenizer
MODEL_PATH = "finetune-hf-vits/runs/mms_vie_ft_single_spk"
TEST_JSONL = "data/test.jsonl"
WAVS_DIR = "data/wavs_16khz"


# Hàm sinh audio từ batch văn bản
def synthesize_text_batch(model, tokenizer, texts, device):
    # Tham số sinh audio (tham khảo vits_gui_simple.py)
    # Nếu muốn dùng các giá trị động, có thể truyền vào qua biến params
    noise_scale = getattr(model.config, 'noise_scale', 0.667)
    noise_scale_duration = getattr(model.config, 'noise_scale_duration', 0.8)
    speaking_rate = getattr(model.config, 'speaking_rate', 1.0)
    padding_start = getattr(model.config, 'padding_start', '  ')
    padding_end = getattr(model.config, 'padding_end', ' . ')

    # Thêm padding cho từng text
    padded_texts = [padding_start + t + padding_end for t in texts]
    inputs = tokenizer(padded_texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(
            inputs["input_ids"],
            noise_scale=noise_scale,
            noise_scale_duration=noise_scale_duration,
            length_scale=speaking_rate
        )
    # output.waveform shape: (batch, time)
    # sampling_rate lấy từ model.config
    sampling_rate = getattr(model.config, 'sampling_rate', 16000)
    return output.waveform.cpu().numpy(), sampling_rate

# Đánh giá chất lượng bằng PESQ và WER

def evaluate():
    import torch
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VitsModel.from_pretrained(MODEL_PATH)
    tokenizer = VitsTokenizer.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()

    pesq_scores = []
    wers = []
    batch_size = 8
    items = []
    results = []
    transcripts = []
    output_dir = "output"
    audio_out_dir = os.path.join(output_dir, "audio")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audio_out_dir, exist_ok=True)
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                items.append(item)
    num_batches = (len(items) + batch_size - 1) // batch_size
    pbar = tqdm(total=len(items), desc="Đánh giá từng mẫu", unit="mẫu")
    # Khởi tạo Whisper
    whisper_model = WhisperModel("large-v3", device=device.type, compute_type="float16" if device.type == "cuda" else "int8")
    for i in range(num_batches):
        batch_items = items[i*batch_size:(i+1)*batch_size]
        texts = [it["text"] for it in batch_items]
        audio_files = [os.path.join(WAVS_DIR, os.path.basename(it["audio"])) for it in batch_items]
        ref_audios = []
        srs = []
        for audio_file in audio_files:
            ref_audio, sr = sf.read(audio_file)
            ref_audios.append(ref_audio)
            srs.append(sr)
        gen_audios, gen_sr = synthesize_text_batch(model, tokenizer, texts, device)
        for j in range(len(batch_items)):
            ref_audio = ref_audios[j]
            gen_audio = gen_audios[j]
            sr = srs[j]
            # Chuẩn hóa sampling rate nếu cần
            if gen_sr != sr:
                import librosa
                gen_audio = librosa.resample(gen_audio, orig_sr=gen_sr, target_sr=sr)
            min_len = min(len(ref_audio), len(gen_audio))
            ref_audio = ref_audio[:min_len]
            gen_audio = gen_audio[:min_len]
            pesq_score = pesq(sr, ref_audio, gen_audio, "wb")
            pesq_scores.append(pesq_score)
            audio_name = os.path.basename(audio_files[j])
            base, ext = os.path.splitext(audio_name)
            gen_audio_name = f"{base}_gen{ext}"
            gen_audio_path = os.path.join(audio_out_dir, gen_audio_name)
            sf.write(gen_audio_path, gen_audio, sr)
            # Nhận diện text từ audio sinh ra bằng Whisper
            segments, _ = whisper_model.transcribe(gen_audio_path, language="vi")
            transcript = " ".join([seg.text.strip() for seg in segments])
            transcripts.append({
                "audio": gen_audio_name,
                "transcript": transcript
            })
            wer_score = wer(texts[j], transcript)
            wers.append(wer_score)
            results.append({
                "text": texts[j],
                "audio_file": audio_files[j],
                "gen_audio_file": gen_audio_path,
                "pesq": pesq_score,
                "wer": wer_score,
                "transcript": transcript
            })
            pbar.update(1)
    pbar.close()
    print(f"Trung bình PESQ: {np.mean(pesq_scores):.3f}")
    print(f"Trung bình WER: {np.mean(wers):.3f}")
    # Lưu kết quả chi tiết
    with open(os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # Lưu transcript nhận diện được
    with open(os.path.join(output_dir, "gen_transcripts.json"), "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    evaluate()
