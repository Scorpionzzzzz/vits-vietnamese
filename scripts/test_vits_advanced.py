#!/usr/bin/env python3
"""
VITS testing script with configurable parameters
Loads model directly to allow parameter tuning
"""

import torch
import scipy.io.wavfile as wav
import os
import numpy as np
from pathlib import Path
from transformers import VitsModel, AutoTokenizer
import random

def test_vits():
    """Test VITS model with configurable parameters"""
    
    # Đường dẫn đến mô hình đã train
    model_dir = r"D:\Workspace\NLP\TEXT-TO-SPEECH\finetune-hf-vits\runs\mms_vie_ft_single_spk"
    
    # Kiểm tra mô hình có tồn tại không
    if not os.path.exists(model_dir):
        print(f"❌ Không tìm thấy mô hình ở: {model_dir}")
        return
    
    print(f"✅ Tìm thấy mô hình ở: {model_dir}")
    
    try:
        # Load model và tokenizer trực tiếp
        print("🔄 Đang load model và tokenizer...")
        model = VitsModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Chuyển model lên GPU nếu có
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        # Thiết lập seed để tái thiết lập kết quả
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
        torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        
        print(f"🔢 Seed đã được thiết lập: {seed}")
        print(f"   🎲 PyTorch seed: {seed}")
        print(f"   🎲 NumPy seed: {seed}")
        print(f"   🔒 Deterministic mode: Bật")
        
        print("✅ Đã load model thành công!")
        print(f"📊 Model: {model_dir}")
        print(f"🔧 Device: {device}")
        print(f"📝 Vocab size: {model.config.vocab_size}")
        
        # Hiển thị config hiện tại
        print(f"\n📋 Config hiện tại:")
        print(f"   🎵 Noise scale: {model.config.noise_scale}")
        print(f"   🎵 Noise scale duration: {model.config.noise_scale_duration}")
        print(f"   🎵 Speaking rate: {model.config.speaking_rate}")
        print(f"   🎵 Sampling rate: {model.config.sampling_rate}")
        
        # Cập nhật config tùy chỉnh
        print(f"\n🔧 Cập nhật config...")
        # model.config.noise_scale = 0.5          # Giảm noise (ổn định hơn)
        # model.config.noise_scale_duration = 1.2 # Giảm variation trong duration
        # model.config.speaking_rate = 0.7       # Chậm hơn một chút
        model.config.noise_scale = 0.5          # Giảm noise (ổn định hơn)
        model.config.noise_scale_duration = 0.5 # Giảm variation trong duration
        model.config.speaking_rate = 0.9       # Chậm hơn một chút
        
        print(f"   🎵 Noise scale: {model.config.noise_scale}")
        print(f"   🎵 Noise scale duration: {model.config.noise_scale_duration}")
        print(f"   🎵 Speaking rate: {model.config.speaking_rate}")
        
        # Tạo thư mục output
        output_dir = Path(r"D:\Workspace\NLP\TEXT-TO-SPEECH\generated_audio")
        output_dir.mkdir(exist_ok=True)
        
        # Cấu hình padding
        padding_start = "  "  # Ký tự padding ở đầu
        padding_end = " . "    # Ký tự padding ở cuối
        
        # Hiển thị cấu hình padding
        print(f"\n🔧 Cấu hình padding:")
        print(f"   📝 Padding start: '{padding_start}'")
        print(f"   📝 Padding end: '{padding_end}'")
        
        # Danh sách câu test
        test_texts = [
            " Vậy là không được đâu nhé  "
        ]
        
        print(f"\n🎯 Sẽ test {len(test_texts)} câu với cấu hình tùy chỉnh")
        
        # Test từng câu
        for text_idx, text in enumerate(test_texts, 1):
            print(f"\n🎵 Đang tạo âm thanh cho câu {text_idx}: {text[:50]}...")
            
            try:
                # Thêm ký tự im lặng/ngắt câu để tránh mất chữ ở đầu/cuối
                padded_text = padding_start + text + padding_end
                print(f"   📝 Text gốc: {text}")
                print(f"   📝 Text đã pad: {padded_text}")
                
                # Tokenize text đã pad
                inputs = tokenizer(padded_text, return_tensors="pt").to(device)
                
                # Generate speech
                with torch.no_grad():
                    output = model(**inputs)
                
                # Lấy waveform
                waveform = output.waveform.squeeze().cpu().numpy()
                sampling_rate = model.config.sampling_rate
                
                # Lưu file
                output_file = output_dir / f"test_custom_{text_idx}.wav"
                wav.write(str(output_file), sampling_rate, waveform)
                
                # Thông tin về audio
                audio_duration = len(waveform) / sampling_rate
                print(f"✅ Đã lưu: {output_file.name}")
                print(f"   📊 Sampling rate: {sampling_rate} Hz")
                print(f"   ⏱️  Duration: {audio_duration:.2f} seconds")
                print(f"   📏 Audio length: {len(waveform)} samples")
                
            except Exception as e:
                print(f"❌ Lỗi khi tạo audio cho câu {text_idx}: {e}")
                continue
        
        print(f"\n🎉 Hoàn thành! Đã tạo {len(test_texts)} file âm thanh trong: {output_dir}")
        print(f"\n💡 Các file được tạo:")
        for text_idx in range(1, len(test_texts) + 1):
            filename = f"test_custom_{text_idx}.wav"
            print(f"   📁 {filename}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vits()
