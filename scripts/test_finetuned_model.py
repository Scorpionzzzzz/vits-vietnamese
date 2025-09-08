#!/usr/bin/env python3
"""
Simple script to test the finetuned MMS-TTS-VIE model
"""

from transformers import pipeline
import scipy.io.wavfile as wav
import os
from pathlib import Path

def test_model():
    # Đường dẫn đến mô hình đã train
    model_dir = r"D:\Workspace\NLP\TEXT-TO-SPEECH\finetune-hf-vits\runs\mms_vie_ft_single_spk" #\Model_Train_P1
    
    # Kiểm tra mô hình có tồn tại không
    if not os.path.exists(model_dir):
        print(f"❌ Không tìm thấy mô hình ở: {model_dir}")
        return
    
    print(f"✅ Tìm thấy mô hình ở: {model_dir}")
    
    try:
        # Load mô hình
        print("🔄 Đang load mô hình...")
        tts = pipeline("text-to-speech", model=model_dir, device=0)
        print("✅ Đã load mô hình thành công!")
        
        # Thêm thông tin về model
        print(f"📊 Model: {model_dir}")
        print(f"🔧 Device: {tts.device}")
        
        # Tạo thư mục output nếu chưa có
        output_dir = Path(r"D:\Workspace\NLP\TEXT-TO-SPEECH\generated_audio")
        output_dir.mkdir(exist_ok=True)
        
        # Danh sách câu test
        test_texts = [
            "Nhưng mà thường thường những cái câu chuyện của người lớn nói Sẽ không phải là những cái chủ đề mà mình quan tâm Mà cái lúc đấy thì mình chỉ ngồi thẫn thờ thôi Và tự dưng có một cái suy nghĩ len lỏi trong đầu mình"
        ]
        
        # Tạo âm thanh cho từng câu
        for i, text in enumerate(test_texts, 1):
            print(f"\n🎵 Đang tạo âm thanh cho câu {i}: {text}")
            
            # Tạo âm thanh - chỉ dùng tham số cơ bản
            output = tts(text)
            
            # Lưu file
            output_file = output_dir / f"test_output_{i}.wav"
            wav.write(str(output_file), output["sampling_rate"], output["audio"][0])
            
            # Thông tin về audio
            audio_duration = len(output["audio"][0]) / output["sampling_rate"]
            print(f"✅ Đã lưu: {output_file}")
            print(f"   📊 Sampling rate: {output['sampling_rate']} Hz")
            print(f"   ⏱️  Duration: {audio_duration:.2f} seconds")
        
        print(f"\n🎉 Hoàn thành! Đã tạo {len(test_texts)} file âm thanh trong: {output_dir}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    test_model()
