# HuggingFace MMS-TTS-VIE Finetuning Guide

This guide explains how to finetune the [Facebook MMS-TTS-VIE](https://huggingface.co/facebook/mms-tts-vie) model using the [finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits) repository.

## 🎯 Why This Approach?

- ✅ **Vietnamese pretrained model**: MMS-TTS-VIE is already trained on Vietnamese
- ✅ **Better quality**: Higher quality than finetuning from English VITS
- ✅ **Faster convergence**: Less epochs needed for good results
- ✅ **Industry standard**: Uses HuggingFace ecosystem
- ✅ **Active development**: Well-maintained repository

## 📋 Prerequisites

1. **ffmpeg**: For audio resampling
   ```bash
   conda install ffmpeg
   ```

2. **git**: For cloning repositories
   ```bash
   conda install git
   ```

3. **Dataset**: Your Vietnamese dataset in the expected format

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
python scripts/setup_huggingface_finetune.py
```

This script will:
- ✅ Resample audio to 16kHz mono
- ✅ Convert metadata to JSONL format
- ✅ Clone finetune repository
- ✅ Install requirements
- ✅ Build monotonic alignment
- ✅ Create Vietnamese checkpoint

### Option 2: Manual Setup

#### Step 1: Resample Audio
```bash
python scripts/resample_to_16khz.py data/training_data/dataset/wavs data/wavs_16khz
```

#### Step 2: Create JSONL Files
```bash
python scripts/create_jsonl_for_hf.py data/training_data/dataset/metadata_train.csv data/train.jsonl data/wavs_16khz --text-column 2
python scripts/create_jsonl_for_hf.py data/training_data/dataset/metadata_val.csv data/val.jsonl data/wavs_16khz --text-column 2
```

#### Step 3: Clone Repository
```bash
git clone https://github.com/ylacombe/finetune-hf-vits
cd finetune-hf-vits
pip install -r requirements.txt
```

#### Step 4: Build Monotonic Alignment
```bash
cd monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd ..
```

#### Step 5: Create Vietnamese Checkpoint
```bash
python convert_original_discriminator_checkpoint.py \
  --language_code vie \
  --pytorch_dump_folder_path checkpoints/mms-vie-train
```

## 🎯 Finetuning

### Basic Command
```bash
cd finetune-hf-vits

accelerate launch run_vits_finetuning.py \
  --model_name_or_path checkpoints/mms-vie-train \
  --output_dir runs/mms_vie_ft_single_spk \
  --train_file ../data/train.jsonl \
  --validation_file ../data/val.jsonl \
  --audio_column_name audio \
  --text_column_name text \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 8 \
  --num_train_epochs 50 \
  --save_steps 500 \
  --logging_steps 50 \
  --weight_mel 45.0 \
  --weight_disc 1.0 \
  --weight_gen 1.0 \
  --weight_fmaps 1.0 \
  --weight_kl 1.0 \
  --weight_duration 1.0
```

### Key Parameters

- **`--model_name_or_path`**: Path to Vietnamese checkpoint
- **`--train_file`**: Training data JSONL file
- **`--validation_file`**: Validation data JSONL file
- **`--learning_rate`**: Start with 2e-4, adjust as needed
- **`--per_device_train_batch_size`**: Adjust based on GPU memory
- **`--num_train_epochs`**: Start with 50, increase if needed

### Loss Weights (Important!)

- **`--weight_mel 45.0`**: Mel-spectrogram loss (audio quality)
- **`--weight_disc 1.0`**: Discriminator loss (realism)
- **`--weight_gen 1.0`**: Generator loss (stability)
- **`--weight_fmaps 1.0`**: Feature matching loss (detail)
- **`--weight_kl 1.0`**: KL divergence loss (latent space)
- **`--weight_duration 1.0`**: Duration prediction loss (timing)

## 🧪 Testing

### Test During Training
The model will generate test samples every `--logging_steps` epochs.

### Test After Training
```python
from transformers import pipeline
import scipy

# Load finetuned model
synth = pipeline("text-to-speech", "runs/mms_vie_ft_single_spk")

# Generate speech
text = "Xin chào, đây là kiểm thử sau khi fine-tune!"
output = synth(text)

# Save audio
scipy.io.wavfile.write("test_output.wav", 
                       rate=output["sampling_rate"], 
                       data=output["audio"][0])
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--per_device_train_batch_size`
   - Use gradient accumulation: `--gradient_accumulation_steps 2`

2. **Training Diverges**
   - Reduce learning rate: `--learning_rate 1e-4`
   - Adjust loss weights
   - Check data quality

3. **Poor Audio Quality**
   - Increase `--weight_mel` (try 60.0)
   - Increase `--weight_disc` (try 2.0)
   - Train for more epochs

4. **Monotonic Alignment Build Fails**
   - Ensure you're in the right directory
   - Check Python version compatibility
   - Try: `pip install --upgrade setuptools wheel`

### Performance Tips

- **Start small**: Use 100-500 samples first
- **Monitor loss**: Watch for convergence
- **Save checkpoints**: Use `--save_steps 100` for frequent saves
- **Use mixed precision**: Add `--fp16` if supported

## 📊 Expected Results

- **Epochs needed**: 20-50 for good quality
- **Training time**: 2-8 hours depending on dataset size
- **Audio quality**: Should be better than English VITS finetuning
- **Vietnamese accent**: Should be more natural

## 🔗 Useful Links

- [MMS-TTS-VIE Model](https://huggingface.co/facebook/mms-tts-vie)
- [Finetune Repository](https://github.com/ylacombe/finetune-hf-vits)
- [MMS Paper](https://arxiv.org/abs/2305.13516)
- [VITS Paper](https://arxiv.org/abs/2106.06103)

## 💡 Next Steps

After successful finetuning:

1. **Evaluate quality** on test sentences
2. **Optimize hyperparameters** if needed
3. **Export model** for production use
4. **Share results** with the community

---

**Happy finetuning! 🎉**
