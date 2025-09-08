@echo off
REM Batch script to run VITS finetuning
REM This script handles Windows command line formatting properly

echo 🚀 Starting VITS finetuning...

REM Change to finetune directory
cd finetune-hf-vits

REM Run the finetuning command
accelerate launch run_vits_finetuning.py ^
    --model_name_or_path "checkpoints/mms-vie-train" ^
    --output_dir "runs/mms_vie_ft_single_spk" ^
    --train_file "../data/train.jsonl" ^
    --validation_file "../data/val.jsonl" ^
    --audio_column_name "audio" ^
    --text_column_name "text" ^
    --learning_rate 2e-4 ^
    --per_device_train_batch_size 8 ^
    --num_train_epochs 50 ^
    --save_steps 500 ^
    --logging_steps 50 ^
    --weight_mel 45.0 ^
    --weight_disc 1.0 ^
    --weight_gen 1.0 ^
    --weight_fmaps 1.0 ^
    --weight_kl 1.0 ^
    --weight_duration 1.0

echo ✅ Finetuning completed!
pause
