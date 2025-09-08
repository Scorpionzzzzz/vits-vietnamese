#!/usr/bin/env python3
"""
Setup script for HuggingFace MMS-TTS-VIE finetuning.
This script automates the entire data preparation process.
"""

import os
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   Stderr: {e.stderr.strip()}")
        return False

def check_requirements():
    """Check if required tools are installed."""
    print("🔍 Checking requirements...")
    
    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ ffmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg not found")
        print("   Install with: conda install ffmpeg")
        return False
    
    # Check git
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        print("✅ git found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ git not found")
        print("   Install with: conda install git")
        return False
    
    return True

def setup_huggingface_finetune():
    """Main setup function."""
    print("🚀 Setting up HuggingFace MMS-TTS-VIE finetuning")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        return False
    
    # Get current directory and project root
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Determine project root based on current location
    if current_dir.name == "scripts":
        project_root = current_dir.parent
        print(f"📁 Running from scripts directory, project root: {project_root}")
    elif current_dir.name == "TEXT-TO-SPEECH":
        project_root = current_dir
        print(f"📁 Running from project root: {project_root}")
    else:
        # Try to find project root by looking for key directories
        potential_roots = [
            current_dir,
            current_dir.parent,
            current_dir.parent.parent,
            Path("D:/Workspace/NLP/TEXT-TO-SPEECH")  # Absolute path as fallback
        ]
        
        for root in potential_roots:
            if (root / "data").exists() and (root / "scripts").exists():
                project_root = root
                print(f"📁 Found project root: {project_root}")
                break
        else:
            print("❌ Could not determine project root")
            print("   Please run this script from the project directory or scripts subdirectory")
            return False
    
    # Define paths based on actual structure
    data_dir = project_root / "data"
    training_data_dir = data_dir / "training_data"
    dataset_dir = training_data_dir / "dataset"
    wavs_dir = dataset_dir / "wavs"
    wavs_16khz_dir = data_dir / "wavs_16khz"
    
    print(f"\n📁 Project root: {project_root}")
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Training data: {training_data_dir}")
    print(f"📁 Dataset: {dataset_dir}")
    print(f"📁 WAVs: {wavs_dir}")
    print(f"📁 16kHz output: {wavs_16khz_dir}")
    
    # Check if dataset exists
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print("   Please run your data preparation pipeline first")
        print("   Expected structure: data/training_data/dataset/")
        return False
    
    if not wavs_dir.exists():
        print(f"❌ WAVs directory not found: {wavs_dir}")
        print("   Please run your data preparation pipeline first")
        print("   Expected structure: data/training_data/dataset/wavs/")
        return False
    
    # Step 1: Resample audio to 16kHz
    print(f"\n📊 Step 1: Resampling audio to 16kHz mono")
    
    # Use absolute paths for the resample command
    resample_cmd = [
        "python", str(project_root / "scripts" / "resample_to_16khz.py"),
        str(wavs_dir), str(wavs_16khz_dir)
    ]
    
    if not run_command(resample_cmd, "Resampling audio files"):
        return False
    
    # Step 2: Create JSONL files
    print(f"\n📝 Step 2: Creating JSONL files for HuggingFace")
    
    # Check if metadata files exist
    train_csv = dataset_dir / "metadata_train.csv"
    val_csv = dataset_dir / "metadata_val.csv"
    
    if not train_csv.exists():
        print(f"❌ Training metadata not found: {train_csv}")
        print("   Please run your data preparation pipeline first")
        print("   Expected file: data/training_data/dataset/metadata_train.csv")
        return False
    
    if not val_csv.exists():
        print(f"❌ Validation metadata not found: {val_csv}")
        print("   Please run your data preparation pipeline first")
        print("   Expected file: data/training_data/dataset/metadata_val.csv")
        return False
    
    # Create JSONL files
    train_jsonl = data_dir / "train.jsonl"
    val_jsonl = data_dir / "val.jsonl"
    
    # Convert training data
    train_cmd = [
        "python", str(project_root / "scripts" / "create_jsonl_for_hf.py"),
        str(train_csv), str(train_jsonl), str(wavs_16khz_dir),
        "--text-column", "2"  # Use normalized_text column
    ]
    
    if not run_command(train_cmd, "Converting training metadata to JSONL"):
        return False
    
    # Convert validation data
    val_cmd = [
        "python", str(project_root / "scripts" / "create_jsonl_for_hf.py"),
        str(val_csv), str(val_jsonl), str(wavs_16khz_dir),
        "--text-column", "2"  # Use normalized_text column
    ]
    
    if not run_command(val_cmd, "Converting validation metadata to JSONL"):
        return False
    
    # Step 2.5: Create HuggingFace dataset
    print(f"\n📊 Step 2.5: Creating HuggingFace dataset")
    
    hf_dataset_dir = data_dir / "hf_dataset"
    
    create_dataset_cmd = [
        "python", str(project_root / "scripts" / "create_hf_dataset.py"),
        str(train_jsonl), str(val_jsonl), str(hf_dataset_dir)
    ]
    
    if not run_command(create_dataset_cmd, "Creating HuggingFace dataset"):
        return False
    
    # Step 3: Clone finetune repository
    print(f"\n📥 Step 3: Cloning finetune-hf-vits repository")
    
    finetune_dir = project_root / "finetune-hf-vits"
    
    if finetune_dir.exists():
        print(f"📁 Repository already exists: {finetune_dir}")
        print("   Skipping clone...")
    else:
        clone_cmd = [
            "git", "clone", "https://github.com/ylacombe/finetune-hf-vits"
        ]
        
        if not run_command(clone_cmd, "Cloning repository"):
            return False
    
    # Step 4: Install requirements
    print(f"\n📦 Step 4: Installing requirements")
    
    requirements_cmd = [
        "pip", "install", "-r", str(finetune_dir / "requirements.txt")
    ]
    
    if not run_command(requirements_cmd, "Installing requirements"):
        print("⚠️  Requirements installation failed, but continuing...")
    
    # Step 5: Build monotonic alignment
    print(f"\n🔨 Step 5: Building monotonic alignment")
    
    align_dir = finetune_dir / "monotonic_align"
    if not align_dir.exists():
        print(f"❌ Monotonic align directory not found: {align_dir}")
        return False
    
    # Change to align directory and build
    original_dir = os.getcwd()
    os.chdir(align_dir)
    
    # Create build directory using Python instead of mkdir
    build_dir = align_dir / "monotonic_align"
    build_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created build directory: {build_dir}")
    
    # Build extension
    build_ext_cmd = ["python", "setup.py", "build_ext", "--inplace"]
    if not run_command(build_ext_cmd, "Building monotonic alignment extension"):
        os.chdir(original_dir)
        return False
    
    # Return to original directory
    os.chdir(original_dir)
    
    # Step 6: Create training checkpoint
    print(f"\n🎯 Step 6: Creating training checkpoint for Vietnamese")
    
    checkpoints_dir = finetune_dir / "checkpoints"
    checkpoint_dir = checkpoints_dir / "mms-vie-train"
    
    # Create checkpoints directory
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert checkpoint
    convert_cmd = [
        "python", str(finetune_dir / "convert_original_discriminator_checkpoint.py"),
        "--language_code", "vie",
        "--pytorch_dump_folder_path", str(checkpoint_dir)
    ]
    
    if not run_command(convert_cmd, "Converting checkpoint for Vietnamese"):
        return False
    
    # Summary
    print(f"\n🎉 Setup completed successfully!")
    print("=" * 60)
    print(f"📁 16kHz audio: {wavs_16khz_dir}")
    print(f"📁 Training JSONL: {train_jsonl}")
    print(f"📁 Validation JSONL: {val_jsonl}")
    print(f"📁 HuggingFace dataset: {hf_dataset_dir}")
    print(f"📁 Finetune repo: {finetune_dir}")
    print(f"📁 Checkpoint: {checkpoint_dir}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Test the finetuned model:")
    print(f"      cd {finetune_dir}")
    print(f"      accelerate launch run_vits_finetuning.py \\")
    print(f"        --model_name_or_path {checkpoint_dir} \\")
    print(f"        --output_dir runs/mms_vie_ft_single_spk \\")
    print(f"        --dataset_name {hf_dataset_dir} \\")
    print(f"        --audio_column_name audio \\")
    print(f"        --text_column_name text \\")
    print(f"        --learning_rate 2e-4 \\")
    print(f"        --per_device_train_batch_size 8 \\")
    print(f"        --num_train_epochs 50 \\")
    print(f"        --save_steps 500 \\")
    print(f"        --logging_steps 50")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup HuggingFace MMS-TTS-VIE finetuning")
    args = parser.parse_args()
    
    if setup_huggingface_finetune():
        print(f"\n✅ All done! Ready for finetuning.")
    else:
        print(f"\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()
