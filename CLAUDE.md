# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fast-SE is a speech enhancement system based on SEMamba (Speech Enhancement Mamba), which uses Mamba blocks for denoising audio. The model processes noisy speech by separating magnitude and phase components in the frequency domain, applying temporal-frequency Mamba blocks, and reconstructing clean audio.

## Development Commands

### Environment Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Mamba SSM (required for Mamba blocks)
# The mamba_install/ or mamba-1_2_0_post1/ directory contains the Mamba implementation
cd mamba_install && pip install -e . && cd ..

# Configure Accelerate (one-time setup for multi-GPU training)
accelerate config
# Follow prompts to configure for your hardware setup
```

### Data Preparation
```bash
# Generate dataset JSON files from audio directories
python data/make_dataset_json.py --prefix_path /path/to/noisy_vctk/

# This creates train/valid/test JSON files in data/ directory:
# - train_clean.json, train_noisy.json
# - valid_clean.json, valid_noisy.json
# - test_clean.json, test_noisy.json
```

### Training
```bash
# Single-GPU or simple multi-GPU training
python train.py --config recipes/SEMamba_advanced/SEMamba_advanced.yaml

# Multi-GPU training with Accelerate launcher (recommended for distributed training)
accelerate launch train.py --config recipes/SEMamba_advanced/SEMamba_advanced.yaml

# Train with custom config
python train.py --config path/to/config.yaml --exp_folder exp --exp_name experiment_name

# Training with pretrained discriminator
python train.py --config recipes/SEMamba_advanced/SEMamba_advanced_pretrainedD.yaml

# Training with PCS (Perceptually Constrained Speech) processing
python train.py --config recipes/SEMamba_advanced_PCS/SEMamba_advanced_PCS.yaml

# Note: Both 'python train.py' and 'accelerate launch train.py' work the same way.
# The code automatically detects and uses available GPUs via Accelerate.
```

### Inference
```bash
# Run inference on noisy audio files
python inference.py --checkpoint_file ckpts/SEMamba_advanced.pth \
    --config ckpts/config.yaml \
    --input_folder /path/to/noisy/audio \
    --output_folder results

# With PCS post-processing
python inference.py --checkpoint_file ckpts/SEMamba_advanced.pth \
    --config ckpts/config.yaml \
    --input_folder /path/to/noisy/audio \
    --output_folder results \
    --post_processing_PCS true
```

## Architecture Overview

### Model Structure (models/)

**SEMamba (generator.py)**: Main speech enhancement model
- DenseEncoder: Encodes magnitude and phase inputs (concatenated as 2-channel input)
- TFMambaBlock: 4 temporal-frequency Mamba blocks for sequence modeling
- MagDecoder: Decodes magnitude mask (multiplied with noisy magnitude)
- PhaseDecoder: Decodes phase information

**TFMambaBlock (mamba_block.py)**: Core sequence modeling component
- Applies bidirectional Mamba processing (forward + backward) on temporal dimension
- Applies bidirectional Mamba processing on frequency dimension
- Uses residual connections with 1D ConvTranspose projection layers

**MetricDiscriminator (discriminator.py)**: Metric-based discriminator
- Takes clean and enhanced magnitude spectrograms as input
- Predicts PESQ-like quality score (0-1 range)
- Uses spectral normalization and learnable sigmoid activation

**DenseEncoder/Decoder (codec_module.py)**:
- DenseBlock: Multiple dilated convolutions with skip connections
- Encoder: Compresses frequency dimension by 2x
- Decoders: Separate paths for magnitude mask and phase estimation

### Training Process (train.py)

**Multi-GPU Training**: Uses Hugging Face Accelerate for seamless single/multi-GPU training
- Accelerate automatically handles device placement and distributed training
- No manual process spawning or DDP wrapping needed
- Works with both `python train.py` and `accelerate launch train.py`
- Supports future extensions: mixed precision, gradient accumulation, DeepSpeed

**Loss Functions**:
- Metric loss: Discriminator prediction vs ground truth
- Magnitude loss: MSE between clean and enhanced magnitude
- Phase loss: Anti-wrapping IP + GD + IAF losses
- Complex loss: MSE between clean and enhanced complex spectrograms
- Time loss: L1 between clean and enhanced waveforms
- Consistency loss: Ensures STFT(enhanced audio) matches predicted complex spectrogram

**Training Loop**:
1. Models, optimizers, and dataloaders prepared with `accelerator.prepare()`
2. Generator predicts enhanced magnitude, phase, complex from noisy inputs
3. Discriminator training: Distinguishes clean pairs from enhanced pairs
4. Generator training: Multi-objective loss (weighted combination)
5. Backward passes use `accelerator.backward()` for automatic gradient handling
6. Validation every N steps with PESQ score calculation (main process only)
7. Checkpointing: Saves unwrapped models using `accelerator.unwrap_model()`

### Data Pipeline (dataloaders/dataloader_vctk.py)

**VCTKDemandDataset**: Loads paired clean/noisy audio
- Reads file paths from JSON (generated by make_dataset_json.py)
- Performs on-the-fly STFT transformation
- Segments audio to fixed length (default 32000 samples = 2 seconds at 16kHz)
- Optional PCS400 processing during training
- DataLoader prepared by Accelerate for automatic distributed data sharding
- Returns: clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha

### STFT Processing (models/stfts.py)

- `mag_phase_stft()`: Converts waveform to compressed magnitude + phase
- `mag_phase_istft()`: Reconstructs waveform from magnitude + phase
- Uses compression factor (default 0.3) for magnitude: mag^compress_factor

## Configuration Files (recipes/)

YAML configs control all training parameters:
- `env_setting`: GPU count (reference only), workers, intervals for logging/checkpointing
- `data_cfg`: Paths to train/valid/test JSON files
- `training_cfg`: Epochs, batch size, learning rate, loss weights, PCS usage
- `stft_cfg`: Sampling rate (16kHz), n_fft (400), hop_size (100), win_size (400)
- `model_cfg`: hid_feature (64), compress_factor (0.3), num_tfmamba (4), d_state (16), d_conv (4), expand (4)

Note: `num_gpus` in `env_setting` is for reference only - Accelerate auto-detects available GPUs.

## Key Implementation Details

**Mamba Integration**:
- Uses mamba_ssm library (in mamba_install/ or mamba-1_2_0_post1/)
- Bidirectional processing: Forward + backward passes, then concatenate
- Applied separately on temporal and frequency dimensions

**Magnitude-Phase Processing**:
- Model inputs: Noisy magnitude and phase (2 channels)
- Magnitude decoder outputs mask (0-1 via learnable sigmoid), multiplied with noisy magnitude
- Phase decoder outputs phase directly (atan2 from real/imaginary components)
- Final output: Magnitude × cos(phase) + i × sin(phase) for complex spectrogram

**PESQ Metric**:
- Used for validation and discriminator training
- Computed in parallel (joblib) for batch processing
- Normalized to 0-1 range: (pesq - 1) / 3.5

**Checkpointing**:
- Models unwrapped with `accelerator.unwrap_model()` before saving
- Generator checkpoints: `exp_folder/exp_name/g_{steps:08d}.pth`
- Discriminator+optimizer checkpoints: `exp_folder/exp_name/do_{steps:08d}.pth`
- Config copied to: `exp_folder/exp_name/config.yaml`

**Accelerate Integration**:
- Single `Accelerator()` instance handles all device/distributed logic
- `accelerator.prepare()` wraps models, optimizers, dataloaders, schedulers
- `accelerator.backward()` replaces standard `loss.backward()`
- `accelerator.is_main_process` gates logging/checkpointing operations
- Automatic gradient synchronization across GPUs (no manual barriers needed)

## GPU Requirements

- Training requires CUDA (CPU mode not supported for Mamba operations)
- Multi-GPU training handled automatically by Accelerate
- Use `accelerate config` for initial setup, or run directly with `python train.py`
- Accelerate auto-detects available GPUs

## Special Features

**PCS400 (Perceptually Constrained Speech)**: Optional post-processing (models/pcs400.py)
- Applied during training (use_PCS400: true in config)
- Applied during inference (--post_processing_PCS true flag)

**Pretrained Discriminator**:
- Load with use_pretrainedD: true in config
- Path: ckpts/pretrained_discriminator.pth

## Future Improvements (TODOs)

The following features can be easily added with Accelerate:

**Mixed Precision Training**:
- Enable with `Accelerator(mixed_precision='fp16')` or `'bf16'`
- Significant speedup on modern GPUs (A100, RTX 30xx/40xx series)
- Reduces memory usage, allows larger batch sizes

**Gradient Accumulation**:
- Add `gradient_accumulation_steps` parameter to Accelerator
- Simulate larger batch sizes on limited GPU memory

**DeepSpeed Integration**:
- Configure via `accelerate config` for ZeRO optimization
- Enables training very large models

**Hydra Configuration** (Planned):
- Replace YAML loading with Hydra for better config management
- Support config composition and command-line overrides
