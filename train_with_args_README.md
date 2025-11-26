# Train with Arguments - PCA Pipeline Training

This script provides a configurable training pipeline for PCA-based VAE models with command-line argument support.

## Overview

`train_with_args.py` is a flexible training script that replicates the functionality of `train.py` but adds comprehensive command-line argument support for easy configuration and automation.

## Quick Start

```bash
# Basic training with default parameters
python train_with_args.py

# Training with custom configuration
python train_with_args.py \
    --train_data_dir my_train_images \
    --eval_data_dir my_eval_images \
    --epochs 200 \
    --batch_size 8 \
    --device cuda:0
```

## Command-line Arguments

### Model Configuration

- `--model_path`: Path to VAE model (default: "black-forest-labs/FLUX.1-dev")
- `--pca_components_path`: Path to PCA components CSV file
- `--pca_mean_path`: Path to PCA mean CSV file
- `--predictor_type`: Type of PCA predictor ("color_aware" or "basic", default: "color_aware")
- `--residual_detail`: Enable residual detail prediction (default: True)

### Data Configuration

- `--train_data_dir`: Directory containing training images (default: "train_images")
- `--eval_data_dir`: Directory containing evaluation images (default: "eval_images")
- `--train_batch_size`: Batch size for training (default: 4)
- `--eval_batch_size`: Batch size for evaluation (default: 1)
- `--num_workers`: Number of data loader workers (default: 4)

### Training Configuration

- `--stage`: Training stage (1: both predictors, 2: residual only, default: 1)
- `--epochs`: Number of training epochs (default: 400)
- `--warmup_steps`: Number of warmup steps for learning rate (default: 150)
- `--device`: Device to use for training (default: "cuda")
- `--precision`: Precision for training ("float32", "bfloat16", "float16", default: "bfloat16")

### Optimizer Configuration

- `--stage1_pca_lr`: Learning rate for PCA predictor in stage 1 (default: 5e-3)
- `--stage1_residual_lr`: Learning rate for residual predictor in stage 1 (default: 1.2e-2)
- `--stage2_lr`: Learning rate for stage 2 training (default: 8e-3)
- `--delta_lr`: Learning rate for PCA delta parameters (default: 5e-3)

### Scheduler Configuration

- `--stage1_pca_step_size`: Step size for PCA scheduler in stage 1 (default: 120)
- `--stage1_pca_gamma`: Gamma for PCA scheduler in stage 1 (default: 0.5)
- `--stage1_delta_step_size`: Step size for delta scheduler in stage 1 (default: 150)
- `--stage1_delta_gamma`: Gamma for delta scheduler in stage 1 (default: 0.1)
- `--stage2_pca_step_size`: Step size for PCA scheduler in stage 2 (default: 40)
- `--stage2_pca_gamma`: Gamma for PCA scheduler in stage 2 (default: 0.8)
- `--stage2_delta_step_size`: Step size for delta scheduler in stage 2 (default: 100)
- `--stage2_delta_gamma`: Gamma for delta scheduler in stage 2 (default: 0.1)

### Loss Configuration

- `--diff_dist_weight`: Weight for distribution difference loss (default: 0.3)
- `--recon_latent_weight`: Weight for latent reconstruction loss (default: 1.0)
- `--kl_weight`: Weight for KL divergence loss (default: 0.03)
- `--diff_weight`: Weight for residual difference loss (default: 1.0)

### Logging & Checkpointing

- `--log_dir`: Directory for TensorBoard logs (default: "runs")
- `--save_dir`: Directory for saving checkpoints (default: "ckpt")
- `--eval_frequency`: Frequency of evaluation in epochs (default: 10)
- `--save_frequency`: Frequency of checkpoint saving in epochs (default: 50)
- `--config_save_path`: Path to save training configuration (default: "training_config.json")

## Usage Examples

### Basic Training
```bash
python train_with_args.py
```

### Custom Dataset Training
```bash
python train_with_args.py \
    --train_data_dir /path/to/train \
    --eval_data_dir /path/to/eval \
    --train_batch_size 8 \
    --epochs 500
```

### Multi-GPU Training
```bash
python train_with_args.py \
    --device cuda:0,1 \
    --train_batch_size 16 \
    --num_workers 8
```

### Mixed Precision Training
```bash
python train_with_args.py \
    --precision bfloat16 \
    --train_batch_size 16
```

### Two-Stage Training
```bash
# Stage 1: Train both predictors
python train_with_args.py --stage 1 --epochs 200

# Stage 2: Train residual predictor only
python train_with_args.py --stage 2 --epochs 100
```

### Custom Loss Weights
```bash
python train_with_args.py \
    --diff_dist_weight 0.5 \
    --recon_latent_weight 2.0 \
    --kl_weight 0.01
```

## Output Files

### Checkpoints
- Saved in `--save_dir` directory
- Format: `pca_pipeline_stage{stage}_{timestamp}.pth`
- Contains: PCA predictor, residual detail predictor, and PCA model state

### Logs
- TensorBoard logs in `--log_dir` directory
- Includes: Loss curves, images, gradients, model statistics

### Configuration
- Training configuration saved as JSON file
- Default path: `training_config.json`

## Environment Variables

The script uses the following environment variables:

- `HUGGINGFACE_TOKEN`: HuggingFace API token for model access
- `HF_CACHE_DIR`: Cache directory for HuggingFace models

## Training Process

### Stage 1 (Default)
- Trains both PCA predictor and residual detail predictor
- Uses warmup learning rate scheduling
- Optimizes PCA delta parameters
- Comprehensive loss function with multiple components

### Stage 2
- Freezes PCA predictor
- Trains only residual detail predictor
- Different learning rate schedule
- Focused on residual detail refinement

## Monitoring

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir runs/
```

Available metrics include:
- Loss curves (total, PCA, reconstruction, KL, etc.)
- Learning rate schedules
- Gradient norms
- Image reconstructions
- Frequency domain analysis
- Model statistics

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use mixed precision
2. **Slow Training**: Increase number of workers or use faster storage
3. **Model Loading Issues**: Check HuggingFace token and network connectivity

### Performance Tips

- Use `--precision bfloat16` for faster training and reduced memory usage
- Increase `--num_workers` for faster data loading
- Use larger batch sizes with gradient accumulation if needed
- Monitor GPU utilization with `nvidia-smi`

## Dependencies

- PyTorch
- HuggingFace Transformers
- TensorBoard
- NumPy
- scikit-learn
- tqdm

## Related Files

- `train.py`: Original training script without argument parsing
- `vae_slim.py`: Core model definitions
- `dataloader.py`: Data loading utilities
- `losses.py`: Custom loss functions
- `utils.py`: Utility functions for visualization