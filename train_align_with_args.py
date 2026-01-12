import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
import logging
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import project modules
from vae_align import AlignPipeline
from dataloader import image_dataloader
from losses import WeightedL1Loss
from utils import visualize_spectrum_comparison, radial_profile, rgb_grad_map, rgb_grad_comparison

# HuggingFace imports
from huggingface_hub import login
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments for VAE alignment training configuration"""
    parser = argparse.ArgumentParser(description="Train VAE Alignment Pipeline")
    
    # General Configuration
    parser.add_argument('--load_config', type=str, default=None,
                       help="Path to a JSON configuration file to load parameters from")
    parser.add_argument('--load_checkpoint', type=str, default=None,
                       help="Path to a checkpoint to load the model from")
    
    # VAE Model Configuration
    parser.add_argument('--vae1_path', type=str, default="black-forest-labs/FLUX.1-dev",
                       help="Path to the first VAE model, e.g., black-forest-labs/FLUX.1-dev")
    parser.add_argument('--vae2_path', type=str, default="sd-legacy/stable-diffusion-v1-5",
                       help="Path to the second VAE model, e.g., sd-legacy/stable-diffusion-v1-5, \
                                                                Tongyi-MAI/Z-Image-Turbo, stabilityai/stable-diffusion-3.5-large, \
                                                                    stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument('--vae1_subfolder', type=str, default="vae",
                       help="Subfolder for VAE1 model")
    parser.add_argument('--vae2_subfolder', type=str, default="vae",
                       help="Subfolder for VAE2 model")
    
    # Alignment Module Configuration
    parser.add_argument('--model_version', type=str, default="base",
                       choices=["base", "longtail", "light"],
                       help="Version of the alignment module to use")
    parser.add_argument('--img_in_channels', type=int, default=3,
                       help="Input image channels for alignment module")
    parser.add_argument('--in_channels', type=int, default=4,
                       help="Input channels for alignment module (should match VAE latent channels)")
    parser.add_argument('--hidden_channels', type=int, default=32,
                       help="Hidden channels in alignment module")
    parser.add_argument('--out_channels', type=int, default=16,
                       help="Output channels in alignment module (should match VAE latent channels)")
    parser.add_argument('--num_blocks', type=int, default=2,
                       help="Number of residual blocks in each downsample stage")
    parser.add_argument('--downsample_times', type=int, default=3,
                       help="Number of downsample times in alignment module")
    parser.add_argument('--channel_times', type=int, default=4,
                          help="Channel expansion factor for each downsample stage")
    parser.add_argument('--input_types', type=str, nargs='+', default=['image', 'latent'],
                       choices=['image', 'latent', 'DWT'],
                       help="Input types for alignment module")
    parser.add_argument('--image_normalize', action='store_true', default=True,
                       help="Whether to normalize images to [-1, 1] range before forward")
    
    # Data Configuration
    parser.add_argument('--train_data_dir', type=str, default="train_images",
                       help="Directory containing training images")
    parser.add_argument('--eval_data_dir', type=str, default="eval_images",
                       help="Directory containing evaluation images")
    parser.add_argument('--train_batch_size', type=int, default=4,
                       help="Batch size for training")
    parser.add_argument('--eval_batch_size', type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument('--num_workers', type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument('--dataset_type', type=str, default="imagenet",
                       choices=["default", "imagenet"],
                       help="Type of dataset to use")
    
    # Training Configuration
    parser.add_argument('--it_or_epochs', type=str, default="epochs",
                       choices=["iterations", "epochs"],
                       help="Training mode: iterations or epochs")
    parser.add_argument('--iterations', type=int, default=10000,
                       help="Total number of training iterations")
    parser.add_argument('--epochs', type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help="Number of warmup steps for learning rate")
    parser.add_argument('--device', type=str, default="cuda",
                       help="Device to use for training")
    parser.add_argument('--precision', type=str, default="bfloat16", 
                       choices=["float32", "bfloat16", "float16"],
                       help="Precision for training")
    
    # Optimizer Configuration
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help="Learning rate for alignment module")
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help="Weight decay for optimizer")
    
    # Scheduler Configuration
    parser.add_argument('--scheduler_step_size', type=int, default=50,
                       help="Step size for learning rate scheduler")
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                       help="Gamma for learning rate scheduler")
    
    # Loss Configuration
    parser.add_argument('--training_stage', type=int, default=1,
                       help="Stage of training: align or reconstruct")

    parser.add_argument('--loss_type', type=str, nargs='+', default=["l1"],
                       choices=["mse", "l1", "perceptual", "lpips"],
                       help="Type of loss function to use")
    parser.add_argument('--loss_weight', type=float, nargs='+',default=[1.0],
                       help="Weight for the loss function")
    
    # Gradient Clipping Configuration
    parser.add_argument('--grad_clip', type=float, default=2.5,
                       help="Gradient clipping norm (0.0 means no clipping)")
    
    # Logging & Checkpointing
    parser.add_argument('--log_dir', type=str, default="runs",
                       help="Directory for TensorBoard logs")
    parser.add_argument('--save_dir', type=str, default="ckpt_align",
                       help="Directory for saving checkpoints")
    parser.add_argument('--eval_frequency', type=int, default=1,
                       help="Frequency of evaluation")
    parser.add_argument('--save_frequency', type=int, default=1,
                       help="Frequency of checkpoint saving")
    parser.add_argument('--config_save_path', type=str, default="config/",
                       help="Path to save training configuration")
    
    # Visualization Configuration
    parser.add_argument('--visualize_frequency', type=int, default=1,
                       help="Frequency of visualization")
    parser.add_argument('--num_visualize_samples', type=int, default=4,
                       help="Number of samples to visualize")
    
    return parser.parse_args()


def get_dtype(precision):
    """Get torch dtype from precision string"""
    if precision == "bfloat16":
        return torch.bfloat16
    elif precision == "float16":
        return torch.float16
    else:
        return torch.float32


def load_vae_model(model_path, subfolder, cache_dir, device, dtype):
    """Load VAE model with configurable parameters"""
    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder=subfolder,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        proxies={'http': '127.0.0.1:7890'}
    )
    vae.to(device)
    return vae


def create_align_pipeline(vae1, vae2, args):
    """Create alignment pipeline with configurable parameters"""
    pipeline = AlignPipeline(
        VAE_1=vae1,
        VAE_2=vae2,
        model_version=args.model_version,
        img_in_channels=args.img_in_channels,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_blocks=args.num_blocks,
        downsample_times=args.downsample_times,
        channel_times=args.channel_times,
        input_types=args.input_types,
        device=args.device,
        dtype=get_dtype(args.precision)
    )
    
    # Freeze VAE models
    pipeline.freeze_vae()
    
    return pipeline


def create_data_loaders(train_data_dir, eval_data_dir, train_batch_size, eval_batch_size, num_workers, dataset_type):
    """Create train and eval data loaders with configurable parameters"""
    if dataset_type == "imagenet":
        from dataloader import ImageNetDataloader, get_imagenet_dataset
        train_dataset = get_imagenet_dataset(split="train")
        eval_dataset = get_imagenet_dataset(split="test")
        
        train_loader = ImageNetDataloader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
        eval_loader = ImageNetDataloader(eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
        
    else:
        train_loader = image_dataloader(
            data_dir=train_data_dir,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        eval_loader = image_dataloader(
            data_dir=eval_data_dir,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0
        )
    
    return train_loader, eval_loader


def setup_optimizer(pipeline, args):
    """Setup optimizer for alignment module"""
    optimizer = torch.optim.Adam(
        pipeline.align_module.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    return optimizer


def setup_scheduler(optimizer, args):
    """Setup learning rate scheduler"""
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma
    )
    return scheduler


def get_loss_function(loss_type: list, args):
    """Get loss function based on type"""
    
    loss_list = []
    for loss in loss_type:
        if loss == "lpips":
            import lpips
            addition = lpips.LPIPS(net='alex').to(args.device, dtype=get_dtype(args.precision))
        else:
            addition = None
        loss_list.append((loss, args.loss_weight[loss_type.index(loss)], addition))

    return loss_list


def train_align_pipeline(args, vae1, vae2, train_loader, eval_loader, generator):
    """Main training loop for alignment pipeline"""
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{args.log_dir}/align_pipeline_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create pipeline
    pipeline = create_align_pipeline(vae1, vae2, args)
    if args.load_checkpoint:
        # Load checkpoint if provided
        pipeline.load(args.load_checkpoint)
        print(f"Loaded checkpoint from: {args.load_checkpoint}")
    
    # Setup optimizer and scheduler
    optimizer = setup_optimizer(pipeline, args)
    scheduler = setup_scheduler(optimizer, args)
    
    # Loss function
    criterion = get_loss_function(args.loss_type, args)
    
    dtype = get_dtype(args.precision)
    global_step = 0
    
    # Main training loop
    for epoch in tqdm(range(args.epochs)):
        pipeline.align_module.train()
        epoch_losses = {k: 0.0 for k in args.loss_type}
        epoch_losses['total_loss'] = 0.0
        
        for batch_idx, x in tqdm(enumerate(train_loader)):
            # Warmup learning rate
            if global_step < args.warmup_steps:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = min(args.learning_rate, args.learning_rate * (global_step + 1) / args.warmup_steps)
            
            x = x['pixel_values'].to(args.device).to(dtype)
            
            # Training step
            optimizer.zero_grad()
            
            # Forward pass and loss calculation
            loss_dict = pipeline.train_step(x, optimizer, criterion, generator, args.image_normalize)
            
            # Record losses
            for k, v in loss_dict.items():
                epoch_losses[k] += v
                writer.add_scalar(f'Batch/{k}', v, global_step)
            
            # Apply gradient clipping if enabled
            if args.grad_clip > 0:
                clip_norm = torch.nn.utils.clip_grad_norm_(
                    pipeline.align_module.parameters(), 
                    max_norm=args.grad_clip
                )
                writer.add_scalar('Training/Gradient_Clip_Norm', clip_norm.item(), global_step)
            
            # Record gradient norm
            total_norm = 0
            for p in pipeline.align_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            writer.add_scalar('Training/Gradient_Norm', total_norm, global_step)
            
            global_step += 1
        
        # Evaluation and logging
        if epoch % args.eval_frequency == 0:
            evaluate_and_log(pipeline, eval_loader, writer, epoch, args, dtype)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        
        # Record epoch statistics
        avg_losses = {k: v/len(train_loader) for k, v in epoch_losses.items()}
        for k, v in avg_losses.items():
            writer.add_scalar(f'Epoch/{k}', v, epoch)
        
        # Log model output statistics
        log_model_stats(pipeline, eval_loader, writer, epoch, args, dtype)
        
        # Visualization
        if epoch % args.visualize_frequency == 0:
            visualize_results(pipeline, eval_loader, writer, epoch, args, dtype)
        
        print(f"Epoch {epoch}: {avg_losses}, LR: {current_lr:.2e}")
        
        if args.save_frequency > 0 and epoch % args.save_frequency == 0:
            # Save checkpoint
            checkpoint_path = f"{args.save_dir}/align_pipeline_epoch_{epoch}.pth"
            pipeline.save(checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
    
    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")
    
    return pipeline


def evaluate_and_log(pipeline, eval_loader, writer, epoch, args, dtype):
    """Evaluate model and log results to TensorBoard"""
    pipeline.align_module.eval()
    
    with torch.no_grad():
        x = next(iter(eval_loader))['pixel_values'].to(args.device).to(dtype)
        
        if args.image_normalize:
            # Normalize images to [-1, 1] range
            x = x * 2 - 1
        # Get latent representations
        z_vae1 = pipeline._encode_vae_image(pipeline.VAE_1, x, generator=None)
        z_vae2 = pipeline._encode_vae_image(pipeline.VAE_2, x, generator=None)
        
        # Get aligned latent
        z_vae1_aligned = pipeline.align_module(x, z_vae1)
        
        # Calculate alignment error
        alignment_error = F.mse_loss(z_vae1_aligned, z_vae2)
        
        # Log metrics
        writer.add_scalar('Evaluation/Alignment_Error', alignment_error.item(), epoch)
        writer.add_scalar('Evaluation/z_vae1_mean', z_vae1.mean().item(), epoch)
        writer.add_scalar('Evaluation/z_vae2_mean', z_vae2.mean().item(), epoch)
        writer.add_scalar('Evaluation/z_vae1_aligned_mean', z_vae1_aligned.mean().item(), epoch)
        writer.add_scalar('Evaluation/z_vae1_std', z_vae1.std().item(), epoch)
        writer.add_scalar('Evaluation/z_vae2_std', z_vae2.std().item(), epoch)
        writer.add_scalar('Evaluation/z_vae1_aligned_std', z_vae1_aligned.std().item(), epoch)


def log_model_stats(pipeline, eval_loader, writer, epoch, args, dtype):
    """Log model output statistics"""
    pipeline.align_module.eval()
    
    with torch.no_grad():
        test_batch = next(iter(eval_loader))['pixel_values'].to(args.device).to(dtype)
        
        if args.image_normalize:
            # Normalize images to [-1, 1] range
            test_batch = test_batch * 2 - 1

        # Get latent representations
        z_vae1 = pipeline._encode_vae_image(pipeline.VAE_1, test_batch, generator=None)
        z_vae1_aligned = pipeline.align_module(test_batch, z_vae1)
        
        # Log alignment module output statistics
        writer.add_scalar('Stats/z_vae1_aligned_mean', z_vae1_aligned.mean().item(), epoch)
        writer.add_scalar('Stats/z_vae1_aligned_std', z_vae1_aligned.std().item(), epoch)
        
        # Log alignment module parameters
        for name, param in pipeline.align_module.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            writer.add_histogram(f'Parameters/{name}', param, epoch)


def visualize_results(pipeline, eval_loader, writer, epoch, args, dtype):
    """Visualize alignment results"""
    pipeline.align_module.eval()
    
    with torch.no_grad():
        x = next(iter(eval_loader))['pixel_values'].to(args.device).to(dtype)
        x = x[:args.num_visualize_samples]  # Take only a few samples
        
        if args.image_normalize:
            # Normalize images to [-1, 1] range
            x = x * 2 - 1

        # Get reconstructions from both VAEs
        recon_vae1 = pipeline.latent_reconstruction(pipeline.VAE_1, x, generator=None, do_normalize=args.image_normalize)
        recon_vae2 = pipeline.latent_reconstruction(pipeline.VAE_2, x, generator=None, do_normalize=args.image_normalize)

        # Get latent representations
        z_vae1 = pipeline._encode_vae_image(pipeline.VAE_1, x, generator=None)
        z_vae2 = pipeline._encode_vae_image(pipeline.VAE_2, x, generator=None)
        
        # Get aligned latent and reconstruction
        z_vae1_aligned = pipeline.align_module(x, z_vae1)
        recon_aligned = pipeline._decode_vae_latents(pipeline.VAE_2, z_vae1_aligned, do_normalize=args.image_normalize)

        if args.image_normalize:
            x = (x + 1) / 2  # Convert back to [0, 1] range for visualization

        # Log images
        writer.add_images('Visualization/Original', x.float(), epoch)
        writer.add_images('Visualization/Reconstruction_VAE1', recon_vae1.float(), epoch)
        writer.add_images('Visualization/Reconstruction_VAE2', recon_vae2.float(), epoch)
        writer.add_images('Visualization/Reconstruction_Aligned', recon_aligned.float(), epoch)
        
        # Log difference maps
        diff_vae1 = (x - recon_vae1).abs()
        diff_vae2 = (x - recon_vae2).abs()
        diff_aligned = (x - recon_aligned).abs()
        
        writer.add_images('Visualization/Difference_VAE1', diff_vae1, epoch)
        writer.add_images('Visualization/Difference_VAE2', diff_vae2, epoch)
        writer.add_images('Visualization/Difference_Aligned', diff_aligned, epoch)
        
        # Log latent space visualizations (first 3 channels)
        if z_vae1.shape[1] >= 3:
            z_vae1_rgb = z_vae1[:, :3, :, :]
            z_vae2_rgb = z_vae2[:, :3, :, :]
            z_vae1_aligned_rgb = z_vae1_aligned[:, :3, :, :]
            
            # Normalize for visualization
            z_vae1_rgb = (z_vae1_rgb - z_vae1_rgb.min()) / (z_vae1_rgb.max() - z_vae1_rgb.min() + 1e-8)
            z_vae2_rgb = (z_vae2_rgb - z_vae2_rgb.min()) / (z_vae2_rgb.max() - z_vae2_rgb.min() + 1e-8)
            z_vae1_aligned_rgb = (z_vae1_aligned_rgb - z_vae1_aligned_rgb.min()) / (z_vae1_aligned_rgb.max() - z_vae1_aligned_rgb.min() + 1e-8)
            
            writer.add_images('Latent/z_vae1_rgb', z_vae1_rgb, epoch)
            writer.add_images('Latent/z_vae2_rgb', z_vae2_rgb, epoch)
            writer.add_images('Latent/z_vae1_aligned_rgb', z_vae1_aligned_rgb, epoch)


def save_config(args, save_path):
    """Save training configuration to JSON file"""
    config_dict = vars(args)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_path, f"align_training_config_{datetime_str}.json")
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Training configuration saved to: {save_path}")


def load_config(config_path):
    """Load training configuration from JSON file"""
    from types import SimpleNamespace
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return SimpleNamespace(**config_dict)


def main():
    """Main execution function"""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    if args.load_config:
        args = load_config(args.load_config)
    
    # Save configuration
    save_config(args, args.config_save_path)
    
    # Setup environment
    Token = os.getenv("HUGGINGFACE_TOKEN", "")
    cache_dir = os.getenv("HF_CACHE_DIR", "")
    
    if Token:
        login(token=Token)
        logger.info("Logged into HuggingFace Hub")
    else:
        logger.warning("No HuggingFace token found, proceeding without login")
    
    # Load VAE models
    logger.info(f"Loading VAE1 from: {args.vae1_path}")
    logger.info(f"Loading VAE2 from: {args.vae2_path}")
    dtype = get_dtype(args.precision)
    
    vae1 = load_vae_model(args.vae1_path, args.vae1_subfolder, cache_dir, args.device, dtype)
    vae2 = load_vae_model(args.vae2_path, args.vae2_subfolder, cache_dir, args.device, dtype)
    
    # Create data loaders
    logger.info(f"Creating data loaders from train: {args.train_data_dir}, eval: {args.eval_data_dir}")
    train_loader, eval_loader = create_data_loaders(
        args.train_data_dir, args.eval_data_dir,
        args.train_batch_size, args.eval_batch_size, args.num_workers,
        args.dataset_type
    )
    
    # Setup generator
    generator = torch.manual_seed(42)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training
    logger.info(f"Starting alignment training with {args.epochs} epochs")
    pipeline = train_align_pipeline(args, vae1, vae2, train_loader, eval_loader, generator)
    
    # Save final checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_ckpt_path = f"{args.save_dir}/align_pipeline_{timestamp}.pth"
    pipeline.save(final_ckpt_path)
    logger.info(f"Final checkpoint saved to: {final_ckpt_path}")
    
    logger.info("Alignment training completed successfully!")


if __name__ == "__main__":
    main()
