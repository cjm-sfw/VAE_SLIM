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
from vae_slim import PCAPipeline, PCAModel, ColorAwarePCAPredictor, PCAPredictor
from dataloader import image_dataloader
from losses import WeightedL1Loss
from utils import visualize_spectrum_comparison, radial_profile, rgb_grad_map, rgb_grad_comparison

# HuggingFace imports
from huggingface_hub import login
from diffusers import AutoencoderKL
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments for training configuration"""
    parser = argparse.ArgumentParser(description="Train PCA Pipeline with VAE")
    
    # General Configuration
    parser.add_argument('--load_config', type=str, default=None,
                       help="Path to a JSON configuration file to load parameters from")
    
    # Model Configuration
    parser.add_argument('--model_path', type=str, default="black-forest-labs/FLUX.1-dev",
                       help="Path to the VAE model, e.g., black-forest-labs/FLUX.1-dev, sd-legacy/stable-diffusion-v1-5")
    parser.add_argument('--freeze_pca', action='store_true', default=False,
                        help="Freeze PCA components during training")
    parser.add_argument('--n_components', type=int, default=16,
                       help="The number of PCA components to use, 16 for FLUX.1-dev, 4 for SD1.5")
    parser.add_argument('--n_channels', type=int, default=16,
                       help="Number of channels in the PCA model, 16 for FLUX.1-dev, 4 for SD1.5")
    parser.add_argument('--valid_n_components', type=int, nargs='+', default=None,
                        help="List of PCA components to validate against, e.g., [0, 1, 2]")
    parser.add_argument('--high_freq_enable', action='store_true', default=False,
                       help="Enable high frequency prediction")
    parser.add_argument('--pca_components_path', type=str, 
                       default="/workspace/DiffBrush/VIS/pca3d_pca_components.csv",
                       help="Path to PCA components CSV file")
    parser.add_argument('--pca_mean_path', type=str, 
                       default="/workspace/DiffBrush/VIS/pca3d_pca_mean.csv",
                       help="Path to PCA mean CSV file")
    parser.add_argument('--residual_detail', action='store_true', default=True,
                       help="Enable residual detail prediction")
    
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
    
    # Training Configuration
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                       help="Training stage (1: both predictors, 2: residual only)")
    parser.add_argument('--epochs', type=int, default=400,
                       help="Number of training epochs")
    parser.add_argument('--warmup_steps', type=int, default=150,
                       help="Number of warmup steps for learning rate")
    parser.add_argument('--device', type=str, default="cuda",
                       help="Device to use for training")
    parser.add_argument('--precision', type=str, default="bfloat16", 
                       choices=["float32", "bfloat16", "float16"],
                       help="Precision for training")
    
    # Optimizer Configuration
    parser.add_argument('--stage1_pca_lr', type=float, default=5e-3,
                       help="Learning rate for PCA predictor in stage 1")
    parser.add_argument('--stage1_residual_lr', type=float, default=1.2e-2,
                       help="Learning rate for residual predictor in stage 1")
    parser.add_argument('--stage2_lr', type=float, default=8e-3,
                       help="Learning rate for stage 2 training")
    parser.add_argument('--delta_lr', type=float, default=5e-3,
                       help="Learning rate for PCA delta parameters")
    
    # Scheduler Configuration
    parser.add_argument('--stage1_pca_step_size', type=int, default=120,
                       help="Step size for PCA scheduler in stage 1")
    parser.add_argument('--stage1_pca_gamma', type=float, default=0.5,
                       help="Gamma for PCA scheduler in stage 1")
    parser.add_argument('--stage1_delta_step_size', type=int, default=150,
                       help="Step size for delta scheduler in stage 1")
    parser.add_argument('--stage1_delta_gamma', type=float, default=0.1,
                       help="Gamma for delta scheduler in stage 1")
    parser.add_argument('--stage2_pca_step_size', type=int, default=40,
                       help="Step size for PCA scheduler in stage 2")
    parser.add_argument('--stage2_pca_gamma', type=float, default=0.8,
                       help="Gamma for PCA scheduler in stage 2")
    parser.add_argument('--stage2_delta_step_size', type=int, default=100,
                       help="Step size for delta scheduler in stage 2")
    parser.add_argument('--stage2_delta_gamma', type=float, default=0.1,
                       help="Gamma for delta scheduler in stage 2")
    
    # Loss Configuration
    parser.add_argument('--diff_dist_weight', type=float, default=0.3,
                       help="Weight for distribution difference loss")
    parser.add_argument('--recon_latent_weight', type=float, default=1.0,
                       help="Weight for latent reconstruction loss")
    parser.add_argument('--kl_weight', type=float, default=0.03,
                       help="Weight for KL divergence loss")
    parser.add_argument('--diff_weight', type=float, default=1.0,
                       help="Weight for residual difference loss")
    
    # Logging & Checkpointing
    parser.add_argument('--log_dir', type=str, default="runs",
                       help="Directory for TensorBoard logs")
    parser.add_argument('--save_dir', type=str, default="ckpt",
                       help="Directory for saving checkpoints")
    parser.add_argument('--eval_frequency', type=int, default=10,
                       help="Frequency of evaluation (in epochs)")
    parser.add_argument('--save_frequency', type=int, default=50,
                       help="Frequency of checkpoint saving (in epochs)")
    parser.add_argument('--config_save_path', type=str, default="config/training_config.json",
                       help="Path to save training configuration")
    
    return parser.parse_args()


def get_dtype(precision):
    """Get torch dtype from precision string"""
    if precision == "bfloat16":
        return torch.bfloat16
    elif precision == "float16":
        return torch.float16
    else:
        return torch.float32


def load_vae_model(model_path, cache_dir, device, dtype):
    """Load VAE model with configurable parameters"""
    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=dtype,
        cache_dir=cache_dir,
        proxies={'http': '127.0.0.1:7890'}
    )
    vae.to(device)
    return vae


def load_pca_model(pca_components_path, pca_mean_path, device, args):
    """Load PCA model with configurable paths"""
    if pca_components_path.endswith('.npy'):
        pca_components = np.load(pca_components_path)
    else:
        pca_components = np.loadtxt(pca_components_path, delimiter=',', dtype=np.float16)
    
    if pca_mean_path.endswith('.npy'):
        pca_mean = np.load(pca_mean_path)
    else:
        pca_mean = np.loadtxt(pca_mean_path, delimiter=',', dtype=np.float16)
    
    if args.valid_n_components:
        pca_components = pca_components[args.valid_n_components]
    
    pca_model = PCAModel(
        pca_components_freeze=pca_components,
        pca_mean=pca_mean,
        device=device
    )
    return pca_model


def create_pipeline(vae, pca_model, residual_detail, device, args):
    """Create PCAPipeline with configurable predictor type"""
    pipeline = PCAPipeline(vae, pca_model, args.high_freq_enable, residual_detail, args.n_channels, device)
    
    return pipeline


def create_data_loaders(train_data_dir, eval_data_dir, train_batch_size, eval_batch_size, num_workers):
    """Create train and eval data loaders with configurable parameters"""
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


def setup_optimizers(pipeline, stage, stage1_pca_lr, stage1_residual_lr, stage2_lr, delta_lr):
    """Setup optimizers based on training stage"""
    if stage == 1:
        if pipeline.residual_detail:
            optimizer_for_z_pca = torch.optim.Adam([
                {'params': pipeline.pca_predictor.parameters(), 'lr': stage1_pca_lr},
                {'params': pipeline.residual_detail_predictor.parameters(), 'lr': stage1_residual_lr}
            ])
        else:
            optimizer_for_z_pca = torch.optim.Adam(
                pipeline.pca_predictor.parameters(), 
                lr=stage1_pca_lr
            )
    else:  # stage 2
        pipeline.pca_predictor.requires_grad_(False)
        optimizer_for_z_pca = torch.optim.Adam(
            pipeline.residual_detail_predictor.parameters(), 
            lr=stage2_lr
        )
    
    optimizer_for_delta = torch.optim.Adam(
        [pipeline.pca_model.pca_components_delta, pipeline.pca_model.pca_mean], 
        lr=delta_lr
    )
    
    return optimizer_for_z_pca, optimizer_for_delta


def setup_schedulers(optimizer_for_z_pca, optimizer_for_delta, stage, args):
    """Setup learning rate schedulers"""
    if stage == 1:
        scheduler_for_z_pca = torch.optim.lr_scheduler.StepLR(
            optimizer_for_z_pca, 
            step_size=args.stage1_pca_step_size, 
            gamma=args.stage1_pca_gamma
        )
        scheduler_for_delta = torch.optim.lr_scheduler.StepLR(
            optimizer_for_delta, 
            step_size=args.stage1_delta_step_size, 
            gamma=args.stage1_delta_gamma
        )
    else:  # stage 2
        scheduler_for_z_pca = torch.optim.lr_scheduler.StepLR(
            optimizer_for_z_pca, 
            step_size=args.stage2_pca_step_size, 
            gamma=args.stage2_pca_gamma
        )
        scheduler_for_delta = torch.optim.lr_scheduler.StepLR(
            optimizer_for_delta, 
            step_size=args.stage2_delta_step_size, 
            gamma=args.stage2_delta_gamma
        )
    
    return scheduler_for_z_pca, scheduler_for_delta


def train_pca_pipeline(args, vae, pca_model, train_loader, eval_loader, generator):
    """Main training loop with configurable parameters"""
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{args.log_dir}/pca_pipeline_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create pipeline
    pipeline = create_pipeline(vae, pca_model, args.residual_detail, args.device, args)
    
    # Freeze VAE
    pipeline.vae.requires_grad_(False)
    
    n_components = args.n_components
    n_channels = args.n_channels
    
    # Setup optimizers and schedulers
    optimizer_for_z_pca, optimizer_for_delta = setup_optimizers(
        pipeline, args.stage, args.stage1_pca_lr, args.stage1_residual_lr, args.stage2_lr, args.delta_lr
    )
    
    scheduler_for_z_pca, scheduler_for_delta = setup_schedulers(
        optimizer_for_z_pca, optimizer_for_delta, args.stage, args
    )
    
    # Loss functions
    criterion = nn.MSELoss()
    criterion_diff = WeightedL1Loss()
    
    # Training metrics
    train_metrics = {
        'losses': [],
        'learning_rates': [],
        'grad_norms': []
    }
    
    dtype = get_dtype(args.precision)
    global_step = 0
    
    # Main training loop
    for epoch in tqdm(range(args.epochs)):
        pipeline.pca_predictor.train()
        epoch_losses = {
            'total_loss': 0, 'pca_loss': 0, 'diff_loss': 0,
            'diff_dist_loss': 0, 'recon_latent_loss': 0, 'kl_loss': 0
        }
        
        for batch_idx, x in enumerate(train_loader):
            # Warmup learning rate
            if global_step < args.warmup_steps:
                if args.stage == 1:
                    for param_group in optimizer_for_z_pca.param_groups:
                        param_group['lr'] = min(args.stage1_pca_lr, args.stage1_pca_lr * (global_step + 1) / args.warmup_steps)
                elif args.stage == 2:
                    for param_group in optimizer_for_z_pca.param_groups:
                        param_group['lr'] = min(args.stage2_lr, args.stage2_lr * (global_step + 1) / args.warmup_steps)
                for param_group in optimizer_for_delta.param_groups:
                    param_group['lr'] = min(args.delta_lr, args.delta_lr * (global_step + 1) / args.warmup_steps)
            
            x = x.to(args.device).to(dtype)
            
            # Training step
            optimizer_for_z_pca.zero_grad()
            if hasattr(pipeline.pca_model, 'pca_components_delta'):
                optimizer_for_delta.zero_grad()
            
            with torch.no_grad():
                z_true = pipeline._encode_vae_image(x, generator)  # [batch, 16, H/8, W/8]
                
            z_pca_true = pipeline.pca_transform_batch(z_true, n_components=n_components, n_channels=n_channels)  # [batch, 3, H/8, W/8]
            z_pca_true = z_pca_true.to(dtype).to(pipeline.device)
            
            # PCA predictor forward pass
            z_pca_pred = pipeline.pca_predictor(x)
            
            # Calculate losses
            pca_loss = criterion_diff(z_pca_pred, z_pca_true)
            
            z_pred = pipeline.pca_inverse_transform_batch(z_pca_pred, n_components=n_components, n_channels=n_channels)  # [batch, 16, H/8, W/8]
            
            if args.residual_detail:
                diff_pred = pipeline.residual_detail_predictor(x)  # [batch, 16, H/8, W/8]
                diff_true = (z_true - z_pred).detach()
                diff_loss = criterion_diff(diff_pred, diff_true)
            else:
                diff_pred = torch.zeros_like(z_true)
                diff_loss = torch.tensor(0.0, device=args.device)
            
            recon_latent_loss = criterion_diff(z_pred + diff_pred.detach(), z_true)
            
            # Distribution difference loss
            mean_diff_pca = (z_pca_pred.mean(dim=(0, 2, 3)) - z_pca_true.mean(dim=(0, 2, 3)))**2
            std_diff_pca = (z_pca_pred.std(dim=(0, 2, 3)) - z_pca_true.std(dim=(0, 2, 3)))**2
            diff_dist_loss = (mean_diff_pca.mean() + std_diff_pca.mean())
            
            # KL divergence loss
            kl_loss = F.kl_div(z_pca_true.log_softmax(dim=1), z_pca_pred.softmax(dim=1), reduction='batchmean')
            
            # Total loss with configurable weights
            total_loss = (pca_loss +
                         args.diff_dist_weight * diff_dist_loss +
                         args.recon_latent_weight * recon_latent_loss +
                         args.kl_weight * kl_loss +
                         args.diff_weight * diff_loss)
            
            # Backward pass
            total_loss.backward()
            
            # Record gradient norm
            total_norm = 0
            for p in pipeline.pca_predictor.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            writer.add_scalar('Training/Gradient_Norm', total_norm, global_step)
            
            # Optimization step
            optimizer_for_z_pca.step()
            if hasattr(pipeline.pca_model, 'pca_components_delta'):
                optimizer_for_delta.step()
            
            # Record losses
            losses = {
                'total_loss': total_loss.item(),
                'pca_loss': pca_loss.item(),
                'diff_dist_loss': diff_dist_loss.item(),
                'diff_loss': diff_loss.item(),
                'recon_latent_loss': recon_latent_loss.item(),
                'kl_loss': kl_loss.item()
            }
            
            for k, v in losses.items():
                epoch_losses[k] += v
                writer.add_scalar(f'Batch/{k}', v, global_step)
            
            global_step += 1
        
        # Evaluation and logging
        if epoch % args.eval_frequency == 0:
            evaluate_and_log(pipeline, eval_loader, writer, epoch, args, dtype)
        
        # Learning rate scheduling
        scheduler_for_z_pca.step()
        scheduler_for_delta.step()
        current_lr = scheduler_for_z_pca.get_last_lr()[0]
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        
        # Record epoch statistics
        avg_losses = {k: v/len(train_loader) for k, v in epoch_losses.items()}
        for k, v in avg_losses.items():
            writer.add_scalar(f'Epoch/{k}', v, epoch)
        
        # Log model output statistics
        log_model_stats(pipeline, eval_loader, writer, epoch, args, dtype)
        
        print(f"Epoch {epoch}: {avg_losses}, LR: {current_lr:.2e}")
    
    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")
    
    return pipeline


def evaluate_and_log(pipeline, eval_loader, writer, epoch, args, dtype):
    """Evaluate model and log results to TensorBoard"""
    with torch.no_grad():
        x = next(iter(eval_loader)).to(args.device).to(dtype)
        
        # Generate reconstruction
        recon, z_pca_pred, x_recon = pipeline.generate_for_comparsion(x, generator=None, x_recon=True, n_components=args.n_components, n_channels=args.n_channels)
        diff_map = (x[:4] - recon[:4]).abs().float()
        
        # Log images
        writer.add_images('Input/x/Original', x[:4].float(), epoch)
        writer.add_images('Output/x/Ori_Reconstruction', x_recon[:4].float(), epoch)
        writer.add_images('Output/x/Reconstruction', recon[:4].float(), epoch)
        writer.add_images('Output/x/Difference/Space', diff_map, epoch)
        
        # Log gradient maps
        (grad_maps_x, grad_maps_x_fig), (grad_maps_recon, grad_maps_recon_fig) = rgb_grad_map(x[:4].float()), rgb_grad_map(recon[:4].float())
        writer.add_figure('Output/x/Gradient/Original', grad_maps_x_fig, epoch)
        writer.add_figure('Output/x/Gradient/Reconstruction', grad_maps_recon_fig, epoch)
        grad_comparison_x = rgb_grad_comparison(grad_maps_x, diff_map)
        writer.add_figure('Output/x/Gradient/Comparison', grad_comparison_x, epoch)
        grad_comparison_recon = rgb_grad_comparison(grad_maps_recon, diff_map)
        writer.add_figure('Output/x/Gradient/Comparison_Reconstruction', grad_comparison_recon, epoch)
        
        writer.add_scalar('Output/x/Difference/Mean', (x[:4] - recon[:4]).abs().mean().item(), epoch)
        writer.add_scalar('Output/x/Difference/Std', (x[:4] - recon[:4]).abs().std().item(), epoch)
        
        # Log frequency domain analysis
        original = x[:4].to(dtype)
        pca_reconstructed = recon[:4].to(dtype)
        ori_freq_fig = visualize_spectrum_comparison(original, pipeline.vae, pipeline.pca_model, n_components=args.n_components, n_channels=args.n_channels)
        recon_freq_fig = visualize_spectrum_comparison(pca_reconstructed, pipeline.vae, pipeline.pca_model, n_components=args.n_components, n_channels=args.n_channels)
        diff_freq_fig = visualize_spectrum_comparison((x[:4] - recon[:4]).abs(), pipeline.vae, pipeline.pca_model, n_components=args.n_components, n_channels=args.n_channels)
        writer.add_figure('Frequency/Original', ori_freq_fig, epoch)
        writer.add_figure('Frequency/Reconstruction', recon_freq_fig, epoch)
        writer.add_figure('Frequency/Difference', diff_freq_fig, epoch)
        
        # Log PCA parameters
        if hasattr(pipeline.pca_model, 'pca_components_delta'):
            delta = pipeline.pca_model.pca_components_delta
            writer.add_histogram('Parameters/pca_components_delta', delta, epoch)
            writer.add_histogram('Parameters/pca_mean', pipeline.pca_model.pca_mean, epoch)
        
        # Log gradients and parameters
        for name, param in pipeline.pca_predictor.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            writer.add_histogram(f'Parameters/{name}', param, epoch)
        
        if args.residual_detail:
            for name, param in pipeline.residual_detail_predictor.named_parameters():
                writer.add_histogram(f'Parameters/Residual_Detail_Parameters/{name}', param, epoch)


def log_model_stats(pipeline, eval_loader, writer, epoch, args, dtype):
    """Log model output statistics"""
    pipeline.pca_predictor.eval()
    
    with torch.no_grad():
        test_batch = next(iter(eval_loader)).to(args.device).to(dtype)
        
        z_pca_pred_test = pipeline.pca_predictor(test_batch)
        z_pred_test = pipeline.pca_inverse_transform_batch(z_pca_pred_test, n_components=args.n_components, n_channels=args.n_channels)
        
        z_true_test = pipeline._encode_vae_image(test_batch, generator=None)
        z_pca_true_test = pipeline.pca_transform_batch(z_true_test, n_components=args.n_components, n_channels=args.n_channels)
        
        # Log PCA output statistics
        writer.add_scalar('Stats/z_pca_pred_mean', z_pca_pred_test.mean().item(), epoch)
        writer.add_scalar('Stats/z_pca_pred_std', z_pca_pred_test.std().item(), epoch)
        writer.add_scalar('Stats/z_pca_true_mean', z_pca_true_test.mean().item(), epoch)
        writer.add_scalar('Stats/z_pca_true_std', z_pca_true_test.std().item(), epoch)
        writer.add_scalar('Stats/z_pca_diff_mean', (z_pca_pred_test.mean(dim=(0, 2, 3))-z_pca_true_test.mean(dim=(0, 2, 3))).mean().item(), epoch)
        writer.add_scalar('Stats/z_pca_diff_std', (z_pca_pred_test.std(dim=(0, 2, 3))-z_pca_true_test.std(dim=(0, 2, 3))).mean().item(), epoch)
        
        # Log reconstruction output statistics
        writer.add_scalar('Stats/z_pred_mean', z_pred_test.mean().item(), epoch)
        writer.add_scalar('Stats/z_pred_std', z_pred_test.std().item(), epoch)
        writer.add_scalar('Stats/z_true_mean', z_true_test.mean().item(), epoch)
        writer.add_scalar('Stats/z_true_std', z_true_test.std().item(), epoch)
        writer.add_scalar('Stats/z_diff_mean', (z_pred_test.mean(dim=(0, 2, 3))-z_true_test.mean(dim=(0, 2, 3))).mean().item(), epoch)
        writer.add_scalar('Stats/z_diff_std', (z_pred_test.std(dim=(0, 2, 3))-z_true_test.std(dim=(0, 2, 3))).mean().item(), epoch)


def save_config(args, save_path):
    """Save training configuration to JSON file"""
    config_dict = vars(args)
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
    
    # Load models
    logger.info(f"Loading VAE from: {args.model_path}")
    dtype = get_dtype(args.precision)
    vae = load_vae_model(args.model_path, cache_dir, args.device, dtype)
    
    logger.info(f"Loading PCA model from components: {args.pca_components_path}, mean: {args.pca_mean_path}")
    pca_model = load_pca_model(args.pca_components_path, args.pca_mean_path, args.device, args)
    
    # Create data loaders
    logger.info(f"Creating data loaders from train: {args.train_data_dir}, eval: {args.eval_data_dir}")
    train_loader, eval_loader = create_data_loaders(
        args.train_data_dir, args.eval_data_dir,
        args.train_batch_size, args.eval_batch_size, args.num_workers
    )
    
    # Setup generator
    generator = torch.manual_seed(42)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Stage 1 training
    logger.info(f"Starting Stage {args.stage} training with {args.epochs} epochs")
    pipeline = train_pca_pipeline(args, vae, pca_model, train_loader, eval_loader, generator)
    
    # Save checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = f"{args.save_dir}/pca_pipeline_stage{args.stage}_{timestamp}.pth"
    pipeline.save(ckpt_path)
    logger.info(f"Stage {args.stage} checkpoint saved to: {ckpt_path}")
    
    # Stage 2 training (if specified)
    if args.stage == 1:
        logger.info("Starting Stage 2 training")
        args.stage = 2
        args.epochs = 400  # Reset epochs for stage 2
        
        # Update configuration
        save_config(args, f"{args.config_save_path}.stage2")
        
        # Stage 2 training
        pipeline = train_pca_pipeline(args, vae, pca_model, train_loader, eval_loader, generator)
        
        # Save final checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_ckpt_path = f"{args.save_dir}/pca_pipeline_stage2_{timestamp}.pth"
        pipeline.save(final_ckpt_path)
        logger.info(f"Final checkpoint saved to: {final_ckpt_path}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()