#!/usr/bin/env python3
"""
eval_align.py - 独立的 VAE 对齐模型评估脚本

该脚本用于评估已训练的 VAE 对齐模型检查点，提供：
1. 定量指标计算（MSE, PSNR, SSIM, LPIPS, rFID）
2. 可视化对比（原始图像、重建图像、差异图）
3. 潜在空间分析
4. 批量评估和报告生成

使用方法：
python eval_align.py --checkpoint ckpt_align/align_pipeline_20251231_034701.pth \
                     --eval_data_dir eval_images \
                     --output_dir eval_results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from vae_align import AlignPipeline
from dataloader import image_dataloader

# HuggingFace imports
from huggingface_hub import login
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import evaluation metrics libraries
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips package not installed. LPIPS metric will be unavailable.")
    print("Install with: pip install lpips")

try:
    from cleanfid import fid
    CLEANFID_AVAILABLE = True
except ImportError:
    CLEANFID_AVAILABLE = False
    print("Warning: cleanfid package not installed. rFID metric will be unavailable.")
    print("Install with: pip install clean-fid")

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics package not installed. SSIM metric will use scikit-image.")
    print("Install with: pip install torchmetrics")

try:
    from skimage.metrics import structural_similarity as ssim_skimage
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image package not installed. SSIM metric will be unavailable.")
    print("Install with: pip install scikit-image")


class AlignVAEEvaluator:
    """
    用于评估VAE对齐模型重建质量的综合评估器
    复用VAEReconstructionEvaluator的功能，支持多模型并行评估
    """
    
    def __init__(self, device='cuda', save_dir='eval_temp/', prefix='align_', dtype=torch.bfloat16):
        # 导入基类
        from eval_class import VAEReconstructionEvaluator
        
        # 为每个模型创建独立的评估器实例
        self.evaluators = {}
        self.model_names = ['vae1', 'vae2', 'aligned']
        
        for model_name in self.model_names:
            model_prefix = f'{prefix}{model_name}_'
            evaluator = VAEReconstructionEvaluator(
                device=device,
                save_dir=save_dir,
                prefix=model_prefix,
                dtype=dtype
            )
            self.evaluators[model_name] = evaluator
        
        # 存储所有结果
        self.results = {
            'vae1': {'mse': [], 'psnr': [], 'ssim': [], 'lpips': []},
            'vae2': {'mse': [], 'psnr': [], 'ssim': [], 'lpips': []},
            'aligned': {'mse': [], 'psnr': [], 'ssim': [], 'lpips': []},
            'rfid': {}
        }
        
        # 用于rFID计算的临时目录（保持向后兼容）
        self.real_dir = os.path.join(save_dir, prefix + 'real/')
        self.recon_vae1_dir = os.path.join(save_dir, prefix + 'vae1_recon/')
        self.recon_vae2_dir = os.path.join(save_dir, prefix + 'vae2_recon/')
        self.recon_aligned_dir = os.path.join(save_dir, prefix + 'aligned_recon/')
        
        for dir_path in [self.real_dir, self.recon_vae1_dir, self.recon_vae2_dir, self.recon_aligned_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.device = device
        self.dtype = dtype
        self.image_counter = 0
        self.prefix = prefix
        self.save_dir = save_dir
    
    def evaluate_batch(self, original, recon_vae1, recon_vae2, recon_aligned, batch_idx, save_for_fid=True, max_fid_batches=10):
        """
        评估单个批次
        
        参数:
            original: 原始图像 [B, C, H, W], 值范围[0, 1]
            recon_vae1: VAE1重建图像 [B, C, H, W], 值范围[0, 1]
            recon_vae2: VAE2重建图像 [B, C, H, W], 值范围[0, 1]
            recon_aligned: 对齐重建图像 [B, C, H, W], 值范围[0, 1]
            batch_idx: 批次索引
            save_for_fid: 是否保存图像用于rFID计算
            max_fid_batches: 用于FID计算的最大批次数量
        """
        # 确保张量在正确设备上
        original = original.to(self.device).float()
        recon_vae1 = recon_vae1.to(self.device).float()
        recon_vae2 = recon_vae2.to(self.device).float()
        recon_aligned = recon_aligned.to(self.device).float()
        
        batch_results = {}
        
        # 使用各个评估器评估每个模型
        models = {
            'vae1': recon_vae1,
            'vae2': recon_vae2,
            'aligned': recon_aligned
        }
        
        for model_name, recon in models.items():
            # 使用基类评估器的 evaluate_batch 方法
            eval_result = self.evaluators[model_name].evaluate_batch(
                original, recon, batch_idx, 
                save_for_fid=False,  # 我们会在下面统一处理FID图像保存
                n_batches=max_fid_batches
            )
            
            # 保存结果
            self.results[model_name]['mse'].append(eval_result['mse'])
            self.results[model_name]['psnr'].append(eval_result['psnr'])
            self.results[model_name]['ssim'].append(eval_result['ssim'])
            self.results[model_name]['lpips'].append(eval_result['lpips'])
            
            batch_results[model_name] = eval_result
        
        # 保存图像用于rFID（仅对前几个批次，避免存储过大）
        if save_for_fid and batch_idx < max_fid_batches:
            self.save_images_for_fid(original, recon_vae1, recon_vae2, recon_aligned, batch_idx)
        
        # 打印当前批次结果
        print(f"批次 {batch_idx}:")
        for model_name in ['vae1', 'vae2', 'aligned']:
            res = batch_results[model_name]
            print(f"  {model_name:8s} - MSE={res['mse']:.6f}, PSNR={res['psnr']:.2f}dB, "
                  f"SSIM={res['ssim']:.4f}, LPIPS={res['lpips']:.4f}")
        
        return batch_results
    
    def save_images_for_fid(self, original, recon_vae1, recon_vae2, recon_aligned, batch_idx):
        """保存图像用于rFID计算"""
        batch_size = original.shape[0]
        
        for i in range(batch_size):
            idx = self.image_counter
            
            # 保存原始图像
            img_orig = original[i].permute(1, 2, 0).cpu().numpy()
            img_orig = np.clip(img_orig * 255, 0, 255).astype(np.uint8)
            Image.fromarray(img_orig).save(f'{self.real_dir}/img_{idx:05d}.png')
            
            # 保存VAE1重建图像
            img_vae1 = recon_vae1[i].permute(1, 2, 0).cpu().numpy()
            img_vae1 = np.clip(img_vae1 * 255, 0, 255).astype(np.uint8)
            Image.fromarray(img_vae1).save(f'{self.recon_vae1_dir}/img_{idx:05d}.png')
            
            # 保存VAE2重建图像
            img_vae2 = recon_vae2[i].permute(1, 2, 0).cpu().numpy()
            img_vae2 = np.clip(img_vae2 * 255, 0, 255).astype(np.uint8)
            Image.fromarray(img_vae2).save(f'{self.recon_vae2_dir}/img_{idx:05d}.png')
            
            # 保存对齐重建图像
            img_aligned = recon_aligned[i].permute(1, 2, 0).cpu().numpy()
            img_aligned = np.clip(img_aligned * 255, 0, 255).astype(np.uint8)
            Image.fromarray(img_aligned).save(f'{self.recon_aligned_dir}/img_{idx:05d}.png')
            
            self.image_counter += 1
    
    def calculate_rfid(self):
        """计算重建FID（rFID）"""
        if not CLEANFID_AVAILABLE:
            print("Warning: cleanfid package not available. rFID cannot be calculated.")
            return None
        
        try:
            rfid_scores = {}
            
            # 计算VAE1的rFID
            rfid_vae1 = fid.compute_fid(self.real_dir, self.recon_vae1_dir, mode="legacy_pytorch")
            rfid_scores['vae1'] = rfid_vae1
            
            # 计算VAE2的rFID
            rfid_vae2 = fid.compute_fid(self.real_dir, self.recon_vae2_dir, mode="legacy_pytorch")
            rfid_scores['vae2'] = rfid_vae2
            
            # 计算对齐模型的rFID
            rfid_aligned = fid.compute_fid(self.real_dir, self.recon_aligned_dir, mode="legacy_pytorch")
            rfid_scores['aligned'] = rfid_aligned
            
            return rfid_scores
        except Exception as e:
            print(f"计算rFID时出错: {e}")
            return None
    
    def get_final_results(self, compute_rfid=True):
        """获取最终评估结果"""
        final_results = {}
        
        # 计算每个模型的平均值和标准差
        for model_name in ['vae1', 'vae2', 'aligned']:
            final_results[model_name] = {}
            for metric in ['mse', 'psnr', 'ssim', 'lpips']:
                values = self.results[model_name][metric]
                if values:  # 检查列表是否非空
                    final_results[model_name][f'{metric}_mean'] = np.mean(values).item()
                    final_results[model_name][f'{metric}_std'] = np.std(values).item()
                else:
                    final_results[model_name][f'{metric}_mean'] = None
                    final_results[model_name][f'{metric}_std'] = None
        
        # 计算rFID（如果需要）
        if compute_rfid and CLEANFID_AVAILABLE:
            print("正在计算rFID，这可能需要一些时间...")
            rfid_scores = self.calculate_rfid()
            if rfid_scores:
                final_results['rfid'] = rfid_scores
                self.results['rfid'] = rfid_scores
        
        return final_results
    
    def print_summary(self):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("VAE对齐模型评估摘要")
        print("="*60)
        
        final = self.get_final_results(compute_rfid=False)
        
        # 打印每个模型的指标
        for model_name in ['vae1', 'vae2', 'aligned']:
            print(f"\n{model_name.upper():8s}:")
            print("-"*40)
            
            metrics = ['mse', 'psnr', 'ssim', 'lpips']
            for metric in metrics:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                
                if final[model_name][mean_key] is not None:
                    if metric == 'psnr':
                        print(f"  {metric.upper():6s}: {final[model_name][mean_key]:.2f} ± {final[model_name][std_key]:.2f} dB")
                    else:
                        print(f"  {metric.upper():6s}: {final[model_name][mean_key]:.6f} ± {final[model_name][std_key]:.6f}")
        
        # 打印rFID结果
        if 'rfid' in self.results and self.results['rfid']:
            print("\nrFID Scores:")
            print("-"*40)
            for model_name, score in self.results['rfid'].items():
                print(f"  {model_name:8s}: {score:.2f}")
        
        print("="*60)
    
    def cleanup(self):
        """清理临时文件"""
        import shutil
        dirs_to_clean = [self.real_dir, self.recon_vae1_dir, self.recon_vae2_dir, self.recon_aligned_dir]
        
        for dir_path in dirs_to_clean:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"Cleaned up directory: {dir_path}")
        
        # 同时清理各个评估器的临时文件
        for evaluator in self.evaluators.values():
            evaluator.cleanup()


def parse_args():
    """Parse command line arguments for evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate VAE Alignment Pipeline")
    
    # Model Configuration
    parser.add_argument('--checkpoint', type=str, required=True,
                       help="Path to the trained alignment checkpoint (.pth file)")
    parser.add_argument('--vae1_path', type=str, default="sd-legacy/stable-diffusion-v1-5",
                       help="Path to the first VAE model")
    parser.add_argument('--vae2_path', type=str, default="black-forest-labs/FLUX.1-dev",
                       help="Path to the second VAE model")
    parser.add_argument('--vae1_subfolder', type=str, default="vae",
                       help="Subfolder for VAE1 model")
    parser.add_argument('--vae2_subfolder', type=str, default="vae",
                       help="Subfolder for VAE2 model")
    parser.add_argument('--config_load', type=str, default=None,
                       help="Path to a configuration file to load additional parameters")
    parser.add_argument('--image_normalize', action='store_true', default=False,
                       help="Whether to normalize images to [0, 1] range")
    parser.add_argument('--sample_mode', type=str, default="sample",
                       choices=["sample", "mean"],
                       help="Sampling mode for VAE: 'sample' or 'mean'")
    
    # Data Configuration
    parser.add_argument('--eval_data_dir', type=str, required=True,
                       help="Directory containing evaluation images")
    parser.add_argument('--eval_batch_size', type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument('--num_workers', type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument('--dataset_type', type=str, default="imagenet",
                       choices=["default", "imagenet"],
                       help="Type of dataset to use")
    parser.add_argument('--num_samples', type=int, default=10,
                       help="Number of samples to evaluate")
    
    # Evaluation Configuration
    parser.add_argument('--metrics', type=str, nargs='+', 
                       default=['mse', 'psnr', 'ssim', 'lpips'],
                       choices=['mse', 'psnr', 'ssim', 'lpips', 'rfid', 'all'],
                       help="Metrics to compute")
    parser.add_argument('--compute_rfid', action='store_true',
                       help="Compute rFID metric (requires cleanfid package)")
    parser.add_argument('--max_fid_batches', type=int, default=10,
                       help="Maximum number of batches to use for rFID calculation")
    parser.add_argument('--batch_evaluation', action='store_true',
                       help="Use batch evaluation instead of sample-by-sample")
    parser.add_argument('--device', type=str, default="cuda",
                       help="Device to use for evaluation")
    parser.add_argument('--precision', type=str, default="bfloat16", 
                       choices=["float32", "bfloat16", "float16"],
                       help="Precision for evaluation")
    
    # Output Configuration
    parser.add_argument('--output_dir', type=str, default="eval_results",
                       help="Directory to save evaluation results")
    parser.add_argument('--save_visualizations', action='store_true',
                       help="Save visualization images")
    parser.add_argument('--verbose', action='store_true',
                       help="Enable verbose logging")
    
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
    vae.eval()
    return vae


def create_data_loader(eval_data_dir, eval_batch_size, num_workers, dataset_type, num_samples=10):
    """Create evaluation data loader"""
    if dataset_type == "imagenet":
        from dataloader import ImageNetDataloader, get_imagenet_dataset
        eval_dataset = get_imagenet_dataset(split="test")
        # Convert to list and take subset
        # indices = list(range(min(num_samples, len(eval_dataset))))
        # eval_dataset = torch.utils.data.Subset(eval_dataset, indices)
        eval_loader = ImageNetDataloader(eval_dataset, batch_size=eval_batch_size, 
                                        shuffle=False, num_workers=num_workers)
    else:
        # For default dataset, we'll load and limit samples
        eval_loader = image_dataloader(
            data_dir=eval_data_dir,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        # We'll limit during iteration
    
    return eval_loader


def compute_metrics(original, reconstructed, metrics_list, device='cuda'):
    """Compute various image quality metrics"""
    results = {}
    
    # Ensure tensors are on correct device and dtype
    original = original.to(device).float()
    reconstructed = reconstructed.to(device).float()
    
    # Normalize to [0, 1] if needed
    if original.min() < 0 or original.max() > 1:
        original = (original + 1) / 2
        reconstructed = (reconstructed + 1) / 2
    
    # MSE
    if 'mse' in metrics_list or 'all' in metrics_list:
        mse = F.mse_loss(original, reconstructed)
        results['mse'] = mse.item()
    
    # PSNR
    if 'psnr' in metrics_list or 'all' in metrics_list:
        mse = F.mse_loss(original, reconstructed)
        if mse > 0:
            psnr_val = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)
            results['psnr'] = psnr_val.item()
        else:
            results['psnr'] = float('inf')
    
    return results


def visualize_comparison(original, recon_vae1, recon_vae2, recon_aligned, 
                        save_path, sample_idx=0):
    """Create visualization comparison image"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Convert tensors to numpy for visualization
    def tensor_to_numpy(img_tensor):
        img = img_tensor.float().squeeze().cpu().detach().numpy()
        if img.shape[0] == 3:  # CHW to HWC
            img = np.transpose(img, (1, 2, 0))
        # Clip to valid range
        img = np.clip(img, 0, 1)
        return img
    
    titles = ['Original', 'VAE1 Recon', 'VAE2 Recon', 'Aligned Recon',
              'Diff VAE1', 'Diff VAE2', 'Diff Aligned', 'Latent Stats']
    
    # Row 1: Images
    images = [original, recon_vae1, recon_vae2, recon_aligned]
    for idx, (ax, img, title) in enumerate(zip(axes[0], images, titles[:4])):
        img_np = tensor_to_numpy(img)
        ax.imshow(img_np)
        ax.set_title(title)
        ax.axis('off')
    
    # Row 2: Differences and stats
    # Difference maps
    diff_vae1 = torch.abs(original - recon_vae1)
    diff_vae2 = torch.abs(original - recon_vae2)
    diff_aligned = torch.abs(original - recon_aligned)
    
    diffs = [diff_vae1, diff_vae2, diff_aligned]
    for idx, (ax, diff, title) in enumerate(zip(axes[1][:3], diffs, titles[4:7])):
        diff_np = tensor_to_numpy(diff)
        # Normalize for visualization
        if diff_np.max() > 0:
            diff_np = diff_np / diff_np.max()
        ax.imshow(diff_np, cmap='hot')
        ax.set_title(title)
        ax.axis('off')
    
    # Statistics text
    mse_vae1 = F.mse_loss(original, recon_vae1).item()
    mse_vae2 = F.mse_loss(original, recon_vae2).item()
    mse_aligned = F.mse_loss(original, recon_aligned).item()
    
    stats_text = f'MSE:\nVAE1: {mse_vae1:.4f}\nVAE2: {mse_vae2:.4f}\nAligned: {mse_aligned:.4f}'
    axes[1][3].text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10)
    axes[1][3].set_title(titles[7])
    axes[1][3].axis('off')
    
    plt.suptitle(f'Sample {sample_idx}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """Main evaluation function using AlignVAEEvaluator"""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Create alignment pipeline
    pipeline = AlignPipeline(
        VAE_1=vae1,
        VAE_2=vae2,
        img_in_channels=3,
        in_channels=4,
        hidden_channels=32,
        out_channels=16,
        num_blocks=2,
        downsample_times=3,
        input_types=['image', 'latent'],
        device=args.device,
        dtype=dtype
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    pipeline.load(args.checkpoint)
    pipeline.freeze_vae()
    
    # Create data loader
    logger.info(f"Creating data loader from: {args.eval_data_dir}")
    eval_loader = create_data_loader(
        args.eval_data_dir, args.eval_batch_size, args.num_workers,
        args.dataset_type, args.num_samples
    )
    
    # Initialize evaluator
    temp_dir = os.path.join(args.output_dir, 'temp_fid')
    evaluator = AlignVAEEvaluator(
        device=args.device,
        save_dir=temp_dir,
        prefix='align_eval_',
        dtype=dtype
    )
    
    # Evaluation loop
    logger.info(f"Starting evaluation of {args.num_samples} samples...")
    pipeline.align_module.eval()
    
    sample_count = 0
    batch_count = 0
    
    sample_mode = args.sample_mode
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if sample_count >= args.num_samples:
                break
                
            x = batch['pixel_values'].to(args.device).to(dtype)
            batch_size = x.shape[0]
            
            # Adjust batch size if we're near the sample limit
            if sample_count + batch_size > args.num_samples:
                x = x[:args.num_samples - sample_count]
                batch_size = x.shape[0]
            
            if args.image_normalize:
                # Normalize to [-1, 1] range for VAE processing
                x_normalized = x * 2 - 1
            else:
                x_normalized = x
            
            # Get reconstructions
            recon_vae1 = pipeline.latent_reconstruction(pipeline.VAE_1,
                                                        x_normalized, 
                                                        generator=None, 
                                                        do_normalize=args.image_normalize, 
                                                        sample_mode=sample_mode)
            
            recon_vae2 = pipeline.latent_reconstruction(pipeline.VAE_2, 
                                                        x_normalized, 
                                                        generator=None, 
                                                        do_normalize=args.image_normalize, 
                                                        sample_mode=sample_mode)
            
            # Get latent representations
            z_vae1 = pipeline._encode_vae_image(pipeline.VAE_1, x_normalized, generator=None, sample_mode=sample_mode)
            z_vae2 = pipeline._encode_vae_image(pipeline.VAE_2, x_normalized, generator=None, sample_mode=sample_mode)
            
            # Get aligned latent and reconstruction
            z_vae1_aligned = pipeline.align_module(x_normalized, z_vae1)
            recon_aligned = pipeline._decode_vae_latents(pipeline.VAE_2, z_vae1_aligned, do_normalize=args.image_normalize)
            
            # Ensure images are in [0, 1] range for evaluation
            if args.image_normalize:
                x_eval = (x_normalized + 1) / 2
            else:
                x_eval = x
            
            # # Normalize reconstructions to [0, 1] if needed
            # if recon_vae1.min() < 0 or recon_vae1.max() > 1:
            #     recon_vae1 = (recon_vae1 + 1) / 2
            # if recon_vae2.min() < 0 or recon_vae2.max() > 1:
            #     recon_vae2 = (recon_vae2 + 1) / 2
            # if recon_aligned.min() < 0 or recon_aligned.max() > 1:
            #     recon_aligned = (recon_aligned + 1) / 2
            
            # Evaluate batch using AlignEvaluator
            save_for_fid = args.compute_rfid and ('rfid' in args.metrics or 'all' in args.metrics)
            batch_results = evaluator.evaluate_batch(
                x_eval, recon_vae1, recon_vae2, recon_aligned,
                batch_idx=batch_count,
                save_for_fid=save_for_fid,
                max_fid_batches=args.max_fid_batches
            )
            
            # Save visualizations if requested
            if args.save_visualizations:
                for i in range(min(batch_size, 5)):  # Save at most 5 samples per batch
                    if sample_count + i < args.num_samples:
                        vis_path = os.path.join(args.output_dir, f'comparison_sample_{sample_count + i}.png')
                        visualize_comparison(
                            x_eval[i:i+1], 
                            recon_vae1[i:i+1], 
                            recon_vae2[i:i+1], 
                            recon_aligned[i:i+1],
                            vis_path, 
                            sample_count + i
                        )
            
            sample_count += batch_size
            batch_count += 1
            logger.info(f"Processed batch {batch_count}: {sample_count}/{args.num_samples} samples")
            
            if sample_count >= args.num_samples:
                break
    
    # Get final results
    compute_rfid_flag = args.compute_rfid and ('rfid' in args.metrics or 'all' in args.metrics)
    final_results = evaluator.get_final_results(compute_rfid=compute_rfid_flag)
    
    # Add metadata to results
    final_results['metadata'] = {
        'config': vars(args),
        'timestamp': datetime.now().isoformat(),
        'num_samples': sample_count,
        'num_batches': batch_count
    }
    
    # Save results
    results_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Print summary
    evaluator.print_summary()
    
    # Cleanup temporary files
    if compute_rfid_flag:
        evaluator.cleanup()
    
    # Print improvement analysis
    if 'vae1' in final_results and 'aligned' in final_results:
        if final_results['vae1'].get('mse_mean') and final_results['aligned'].get('mse_mean'):
            mse_improvement = ((final_results['vae1']['mse_mean'] - final_results['aligned']['mse_mean']) / 
                              final_results['vae1']['mse_mean'] * 100)
            logger.info(f"\nMSE Improvement from VAE1 to Aligned: {mse_improvement:.1f}%")
    
    logger.info(f"\nResults saved to: {results_path}")
    if args.save_visualizations:
        logger.info(f"Visualizations saved to: {args.output_dir}")
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
