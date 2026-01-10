#!/usr/bin/env python3
"""
infer_align_vis.py - VAE对齐模块的latent分布可视化分析脚本

功能：
1. 从ImageNet数据集中随机取一张256×256图片
2. 下采样到32×32，再分别用双线性插值和最近邻插值上采样回256×256
3. 将三张图片分别送入VAE1、VAE2和对齐模块得到对应的latent
4. 对这些latent进行可视化分析和统计分析

输出：
- 图像文件：原始图像、采样图像、latent可视化、统计图表
- 数据文件：JSON格式的统计报告
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import json
from datetime import datetime
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
from vae_align import AlignPipeline
from dataloader import get_imagenet_dataset, ImageNetDataloader
from utils import plot_images, plot_figure, pca_visualize_and_save, rgb_color_gradient

# HuggingFace imports
from huggingface_hub import login
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="VAE对齐模块的latent分布可视化分析")
    
    # 模型配置
    parser.add_argument('--vae1_path', type=str, default="sd-legacy/stable-diffusion-v1-5",
                       help="VAE1模型路径，默认: sd-legacy/stable-diffusion-v1-5")
    parser.add_argument('--vae2_path', type=str, default="black-forest-labs/FLUX.1-dev",
                       help="VAE2模型路径，默认: black-forest-labs/FLUX.1-dev")
    parser.add_argument('--vae1_subfolder', type=str, default="vae",
                       help="VAE1子文件夹，默认: vae")
    parser.add_argument('--vae2_subfolder', type=str, default="vae",
                       help="VAE2子文件夹，默认: vae")
    parser.add_argument('--align_ckpt', type=str, default="ckpt_align/align_pipeline_20260109_004958.pth",
                       help="对齐模块检查点路径，默认: ckpt_align/align_pipeline_20260109_004958.pth")

    # 数据配置
    parser.add_argument('--dataset_split', type=str, default="test",
                       choices=["train", "test", "validation"],
                       help="ImageNet数据集分割，默认: test")
    parser.add_argument('--sample_index', type=int, default=None,
                       help="指定样本索引，如果为None则随机选择")
    parser.add_argument('--random_seed', type=int, default=42,
                       help="随机种子，默认: 42")
    
    # 处理配置
    parser.add_argument('--downsample_size', type=int, default=32,
                       help="下采样尺寸，默认: 32")
    parser.add_argument('--upsample_size', type=int, default=256,
                       help="上采样尺寸，默认: 256")
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default="infer_align_vis_results",
                       help="输出目录，默认: infer_align_vis_results")
    parser.add_argument('--save_latents', action='store_true',
                       help="是否保存latent数据为numpy文件")
    parser.add_argument('--device', type=str, default="cuda",
                       help="设备，默认: cuda")
    parser.add_argument('--precision', type=str, default="bfloat16",
                       choices=["float32", "bfloat16", "float16"],
                       help="精度，默认: bfloat16")
    
    return parser.parse_args()


def get_dtype(precision: str):
    """获取torch dtype"""
    if precision == "bfloat16":
        return torch.bfloat16
    elif precision == "float16":
        return torch.float16
    else:
        return torch.float32


def load_vae_model(model_path: str, subfolder: str, device: str, dtype: torch.dtype):
    """加载VAE模型"""
    print(f"加载VAE模型: {model_path}")
    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder=subfolder,
        torch_dtype=dtype,
        cache_dir=os.getenv("HF_CACHE_DIR", ""),
        proxies={'http': '127.0.0.1:7890'} if os.getenv("USE_PROXY", "0") == "1" else None
    )
    vae.to(device)
    vae.eval()
    return vae


def load_align_pipeline(vae1, vae2, align_ckpt: str, args):
    """加载对齐管道"""
    print(f"加载对齐管道，检查点: {align_ckpt}")
    
    # 创建对齐管道
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
        dtype=get_dtype(args.precision)
    )
    
    # 冻结VAE模型
    pipeline.freeze_vae()
    
    # 加载对齐模块权重
    if os.path.exists(align_ckpt):
        pipeline.load(align_ckpt)
        print(f"成功加载对齐模块权重: {align_ckpt}")
    else:
        print(f"警告: 对齐模块检查点不存在: {align_ckpt}")
        print("将使用随机初始化的对齐模块")
    
    return pipeline


def get_random_image_from_imagenet(split: str = "test", index: int = None, seed: int = 42):
    """从ImageNet数据集中获取随机图像"""
    print(f"从ImageNet {split}分割中加载图像...")
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 获取数据集
    dataset = get_imagenet_dataset(split=split)
    
    # 选择索引
    if index is None:
        index = np.random.randint(0, len(dataset))
    
    print(f"选择图像索引: {index}")
    
    # 获取图像
    item = dataset[index]
    image = item['image']
    
    # 转换为RGB并调整大小到256×256
    image = image.convert('RGB')
    image = image.resize((256, 256), Image.LANCZOS)
    
    # 转换为numpy数组并归一化到[0, 1]
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # 转换为torch tensor [C, H, W]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    
    return image_tensor, index


def create_image_variants(image_tensor: torch.Tensor, downsample_size: int = 32, upsample_size: int = 256):
    """创建图像变体：原始、双线性插值、最近邻插值"""
    print(f"创建图像变体: 下采样到{downsample_size}×{downsample_size}, 上采样到{upsample_size}×{upsample_size}")
    
    # 确保图像在[0, 1]范围
    if image_tensor.max() > 1.0:
        image_tensor = image_tensor / 255.0
    
    # 原始图像
    original = image_tensor.clone()
    
    # 下采样
    downsampled = F.interpolate(
        original.unsqueeze(0),
        size=(downsample_size, downsample_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    
    # 双线性插值上采样
    bilinear = F.interpolate(
        downsampled.unsqueeze(0),
        size=(upsample_size, upsample_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    
    # 最近邻插值上采样
    nearest = F.interpolate(
        downsampled.unsqueeze(0),
        size=(upsample_size, upsample_size),
        mode='nearest'
    ).squeeze(0)
    
    return {
        'original': original,
        'bilinear': bilinear,
        'nearest': nearest,
        'downsampled': downsampled
    }


def encode_images(pipeline, images_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype):
    """编码图像获取latent"""
    print("编码图像获取latent...")
    
    latents_dict = {}
    
    with torch.no_grad():
        for name, image in images_dict.items():
            # 跳过下采样图像
            if name == 'downsampled':
                continue
                
            # 准备图像 [1, C, H, W]
            img_batch = image.unsqueeze(0).to(device).to(dtype)
            
            # 归一化到[-1, 1]范围（与训练一致）
            img_batch_norm = img_batch * 2 - 1
            
            # VAE1编码
            z_vae1 = pipeline._encode_vae_image(pipeline.VAE_1, img_batch_norm, generator=None, sample_mode='mean')
            
            # VAE2编码
            z_vae2 = pipeline._encode_vae_image(pipeline.VAE_2, img_batch_norm, generator=None, sample_mode='mean')
            
            # 对齐模块
            z_aligned = pipeline.align_module(img_batch_norm, z_vae1)
            
            latents_dict[name] = {
                'z_vae1': z_vae1.cpu(),
                'z_vae2': z_vae2.cpu(),
                'z_aligned': z_aligned.cpu()
            }
            
            print(f"  {name}: z_vae1 shape={z_vae1.shape}, z_vae2 shape={z_vae2.shape}, z_aligned shape={z_aligned.shape}")
    
    return latents_dict


def visualize_images(images_dict: Dict[str, torch.Tensor], save_dir: str):
    """可视化原始图像和变体"""
    print("可视化图像...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备图像列表
    image_list = []
    title_list = []
    
    for name, image in images_dict.items():
        image_list.append(image)
        title_list.append(name.capitalize())
    
    # 创建子图
    fig, axes = plt.subplots(1, len(image_list), figsize=(5*len(image_list), 5))
    
    if len(image_list) == 1:
        axes = [axes]
    
    for idx, (img, title, ax) in enumerate(zip(image_list, title_list, axes)):
        # 转换为[H, W, C]格式
        if img.dim() == 3:
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = img.numpy()
        
        # 确保值在[0, 1]范围
        img_np = np.clip(img_np, 0, 1)
        
        ax.imshow(img_np)
        ax.set_title(f"{title} ({img.shape[1]}×{img.shape[2]})")
        ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "image_variants.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图像可视化保存到: {save_path}")
    
    # 单独保存每个图像
    for name, image in images_dict.items():
        if name == 'downsampled':
            continue
            
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        img_np = image.permute(1, 2, 0).numpy() if image.dim() == 3 else image.numpy()
        img_np = np.clip(img_np, 0, 1)
        
        ax.imshow(img_np)
        ax.set_title(f"{name.capitalize()} Image")
        ax.axis('off')
        
        save_path = os.path.join(save_dir, f"image_{name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def visualize_latent_channels(latent: torch.Tensor, title: str, save_path: str, max_channels: int = 3):
    """可视化latent的前几个通道"""
    # latent形状: [1, C, H, W]
    
    latent = latent.float()
    if latent.dim() == 4:
        latent = latent.squeeze(0)  # [C, H, W]
    
    # 限制通道数
    n_channels = min(latent.shape[0], max_channels)
    
    fig, axes = plt.subplots(1, n_channels, figsize=(4*n_channels, 4))
    
    if n_channels == 1:
        axes = [axes]
    
    for i in range(n_channels):
        channel = latent[i].numpy()
        
        # 归一化每个通道以便可视化
        channel_min = channel.min()
        channel_max = channel.max()
        if channel_max > channel_min:
            channel_norm = (channel - channel_min) / (channel_max - channel_min)
        else:
            channel_norm = channel
        
        im = axes[i].imshow(channel_norm, cmap='viridis')
        axes[i].set_title(f"Channel {i}")
        axes[i].axis('off')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"{title} (First {n_channels} Channels)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_latent_distributions(latents_dict: Dict, save_dir: str):
    """可视化latent分布"""
    print("可视化latent分布...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个图像变体创建分布图
    for img_name, latents in latents_dict.items():
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        model_names = ['VAE1', 'VAE2', 'Aligned']
        latent_keys = ['z_vae1', 'z_vae2', 'z_aligned']
        
        for row, (model_name, latent_key) in enumerate(zip(model_names, latent_keys)):
            latent = latents[latent_key]  # [1, C, H, W]
            
            if latent.dim() == 4:
                latent_flat = latent.view(-1).float().numpy()
            else:
                latent_flat = latent.float().numpy().flatten()
            
            # 直方图
            axes[row, 0].hist(latent_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[row, 0].set_title(f'{model_name} - Histogram')
            axes[row, 0].set_xlabel('Value')
            axes[row, 0].set_ylabel('Frequency')
            axes[row, 0].grid(True, alpha=0.3)
            
            # 箱线图
            axes[row, 1].boxplot(latent_flat, vert=True)
            axes[row, 1].set_title(f'{model_name} - Box Plot')
            axes[row, 1].set_ylabel('Value')
            axes[row, 1].grid(True, alpha=0.3)
            
            # 密度图
            sns.kdeplot(latent_flat, ax=axes[row, 2], fill=True, color='orange', alpha=0.5)
            axes[row, 2].set_title(f'{model_name} - Density Plot')
            axes[row, 2].set_xlabel('Value')
            axes[row, 2].set_ylabel('Density')
            axes[row, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Latent Distributions - {img_name.capitalize()} Image', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"latent_distributions_{img_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  {img_name}分布图保存到: {save_path}")
        
        # 单独可视化每个latent的前几个通道
        for latent_key, latent in latents.items():
            model_name = latent_key.replace('z_', '').capitalize()
            save_path = os.path.join(save_dir, f"latent_channels_{img_name}_{model_name}.png")
            visualize_latent_channels(latent, f"{img_name.capitalize()} - {model_name}", save_path)


def compute_statistics(latents_dict: Dict) -> Dict:
    """计算latent统计信息"""
    print("计算统计信息...")
    
    stats_dict = {}
    
    for img_name, latents in latents_dict.items():
        stats_dict[img_name] = {}
        
        for latent_key, latent in latents.items():
            latent = latent.float()  # 确保是float类型
            if latent.dim() == 4:
                latent_flat = latent.view(-1).numpy()
            else:
                latent_flat = latent.numpy().flatten()
            
            # 计算基本统计量
            stats = {
                'mean': float(np.mean(latent_flat)),
                'std': float(np.std(latent_flat)),
                'min': float(np.min(latent_flat)),
                'max': float(np.max(latent_flat)),
                'median': float(np.median(latent_flat)),
                'q1': float(np.percentile(latent_flat, 25)),
                'q3': float(np.percentile(latent_flat, 75)),
                'skewness': float(float(np.mean((latent_flat - np.mean(latent_flat))**3) / (np.std(latent_flat)**3)) if np.std(latent_flat) > 0 else 0),
                'kurtosis': float(float(np.mean((latent_flat - np.mean(latent_flat))**4) / (np.std(latent_flat)**4)) if np.std(latent_flat) > 0 else 0),
                'shape': latent.shape if isinstance(latent, torch.Tensor) else latent.shape
            }
            
            stats_dict[img_name][latent_key] = stats
    
    return stats_dict


def compute_comparison_metrics(latents_dict: Dict) -> Dict:
    """计算latent之间的比较指标"""
    print("计算比较指标...")
    
    comparison_dict = {}
    
    # 对于每个图像变体
    for img_name, latents in latents_dict.items():
        comparison_dict[img_name] = {}
        
        # 计算不同模型之间的MSE
        z_vae1 = latents['z_vae1']
        z_vae2 = latents['z_vae2']
        z_aligned = latents['z_aligned']
        
        # 展平以便计算
        if z_vae1.dim() == 4:
            z_vae1_flat = z_vae1.view(1, -1)
            z_vae2_flat = z_vae2.view(1, -1)
            z_aligned_flat = z_aligned.view(1, -1)
        else:
            z_vae1_flat = z_vae1.flatten().unsqueeze(0)
            z_vae2_flat = z_vae2.flatten().unsqueeze(0)
            z_aligned_flat = z_aligned.flatten().unsqueeze(0)
        
        mse_vae2_aligned = F.mse_loss(z_vae2_flat, z_aligned_flat).item()
        
        # 余弦相似度
        cos_sim_vae2_aligned = F.cosine_similarity(z_vae2_flat, z_aligned_flat).mean().item()
        
        comparison_dict[img_name] = {
            'mse_vae2_vs_aligned': mse_vae2_aligned,
            'cosine_sim_vae2_vs_aligned': cos_sim_vae2_aligned
        }
    
    # 计算不同图像变体之间的比较
    if len(latents_dict) > 1:
        comparison_dict['cross_image_comparison'] = {}
        
        image_names = list(latents_dict.keys())
        
        for i, img_name1 in enumerate(image_names):
            for j, img_name2 in enumerate(image_names[i+1:], i+1):
                key = f"{img_name1}_vs_{img_name2}"
                comparison_dict['cross_image_comparison'][key] = {}
                
                for latent_key in ['z_vae1', 'z_vae2', 'z_aligned']:
                    latent1 = latents_dict[img_name1][latent_key]
                    latent2 = latents_dict[img_name2][latent_key]
                    
                    if latent1.dim() == 4:
                        latent1_flat = latent1.view(1, -1)
                        latent2_flat = latent2.view(1, -1)
                    else:
                        latent1_flat = latent1.flatten().unsqueeze(0)
                        latent2_flat = latent2.flatten().unsqueeze(0)
                    
                    mse = F.mse_loss(latent1_flat, latent2_flat).item()
                    cos_sim = F.cosine_similarity(latent1_flat, latent2_flat).mean().item()
                    
                    comparison_dict['cross_image_comparison'][key][f"{latent_key}_mse"] = mse
                    comparison_dict['cross_image_comparison'][key][f"{latent_key}_cosine_sim"] = cos_sim
    
    return comparison_dict


def visualize_comparison_metrics(comparison_dict: Dict, save_dir: str):
    """可视化比较指标"""
    print("可视化比较指标...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个图像变体创建比较图
    for img_name, metrics in comparison_dict.items():
        if img_name == 'cross_image_comparison':
            continue
            
        # MSE比较
        mse_metrics = {
            'VAE2 vs Aligned': metrics['mse_vae2_vs_aligned']
        }
        
        # 余弦相似度比较
        cos_sim_metrics = {
            'VAE2 vs Aligned': metrics['cosine_sim_vae2_vs_aligned']
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # MSE柱状图
        bars1 = axes[0].bar(mse_metrics.keys(), mse_metrics.values(), color=['red', 'blue', 'green'])
        axes[0].set_title(f'MSE Comparison - {img_name.capitalize()}')
        axes[0].set_ylabel('MSE')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加数值
        for bar, value in zip(bars1, mse_metrics.values()):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_metrics.values())*0.01,
                        f'{value:.6f}', ha='center', va='bottom', fontsize=9)
        
        # 余弦相似度柱状图
        bars2 = axes[1].bar(cos_sim_metrics.keys(), cos_sim_metrics.values(), color=['red', 'blue', 'green'])
        axes[1].set_title(f'Cosine Similarity - {img_name.capitalize()}')
        axes[1].set_ylabel('Cosine Similarity')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加数值
        for bar, value in zip(bars2, cos_sim_metrics.values()):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cos_sim_metrics.values())*0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"comparison_metrics_{img_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  比较指标图保存到: {save_path}")
    
    # 可视化跨图像比较
    if 'cross_image_comparison' in comparison_dict:
        cross_metrics = comparison_dict['cross_image_comparison']
        
        if cross_metrics:
            # 创建热图数据
            comparison_pairs = list(cross_metrics.keys())
            metric_types = ['z_vae1_mse', 'z_vae2_mse', 'z_aligned_mse']
            
            heatmap_data = []
            for pair in comparison_pairs:
                row = []
                for metric in metric_types:
                    if metric in cross_metrics[pair]:
                        row.append(cross_metrics[pair][metric])
                    else:
                        row.append(0)
                heatmap_data.append(row)
            
            heatmap_data = np.array(heatmap_data)
            
            # 创建热图
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(heatmap_data, cmap='YlOrRd')
            
            # 设置坐标轴
            ax.set_xticks(np.arange(len(metric_types)))
            ax.set_yticks(np.arange(len(comparison_pairs)))
            ax.set_xticklabels([m.replace('_mse', '') for m in metric_types])
            ax.set_yticklabels(comparison_pairs)
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # 添加数值标签
            for i in range(len(comparison_pairs)):
                for j in range(len(metric_types)):
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.6f}',
                                  ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title("Cross-Image MSE Comparison")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, "cross_image_comparison_heatmap.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  跨图像比较热图保存到: {save_path}")


def analyze_same_vae_cross_image(latents_dict: Dict) -> Dict:
    """
    分析同一VAE内不同图像的latent差异
    
    对于每个VAE模型（VAE1、VAE2、Aligned），分析三种图像（原始、双线性、最近邻）
    编码出的latent之间的差异。
    """
    print("分析同一VAE内不同图像的latent差异...")
    
    analysis_dict = {}
    
    # 定义VAE模型
    vae_models = ['z_vae1', 'z_vae2', 'z_aligned']
    image_names = list(latents_dict.keys())  # ['original', 'bilinear', 'nearest']
    
    for vae_model in vae_models:
        analysis_dict[vae_model] = {}
        
        # 收集该VAE模型下所有图像的latent
        latents_by_image = {}
        for img_name in image_names:
            latents_by_image[img_name] = latents_dict[img_name][vae_model]
        
        # 计算两两之间的差异
        for i, img_name1 in enumerate(image_names):
            for j, img_name2 in enumerate(image_names[i+1:], i+1):
                key = f"{img_name1}_vs_{img_name2}"
                
                latent1 = latents_by_image[img_name1].float()
                latent2 = latents_by_image[img_name2].float()
                
                # 展平以便计算
                if latent1.dim() == 4:
                    latent1_flat = latent1.view(1, -1)
                    latent2_flat = latent2.view(1, -1)
                else:
                    latent1_flat = latent1.flatten().unsqueeze(0)
                    latent2_flat = latent2.flatten().unsqueeze(0)
                
                # 计算多种差异指标
                mse = F.mse_loss(latent1_flat, latent2_flat).item()
                cos_sim = F.cosine_similarity(latent1_flat, latent2_flat).mean().item()
                
                # L1距离
                l1_dist = F.l1_loss(latent1_flat, latent2_flat).item()
                
                # 计算分布统计量
                latent1_np = latent1_flat.numpy().flatten()
                latent2_np = latent2_flat.numpy().flatten()
                
                # KL散度（近似）
                from scipy import stats
                try:
                    # 使用直方图近似KL散度
                    hist1, bin_edges = np.histogram(latent1_np, bins=50, density=True)
                    hist2, _ = np.histogram(latent2_np, bins=bin_edges, density=True)
                    
                    # 避免零值
                    hist1 = np.clip(hist1, 1e-10, None)
                    hist2 = np.clip(hist2, 1e-10, None)
                    
                    kl_div = stats.entropy(hist1, hist2)
                    kl_div = float(kl_div) if not np.isnan(kl_div) else 0.0
                except:
                    kl_div = 0.0
                
                # Wasserstein距离（Earth Mover's Distance）
                try:
                    from scipy.stats import wasserstein_distance
                    wasserstein_dist = wasserstein_distance(latent1_np, latent2_np)
                except:
                    wasserstein_dist = 0.0
                
                analysis_dict[vae_model][key] = {
                    'mse': mse,
                    'cosine_similarity': cos_sim,
                    'l1_distance': l1_dist,
                    'kl_divergence': float(kl_div),
                    'wasserstein_distance': float(wasserstein_dist),
                    'mean_diff': float(np.mean(latent1_np) - np.mean(latent2_np)),
                    'std_diff': float(np.std(latent1_np) - np.std(latent2_np))
                }
        
        # 计算三种图像之间的整体差异（方差分析思路）
        all_latents = []
        for img_name in image_names:
            latent = latents_by_image[img_name].float()
            if latent.dim() == 4:
                all_latents.append(latent.view(-1).numpy())
            else:
                all_latents.append(latent.numpy().flatten())
        
        # 计算组间方差和组内方差
        all_data = np.concatenate(all_latents)
        group_means = [np.mean(group) for group in all_latents]
        overall_mean = np.mean(all_data)
        
        # 组间方差
        ss_between = 0
        for i, group in enumerate(all_latents):
            ss_between += len(group) * (group_means[i] - overall_mean) ** 2
        
        # 组内方差
        ss_within = 0
        for i, group in enumerate(all_latents):
            ss_within += np.sum((group - group_means[i]) ** 2)
        
        # F统计量
        k = len(all_latents)  # 组数
        n = len(all_data)     # 总样本数
        
        if k > 1 and n > k:
            ms_between = ss_between / (k - 1)
            ms_within = ss_within / (n - k)
            f_statistic = ms_between / ms_within if ms_within > 0 else 0
        else:
            f_statistic = 0
        
        analysis_dict[vae_model]['anova_summary'] = {
            'f_statistic': float(f_statistic),
            'ss_between': float(ss_between),
            'ss_within': float(ss_within),
            'group_means': [float(m) for m in group_means],
            'overall_mean': float(overall_mean)
        }
    
    return analysis_dict


def compute_frequency_domain_analysis(latents_dict: Dict) -> Dict:
    """
    计算latent的频域分析
    
    对每个latent进行傅里叶变换，分析频域特性。
    """
    print("计算频域分析...")
    
    freq_dict = {}
    
    for img_name, latents in latents_dict.items():
        freq_dict[img_name] = {}
        
        for latent_key, latent in latents.items():
            latent = latent.float()  # 确保是float类型
            # latent形状: [1, C, H, W] 或 [C, H, W]
            if latent.dim() == 4:
                latent = latent.squeeze(0)  # [C, H, W]
            
            # 对每个通道进行FFT
            channels = latent.shape[0]
            height = latent.shape[1]
            width = latent.shape[2]
            
            # 计算2D FFT
            fft_results = []
            magnitude_spectra = []
            power_spectra = []
            
            for c in range(channels):
                channel_data = latent[c].numpy()
                
                # 2D FFT
                fft2 = np.fft.fft2(channel_data)
                fft_shifted = np.fft.fftshift(fft2)  # 将零频移到中心
                
                # 幅度谱
                magnitude = np.abs(fft_shifted)
                
                # 功率谱
                power = magnitude ** 2
                
                fft_results.append(fft_shifted)
                magnitude_spectra.append(magnitude)
                power_spectra.append(power)
            
            # 计算频域统计量
            # 平均功率谱（跨通道）
            avg_power_spectrum = np.mean(power_spectra, axis=0)
            
            # 计算径向平均功率谱
            center_y, center_x = height // 2, width // 2
            y, x = np.indices((height, width))
            r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            r = r.astype(int)
            
            # 计算每个半径的平均功率
            radial_power = np.zeros(np.max(r) + 1)
            radial_count = np.zeros(np.max(r) + 1)
            
            for i in range(height):
                for j in range(width):
                    radius = r[i, j]
                    radial_power[radius] += avg_power_spectrum[i, j]
                    radial_count[radius] += 1
            
            # 避免除以零
            radial_count = np.maximum(radial_count, 1)
            radial_avg_power = radial_power / radial_count
            
            # 计算频带能量
            max_radius = np.max(r)
            low_freq_band = int(max_radius * 0.125)  # 0-1/8
            mid_freq_band = int(max_radius * 0.5)    # 1/8-1/2
            high_freq_band = max_radius              # 1/2-1
            
            low_freq_energy = np.sum(radial_avg_power[:low_freq_band])
            mid_freq_energy = np.sum(radial_avg_power[low_freq_band:mid_freq_band])
            high_freq_energy = np.sum(radial_avg_power[mid_freq_band:])
            total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
            
            # 计算频带能量比
            if total_energy > 0:
                low_freq_ratio = low_freq_energy / total_energy
                mid_freq_ratio = mid_freq_energy / total_energy
                high_freq_ratio = high_freq_energy / total_energy
            else:
                low_freq_ratio = mid_freq_ratio = high_freq_ratio = 0.0
            
            # 计算频谱熵
            power_normalized = avg_power_spectrum / (np.sum(avg_power_spectrum) + 1e-10)
            spectral_entropy = -np.sum(power_normalized * np.log(power_normalized + 1e-10))
            
            # 保存结果
            freq_dict[img_name][latent_key] = {
                'avg_power_spectrum_shape': list(avg_power_spectrum.shape),
                'radial_avg_power': radial_avg_power.tolist(),
                'frequency_band_energy': {
                    'low_freq': float(low_freq_energy),
                    'mid_freq': float(mid_freq_energy),
                    'high_freq': float(high_freq_energy),
                    'total': float(total_energy)
                },
                'frequency_band_ratio': {
                    'low_freq_ratio': float(low_freq_ratio),
                    'mid_freq_ratio': float(mid_freq_ratio),
                    'high_freq_ratio': float(high_freq_ratio)
                },
                'spectral_entropy': float(spectral_entropy),
                'max_power': float(np.max(avg_power_spectrum)),
                'mean_power': float(np.mean(avg_power_spectrum)),
                'std_power': float(np.std(avg_power_spectrum))
            }
    
    return freq_dict


def visualize_same_vae_comparison(same_vae_analysis_dict: Dict, save_dir: str):
    """可视化同一VAE内不同图像的比较结果"""
    print("可视化同一VAE内不同图像的比较...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    vae_models = list(same_vae_analysis_dict.keys())
    
    for vae_model in vae_models:
        analysis = same_vae_analysis_dict[vae_model]
        
        # 创建热图显示两两比较的MSE
        comparison_pairs = [k for k in analysis.keys() if k != 'anova_summary']
        
        if not comparison_pairs:
            continue
        
        # 提取MSE值
        mse_values = []
        pair_names = []
        
        for pair in comparison_pairs:
            mse_values.append(analysis[pair]['mse'])
            pair_names.append(pair)
        
        # 创建柱状图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. MSE柱状图
        bars = axes[0, 0].bar(pair_names, mse_values, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title(f'{vae_model} - MSE Comparison')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加数值
        for bar, value in zip(bars, mse_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01,
                          f'{value:.6f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 余弦相似度柱状图
        cos_sim_values = [analysis[pair]['cosine_similarity'] for pair in comparison_pairs]
        bars2 = axes[0, 1].bar(pair_names, cos_sim_values, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 1].set_title(f'{vae_model} - Cosine Similarity')
        axes[0, 1].set_ylabel('Cosine Similarity')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars2, cos_sim_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cos_sim_values)*0.01,
                          f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 3. 分布距离指标（KL散度、Wasserstein距离）
        kl_values = [analysis[pair]['kl_divergence'] for pair in comparison_pairs]
        wasserstein_values = [analysis[pair]['wasserstein_distance'] for pair in comparison_pairs]
        
        x = np.arange(len(pair_names))
        width = 0.35
        
        bars3_kl = axes[1, 0].bar(x - width/2, kl_values, width, label='KL Divergence', color='skyblue')
        bars3_wass = axes[1, 0].bar(x + width/2, wasserstein_values, width, label='Wasserstein Distance', color='lightcoral')
        
        axes[1, 0].set_title(f'{vae_model} - Distribution Distances')
        axes[1, 0].set_ylabel('Distance')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(pair_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. 均值差异和标准差差异
        mean_diff_values = [analysis[pair]['mean_diff'] for pair in comparison_pairs]
        std_diff_values = [analysis[pair]['std_diff'] for pair in comparison_pairs]
        
        bars4_mean = axes[1, 1].bar(x - width/2, mean_diff_values, width, label='Mean Difference', color='lightgreen')
        bars4_std = axes[1, 1].bar(x + width/2, std_diff_values, width, label='Std Difference', color='orange')
        
        axes[1, 1].set_title(f'{vae_model} - Statistical Differences')
        axes[1, 1].set_ylabel('Difference')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(pair_names, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Same VAE Cross-Image Analysis - {vae_model}', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"same_vae_comparison_{vae_model}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  同一VAE比较图保存到: {save_path}")


def visualize_frequency_domain(freq_dict: Dict, save_dir: str):
    """可视化频域分析结果"""
    print("可视化频域分析结果...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    for img_name, latents_freq in freq_dict.items():
        for latent_key, freq_data in latents_freq.items():
            # 创建频域可视化
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. 径向平均功率谱
            radial_power = np.array(freq_data['radial_avg_power'])
            axes[0, 0].plot(radial_power, 'b-', linewidth=2)
            axes[0, 0].set_title(f'{img_name} - {latent_key}\nRadial Average Power Spectrum')
            axes[0, 0].set_xlabel('Frequency Radius')
            axes[0, 0].set_ylabel('Average Power')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 标记频带边界
            max_radius = len(radial_power) - 1
            low_band = int(max_radius * 0.125)
            mid_band = int(max_radius * 0.5)
            
            axes[0, 0].axvline(x=low_band, color='r', linestyle='--', alpha=0.5, label='Low/Mid Boundary')
            axes[0, 0].axvline(x=mid_band, color='g', linestyle='--', alpha=0.5, label='Mid/High Boundary')
            axes[0, 0].legend()
            
            # 2. 频带能量比饼图
            band_ratios = [
                freq_data['frequency_band_ratio']['low_freq_ratio'],
                freq_data['frequency_band_ratio']['mid_freq_ratio'],
                freq_data['frequency_band_ratio']['high_freq_ratio']
            ]
            band_labels = ['Low Freq', 'Mid Freq', 'High Freq']
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            
            axes[0, 1].pie(band_ratios, labels=band_labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Frequency Band Energy Ratio')
            
            # 3. 频带能量柱状图
            band_energies = [
                freq_data['frequency_band_energy']['low_freq'],
                freq_data['frequency_band_energy']['mid_freq'],
                freq_data['frequency_band_energy']['high_freq']
            ]
            
            x_pos = np.arange(len(band_labels))
            bars = axes[1, 0].bar(x_pos, band_energies, color=colors)
            axes[1, 0].set_title('Frequency Band Energy')
            axes[1, 0].set_ylabel('Energy')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(band_labels)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # 在柱子上添加数值
            for bar, energy in zip(bars, band_energies):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(band_energies)*0.01,
                              f'{energy:.2e}', ha='center', va='bottom', fontsize=9)
            
            # 4. 频谱统计信息
            stats_text = f"""
            Spectral Entropy: {freq_data['spectral_entropy']:.4f}
            Max Power: {freq_data['max_power']:.2e}
            Mean Power: {freq_data['mean_power']:.2e}
            Std Power: {freq_data['std_power']:.2e}
            Total Energy: {freq_data['frequency_band_energy']['total']:.2e}
            """
            
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 1].set_title('Spectral Statistics')
            axes[1, 1].axis('off')
            
            plt.suptitle(f'Frequency Domain Analysis - {img_name.capitalize()} - {latent_key}', fontsize=14)
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, f"frequency_domain_{img_name}_{latent_key}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  频域分析图保存到: {save_path}")
        
        # 创建跨模型比较图
        if len(latents_freq) > 1:
            # 比较不同模型的频带能量比
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            latent_keys = list(latents_freq.keys())
            x = np.arange(len(latent_keys))
            width = 0.25
            
            # 提取频带能量比
            low_ratios = [latents_freq[key]['frequency_band_ratio']['low_freq_ratio'] for key in latent_keys]
            mid_ratios = [latents_freq[key]['frequency_band_ratio']['mid_freq_ratio'] for key in latent_keys]
            high_ratios = [latents_freq[key]['frequency_band_ratio']['high_freq_ratio'] for key in latent_keys]
            
            bars1 = axes[0].bar(x - width, low_ratios, width, label='Low Freq', color='lightblue')
            bars2 = axes[0].bar(x, mid_ratios, width, label='Mid Freq', color='lightgreen')
            bars3 = axes[0].bar(x + width, high_ratios, width, label='High Freq', color='lightcoral')
            
            axes[0].set_title(f'{img_name.capitalize()} - Frequency Band Ratios by Model')
            axes[0].set_ylabel('Ratio')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([k.replace('z_', '').capitalize() for k in latent_keys])
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # 提取频谱熵
            spectral_entropies = [latents_freq[key]['spectral_entropy'] for key in latent_keys]
            
            bars4 = axes[1].bar(latent_keys, spectral_entropies, color=['skyblue', 'lightgreen', 'lightcoral'])
            axes[1].set_title(f'{img_name.capitalize()} - Spectral Entropy by Model')
            axes[1].set_ylabel('Spectral Entropy')
            axes[1].set_xticklabels([k.replace('z_', '').capitalize() for k in latent_keys], rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # 在柱子上添加数值
            for bar, entropy in zip(bars4, spectral_entropies):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(spectral_entropies)*0.01,
                           f'{entropy:.4f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"frequency_comparison_{img_name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  频域比较图保存到: {save_path}")


def visualize_same_vae_latent_distributions(latents_dict: Dict, save_dir: str):
    """
    可视化同一VAE内三种图像latent的对比图
    
    并排显示同一VAE对三种图像（原始、双线性、最近邻）编码的latent分布
    """
    print("可视化同一VAE内三种图像latent的对比图...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义VAE模型和图像名称
    vae_models = ['z_vae1', 'z_vae2', 'z_aligned']
    image_names = list(latents_dict.keys())  # ['original', 'bilinear', 'nearest']
    image_labels = ['Original', 'Bilinear', 'Nearest']
    
    for vae_model in vae_models:
        # 创建图形：一行三列，显示三种图像的latent分布
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 收集该VAE模型下所有图像的latent
        latents_by_image = {}
        for img_name in image_names:
            latents_by_image[img_name] = latents_dict[img_name][vae_model]
        
        # 确定统一的坐标轴范围
        all_values = []
        for img_name in image_names:
            latent = latents_by_image[img_name].float()
            if latent.dim() == 4:
                all_values.extend(latent.view(-1).numpy())
            else:
                all_values.extend(latent.numpy().flatten())
        
        value_min = np.min(all_values)
        value_max = np.max(all_values)
        value_range = value_max - value_min
        
        # 为每种图像创建分布图
        for idx, (img_name, img_label) in enumerate(zip(image_names, image_labels)):
            latent = latents_by_image[img_name].float()
            
            # 展平latent以便可视化分布
            if latent.dim() == 4:
                latent_flat = latent.view(-1).numpy()
            else:
                latent_flat = latent.numpy().flatten()
            
            # 计算统计信息
            mean_val = np.mean(latent_flat)
            std_val = np.std(latent_flat)
            median_val = np.median(latent_flat)
            
            # 创建直方图 + 密度曲线
            ax = axes[idx]
            
            # 直方图
            n, bins, patches = ax.hist(latent_flat, bins=50, alpha=0.6, color='skyblue', 
                                      edgecolor='black', density=True)
            
            # 密度曲线
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(latent_flat)
                x_range = np.linspace(value_min, value_max, 200)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, alpha=0.8)
            except:
                # 如果KDE失败，使用简单的平滑曲线
                pass
            
            # 添加统计信息标注
            stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMedian: {median_val:.4f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 设置标题和标签
            ax.set_title(f'{img_label}\n{vae_model.replace("z_", "").capitalize()}', fontsize=12)
            ax.set_xlabel('Latent Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            
            # 设置统一的x轴范围
            ax.set_xlim(value_min - 0.1 * value_range, value_max + 0.1 * value_range)
        
        # 添加总标题
        model_name_display = vae_model.replace('z_', '').capitalize()
        plt.suptitle(f'Same VAE Latent Distributions Comparison - {model_name_display}', fontsize=14)
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f"same_vae_latent_distributions_{vae_model}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  同一VAE latent分布对比图保存到: {save_path}")


def visualize_same_vae_difference_heatmaps(latents_dict: Dict, save_dir: str):
    """
    可视化同一VAE内不同图像latent的差异热图
    
    显示同一VAE内不同图像latent的逐元素差异
    """
    print("可视化同一VAE内不同图像latent的差异热图...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义VAE模型和图像名称
    vae_models = ['z_vae1', 'z_vae2', 'z_aligned']
    image_names = list(latents_dict.keys())  # ['original', 'bilinear', 'nearest']
    image_labels = ['Original', 'Bilinear', 'Nearest']
    
    for vae_model in vae_models:
        # 收集该VAE模型下所有图像的latent
        latents_by_image = {}
        for img_name in image_names:
            latents_by_image[img_name] = latents_dict[img_name][vae_model]
        
        # 计算两两之间的差异
        comparison_pairs = [
            ('original', 'bilinear', 'Original vs Bilinear'),
            ('original', 'nearest', 'Original vs Nearest'),
            ('bilinear', 'nearest', 'Bilinear vs Nearest')
        ]
        
        # 创建图形：一行三列，显示三种比较的差异热图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (img1_name, img2_name, title) in enumerate(comparison_pairs):
            latent1 = latents_by_image[img1_name].float()
            latent2 = latents_by_image[img2_name].float()
            
            # 确保latent形状一致
            if latent1.shape != latent2.shape:
                print(f"警告: {img1_name}和{img2_name}的latent形状不一致: {latent1.shape} vs {latent2.shape}")
                continue
            
            # 计算逐元素差异
            # latent形状: [1, C, H, W] 或 [C, H, W]
            if latent1.dim() == 4:
                # 如果是4D，计算跨通道平均差异
                diff = torch.abs(latent1 - latent2)
                # 计算跨通道平均，得到2D差异图
                diff_2d = diff.mean(dim=1).squeeze(0)  # [H, W]
            else:
                # 如果是3D，计算跨通道平均差异
                diff = torch.abs(latent1 - latent2)
                diff_2d = diff.mean(dim=0)  # [H, W]
            
            # 转换为numpy
            diff_np = diff_2d.numpy()
            
            # 创建热图
            ax = axes[idx]
            im = ax.imshow(diff_np, cmap='viridis', aspect='auto')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # 设置标题和标签
            ax.set_title(f'{title}\n{vae_model.replace("z_", "").capitalize()}', fontsize=12)
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            
            # 添加差异统计信息
            mean_diff = np.mean(diff_np)
            max_diff = np.max(diff_np)
            std_diff = np.std(diff_np)
            
            stats_text = f'Mean: {mean_diff:.4f}\nMax: {max_diff:.4f}\nStd: {std_diff:.4f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 添加总标题
        model_name_display = vae_model.replace('z_', '').capitalize()
        plt.suptitle(f'Same VAE Latent Difference Heatmaps - {model_name_display}', fontsize=14)
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f"same_vae_difference_heatmaps_{vae_model}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  同一VAE差异热图保存到: {save_path}")
        
        # 额外：创建差异分布图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (img1_name, img2_name, title) in enumerate(comparison_pairs):
            latent1 = latents_by_image[img1_name].float()
            latent2 = latents_by_image[img2_name].float()
            
            if latent1.shape != latent2.shape:
                continue
            
            # 计算差异并展平
            if latent1.dim() == 4:
                diff = torch.abs(latent1 - latent2)
                diff_flat = diff.view(-1).numpy()
            else:
                diff = torch.abs(latent1 - latent2)
                diff_flat = diff.numpy().flatten()
            
            # 创建差异分布直方图
            ax = axes[idx]
            ax.hist(diff_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            
            # 添加统计信息
            mean_diff = np.mean(diff_flat)
            median_diff = np.median(diff_flat)
            std_diff = np.std(diff_flat)
            
            # 添加垂直线标记均值和中位数
            ax.axvline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_diff:.4f}')
            ax.axvline(median_diff, color='green', linestyle='--', linewidth=2, label=f'Median: {median_diff:.4f}')
            
            ax.set_title(f'{title}\nDifference Distribution', fontsize=12)
            ax.set_xlabel('Absolute Difference')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Same VAE Latent Difference Distributions - {model_name_display}', fontsize=14)
        plt.tight_layout()
        
        # 保存差异分布图
        save_path = os.path.join(save_dir, f"same_vae_difference_distributions_{vae_model}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  同一VAE差异分布图保存到: {save_path}")


def save_results(images_dict: Dict, latents_dict: Dict, stats_dict: Dict, 
                comparison_dict: Dict, save_dir: str, args, sample_index: int):
    """保存所有结果"""
    print("保存结果...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 保存配置信息
    config_info = {
        'timestamp': datetime.now().isoformat(),
        'sample_index': sample_index,
        'random_seed': args.random_seed,
        'dataset_split': args.dataset_split,
        'downsample_size': args.downsample_size,
        'upsample_size': args.upsample_size,
        'vae1_path': args.vae1_path,
        'vae2_path': args.vae2_path,
        'align_ckpt': args.align_ckpt,
        'device': args.device,
        'precision': args.precision
    }
    
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    print(f"  配置信息保存到: {config_path}")
    
    # 2. 保存统计信息
    stats_path = os.path.join(save_dir, "statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    
    print(f"  统计信息保存到: {stats_path}")
    
    # 3. 保存比较指标
    comparison_path = os.path.join(save_dir, "comparison_metrics.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_dict, f, indent=2, ensure_ascii=False)
    
    print(f"  比较指标保存到: {comparison_path}")
    
    # 4. 保存latent数据（如果启用）
    if args.save_latents:
        latents_dir = os.path.join(save_dir, "latents")
        os.makedirs(latents_dir, exist_ok=True)
        
        for img_name, latents in latents_dict.items():
            for latent_key, latent in latents.items():
                # 转换为numpy并保存
                latent_np = latent.numpy()
                save_path = os.path.join(latents_dir, f"{img_name}_{latent_key}.npy")
                np.save(save_path, latent_np)
        
        print(f"  Latent数据保存到: {latents_dir}")
    
    # 5. 保存图像数据
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    for img_name, image in images_dict.items():
        if img_name == 'downsampled':
            continue
            
        # 转换为PIL图像并保存
        if image.dim() == 3:
            img_np = image.permute(1, 2, 0).numpy()
        else:
            img_np = image.numpy()
        
        img_np = np.clip(img_np, 0, 1)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        
        save_path = os.path.join(images_dir, f"{img_name}.png")
        img_pil.save(save_path)
    
    print(f"  图像数据保存到: {images_dir}")
    
    # 6. 生成摘要报告
    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("VAE对齐模块Latent分布分析 - 结果摘要\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"样本索引: {sample_index}\n")
        f.write(f"随机种子: {args.random_seed}\n")
        f.write(f"数据集分割: {args.dataset_split}\n\n")
        
        f.write("图像变体:\n")
        for img_name in images_dict.keys():
            if img_name != 'downsampled':
                f.write(f"  - {img_name.capitalize()}\n")
        
        f.write("\n关键发现:\n")
        
        # 添加关键统计信息
        for img_name, stats in stats_dict.items():
            f.write(f"\n{img_name.capitalize()}图像:\n")
            for latent_key, latent_stats in stats.items():
                f.write(f"  {latent_key}: mean={latent_stats['mean']:.6f}, std={latent_stats['std']:.6f}\n")
        
        f.write("\n对齐效果评估:\n")
        for img_name, metrics in comparison_dict.items():
            if img_name != 'cross_image_comparison':
                f.write(f"\n{img_name.capitalize()}:\n")
                f.write(f"  VAE2 vs Aligned MSE: {metrics['mse_vae2_vs_aligned']:.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("分析完成!\n")
        f.write("=" * 80 + "\n")
    
    print(f"  摘要报告保存到: {summary_path}")
    
    return {
        'config_path': config_path,
        'stats_path': stats_path,
        'comparison_path': comparison_path,
        'summary_path': summary_path,
        'images_dir': images_dir,
        'latents_dir': latents_dir if args.save_latents else None
    }


def main():
    """主函数"""
    print("=" * 80)
    print("VAE对齐模块Latent分布可视化分析")
    print("=" * 80)
    
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    
    try:
        # 1. 加载HuggingFace token
        token = os.getenv("HUGGINGFACE_TOKEN", "")
        if token:
            login(token=token)
            print("成功登录HuggingFace Hub")
        else:
            print("警告: 未找到HuggingFace token，尝试继续...")
        
        # 2. 加载模型
        dtype = get_dtype(args.precision)
        
        print("\n" + "-" * 60)
        print("步骤1: 加载模型")
        print("-" * 60)
        
        vae1 = load_vae_model(args.vae1_path, args.vae1_subfolder, args.device, dtype)
        vae2 = load_vae_model(args.vae2_path, args.vae2_subfolder, args.device, dtype)
        pipeline = load_align_pipeline(vae1, vae2, args.align_ckpt, args)
        
        # 3. 获取图像
        print("\n" + "-" * 60)
        print("步骤2: 获取和处理图像")
        print("-" * 60)
        
        image_tensor, sample_index = get_random_image_from_imagenet(
            split=args.dataset_split,
            index=args.sample_index,
            seed=args.random_seed
        )
        
        # 4. 创建图像变体
        images_dict = create_image_variants(
            image_tensor,
            downsample_size=args.downsample_size,
            upsample_size=args.upsample_size
        )
        
        # 5. 可视化图像
        images_dir = os.path.join(output_dir, "visualizations")
        visualize_images(images_dict, images_dir)
        
        # 6. 编码图像获取latent
        print("\n" + "-" * 60)
        print("步骤3: 编码图像获取latent")
        print("-" * 60)
        
        latents_dict = encode_images(pipeline, images_dict, args.device, dtype)
        
        # 7. 可视化latent分布
        print("\n" + "-" * 60)
        print("步骤4: 可视化latent分布")
        print("-" * 60)
        
        latent_viz_dir = os.path.join(output_dir, "latent_visualizations")
        visualize_latent_distributions(latents_dict, latent_viz_dir)
        
        # 8. 计算统计信息
        print("\n" + "-" * 60)
        print("步骤5: 计算统计信息")
        print("-" * 60)
        
        stats_dict = compute_statistics(latents_dict)
        
        # 9. 计算比较指标
        comparison_dict = compute_comparison_metrics(latents_dict)
        
        # 10. 可视化比较指标
        comparison_viz_dir = os.path.join(output_dir, "comparison_visualizations")
        visualize_comparison_metrics(comparison_dict, comparison_viz_dir)
        
        # 11. 新增分析：同一VAE内跨图像分析
        print("\n" + "-" * 60)
        print("步骤6: 同一VAE内跨图像分析")
        print("-" * 60)
        
        same_vae_analysis_dict = analyze_same_vae_cross_image(latents_dict)
        
        # 12. 新增分析：频域分析
        print("\n" + "-" * 60)
        print("步骤7: 频域分析")
        print("-" * 60)
        
        freq_dict = compute_frequency_domain_analysis(latents_dict)
        
        # 13. 可视化新增分析结果
        print("\n" + "-" * 60)
        print("步骤8: 可视化新增分析结果")
        print("-" * 60)
        
        # 可视化同一VAE内跨图像分析
        same_vae_viz_dir = os.path.join(output_dir, "same_vae_comparison")
        visualize_same_vae_comparison(same_vae_analysis_dict, same_vae_viz_dir)
        
        # 可视化频域分析
        freq_viz_dir = os.path.join(output_dir, "frequency_domain_visualizations")
        visualize_frequency_domain(freq_dict, freq_viz_dir)
        
        # 14. 新增可视化：同一VAE内三种图像latent的对比图
        print("\n" + "-" * 60)
        print("步骤9: 新增可视化 - 同一VAE内三种图像latent对比")
        print("-" * 60)
        
        # 可视化同一VAE内三种图像latent的对比图
        same_vae_dist_dir = os.path.join(output_dir, "same_vae_distributions")
        visualize_same_vae_latent_distributions(latents_dict, same_vae_dist_dir)
        
        # 可视化同一VAE内不同图像latent的差异热图
        same_vae_heatmap_dir = os.path.join(output_dir, "same_vae_heatmaps")
        visualize_same_vae_difference_heatmaps(latents_dict, same_vae_heatmap_dir)
        
        # 15. 保存所有结果（包括新增分析）
        print("\n" + "-" * 60)
        print("步骤10: 保存所有结果")
        print("-" * 60)
        
        save_results(
            images_dict, latents_dict, stats_dict, comparison_dict,
            output_dir, args, sample_index
        )
        
        # 16. 保存新增分析结果
        print("\n" + "-" * 60)
        print("步骤11: 保存新增分析结果")
        print("-" * 60)
        
        # 保存同一VAE内跨图像分析结果
        same_vae_path = os.path.join(output_dir, "same_vae_cross_image_analysis.json")
        with open(same_vae_path, 'w', encoding='utf-8') as f:
            json.dump(same_vae_analysis_dict, f, indent=2, ensure_ascii=False)
        print(f"  同一VAE内跨图像分析结果保存到: {same_vae_path}")
        
        # 保存频域分析结果
        freq_path = os.path.join(output_dir, "frequency_domain_analysis.json")
        with open(freq_path, 'w', encoding='utf-8') as f:
            json.dump(freq_dict, f, indent=2, ensure_ascii=False)
        print(f"  频域分析结果保存到: {freq_path}")
        
        # 12. 打印摘要
        print("\n" + "=" * 80)
        print("分析完成!")
        print("=" * 80)
        
        print(f"\n📊 分析结果摘要:")
        print(f"   样本索引: {sample_index}")
        print(f"   输出目录: {output_dir}")
        
        print(f"\n📈 关键统计信息:")
        for img_name, stats in stats_dict.items():
            print(f"   {img_name.capitalize()}图像:")
            for latent_key, latent_stats in stats.items():
                print(f"     {latent_key}: mean={latent_stats['mean']:.6f}, std={latent_stats['std']:.6f}")
        
        print(f"\n🔍 对齐效果评估:")
        for img_name, metrics in comparison_dict.items():
            if img_name != 'cross_image_comparison':
                print(f"   {img_name.capitalize()}:")
                print(f"     VAE1 vs Aligned MSE: {metrics['mse_vae2_vs_aligned']:.6f}")
                print(f"     VAE1 vs Aligned 余弦相似度: {metrics['cosine_sim_vae2_vs_aligned']:.4f}")
        
        print(f"\n💾 生成的文件:")
        print(f"   配置信息: {output_dir}/config.json")
        print(f"   统计信息: {output_dir}/statistics.json")
        print(f"   比较指标: {output_dir}/comparison_metrics.json")
        print(f"   摘要报告: {output_dir}/summary.txt")
        print(f"   可视化图像: {output_dir}/visualizations/")
        print(f"   Latent可视化: {output_dir}/latent_visualizations/")
        print(f"   比较可视化: {output_dir}/comparison_visualizations/")
        
        if args.save_latents:
            print(f"   Latent数据: {output_dir}/latents/")
        
        print("\n" + "=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 错误发生: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
