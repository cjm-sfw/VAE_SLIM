import torch
import torch.nn as nn
from vae_align import AlignPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from datetime import datetime
import numpy as np
import os

from huggingface_hub import login

from dotenv import load_dotenv
load_dotenv()

# env
Token = os.getenv("HUGGINGFACE_TOKEN", "")
cache_dir = os.getenv("HF_CACHE_DIR", "")
login(token=Token)

def count_parameters(model):
    """计算模型总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def analyze_pipeline_parameters(args):
    """分析AlignPipeline的参数量"""
    # 加载VAE模型（与训练脚本相同）
    dtype = torch.bfloat16 if args.precision == "bfloat16" else torch.float32
    
    vae1 = AutoencoderKL.from_pretrained(
        args.vae1_path,
        subfolder=args.vae1_subfolder,
        torch_dtype=dtype
    )
    
    vae2 = AutoencoderKL.from_pretrained(
        args.vae2_path,
        subfolder=args.vae2_subfolder,
        torch_dtype=dtype
    )
    
    # 创建对齐pipeline
    pipeline = AlignPipeline(
        VAE_1=vae1,
        VAE_2=vae2,
        model_version="longtail",
        img_in_channels=args.img_in_channels,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_blocks=args.num_blocks,
        downsample_times=args.downsample_times,
        input_types=args.input_types,
        device='cpu',  # 在CPU上计算参数量
        dtype=dtype
    )
    
    # 冻结VAE（与训练一致）
    pipeline.freeze_vae()
    
    
    print(pipeline.align_module)
    print("=" * 60)
    print("AlignPipeline 参数分析")
    print("=" * 60)
    
    # 1. 总参数（包括冻结的VAE）
    # total_params, trainable_params = count_parameters(pipeline)
    # print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    # print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    # print(f"冻结参数量: {total_params - trainable_params:,} ({(total_params - trainable_params)/1e6:.2f}M)")
    
    # 2. 按组件细分
    print("\n" + "-" * 60)
    print("按组件细分:")
    print("-" * 60)
    
    # AlignModule参数
    align_total, align_trainable = count_parameters(pipeline.align_module)
    print(f"AlignModule: {align_total:,} ({align_total/1e6:.2f}M)")
    
    # VAE1参数
    vae1_total, _ = count_parameters(pipeline.VAE_1)
    print(f"VAE1 ({args.vae1_path}): {vae1_total:,} ({vae1_total/1e6:.2f}M)")
    
    # VAE2参数
    vae2_total, _ = count_parameters(pipeline.VAE_2)
    print(f"VAE2 ({args.vae2_path}): {vae2_total:,} ({vae2_total/1e6:.2f}M)")
    
    # 3. AlignModule层级细分
    print("\n" + "-" * 60)
    print("AlignModule层级细分:")
    print("-" * 60)
    
    def print_layer_params(name, module):
        total = sum(p.numel() for p in module.parameters())
        if total > 0:
            print(f"  {name}: {total:,}")
    
    # 分析downsample_blocks
    for i, block in enumerate(pipeline.align_module.downsample_blocks):
        print_layer_params(f"downsample_blocks[{i}]", block)
    
    # 分析latent_trans_blocks
    for i, block in enumerate(pipeline.align_module.latent_trans_blocks):
        print_layer_params(f"latent_trans_blocks[{i}]", block)
    
    # 4. 参数类型分布
    print("\n" + "-" * 60)
    print("参数类型分布:")
    print("-" * 60)
    
    conv_params = 0
    norm_params = 0
    other_params = 0
    
    for name, param in pipeline.align_module.named_parameters():
        if 'conv' in name:
            conv_params += param.numel()
        elif 'norm' in name or 'bn' in name:
            norm_params += param.numel()
        else:
            other_params += param.numel()
    
    print(f"卷积层参数: {conv_params:,} ({conv_params/align_total*100:.1f}%)")
    print(f"归一化层参数: {norm_params:,} ({norm_params/align_total*100:.1f}%)")
    print(f"其他参数: {other_params:,} ({other_params/align_total*100:.1f}%)")
    
    return pipeline

# 使用示例
if __name__ == "__main__":
    # 创建模拟参数（使用train_align.sh中的配置）
    class Args:
        vae1_path = "sd-legacy/stable-diffusion-v1-5"
        vae2_path = "black-forest-labs/FLUX.1-dev"
        vae1_subfolder = "vae"
        vae2_subfolder = "vae"
        img_in_channels = 3
        in_channels = 4
        hidden_channels = 32
        out_channels = 16
        num_blocks = 2
        downsample_times = 3
        input_types = ['image', 'latent']
        precision = "bfloat16"
    
    args = Args()
    pipeline = analyze_pipeline_parameters(args)
