import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboard

import numpy as np
import sklearn
from sklearn.decomposition import PCA
import csv

from vae_slim import PCAPipeline, PCAModel

from huggingface_hub import login
from diffusers import (
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    AutoencoderKL
)
from dotenv import load_dotenv
load_dotenv()
from utils import visualize_spectrum_comparison, radial_profile, rgb_grad_map, rgb_grad_comparison

import os
# env
Token = os.getenv("HUGGINGFACE_TOKEN", "")
cache_dir = os.getenv("HF_CACHE_DIR", "/root/autodl-tmp/cache_dir/huggingface/hub/")
ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf"
print("Huggingface token:", Token)
login(token=Token)


model_path = "black-forest-labs/FLUX.1-dev"

print("loading vae from:", model_path)

# base = FluxPipeline.from_pretrained(
#     model_path,
#     use_safetensors=True,
#     torch_dtype=torch.bfloat16,
#     transformer=transformer,
#     use_onnx=False,
#     cache_dir=cache_dir,
# )

vae = AutoencoderKL.from_pretrained(
    model_path,
    subfolder="vae",
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir,
    proxies={'http': '127.0.0.1:7890'}
)
import pdb;
vae.to("cuda")

pca_components_add = "/workspace/DiffBrush/VIS/pca3d_pca_components.csv"
pca_mean_add = "/workspace/DiffBrush/VIS/pca3d_pca_mean.csv"

save_dir = "ckpt/"

pca_model = PCAModel(
    pca_components_freeze=np.loadtxt(pca_components_add, delimiter=',', dtype=np.float16),  # [3, 16]
    pca_mean=np.loadtxt(pca_mean_add, delimiter=',', dtype=np.float16),  # [16]
    device="cuda"
)

from dataloader import image_dataloader


train_loader = image_dataloader(
    data_dir="train_images",
    batch_size=4,  # 根据你的显存大小调整
    shuffle=True,
    num_workers=4  # 根据需要调整
)

eval_loader = image_dataloader(
    data_dir="eval_images",
    batch_size=1,  # 根据你的显存大小调整
    shuffle=False,
    num_workers=0  # 根据需要调整
)

generator=torch.manual_seed(int(42))
from tqdm import tqdm

  
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
def train_pca_pipeline(vae, pca_model, train_loader, generator, residual_detail=False, stage=1, device='cuda', ckpt=None):
    """增强版的TensorBoard监控"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/pca_pipeline{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    
    pipeline = PCAPipeline(vae, pca_model, residual_detail)

    
    # freeze vae
    pipeline.vae.requires_grad_(False)
    
    # 优化器
    if stage == 1:
        optimizer_for_z_pca = torch.optim.Adam([
            {'params': pipeline.pca_predictor.parameters(), 'lr': 5e-3},
            {'params': pipeline.residual_detail_predictor.parameters(), 'lr': 1.2e-2}
        ])
    elif stage == 2:
        pipeline.pca_predictor.requires_grad_(False)
        optimizer_for_z_pca = torch.optim.Adam(pipeline.residual_detail_predictor.parameters(), lr = 8e-3)
    
    optimizer_for_delta = torch.optim.Adam([pipeline.pca_model.pca_components_delta, pipeline.pca_model.pca_mean], lr=5e-3)
    
    
    criterion = nn.MSELoss()
    from losses import WeightedL1Loss
    criterion_diff = WeightedL1Loss()
    
    
    # 学习率调度器
    if stage == 1:
        scheduler_for_z_pca = torch.optim.lr_scheduler.StepLR(optimizer_for_z_pca, step_size=120, gamma=0.5)
        scheduler_for_delta = torch.optim.lr_scheduler.StepLR(optimizer_for_delta, step_size=150, gamma=0.1)
    elif stage == 2:
        scheduler_for_z_pca = torch.optim.lr_scheduler.StepLR(optimizer_for_z_pca, step_size=40, gamma=0.8)
        scheduler_for_delta = torch.optim.lr_scheduler.StepLR(optimizer_for_delta, step_size=100, gamma=0.1)
    
    # 训练指标跟踪
    train_metrics = {
        'losses': [],
        'learning_rates': [],
        'grad_norms': []
    }
    warmup_steps = 150
    global_step = 0
    epoch = 400
    for epoch in tqdm(range(epoch)):
        pipeline.pca_predictor.train()
        epoch_losses = {'total_loss': 0, 'pca_loss': 0, 'diff_loss':0, 'diff_dist_loss':0, 'recon_latent_loss': 0, 'kl_loss': 0}
        
        for batch_idx, x in enumerate(train_loader):
            
            # warm up学习率
            if global_step < warmup_steps:
                if stage == 1:
                    for param_group in optimizer_for_z_pca.param_groups:
                        param_group['lr'] = min(5e-3, 1e-2 * (global_step + 1) / warmup_steps)
                elif stage == 2:
                    for param_group in optimizer_for_z_pca.param_groups:
                        param_group['lr'] = min(8e-3, 8e-3 * (global_step + 1) / warmup_steps)
                for param_group in optimizer_for_delta.param_groups:
                    param_group['lr'] = min(5e-3, 5e-3 * (global_step + 1) / warmup_steps)
            
            x = x.to(device).bfloat16()
            
            # 训练步骤
            optimizer_for_z_pca.zero_grad()
            if hasattr(pipeline.pca_model, 'pca_components_delta'):
                # print("Using PCA components delta for optimization")
                optimizer_for_delta.zero_grad()
            
        
            
            with torch.no_grad():
                z_true = pipeline._encode_vae_image(x, generator)  # [batch, 16, H/8, W/8]
            
            z_pca_true = pipeline.pca_transform_batch(z_true)  # [batch, 3, H/8, W/8]
            z_pca_true = z_pca_true.bfloat16().to(pipeline.device)    
            

            # PCA预测器前向传播
            z_pca_pred = pipeline.pca_predictor(x)
            
            # 计算损失
            pca_loss = criterion_diff(z_pca_pred, z_pca_true)
            
            z_pred = pipeline.pca_inverse_transform_batch(z_pca_pred)  # [batch, 16, H/8, W/8]
            
            if residual_detail:
                diff_pred = pipeline.residual_detail_predictor(x)  # [batch, 16, H/8, W/8]
                diff_true = (z_true-z_pred).detach()
                
                diff_loss = criterion_diff(diff_pred, diff_true)
            else:
                diff_pred = torch.zeros_like(z_true)
                diff_loss = torch.tensor(0.0, device=device)
            
            recon_latent_loss = criterion_diff(z_pred + diff_pred.detach(), z_true)
            
            
            # latents = (
            #     z_pred / pipeline.vae.config.scaling_factor
            # ) + pipeline.vae.config.shift_factor
            # recon = pipeline.vae.decode(latents, return_dict=False)[0]
            
            # # recon = pipeline.vae_decoder(z_pred)
            # recon_loss = criterion(recon, x)
            

            
            mean_diff_pca = (z_pca_pred.mean(dim=(0, 2, 3)) - z_pca_true.mean(dim=(0, 2, 3)))**2
            std_diff_pca = (z_pca_pred.std(dim=(0, 2, 3)) - z_pca_true.std(dim=(0, 2, 3)))**2
            diff_dist_loss = (mean_diff_pca.mean() + std_diff_pca.mean())  # 添加标准差差异的惩罚
            
            # kl_loss
            kl_loss = F.kl_div(z_pca_true.log_softmax(dim=1), z_pca_pred.softmax(dim=1), reduction='batchmean')
            
            total_loss = pca_loss + 0.3 * diff_dist_loss + 1 * recon_latent_loss + 0.03 * kl_loss + 1 * diff_loss

            
            # 反向传播
            total_loss.backward()
            
            # 记录梯度范数
            total_norm = 0
            for p in pipeline.pca_predictor.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            writer.add_scalar('Training/Gradient_Norm', total_norm, global_step)
            
            # 优化步骤
            optimizer_for_z_pca.step()
            if hasattr(pipeline.pca_model, 'pca_components_delta'):
                optimizer_for_delta.step()
            
            # 记录损失
            losses = {
                'total_loss': total_loss.item(),
                'pca_loss': pca_loss.item(),
                'diff_dist_loss': diff_dist_loss.item(),
                'diff_loss': diff_loss.item(),
                'recon_latent_loss': recon_latent_loss.item(),
                # 'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item()
            }
            
            for k, v in losses.items():
                epoch_losses[k] += v
                writer.add_scalar(f'Batch/{k}', v, global_step)
            
            global_step += 1
            
        # 每10个epoch验证并记录更多信息
        if epoch % 10 == 0:
            with torch.no_grad():
                x = next(iter(eval_loader)).to(device).bfloat16()
                # 生成重建图像
                recon, z_pca_pred, x_recon = pipeline.generate_for_comparsion(x, generator, x_recon=True)
                # recon = recon.cpu().bfloat16()
                # x = x.cpu().bfloat16()
                diff_map = (x[:4] - recon[:4]).abs().float()
                
                # 记录图像对比
                writer.add_images('Input/x/Original', x[:4].float(), epoch)
                
                # 记录重建图像
                writer.add_images('Output/x/Ori_Reconstruction', x_recon[:4].float(), epoch)
                writer.add_images('Output/x/Reconstruction', recon[:4].float(), epoch)
                
                writer.add_images('Output/x/Difference/Space', diff_map, epoch)
                
                (grad_maps_x, grad_maps_x_fig), (grad_maps_recon, grad_maps_recon_fig)  = rgb_grad_map(x[:4].float()), rgb_grad_map(recon[:4].float())
                writer.add_figure('Output/x/Gradient/Original', grad_maps_x_fig, epoch)
                writer.add_figure('Output/x/Gradient/Reconstruction', grad_maps_recon_fig, epoch)
                grad_comparison_x = rgb_grad_comparison(grad_maps_x, diff_map)
                writer.add_figure('Output/x/Gradient/Comparison', grad_comparison_x, epoch)
                grad_comparison_recon = rgb_grad_comparison(grad_maps_recon, diff_map)
                writer.add_figure('Output/x/Gradient/Comparison_Reconstruction', grad_comparison_recon, epoch)
                
                writer.add_scalar('Output/x/Difference/Mean', (x[:4] - recon[:4]).abs().mean().item(), epoch)
                writer.add_scalar('Output/x/Difference/Std', (x[:4] - recon[:4]).abs().std().item(), epoch)
                
                
                
                # 计算频域图像
                original = x[:4].bfloat16()
                pca_reconstructed = recon[:4].bfloat16()
                ori_freq_fig = visualize_spectrum_comparison(original, vae, pca_model)
                recon_freq_fig = visualize_spectrum_comparison(pca_reconstructed, vae, pca_model)  
                diff_freq_fig = visualize_spectrum_comparison((x[:4] - recon[:4]).abs(), vae, pca_model)
                writer.add_figure('Frequency/Original', ori_freq_fig, epoch)
                writer.add_figure('Frequency/Reconstruction', recon_freq_fig, epoch)
                writer.add_figure('Frequency/Difference', diff_freq_fig, epoch)


                # 记录PCA组件残差（如果存在）
                if hasattr(pipeline.pca_model, 'pca_components_delta'):
                    delta = pipeline.pca_model.pca_components_delta
                    writer.add_histogram('Parameters/pca_components_delta', delta, epoch)
                    writer.add_histogram('Parameters/pca_mean', pipeline.pca_model.pca_mean, epoch)
                
                # 记录梯度信息（可选）
                for name, param in pipeline.pca_predictor.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                
                # 记录参数分布
                for name, param in pipeline.pca_predictor.named_parameters():
                    writer.add_histogram(f'Parameters/{name}', param, epoch)
                
                if residual_detail:
                    for name, param in pipeline.residual_detail_predictor.named_parameters():
                        writer.add_histogram(f'Parameters/Residual_Detail_Parameters/{name}', param, epoch)
        
        # 学习率调度
        scheduler_for_z_pca.step()
        scheduler_for_delta.step()
        current_lr = scheduler_for_z_pca.get_last_lr()[0]
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        
        # 记录epoch统计
        avg_losses = {k: v/len(train_loader) for k, v in epoch_losses.items()}
        for k, v in avg_losses.items():
            writer.add_scalar(f'Epoch/{k}', v, epoch)
        
        # 记录模型输出统计
        pipeline.pca_predictor.eval()
        
        with torch.no_grad():
            # import pdb;pdb.set_trace()
            test_batch = next(iter(eval_loader)).to(device).bfloat16()
            
            z_pca_pred_test = pipeline.pca_predictor(test_batch)
            z_pred_test = pipeline.pca_inverse_transform_batch(z_pca_pred_test)
            
            z_true_test = pipeline._encode_vae_image(test_batch, generator)
            z_true_flat_test = z_true_test.permute(0, 2, 3, 1).reshape(-1, 16)
            z_pca_true_flat_test = pipeline.pca_model.transform(z_true_flat_test)
            z_pca_true_test = torch.tensor(z_pca_true_flat_test, dtype=torch.bfloat16).to(pipeline.device)
            z_pca_true_test = z_pca_true_test.reshape(test_batch.size(0), -1, 3).permute(0, 2, 1)
            z_pca_true_test = z_pca_true_test.reshape(test_batch.size(0), 3, z_true_test.shape[2], z_true_test.shape[3])
            
            # 记录PCA输出的统计信息
            writer.add_scalar('Stats/z_pca_pred_mean', z_pca_pred_test.mean().item(), epoch)
            writer.add_scalar('Stats/z_pca_pred_std', z_pca_pred_test.std().item(), epoch)
            
            writer.add_scalar('Stats/z_pca_true_mean', z_pca_true_test.mean().item(), epoch)
            writer.add_scalar('Stats/z_pca_true_std', z_pca_true_test.std().item(), epoch)
            
            writer.add_scalar('Stats/z_pca_diff_mean', (z_pca_pred_test.mean(dim=(0, 2, 3))-z_pca_true_test.mean(dim=(0, 2, 3))).mean().item(), epoch)
            writer.add_scalar('Stats/z_pca_diff_std', (z_pca_pred_test.std(dim=(0, 2, 3))-z_pca_true_test.std(dim=(0, 2, 3))).mean().item(), epoch)
            # 记录重建输出的统计信息
            writer.add_scalar('Stats/z_pred_mean', z_pred_test.mean().item(), epoch)
            writer.add_scalar('Stats/z_pred_std', z_pred_test.std().item(), epoch)
            
            writer.add_scalar('Stats/z_true_mean', z_true_test.mean().item(), epoch)
            writer.add_scalar('Stats/z_true_std', z_true_test.std().item(), epoch)
            
            writer.add_scalar('Stats/z_diff_mean', (z_pred_test.mean(dim=(0, 2, 3))-z_true_test.mean(dim=(0, 2, 3))).mean().item(), epoch)
            writer.add_scalar('Stats/z_diff_std', (z_pred_test.std(dim=(0, 2, 3))-z_true_test.std(dim=(0, 2, 3))).mean().item(), epoch)

        
        print(f"Epoch {epoch}: {avg_losses}, LR: {current_lr:.2e}")
    
    
    writer.close()
    print(f"增强版TensorBoard日志保存在: {log_dir}")
    
    
    return pipeline



# train_pca_pipeline(vae, pca_model, train_loader,generator, device='cuda')
pipe = train_pca_pipeline(vae, pca_model, train_loader,generator, residual_detail=True, device='cuda')
ckpt_path = save_dir + f"pca_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
pipe.save(ckpt_path)

pipe = train_pca_pipeline(vae, pca_model, train_loader,generator, residual_detail=True, stage=2, device='cuda', ckpt=ckpt_path)
ckpt_path = save_dir +  f"pca_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
pipe.save(ckpt_path)