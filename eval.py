import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import lpips
from cleanfid import fid
import os
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import warnings
warnings.filterwarnings('ignore')

class VAEReconstructionEvaluator:
    """
    用于评估VAE重建质量的综合评估器
    支持MSE, PSNR, SSIM, LPIPS, rFID
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',save_dir='output/', prefix='test/', dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()
        
        
        # 用于rFID计算的临时目录
        self.real_dir =  save_dir + prefix + 'ori/'
        self.recon_dir = save_dir + prefix + 'recon/'
        os.makedirs(self.real_dir, exist_ok=True)
        os.makedirs(self.recon_dir, exist_ok=True)
        
        # 存储所有结果
        self.results = {
            'mse': [], 'psnr': [], 'ssim': [], 
            'lpips': [], 'rfid': None
        }
    
    def calculate_mse(self, original, reconstructed):
        """计算均方误差"""
        return F.mse_loss(original, reconstructed).item()
    
    def calculate_psnr(self, original, reconstructed, max_val=1.0):
        """计算峰值信噪比"""
        mse = self.calculate_mse(original, reconstructed)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(torch.tensor(mse))
    
    def calculate_ssim(self, original, reconstructed):
        """计算结构相似性指数（批处理版本）"""
        batch_size = original.shape[0]
        ssim_values = []
        
        for i in range(batch_size):
            # 转换为numpy并调整维度为HWC
            img1 = original[i].permute(1, 2, 0).cpu().numpy()
            img2 = reconstructed[i].permute(1, 2, 0).cpu().numpy()
            
            # SSIM计算（对于RGB图像使用多通道）
            ssim_val = ssim(
                img1, img2, 
                win_size=11,  # 使用11x11的高斯窗口
                gaussian_weights=True,
                multichannel=True,  # 对于RGB图像
                data_range=1.0,
                channel_axis=2
            )
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    def calculate_ssim_torchmetrics(self, original, reconstructed):
        """计算结构相似性指数（使用torchmetrics库）"""
        ssim_metric = SSIM(data_range=1.0).to(self.device)
        original = original.to(self.device)
        reconstructed = reconstructed.to(self.device)
        
        # 计算SSIM
        ssim_val = ssim_metric(original, reconstructed)
        return ssim_val.item()
    
    def calculate_lpips(self, original, reconstructed):
        """计算感知相似度（LPIPS）"""
        # LPIPS期望输入在[-1, 1]范围，但这里假设输入在[0, 1]
        # 如果需要，可以调整：input_tensor = input_tensor * 2 - 1
        with torch.no_grad():
            lpips_val = self.lpips_model(original, reconstructed)
        return lpips_val.mean().item()
    
    
    def save_images_for_fid(self, original_batch, reconstructed_batch, batch_idx):
        """保存图像用于rFID计算"""
        for i in range(original_batch.shape[0]):
            idx = batch_idx * original_batch.shape[0] + i
            
            # 原始图像
            img_orig = original_batch[i].permute(1, 2, 0).cpu().numpy()
            img_orig = np.clip(img_orig * 255, 0, 255).astype(np.uint8)
            Image.fromarray(img_orig).save(f'{self.real_dir}/img_{idx:05d}.png')
            
            # 重建图像
            img_recon = reconstructed_batch[i].permute(1, 2, 0).cpu().numpy()
            img_recon = np.clip(img_recon * 255, 0, 255).astype(np.uint8)
            Image.fromarray(img_recon).save(f'{self.recon_dir}/img_{idx:05d}.png')
    
    def calculate_rfid(self):
        """计算重建FID（rFID）"""
        try:
            rfid_score = fid.compute_fid(self.real_dir, self.recon_dir, mode="legacy_pytorch")
            return rfid_score
        except Exception as e:
            print(f"计算rFID时出错: {e}")
            return None
    
    def evaluate_batch(self, original_batch, reconstructed_batch, batch_idx, save_for_fid=True, n_batches=100):
        """
        评估单个批次
        
        参数:
            original_batch: 原始图像 [B, C, H, W], 值范围[0, 1]
            reconstructed_batch: 重建图像 [B, C, H, W], 值范围[0, 1]
            batch_idx: 批次索引
            save_for_fid: 是否保存图像用于rFID计算
        """
        # 确保张量在正确设备上
        original_batch = original_batch.to(self.device)
        reconstructed_batch = reconstructed_batch.to(self.device)
        
        # 计算各项指标
        mse_val = self.calculate_mse(original_batch, reconstructed_batch)
        psnr_val = self.calculate_psnr(original_batch, reconstructed_batch)
        ssim_val = self.calculate_ssim_torchmetrics(original_batch, reconstructed_batch)
        lpips_val = self.calculate_lpips(original_batch, reconstructed_batch)
        
        # 保存结果
        self.results['mse'].append(mse_val)
        self.results['psnr'].append(psnr_val)
        self.results['ssim'].append(ssim_val)
        self.results['lpips'].append(lpips_val)
        
        # 保存图像用于rFID（仅对前几个批次，避免存储过大）
        if save_for_fid and batch_idx < n_batches:  # 调整这个数字控制用于FID的图像数量
            self.save_images_for_fid(original_batch, reconstructed_batch, batch_idx)
        
        # 打印当前批次结果
        print(f"批次 {batch_idx}: MSE={mse_val:.6f}, PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")
        
        return {
            'mse': mse_val, 'psnr': psnr_val, 
            'ssim': ssim_val, 'lpips': lpips_val
        }
    
    def get_final_results(self, compute_rfid=True):
        """获取最终评估结果"""
        final_results = {}
        
        # 计算平均值
        for metric in ['mse', 'psnr', 'ssim', 'lpips']:
            values = self.results[metric]
            final_results[f'{metric}_mean'] = np.mean(values).item()
            final_results[f'{metric}_std'] = np.std(values).item()
        
        # 计算rFID（如果需要）
        if compute_rfid:
            print("正在计算rFID，这可能需要一些时间...")
            rfid_score = self.calculate_rfid()
            final_results['rfid'] = rfid_score.item()
            self.results['rfid'] = rfid_score.item()
        
        return final_results
    
    def print_summary(self):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("重建质量评估摘要")
        print("="*60)
        
        final = self.get_final_results(compute_rfid=False)
        
        print(f"MSE:      {final['mse_mean']:.6f} ± {final['mse_std']:.6f}")
        print(f"PSNR:     {final['psnr_mean']:.2f} ± {final['psnr_std']:.2f} dB")
        print(f"SSIM:     {final['ssim_mean']:.4f} ± {final['ssim_std']:.4f}")
        print(f"LPIPS:    {final['lpips_mean']:.4f} ± {final['lpips_std']:.4f}")
        
        if self.results['rfid'] is not None:
            print(f"rFID:     {self.results['rfid']:.2f}")
        
        print("="*60)
    
    def cleanup(self):
        """清理临时文件"""
        import shutil
        if os.path.exists(self.real_dir):
            shutil.rmtree(self.real_dir)
        if os.path.exists(self.recon_dir):
            shutil.rmtree(self.recon_dir)
            
            
# 主评估流程示例
def main_evaluation_pipeline(prefix="test", device='cuda', batch_size=32, n_batches = 157, cleanup=True, dtype=torch.bfloat16):
    # 初始化评估器
    evaluator = VAEReconstructionEvaluator(device=device, prefix=prefix, dtype=dtype)
    generator=torch.manual_seed(int(42))
    from dataloader import ImageNetDataloader, load_dataset
    from tqdm import tqdm
    data_dir = "benjamin-paine/imagenet-1k-256x256"
    eval_dataset = load_dataset(data_dir, split="validation")
    dataloader = ImageNetDataloader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # import pdb;pdb.set_trace()
    print("开始评估重建质量...")
    print("device: ", device)
    print("dtype: ", dtype)
    print("dataset: ", data_dir)
    
    if prefix == "SD_1_5_VAE_original/":
        from pca_extractor import load_vae, PCAPipeline
        vae = load_vae(model_path="sd-legacy/stable-diffusion-v1-5", dtype=dtype)
        pipe = PCAPipeline(
            vae=vae,
            pca_model=None,
            device=device,
            dtype=dtype
        )
        pipe.eval()
        
        
    elif prefix == "Flux_VAE_original/":
        from pca_extractor import load_vae, PCAPipeline
        vae = load_vae(model_path="black-forest-labs/FLUX.1-dev", dtype=dtype)
        # vae = load_vae(model_path="sd-legacy/stable-diffusion-v1-5")
        pipe = PCAPipeline(
            vae=vae,
            pca_model=None,
            device=device,
            dtype=dtype
        )
        pipe.eval()
    
    elif prefix == "Flux_VAE_mean_sample/":
        from pca_extractor import load_vae, PCAPipeline, PCAModel
        vae = load_vae(model_path="black-forest-labs/FLUX.1-dev", dtype=dtype)
        pipe = PCAPipeline(
            vae=vae,
            pca_model=None,
            device=device,
            dtype=dtype
        )
        pipe.eval()
        
    elif "Flux_VAE_pca" in prefix:
        n = int(prefix[:-1].split("_")[-1])
        from pca_extractor import load_vae, PCAPipeline, PCAModel
        vae = load_vae(model_path="black-forest-labs/FLUX.1-dev", dtype=dtype)
        import os
        pca_dir = os.getcwd()
        full_pca_components_name = os.path.join(pca_dir, "flux_full_channel_pca_components.npy")
        full_pca_mean_name = os.path.join(pca_dir, "flux_full_channel_pca_mean.npy")
        full_pca_components = np.load(full_pca_components_name)
        full_pca_mean = np.load(full_pca_mean_name)
        
        pca_model = PCAModel(pca_components_freeze=full_pca_components[:n], pca_mean=full_pca_mean, device=device, dtype=dtype)
        pipe = PCAPipeline(
            vae=vae,
            pca_model=pca_model,
            device=device, dtype=dtype
        )
        pipe.eval()
        
    elif "SD_1_5_VAE_pca" in prefix:
        n = int(prefix[:-1].split("_")[-1])
        from pca_extractor import load_vae, PCAPipeline, PCAModel
        vae = load_vae(model_path="sd-legacy/stable-diffusion-v1-5", dtype=dtype)
        import os
        pca_dir = os.getcwd()
        full_pca_components_name = os.path.join(pca_dir, "sd_full_channel_pca_components.npy")
        full_pca_mean_name = os.path.join(pca_dir, "sd_full_channel_pca_mean.npy")
        full_pca_components = np.load(full_pca_components_name)
        full_pca_mean = np.load(full_pca_mean_name)
        
        pca_model = PCAModel(pca_components_freeze=full_pca_components[:n], pca_mean=full_pca_mean, device=device, dtype=dtype)
        pipe = PCAPipeline(
            vae=vae,
            pca_model=pca_model,
            device=device, dtype=dtype
        )
        pipe.eval()    
    
    
    # 遍历数据加载器的伪代码
    for batch_idx, batch_data in tqdm(enumerate(dataloader), total=n_batches):
        # 假设batch_data包含原始图像
        original_images = batch_data['pixel_values'].to(evaluator.device)
        
        if prefix == "SD_1_5_VAE_original/" or prefix == "Flux_VAE_original/":
            with torch.no_grad():
                # import pdb;pdb.set_trace()
                latents = pipe._encode_vae_image(original_images.to(dtype), generator)
                reconstructed_images = pipe._decode_vae_latents(latents).float().detach().cpu()
                reconstructed_images= torch.clip(reconstructed_images, 0, 1)
        
        elif prefix == "Flux_VAE_mean_sample/":
            with torch.no_grad():
                latents = pipe._encode_vae_image(original_images.to(dtype), generator, sample_mode="mean")
                reconstructed_images = pipe._decode_vae_latents(latents).float().detach().cpu()
                reconstructed_images= torch.clip(reconstructed_images, 0, 1)
            
        elif "Flux_VAE_pca" in prefix:
            with torch.no_grad():
                # import pdb;pdb.set_trace()
                # pipe.pca_analysis(original_images.to(dtype), generator)
                reconstructed_images = pipe.pca_reconstruction(original_images.to(dtype), 
                                                               generator, 
                                                               n_components=n, 
                                                               n_channels=16, 
                                                               do_normalize=False).float().detach().cpu()
                reconstructed_images= torch.clip(reconstructed_images, 0, 1)
                
        elif "SD_1_5_VAE_pca" in prefix:
            with torch.no_grad():
                # import pdb;pdb.set_trace()
                # pipe.pca_analysis(original_images.to(dtype), generator)
                reconstructed_images = pipe.pca_reconstruction(original_images.to(dtype), 
                                                               generator, 
                                                               n_components=n, 
                                                               n_channels=4, 
                                                               do_normalize=False).float().detach().cpu()
                reconstructed_images= torch.clip(reconstructed_images, 0, 1)

        # 为了演示，我们使用随机重建（实际使用时请替换）
        # reconstructed_images = original_images + torch.randn_like(original_images) * 0.05
        # reconstructed_images = torch.clamp(reconstructed_images, 0, 1)
        
        # 评估当前批次
        evaluator.evaluate_batch(
            original_images, 
            reconstructed_images, 
            batch_idx,
            save_for_fid=(batch_idx < n_batches),  # 仅前20个批次用于rFID
            n_batches=n_batches
        )
        
        # 可选：提前停止，用于测试
        if batch_idx >= n_batches:  # 仅评估50个批次作为示例
            break
    
    # 获取最终结果
    final_results = evaluator.get_final_results(compute_rfid=True)
    
    # 打印摘要
    evaluator.print_summary()
    
    if cleanup:
        # 清理临时文件
        evaluator.cleanup()
    
    
    return final_results

def get_results(prefix="test", batch_size=32, n_batches=10, save_result=True, cleanup=True, dtype=torch.bfloat16):
    """
    获取评估结果
    """
    results = main_evaluation_pipeline(prefix=(prefix+"/"), batch_size=batch_size, n_batches=n_batches, cleanup=cleanup, dtype=dtype)
    
    if save_result:
        output_add = f"output/{prefix}/"
        
        if not os.path.exists(output_add):
            os.makedirs(output_add)
        # 保存结果到文件
        
        import json
        with open(output_add + f'vae_reconstruction_metrics_{prefix}.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"结果已保存到 vae_reconstruction_metrics_{prefix}.json")
    
    return results

if __name__ == "__main__":
    # 运行评估
    prefix = "SD_1_5_VAE_original"
    get_results(prefix=prefix, batch_size=32, n_batches=157, save_result=True, cleanup=True, dtype=torch.bfloat16)
    
    prefix = "Flux_VAE_original"
    # prefix = "Flux_VAE_pca_16"
    # prefix = "SD_1_5_VAE_pca_4"
    
    # prefix = "Flux_VAE_mean_sample"
    
    get_results(prefix=prefix, batch_size=32, n_batches=157, save_result=True, cleanup=True, dtype=torch.bfloat16)
    
    # _prefix = "Flux_VAE_pca_"
    # for i in range(16, 17):
    #     print(f"正在处理 {i}")
    #     prefix = _prefix + str(i)
    #     get_results(prefix=prefix)
        