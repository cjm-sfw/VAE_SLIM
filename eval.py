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
from tqdm import tqdm

import pdb

from eval_class import VAEReconstructionEvaluator  
            
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
        full_pca_components_name = os.path.join(pca_dir + "/vis/", "flux_full_channel_pca_components.npy")
        full_pca_mean_name = os.path.join(pca_dir + "/vis/", "flux_full_channel_pca_mean.npy")
        full_pca_components = np.load(full_pca_components_name)
        full_pca_mean = np.load(full_pca_mean_name)
        
        if "single_iter" in prefix:
            # 单次迭代的PCA
            n_rm = int(prefix[:-1].split("_")[-1])
            n = 15
            channel_list = [i for i in range(16)]
            channel_list.remove(n_rm-1)
            print(f"remove the {n_rm-1} channel")
            # import pdb;pdb.set_trace()
            pca_components_freeze = full_pca_components[channel_list, :]
        else:
            # 多次迭代的PCA
            pca_components_freeze = full_pca_components[:n]
            
        pca_model = PCAModel(pca_components_freeze=pca_components_freeze, pca_mean=full_pca_mean, device=device, dtype=dtype)
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
        full_pca_components_name = os.path.join(pca_dir + "/vis/", "sd_full_channel_pca_components.npy")
        full_pca_mean_name = os.path.join(pca_dir + "/vis/", "sd_full_channel_pca_mean.npy")
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
    # prefix = "SD_1_5_VAE_original"
    # get_results(prefix=prefix, batch_size=32, n_batches=157, save_result=True, cleanup=True, dtype=torch.bfloat16)
    
    # prefix = "Flux_VAE_original"
    # prefix = "Flux_VAE_pca_16"
    # prefix = "SD_1_5_VAE_pca_4"
    
    # prefix = "Flux_VAE_mean_sample"
    
    _prefix = "Flux_VAE_pca_single_iter_"
    for i in tqdm(range(1, 17)):
        print(f"正在处理 {i}")
        prefix = _prefix + str(i)
        get_results(prefix=prefix, batch_size=32, n_batches=157, save_result=True, cleanup=True, dtype=torch.bfloat16)
    
    # get_results(prefix=prefix, batch_size=32, n_batches=157, save_result=True, cleanup=True, dtype=torch.bfloat16)
    
    # _prefix = "Flux_VAE_pca_"
    # for i in range(16, 17):
    #     print(f"正在处理 {i}")
    #     prefix = _prefix + str(i)
    #     get_results(prefix=prefix)
        