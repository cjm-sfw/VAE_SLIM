import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
import os
from vae_slim import PCAPipeline, PCAModel

from huggingface_hub import login
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from sklearn.decomposition import PCA
from dataloader import image_dataloader, ImageNetDataloader
from PIL import Image
from tqdm import tqdm
from utils import plot_images, plot_each_channel_figure_list, plot_image, save_image
# env
from dotenv import load_dotenv
load_dotenv()

def load_vae(model_path = "black-forest-labs/FLUX.1-dev", dtype=torch.bfloat16):
    # model_path = ""
    Token = os.getenv("HUGGINGFACE_TOKEN", "")
    cache_dir = os.getenv("HF_CACHE_DIR", "/root/autodl-tmp/cache_dir/huggingface/hub/")

    if "FLUX.1-dev" in model_path:
        ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf"
    else:
        ckpt_path = None
    
    print("Huggingface token:", Token)
    login(token=Token)

    print("loading vae from:", model_path)

    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=dtype,
        cache_dir=cache_dir,
        proxies={'http': '127.0.0.1:7890'}
    )
    import pdb;
    # Check if CUDA is available before moving to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if vae is not None:
        vae = vae.to(device)
    vae.eval()

    return vae

def load_pca_model(n_components = 16):
    pca = PCA(n_components=n_components)
    return pca

def load_dataloader(data_dir="train_images", dataset_type="imagenet", batch_size=4, shuffle=True, num_workers=0):
    if dataset_type == "images":
        train_loader = image_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,  # 根据你的显存大小调整
            shuffle=shuffle,
            num_workers=num_workers  # 根据需要调整
        )
    elif dataset_type == "imagenet":
        from dataloader import get_imagenet_dataset
        dataset = get_imagenet_dataset(split="test")
        train_loader = ImageNetDataloader(
            dataset=dataset,
            batch_size=batch_size,  # 根据你的显存大小调整
            shuffle=shuffle,
            num_workers=num_workers  # 根据需要调整
        )
    return train_loader



def get_rgb_case(rgb_case_add = "/workspace/VAE_SLIM/input_cases/RGB.png"):
    rgb_case = torch.tensor(np.array(Image.open(rgb_case_add))).to(device).bfloat16()
    print("shape of the rgb case:", rgb_case.shape)
    rgb_case = rgb_case.permute(2, 0, 1).unsqueeze(0)
    return rgb_case

def save_pca_to_csv(reduced_data, pca_components, pca_mean, save_dir="/workspace/VAE_SLIM/vis/", prefix="full_channel"):
    # save to csv
    full_channel_pca_reduced_data_add = f"{save_dir}/{prefix}_pca_reduced_data.npy"
    full_channel_pca_components_add = f"{save_dir}/{prefix}_pca_components.npy"
    full_channel_pca_mean_add = f"{save_dir}/{prefix}_pca_mean.npy"

    with open(full_channel_pca_reduced_data_add, "wb") as f:
        np.save(f, reduced_data)
        print("saved to", full_channel_pca_reduced_data_add)

    with open(full_channel_pca_components_add, "wb") as f:
        np.save(f, pca_components)
        print("saved to", full_channel_pca_components_add)
        print("shape of pca components:", pca_components.shape)

    with open(full_channel_pca_mean_add, "wb") as f:
        np.save(f, pca_mean)
        print("saved to", full_channel_pca_mean_add)
        print("shape of pca mean:", pca_mean.shape)

def visualize_reduced_data_and_rgb(reduced_data, rgb, data_shape=None,save_dir="/workspace/VAE_SLIM/", prefix="3_channel"):
    import torch.nn.functional as F
    rgb = F.interpolate(rgb, size=data_shape[2:], mode='bilinear')
    rgb = rgb.cpu().detach()[0].permute(1, 2, 0).reshape(rgb.shape[2]*rgb.shape[3], 3).numpy()
    print("shape of rgb:", rgb.shape)
    reduced = reduced_data[:,:3]
    print("shape of reduced data:", reduced_data.shape)
    # Save projection results
    import json
    with open(f"{save_dir}/{prefix}_pca3d_projection.json", "w") as f_json:
        json.dump([
            {"x": float(x), "y": float(y), "z": float(z), "r": float(r), "g": float(g), "b": float(b)}
            for (x, y, z), (r, g, b) in zip(reduced.tolist(), rgb.tolist())
        ], f_json, indent=2)
    from matplotlib import pyplot as plt
    # Save scatter plots from different views
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    try:
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=rgb/255, s=2)
    except Exception as e:
        print("Error:", e)
        import pdb;pdb.set_trace()
    ax.set_title(f"PCA 3D Colored by RGB")
    for elev, azim in [(30, 30), (60, 45), (10, 90), (90, 0)]:
        ax.view_init(elev=elev, azim=azim)
        plt.savefig(f"{save_dir}/{prefix}_pca3d_view_e{elev}_a{azim}.png", dpi=300)
    plt.close()

def get_full_channel_pca_model_from_rgb(model_path="black-forest-labs/FLUX.1-dev",
                                        n_components = 16, 
                                        n_channels = 16,
                                        rgb_case_add = "/workspace/VAE_SLIM/input_cases/RGB.png",
                                        save_dir="/workspace/VAE_SLIM/",
                                        prefix="full_channel",
                                        dtype=torch.bfloat16):
    pca = load_pca_model(n_components)
    rgb_case = get_rgb_case(rgb_case_add)
    vae = load_vae(model_path)
    pipe = PCAPipeline(
        vae=vae,
        pca_model=None,
        device=device
    )
    vae.eval()
    pipe.eval()
    latents = pipe._encode_vae_image(rgb_case, generator)
    b, c, h, w = latents.shape
    latents = latents.reshape(latents.shape[0], n_channels, -1).permute(0, 2, 1)

    reduced_data = pca.fit_transform(latents[0].float().cpu().detach().numpy())
    pca_components = pca.components_
    pca_mean = pca.mean_
    print("explained variance ratio:", pca.explained_variance_ratio_)

    save_pca_to_csv(reduced_data=reduced_data, pca_components=pca_components, pca_mean=pca_mean, prefix=prefix)
    visualize_reduced_data_and_rgb(reduced_data=reduced_data, rgb=rgb_case.float(), data_shape=(b, c, h, w), save_dir=save_dir + "vis/", prefix=prefix)

def build_recon_image(pipe, test_batch, generator, n_components=16, n_channels=16):
    pca_recon = pipe.pca_reconstruction(test_batch, generator, n_components=n_components, n_channels=n_channels)
    x_recon = pipe.latent_reconstruction(test_batch, generator)
    # import pdb;pdb.set_trace()
    plot_images([test_batch[0].float().cpu(), x_recon[0].float().detach().cpu(), pca_recon[0].float().detach().cpu()], ncols=3, save_path=f"vis/pca_recon_{n_components}.png")

def build_diff_n_components_image(model_path="black-forest-labs/FLUX.1-dev", 
                                  n_components=16, 
                                  n_channels=16, 
                                  save_dir="/workspace/VAE_SLIM/", 
                                  prefix="full_channel", 
                                  dataset_type="imagenet"):
    vae = load_vae(model_path)
    full_pca_components = np.load(save_dir + prefix + "_pca_components.npy")
    full_pca_mean = np.load(save_dir + prefix + "_pca_mean.npy")
    
    n_range = [i for i in range(1, n_components+1)]

    train_loader = load_dataloader(batch_size=1, dataset_type=dataset_type)
    test_batch = next(iter(train_loader))['pixel_values'].to(device).bfloat16()

    image_list = []
    title_list = []
    
    diff_list = []
    norm_diff_list = []
    diff_title_list = []
    norm_diff_title_list = []
    
    for n in n_range:
        pca_model = PCAModel(pca_components_freeze=full_pca_components[:n], pca_mean=full_pca_mean, device=device)
        pipe = PCAPipeline(
            vae=vae,
            pca_model=pca_model,
            device=device
        )
        pipe.eval()
        # import pdb;pdb.set_trace()
        build_recon_image(pipe, test_batch, generator, n, n_channels)
        
        pca_recon = pipe.pca_reconstruction(test_batch, generator, n_components=n, n_channels=n_channels, do_normalize=False)[0].float().detach().cpu()
        pca_recon = torch.clip(pca_recon, 0, 1)
        image_list.append(pca_recon)
        title_list.append(f"n={n}")
        
        if n != 1: 
            dist = pca_recon - image_list[-2]
            diff_list.append(dist)
        else:
            diff_list.append(pca_recon)
        
        norm_diff_list.append((diff_list[-1] - diff_list[-1].min()) / (diff_list[-1].max() - diff_list[-1].min()))
        diff_title_list.append(f"individual diff n={n}")
        norm_diff_title_list.append(f"individual norm diff n={n}")

    image_list.append(test_batch[0].float().cpu())
    title_list.append("original")

    plot_images(image_list, title_list, save_path=f"vis/{prefix}_pca_recon_diff_n_components.png")
    plot_images(diff_list, diff_title_list, save_path=f"vis/{prefix}_pca_recon_diff_n_components_diff.png")
    plot_each_channel_figure_list(diff_list, ncols=4, save_path=f"vis/{prefix}_pca_recon_diff_n_components_each_channel.png")
    plot_images(norm_diff_list, norm_diff_title_list, save_path=f"vis/{prefix}_pca_recon_diff_n_components_norm_diff.png")
    plot_each_channel_figure_list(norm_diff_list, ncols=4, save_path=f"vis/{prefix}_pca_recon_diff_n_components_each_channel_norm_diff.png")
    
    for image_index in range(len(image_list)):
        save_image(f'vis/image_{image_index}.png', image_list[image_index].float().cpu().detach())
    
    for norm_diff_index in range(len(norm_diff_list)):
        save_image(f'vis/norm_diff_image_{norm_diff_index}.png', norm_diff_list[norm_diff_index].float().cpu().detach())
        
    for diff_index in range(len(diff_list)):
        save_image(f'vis/diff_image_{diff_index}.png', diff_list[diff_index].float().cpu().detach())
  

generator=torch.manual_seed(int(torch.randn(1).item()))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

if __name__ == "__main__":
    # get_full_channel_pca_model_from_rgb(model_path="sd-legacy/stable-diffusion-v1-5", 
    #                                     n_components=4, 
    #                                     n_channels=4, 
    #                                     rgb_case_add="/workspace/VAE_SLIM/input_cases/cases.png",
    #                                     prefix="sd_full_channel"
    #                                     )
    # build_diff_n_components_image(model_path="sd-legacy/stable-diffusion-v1-5", 
    #                               n_components=4, 
    #                               n_channels=4,
    #                               prefix="sd_full_channel")

    get_full_channel_pca_model_from_rgb(model_path="black-forest-labs/FLUX.1-dev", 
                                        n_components=16, 
                                        n_channels=16, 
                                        rgb_case_add="/workspace/VAE_SLIM/input_cases/cases.png",
                                        prefix="flux_full_channel",
                                        dtype=torch.bfloat16
                                        )

    build_diff_n_components_image(model_path="black-forest-labs/FLUX.1-dev", 
                                  n_components=16, 
                                  n_channels=16, 
                                  save_dir="/workspace/VAE_SLIM/vis/",
                                  prefix="flux_full_channel")

        

