import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
import os

from vae_slim import PCAPipeline, PCAModel

from huggingface_hub import login
from diffusers import (
    AutoencoderKL
)

# env
Token = os.getenv("HUGGINGFACE_TOKEN", "")
cache_dir = os.getenv("HF_CACHE_DIR", "/root/autodl-tmp/cache_dir/huggingface/hub/")
ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf"
print("Huggingface token:", Token)
login(token=Token)


model_path = "black-forest-labs/FLUX.1-dev"

print("loading vae from:", model_path)

vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="vae",
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir,
    proxies={'http': '127.0.0.1:7890'}
)
import pdb;
vae.to("cuda")

pca_components_add = "/workspace/DiffBrush/VIS/pca3d_pca_components.csv"
pca_mean_add = "/workspace/DiffBrush/VIS/pca3d_pca_mean.csv"

pca_model = PCAModel(
    pca_components_freeze=np.loadtxt(pca_components_add, delimiter=',', dtype=np.float16),  # [3, 16]
    pca_mean=np.loadtxt(pca_mean_add, delimiter=',', dtype=np.float16),  # [16]
    n_components=3,
    device="cuda"
)

from dataloader import train_dataloader


train_loader = train_dataloader(
    data_dir="train_images",
    batch_size=4,  # 根据你的显存大小调整
    shuffle=True,
    num_workers=4  # 根据需要调整
)

generator=torch.manual_seed(int(42))
from tqdm import tqdm

device = "cuda"

pipe = PCAPipeline(
    vae=vae,
    pca_model=pca_model,
    device=device
)
ckpt_path = ""
pipe.load(ckpt_path)

test_batch = next(iter(train_loader)).to(device).bfloat16()

