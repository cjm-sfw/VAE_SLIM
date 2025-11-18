import torch


from huggingface_hub import login
from diffusers import (
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    AutoencoderKL
)

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipeline,
    PipelineImageInput,
    calculate_shift,
    retrieve_timesteps,
    FluxPipelineOutput,
    is_torch_xla_available,
    randn_tensor,
)
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def get_imagenet_dataset():
    from datasets import load_dataset
    from huggingface_hub import login
    Token = os.getenv("HUGGINGFACE_TOKEN", "")
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("ILSVRC/imagenet-1k")
    return ds

class image_dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        # return tensors
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512), Image.LANCZOS)  # Resize to 512x512
        image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
        image = torch.tensor(image).permute(2, 0, 1)  # Convert to CxHxW
        image = image.to(torch.float32)  # Convert to bfloat16
        
        return image
    


class image_dataloader(DataLoader):
    def __init__(self, data_dir, batch_size=4, shuffle=True, num_workers=0):
        dataset = image_dataset(data_dir)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    


def main():
    # env
    Token = os.getenv("HUGGINGFACE_TOKEN", "")
    cache_dir = os.getenv("HF_CACHE_DIR", "/root/autodl-tmp/cache_dir/huggingface/hub/")
    ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf"
    print("Huggingface token:", Token)
    login(token=Token)


    model_path = "black-forest-labs/FLUX.1-dev"
    print("loading transformer from:", ckpt_path)

    transformer = FluxTransformer2DModel.from_single_file(
        ckpt_path,
        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.bfloat16
        ),
        torch_dtype=torch.bfloat16,
        proxies={'http': '127.0.0.1:7890'}
    )
    print("loading Pipeline from:", model_path)
    base = FluxPipeline.from_pretrained(
        model_path,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        transformer=transformer,
        use_onnx=False,
        cache_dir=cache_dir,
    )
    # base.enable_model_cpu_offload()
    base.to("cuda")

    from datasets import load_dataset

    def init_parti_prompt_dataset():
        parti_prompts = load_dataset("nateraw/parti-prompts", split="train")
        parti_prompts = parti_prompts.shuffle()
        return parti_prompts

    prompts = init_parti_prompt_dataset()
    print(f"Loaded {len(prompts)} prompts from Parti dataset.")

    sample_prompts = [prompts[i+100]["Prompt"] for i in range(36)]
    print(sample_prompts)

    save_dir = "eval_images"

    for i in range(len(sample_prompts)):
        prompt = sample_prompts[i]
        print(f"Generating image for prompt: {prompt}")
        out = base(
            prompt=prompt,
            guidance_scale=3.5,
            height=512,
            width=512,
            num_inference_steps=50,
            num_images_per_prompt=2,
        ).images
        for j, image in enumerate(out):
            image.save(f"{save_dir}/image_{i}_{j}.png")
            print(f"Saved image_{i}_{j}.png")
        
if __name__ == "__main__":
    # main()
    get_imagenet_dataset()