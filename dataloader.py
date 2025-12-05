import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import torchvision.transforms as transforms

from huggingface_hub import login
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig, AutoencoderKL

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipeline,
)
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from dotenv import load_dotenv

load_dotenv()



def get_imagenet_dataset(split="train"):

    token = os.getenv("HUGGINGFACE_TOKEN", "")
    login(token=token)
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("benjamin-paine/imagenet-1k-256x256", split=split)
    return ds
def get_lioncoco_dataset():

    token = os.getenv("HUGGINGFACE_TOKEN", "")
    login(token)

    ds = load_dataset("laion/relaion-coco")
    return ds


class image_dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]

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

        return {'pixel_values': image}


class image_dataloader(DataLoader):
    def __init__(self, data_dir, batch_size=4, shuffle=True, num_workers=0):
        dataset = image_dataset(data_dir)
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )


def main():
    # env
    Token = os.getenv("HUGGINGFACE_TOKEN", "")
    cache_dir = os.getenv("HF_CACHE_DIR", "/root/autodl-tmp/cache_dir/huggingface/hub/")

    ckpt_path = (
        "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf"
    )
    print("Huggingface token:", Token)
    login(token=Token)

    model_path = "black-forest-labs/FLUX.1-dev"
    print("loading transformer from:", ckpt_path)

    transformer = FluxTransformer2DModel.from_single_file(
        ckpt_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        proxies={"http": "127.0.0.1:7890"},
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

    sample_prompts = [prompts[i + 100]["Prompt"] for i in range(36)]
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



class ImageNetDataloader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, transform=None, with_label=False):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),  # 确保图像大小一致
                transforms.ToTensor(),           # 转换为Tensor
                # transforms.Normalize(            # 标准化（使用ImageNet统计量）
                #     mean=[0.485, 0.456, 0.406],
                #     std=[0.229, 0.224, 0.225]
                # )
            ])
        
        if with_label:
            self.collate_fn = self.collate_fn_with_label
        else:
            self.collate_fn = self.collate_fn_without_label
        
        dataset = dataset.with_transform(self.apply_transforms)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=num_workers
        )
    def apply_transforms(self, examples):
        examples['pixel_values'] = [self.transform(image.convert('RGB')) for image in examples['image']]
        return examples
    
    def collate_fn_with_label(self, batch):
        # 提取像素值和标签
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return {'pixel_values': pixel_values, 'labels': labels}
    
    def collate_fn_without_label(self, batch):
        # 提取像素值
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        return {'pixel_values': pixel_values}

if __name__ == "__main__":
    # main()


    # 1. 定义图像预处理转换
    # 根据模型需求调整预处理步骤


    # 2. 自定义转换函数


    # 3. 加载数据集
    print("正在加载数据集...")
    dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split="test")

    train_dataloader = ImageNetDataloader(dataset, batch_size=4)

    # 6. 测试 DataLoader
    print("\n测试 DataLoader...")
    for i, batch in enumerate(train_dataloader):
        print(f"批次 {i+1}:")
        print(f"  像素值形状: {batch['pixel_values'].shape}")
        print(f"  像素值均值: {batch['pixel_values'].mean()}")
        print(f"  像素值标准差: {batch['pixel_values'].std()}")
        # print(f"  标签形状: {batch['labels'].shape}")
        # print(f"  标签示例: {batch['labels'][:5]}")
        
        if i == 2:  # 只测试前3个批次
            break
