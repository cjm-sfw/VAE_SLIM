import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, with_norm=False):
        super(ResidualBlock, self).__init__()
        self.with_norm = with_norm
        if self.with_norm:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0) if in_channels != out_channels else nn.Identity()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        if self.with_norm:
            self.relu = nn.SiLU(inplace=True)
            self.gn1 = nn.GroupNorm(16, out_channels)  # 使用GroupNorm代替LayerNorm
            self.gn2 = nn.GroupNorm(16, out_channels)  # 使用GroupNorm代替LayerNorm

    def forward(self, x):
        
        if self.with_norm:
            residual = self.residual_conv(x)
            
            out = self.relu(self.gn1(self.conv1(x)))
            out = self.relu(self.gn2(self.conv2(out)))
            out += residual
            return out
        else:
            residual = x
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
            out += residual
            return self.relu(out)
        


class AlignModule(nn.Module):
    """对齐模块，使用残差网络构建下采样Module"""

    def __init__(self, 
                 model_version, 
                 img_in_channels, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_blocks=2, 
                 downsample_times=3, 
                 channel_times=4, 
                 input_types=['image', 'latent', 'DWT'],
                 residual_with_norm=False
                 ):
        """
        Args:
            model_version (str): 模型版本，'base' 或 'longtail'
            in_channels (int): 输入通道数
            hidden_channels (int): 隐藏层通道数
            out_channels (int): 输出通道数
            num_blocks (int): 每个下采样阶段的残差块数量
            downsample_times (int): 下采样次数
            input_types (list): 输入类型列表，包含'image', 'latent', 'DWT'
        """
        super(AlignModule, self).__init__()

        self.model_version = model_version  # 模型版本
        
        self.img_in_channels = img_in_channels
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.input_types = input_types
        self.residual_with_norm = residual_with_norm  # 是否使用归一化的残差块

        self.downsample_times = downsample_times
        self.downsample_blocks = nn.ModuleList()

        self.downsample_blocks.append(
            nn.Sequential(
                nn.Conv2d(self.img_in_channels, self.hidden_channels, kernel_size=1),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            )
        )

        for down_index in range(self.downsample_times):
            hidden_channels = self.hidden_channels * (channel_times ** down_index)
            residual_blocks = nn.ModuleList()
            for _ in range(self.num_blocks):
                residual_blocks.append(ResidualBlock(hidden_channels, hidden_channels, with_norm=self.residual_with_norm))
            self.downsample_blocks.append(nn.Sequential(*residual_blocks))
            self.downsample_blocks.append(nn.Conv2d(hidden_channels, hidden_channels * channel_times, kernel_size=3, stride=2, padding=1))

        if self.model_version == 'longtail':
            # 在longtail版本中，最后一个下采样阶段的输出通道数为out_channels
            hidden_channels = hidden_channels // channel_times
            self.downsample_blocks.append(nn.Conv2d(self.hidden_channels * (channel_times ** self.downsample_times), 
                                                    hidden_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0))
            residual_blocks = nn.ModuleList()
            for _ in range(self.out_channels - self.num_blocks * self.downsample_times):
                residual_blocks.append(ResidualBlock(hidden_channels, hidden_channels, with_norm=self.residual_with_norm))
            self.downsample_blocks.append(nn.Sequential(*residual_blocks))
            self.downsample_blocks.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, stride=1))

        elif self.model_version == 'base':
            self.downsample_blocks.append(nn.Conv2d(self.hidden_channels * (channel_times ** self.downsample_times), out_channels, kernel_size=1))

        elif self.model_version == 'variational':
            hidden_channels = hidden_channels // channel_times
            self.downsample_blocks.append(nn.Conv2d(self.hidden_channels * (channel_times ** self.downsample_times), 
                                                    hidden_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0))
            residual_blocks = nn.ModuleList()
            for _ in range(self.out_channels - self.num_blocks * self.downsample_times):
                residual_blocks.append(ResidualBlock(hidden_channels, hidden_channels, with_norm=self.residual_with_norm))
            self.downsample_blocks.append(nn.Sequential(*residual_blocks))
            
            self.out_channels = self.out_channels * 2
            self.downsample_blocks.append(nn.Conv2d(hidden_channels, self.out_channels, kernel_size=3, padding=1, stride=1))


        self.latent_trans_blocks = nn.ModuleList()
        self.latent_trans_blocks.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1))
        self.latent_trans_blocks.append(ResidualBlock(self.out_channels, self.out_channels, with_norm=self.residual_with_norm))
        # self.latent_trans_blocks.append(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1))

    def sample(self, feature, generator):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        """ Args:
            feature (torch.Tensor): 输入特征张量，形状为 [batch_size, channels, height, width]
        Returns:
            torch.Tensor: 采样后的张量，形状为 [batch_size, out_channels, height/2^downsample_times, width/2^downsample_times]
        """
        # 假设feature是潜在变量的均值和对数方差
        if self.model_version == 'variational':
            dist = DiagonalGaussianDistribution(feature)
            z = dist.sample(generator)  # 从分布中采样
            
            return z, dist
        else:
            # 对于其他模型版本，直接返回特征
            return feature, None

    def forward(self, x, z):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, channels, height, width]
            z (torch.Tensor): 潜在变量张量，形状为 [batch_size, latent_channels, height, width]
        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, out_channels, height/2^downsample_times, width/2^downsample_times]
        """
        # resize x to match z's size
        # if x.shape[2:] != z.shape[2:]:
            # print(f"Resizing input x from {x.shape[2:]} to {z.shape[2:]}")
            # x = F.interpolate(x, size=z.shape[2:], mode='bilinear', align_corners=False)
        
        if 'image' in self.input_types and 'latent' not in self.input_types:
            x = x  # 仅图像输入
        elif 'latent' in self.input_types and 'image' not in self.input_types:
            x = z  # 仅潜在变量输入

        # 下采样处理
        for block in self.downsample_blocks:
            x = block(x)

        # 潜在变量处理
        for block in self.latent_trans_blocks:
            z = block(z)

        # 对齐潜在变量
        x = x + z  # 假设对齐方式是简单相加

        return x


class AlignPipeline:
    """不同VAE利用module对齐latent分布的Pipeline"""

    def __init__(self, 
                 VAE_1, 
                 VAE_2, 
                 model_version='base',
                 img_in_channels=3, 
                 in_channels=16,
                 hidden_channels=64, 
                 out_channels=16, 
                 num_blocks=2, 
                 downsample_times=3,
                 channel_times=4,
                 input_types=['image', 'latent'],
                 residual_with_norm=False,
                 device='cuda', 
                 dtype=torch.bfloat16):

        self.VAE_1 = VAE_1
        self.VAE_2 = VAE_2
        self.device = device
        self.dtype = dtype
        self.model_version = model_version  # 模型版本

        self.align_module = AlignModule(
            model_version=model_version,  # 模型版本
            img_in_channels=img_in_channels,  # 输入图像通道数
            in_channels=in_channels,  # 输入潜在变量通道数
            hidden_channels=hidden_channels,  # 隐藏层通道数
            out_channels=out_channels,  # 输出通道数
            num_blocks=num_blocks,  # 每个下采样阶段的残差块数量
            downsample_times=downsample_times,  # 下采样次数
            channel_times=channel_times,  # 通道扩展倍数
            input_types=input_types,  # 输入类型列表
            residual_with_norm=residual_with_norm  # 是否使用归一化的残差块
        ).to(device=self.device, dtype=self.dtype)
        
    def freeze_vae(self):
        """冻结VAE模型的参数"""
        for param in self.VAE_1.parameters():
            param.requires_grad = False
        for param in self.VAE_2.parameters():
            param.requires_grad = False
            
    def _get_latents_dist(self, vae, image: torch.Tensor, generator=None, transform=True):
        """
        获取VAE编码后的潜在变量特征
        Args:
            vae: VAE模型
            image (torch.Tensor): 输入图像张量，形状为 [batch_size, channels, height, width]
            generator: 随机数生成器
            transform (bool): 是否对潜在变量进行变换
        Returns:
            torch.Tensor: 编码后的潜在变量特征
        """
        encode_output = vae.encode(image)

        return encode_output.latent_dist

    def _encode_vae_image(self, vae, image: torch.Tensor, generator=None, transform=True, sample_mode="sample"):
        from diffusers.pipelines.flux.pipeline_flux_img2img import retrieve_latents
        
        if sample_mode == "sample":
            if isinstance(generator, list):
                image_latents = [
                    retrieve_latents(
                        vae.encode(image[i : i + 1]), generator=generator[i]
                    )
                    for i in range(image.shape[0])
                ]
                image_latents = torch.cat(image_latents, dim=0)
            else:
                image_latents = retrieve_latents(
                    vae.encode(image), generator=generator
                )
        elif sample_mode == "mean":
            channels = vae.config.latent_channels
            # import pdb;pdb.set_trace()
            def get_mean_latents(image):
                h = vae._encode(image)
                latent_mean = h[:, :channels, :, :]  # 假设前16个通道是潜变量
                return latent_mean
            
            if isinstance(generator, list):
                image_latents = [
                    retrieve_latents(
                        vae.encode(image[i : i + 1]), generator=generator[i]
                    )
                    for i in range(image.shape[0])
                ]
                image_latents = torch.cat(image_latents, dim=0)
            else:
                image_latents = get_mean_latents(image)
                
        if transform:
            if vae.config.shift_factor:
                image_latents = (
                    image_latents - vae.config.shift_factor
                ) * vae.config.scaling_factor
            else:
                image_latents = image_latents * vae.config.scaling_factor

        return image_latents
    
    def _decode_vae_latents(self, vae, z_pred: torch.Tensor, transform=True, do_normalize=False):

        if transform:
            if vae.config.shift_factor:
                latents = (
                        z_pred / vae.config.scaling_factor
                    ) + vae.config.shift_factor
            else:
                latents = z_pred / vae.config.scaling_factor


        images = vae.decode(latents, return_dict=False)[0]

        if do_normalize:
            # 归一化到[0, 1]
            images = (images + 1) / 2

        return images
    
    def train_step(self, x, optimizer, criterion, generator=None, image_normalize=False):
        """训练步骤"""
        self.VAE_1.eval()
        self.VAE_2.eval()
        self.align_module.train()
        optimizer.zero_grad()

        if image_normalize:
            x = x * 2 - 1  # 将图像归一化到[-1, 1]范围
        
        with torch.no_grad():
            if self.model_version == 'variational':
                # 编码VAE_1的图像
                z_vae1 = self._encode_vae_image(self.VAE_1, x, generator, sample_mode='sample')
                z_vae1 = z_vae1.to(device=self.device, dtype=self.dtype)
                # 编码VAE_2的图像
                z_vae2 = self._encode_vae_image(self.VAE_2, x, generator, sample_mode='sample')
                z_vae2 = z_vae2.to(device=self.device, dtype=self.dtype)
                
                z_vae2_dist = self._get_latents_dist(self.VAE_2, x, generator, transform=True)
                
            else:
                # 编码VAE_1的图像
                z_vae1 = self._encode_vae_image(self.VAE_1, x, generator, sample_mode='mean')
                z_vae1 = z_vae1.to(device=self.device, dtype=self.dtype)
                # 编码VAE_2的图像
                z_vae2 = self._encode_vae_image(self.VAE_2, x, generator, sample_mode='mean')
                z_vae2 = z_vae2.to(device=self.device, dtype=self.dtype)

        # 对齐模块前向传播
        z_vae1_aligned = self.align_module(x, z_vae1)
        
        if self.model_version == 'variational':
            z_vae1_aligned, z_vae1_aligned_dist = self.align_module.sample(z_vae1_aligned, generator)
        
        total_loss = 0
        loss_dict = {}
        
        # 计算损失
        for loss_type, loss_weight, addition in criterion:
            if loss_type == 'l1':
                loss = F.l1_loss(z_vae1_aligned, z_vae2)
            elif loss_type == 'mse':
                loss = F.mse_loss(z_vae1_aligned, z_vae2)
            elif loss_type == 'perceptual':
                # 假设有一个预定义的感知损失函数
                from losses import PerceptualLoss
                perceptual_loss = PerceptualLoss(device=self.device, dtype=self.dtype)
                loss = perceptual_loss(z_vae1_aligned, z_vae2) * loss_weight
            elif loss_type == 'lpips':
                # 假设有一个预定义的LPIPS损失函数
                from losses import lpips_loss
                lpips_criterion = lpips_loss(addition, device=self.device, dtype=self.dtype)
                x_vae1_aligned = self._decode_vae_latents(self.VAE_2, z_vae1_aligned, do_normalize=True)
                x_vae2 = self._decode_vae_latents(self.VAE_2, z_vae2, do_normalize=True)
                loss = lpips_criterion(x_vae1_aligned, x_vae2) * loss_weight
            elif loss_type == 'huber':
                # 假设有一个预定义的Huber损失函数
                from losses import HuberLoss
                huber_criterion = HuberLoss(delta=0.15, reduction='mean')
                loss = huber_criterion(z_vae1_aligned, z_vae2) * loss_weight
            elif loss_type == 'DCT':
                # 假设有一个预定义的DCT损失函数
                from losses import DCTLoss
                dct_criterion = DCTLoss(device=self.device, dtype=self.dtype)
                loss = dct_criterion(z_vae1_aligned, z_vae2) * loss_weight
            elif loss_type == 'KL':
                # 假设有一个预定义的KL散度损失函数
                loss = z_vae1_aligned_dist.kl(z_vae2_dist).mean() * loss_weight
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            loss_dict[loss_type] = loss.item()
            
            total_loss += loss * loss_weight
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            **loss_dict
        }
        
        
        

    def latent_reconstruction(self, vae, x, generator=None, do_normalize=False, sample_mode="sample"):
        """生成重建图像"""
        z_x = self._encode_vae_image(vae, x, generator, sample_mode=sample_mode)
        x_recon = self._decode_vae_latents(vae, z_x, do_normalize=do_normalize)
        return x_recon
        
    def save(self, path):
        """保存模型"""
        torch.save({
            'align_module_state_dict': self.align_module.state_dict()
        }, path)
        
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.align_module.load_state_dict(checkpoint['align_module_state_dict'])
