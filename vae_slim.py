import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import rgb_color_gradient

class PCAPredictor(nn.Module):
    """直接从图像预测PCA降维后的潜变量"""
    
    def __init__(self, input_channels=3, latent_pca_dim=3, reduction_factor=8):
        super().__init__()
        self.reduction_factor = reduction_factor
        
        # 编码器 - 比原始VAE简单很多
        self.encoder = nn.Sequential(
            # 下采样到 H/8 × W/8
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),  # H/2
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # H/4
            nn.SiLU(), 
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # H/8
            nn.SiLU(),
            
            # 特征提取
            nn.Conv2d(256, 512, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.SiLU(),
            
            # 输出PCA潜变量
            nn.Conv2d(512, latent_pca_dim, 1)  # 1x1卷积输出3通道
        )
        
    def forward(self, x):
        # x: [batch, 3, H, W]
        z_pca = self.encoder(x)  # [batch, 3, H/8, W/8]
        return z_pca
    
class ColorAwarePCAPredictor(nn.Module):
    """利用PCA颜色知识的预测器"""
    
    def __init__(self, input_channels=3, latent_pca_dim=3, reduction_factor=8, high_freq_enable=False):
        super().__init__()
        
        self.high_freq_enable = high_freq_enable
        
        # 亮度分支 (对应PC1)
        self.brightness_branch = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 1)  # 输出亮度分量
        )
        
        # 颜色对立分支 (对应PC2和PC3)
        self.color_branch = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 2, 1)  # 输出两个颜色对立分量
        )
        
        if self.high_freq_enable:
            # 高频分支 (可选)
            self.high_freq_branch = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 32, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 1, 1)  # 输出高频分量
                
            )
        
    def forward(self, x):
        brightness = self.brightness_branch(x)  # [batch, 1, H/8, W/8]
        color = self.color_branch(x)            # [batch, 2, H/8, W/8]
        
        if self.high_freq_enable:
            high_freq = self.high_freq_branch(x)  # [batch, 1, H/8, W/8]
            z_pca = torch.cat([brightness, color, high_freq], dim=1)  # [batch, 3, H/8, W/8]
        else:
            z_pca = torch.cat([brightness, color], dim=1)  # [batch, 3, H/8, W/8]

        return z_pca
    
class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)
        
        self.activation = nn.SiLU()
        self.shortcut = nn.Conv2d(hidden_channels, out_channels, 1) if hidden_channels != out_channels else nn.Identity()
        
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, stride=1, padding=1)
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        out += residual  # 残差连接
        return out
        

    
class ResidualDetailPredictor(nn.Module):
    """残差细节预测器"""
    
    def __init__(self, input_channels=8, output_channels=16, reduction_factor=8):
        super().__init__()
        
        self.reduction_factor = reduction_factor
        
        self.encoder_blcoks = nn.ModuleList()
        
        # 编码器 - 简单的卷积网络
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, stride=1, padding=1),  
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # H/2
            nn.SiLU(),
            
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # H/4
            nn.SiLU(),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # H/8
            nn.SiLU(),
            
            # 特征提取
            nn.Conv2d(256, 512, 3, padding=1),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.SiLU(),
            
            # 输出残差细节
            nn.Conv2d(512, output_channels, 1)  # 输出3通道残差细节
        )
    
    def coordinate_generate(self, img):
        # img: [batch, 3, H, W]
        dtype = img.dtype
        x_coord = torch.linspace(-1, 1, img.shape[3]).view(1, 1, 1, img.shape[3]).to(img.device).to(dtype)
        y_coord = torch.linspace(-1, 1, img.shape[2]).view(1, 1, img.shape[2], 1).to(img.device).to(dtype)
        
        x_coord = x_coord.expand(img.shape[0], 1, img.shape[2], img.shape[3])
        y_coord = y_coord.expand(img.shape[0], 1, img.shape[2], img.shape[3])
        
        return torch.cat([x_coord, y_coord], dim=1)

    
    def forward(self, x):
        # x: [batch, 3, H, W]
        
        x_grad = rgb_color_gradient(x, method='sobel', return_magnitude=True, normalized=True)
        coord = self.coordinate_generate(x)
        residual_detail = self.encoder(torch.cat([x, x_grad, coord], dim=1))
        return residual_detail
    
class PCAModel():
    """PCA模型，包含训练和逆变换"""
    
    def __init__(self, pca_components_freeze, pca_mean=None, freeze_pca=False, device='cuda', dtype=torch.bfloat16):
        # from sklearn.decomposition import PCA
        # self.pca = PCA(n_components=n_components)
        self.dtype = dtype
        self.freeze_pca = freeze_pca
        self.pca_components_freeze = torch.tensor(pca_components_freeze, dtype=self.dtype).to(device)  # [3, 16]
        # 如果没有提供pca_mean，则默认为零向量
        self.pca_components_shape = self.pca_components_freeze.shape
        if pca_mean is not None:
            pca_mean = nn.Parameter(torch.tensor(pca_mean, dtype=self.dtype).to(device))   # [16]
        else:
            pca_mean = nn.Parameter(torch.ones(16, dtype=self.dtype, device=device) * 0.004)   # [16]
        self.pca_mean = pca_mean  # [16]
        self.pca_mean_shape = self.pca_mean.shape
        
        # 初始化增量参数
        self.pca_components_delta = nn.Parameter(torch.randn_like(self.pca_components_freeze) * 0.001)  # [3, 16]
        
        self.pca_mean_freeze = pca_mean  # 冻结的PCA均值
        self.pca_components_freeze.requires_grad = False  # 冻结PCA组件
        self.pca_mean.requires_grad = True  # 冻结PCA均值
        self.pca_components_delta.requires_grad = True  # 允许更新的PCA组件增量
        self.pca_components = self.pca_components_freeze + self.pca_components_delta  # 初始PCA组件
        
        self.device = device
        
        if self.freeze_pca:
            self.eval()
    
    def eval(self):
        self.pca_components_delta.requires_grad = False
        self.pca_mean.requires_grad = False
    
    def transform(self, z_flat):
        """PCA变换"""
        # z_flat: [batch_size * h * w, 16]
        # 计算变换: z_pca = (z - mean) × components
        z_flat = z_flat.to(self.device)  # 确保在正确的设备上
        z_pca_flat = (z_flat - self.pca_mean) @ (self.pca_components).t()  # [batch_size * h * w, 3]
        
        return z_pca_flat  # 返回降维后的潜变量 [batch_size * h * w, 3]
    
    def inverse_transform(self, z_pca):
        """PCA逆变换"""
        # z_pca: [batch_size * h * w, 3]
        # 计算逆变换: z = z_pca × components + mean
        z_pca_flat = z_pca.to(self.device)  # 确保在正确的设备上
        z_flat = torch.matmul(z_pca_flat, (self.pca_components)) + self.pca_mean  # [batch_size * h * w, 16]      
        
        return z_flat  # 返回重构的潜变量 [batch_size * h * w, 16]
    
class PCAPipeline:
    """完整的PCA预测Pipeline"""
    
    def __init__(self, vae, pca_model, high_freq_enable=False, residual_detail=False, n_channels=16, device='cuda',dtype=torch.bfloat16):
        self.vae = vae
        self.pca_model = pca_model
        self.device = device
        self.dtype = dtype
        
        self.residual_detail = residual_detail
        if self.residual_detail:
            self.residual_detail_predictor = ResidualDetailPredictor(output_channels=n_channels).to(device=device, dtype=dtype)
        
        # 冻结VAE组件
        if vae:
            for param in self.vae.encoder.parameters():
                param.requires_grad = False
            for param in self.vae.decoder.parameters():
                param.requires_grad = False
            
        # 初始化PCA预测器
        # self.pca_predictor = PCAPredictor().to(device).bfloat16()
        self.pca_predictor = ColorAwarePCAPredictor(high_freq_enable=high_freq_enable).to(device=device, dtype=dtype)
        
    def pca_transform_batch(self, z, n_components=3, n_channels=16):
        """批量进行PCA变换"""
        batch_size, c, h, w = z.shape
        # 重塑为 [batch_size * h * w, 16]
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, n_channels)
        
        # PCA变换
        z_pca_flat = self.pca_model.transform(z_flat)
        
        # 重塑回 [batch_size, 3, h, w]
        z_pca = z_pca_flat.reshape(batch_size, h, w, n_components).permute(0, 3, 1, 2)
        return z_pca
        
    def pca_inverse_transform_batch(self, z_pca, n_components=3, n_channels=16):
        """批量进行PCA逆变换"""
        batch_size, c, h, w = z_pca.shape
        
        # 重塑为 [batch_size * h * w, 3]
        z_pca_flat = z_pca.permute(0, 2, 3, 1).reshape(-1, n_components)
        
        # PCA逆变换
        z_flat = self.pca_model.inverse_transform(z_pca_flat)
        
        # 重塑回 [batch_size, 16, h, w]
        z_reconstructed = z_flat.reshape(batch_size, h, w, n_channels).permute(0, 3, 1, 2)
        return z_reconstructed
    
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator, sample_mode="sample"):
        from diffusers.pipelines.flux.pipeline_flux_img2img import retrieve_latents
        if sample_mode == "sample":
            if isinstance(generator, list):
                image_latents = [
                    retrieve_latents(
                        self.vae.encode(image[i : i + 1]), generator=generator[i]
                    )
                    for i in range(image.shape[0])
                ]
                image_latents = torch.cat(image_latents, dim=0)
            else:
                image_latents = retrieve_latents(
                    self.vae.encode(image), generator=generator
                )
        elif sample_mode == "mean":
            def get_mean_latents(image):
                h = self.vae._encode(image)
                latent_mean = h[:, :16, :, :]  # 假设前16个通道是潜变量
                return latent_mean
            
            if isinstance(generator, list):
                image_latents = [
                    retrieve_latents(
                        self.vae.encode(image[i : i + 1]), generator=generator[i]
                    )
                    for i in range(image.shape[0])
                ]
                image_latents = torch.cat(image_latents, dim=0)
            else:
                image_latents = get_mean_latents(image)
                
        if self.vae.config.shift_factor:
            image_latents = (
                image_latents - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
        else:
            image_latents = image_latents * self.vae.config.scaling_factor
        return image_latents
    
    def _decode_vae_latents(self, z_pred: torch.Tensor, do_normalize=False):
        if self.vae.config.shift_factor:
            latents = (
                    z_pred / self.vae.config.scaling_factor
                ) + self.vae.config.shift_factor
        else:
            latents = z_pred / self.vae.config.scaling_factor
        images = self.vae.decode(latents, return_dict=False)[0]

        if do_normalize:
            # 归一化到[0, 1]
            images = (images - images.min()) / (images.max() - images.min())

        return images
    
    def train_step(self, x, optimizer, criterion, generator=None):
        """训练步骤"""
        self.pca_predictor.train()
        optimizer.zero_grad()
        
        # print("x shape:", x.shape)  # [batch, 3, H, W]
        
        # 通过VAE编码器获取真实的PCA目标
        with torch.no_grad():
            z_true = self._encode_vae_image(x, generator)  # [batch, 16, H/8, W/8]
            z_pca_true = self.pca_transform_batch(z_true)  # [batch, 3, H/8, W/8]
            z_pca_true = z_pca_true.to(device=self.device, dtype=self.dtype)  # 确保在正确的设备上
        
        # PCA预测器前向传播
        z_pca_pred = self.pca_predictor(x) # [batch, 3, H/8, W/8]
        
        # 计算损失
        pca_loss = criterion(z_pca_pred, z_pca_true)
        
        # kl散度损失（可选）
        kl_loss = F.kl_div(z_pca_true.log_softmax(dim=1), z_pca_pred.softmax(dim=1), reduction='batchmean')
        
        # 重建损失（可选）
        z_pred = self.pca_inverse_transform_batch(z_pca_pred)
        recon = self._decode_vae_latents(z_pred)  # [batch, 3, H, W]
        
        recon_loss = criterion(recon, x)
        
        # 总损失
        total_loss = pca_loss + 0.2 * kl_loss + 0.1 * recon_loss  # 可调整权重
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'pca_loss': pca_loss.item(),
            'kl_loss': kl_loss.item(),
            'recon_loss': recon_loss.item()
        }
        
    def train_step_with_residual(self, x, optimizer, criterion, generator=None):
        """训练步骤，包含残差细节预测"""
        self.pca_predictor.train()
        optimizer.zero_grad()
        
        # 通过VAE编码器获取真实的PCA目标
        with torch.no_grad():
            z_true = self._encode_vae_image(x, generator)  # [batch, 16, H/8, W/8]
            z_pca_true = self.pca_transform_batch(z_true)  # [batch, 3, H/8, W/8]
            z_pca_true = z_pca_true.to(device=self.device, dtype=self.dtype)  # 确保在正确的设备上
        
        # PCA预测器前向传播
        z_pca_pred = self.pca_predictor(x)  # [batch, 3, H/8, W/8]
        
        # 残差细节预测
        residual_detail_pred = self.residual_detail_predictor(x)  # [batch, 16, H/8, W/8]
        
        # 计算损失
        pca_loss = criterion(z_pca_pred, z_pca_true)
        
        # kl散度损失（可选）
        kl_loss = F.kl_div(z_pca_true.log_softmax(dim=1), z_pca_pred.softmax(dim=1), reduction='batchmean')
        
        # 重建损失（可选）
        z_pred = self.pca_inverse_transform_batch(z_pca_pred)
        recon = self._decode_vae_latents(z_pred) + residual_detail_pred  # [batch, 3, H, W]
        
        recon_loss = criterion(recon, x)
        
        # 总损失
        total_loss = pca_loss + 0.2 * kl_loss + 0.1 * recon_loss  # 可调整权重
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'pca_loss': pca_loss.item(),
            'kl_loss': kl_loss.item(),
            'recon_loss': recon_loss.item()
        }
    
    def diff_between_pca_and_latents(self, x, generator=None):
        z_x = self._encode_vae_image(x, generator)
        pca_pred = self.pca_predictor(x)
        z_pca_pred = self.pca_inverse_transform_batch(pca_pred)
        diff = self.residual_detail_predictor(x)
        z_pred = z_pca_pred + diff
        return z_x, z_pca_pred, z_pred
        
    def eval(self):
        if self.vae:
            self.vae.eval()
        self.pca_predictor.eval()
        if self.residual_detail:
            self.residual_detail_predictor.eval()
        if self.pca_model:
            self.pca_model.eval()
        
    def generate_for_comparsion(self, x, n_components=16, n_channels=16,generator=None, x_recon=False):
        """生成重建图像"""
        self.pca_predictor.eval()
        
        with torch.no_grad():
            z_pca_pred = self.pca_predictor(x)
            z_pred = self.pca_inverse_transform_batch(z_pca_pred, n_components=n_components, n_channels=n_channels)
            if self.residual_detail:
                self.residual_detail_predictor.eval()
                from utils import rgb_color_gradient
                x_grad = rgb_color_gradient(x, method='sobel', return_magnitude=True, normalized=True, dtype=self.dtype)  # [batch, 3, H, W]
                residual_detail_pred = self.residual_detail_predictor(x_grad)  # [batch, 16, H/8, W/8]
                z_pred += residual_detail_pred  # 将残差细节添加到潜变量中
            pca_recon = self._decode_vae_latents(z_pred)  # [batch, 3, H, W]
            
            if x_recon:
                x_recon = self.pca_reconstruction(x, n_components=n_components, n_channels=n_channels)
            
        return pca_recon, z_pca_pred, x_recon
    
    def pca_analysis(self, x, generator=None, n_components=16, n_channels=16, do_normalize=False):
        """PCA分析"""
        z_x = self._encode_vae_image(x, generator)  # [batch, 16, H/8, W/8]
        z_x_pca = self.pca_transform_batch(z_x,n_components=n_components, n_channels=n_channels)  # [batch, 3, H/8, W/8]
        z_x_pca_inverse = self.pca_inverse_transform_batch(z_x_pca, n_components=n_components, n_channels=n_channels)  # [batch, 16, H/8, W/8]
        x_recon = self._decode_vae_latents(z_x_pca_inverse, do_normalize)  # [batch, 3, H, W]
        
        import torch
        # 假设 z_x, z_x_pca_inverse 的 shape 是 (N, C, ...) 或 (N, D) 且在 cuda:0
        z1 = z_x.detach().to(torch.float32)
        z2 = z_x_pca_inverse.detach().to(torch.float32)

        # 基本统计
        mean1 = z1.mean().item()
        std1  = z1.std().item()
        mean2 = z2.mean().item()
        std2  = z2.std().item()
        print("mean/std z1:", mean1, std1)
        print("mean/std z2:", mean2, std2)

        # 全局 MSE & 相对 MSE (相对于 z 的能量)
        mse = torch.mean((z1 - z2)**2).item()
        energy = torch.mean(z1**2).item()
        rel_mse = mse / (energy + 1e-12)
        print("mse:", mse, "energy:", energy, "relative_mse:", rel_mse)

        # L2 比率与余弦相似度
        l2_diff = (z1 - z2).permute(0, 2, 3, 1).reshape(-1, n_channels)  # reshape to (N, C)
        l2_diff = torch.norm(l2_diff, dim=1) # per-sample
        l2_z1   = torch.norm(z1.permute(0, 2, 3, 1).reshape(-1, n_channels), dim=1)
        l2_ratio = l2_diff / (l2_z1 + 1e-12)
        cos_sim = torch.nn.functional.cosine_similarity(z1.permute(0, 2, 3, 1).reshape(-1, n_channels), z2.permute(0, 2, 3, 1).reshape(-1, n_channels), dim=1).mean().item()
        print("mean per-sample L2 ratio:", l2_ratio, "mean cosine sim:", cos_sim)

        # per-channel mse（若 shape (N, C, ...)）
        # if z1.dim() >= 2:
        #     # collapse spatial dims
        #     z1_flat = z1.view(z1.shape[0], z1.shape[1], -1).mean(dim=2)  # (N,C)
        #     z2_flat = z2.view(z2.shape[0], z2.shape[1], -1).mean(dim=2)
        #     per_channel_mse = torch.mean((z1_flat - z2_flat)**2, dim=0)  # (C,)
        #     print("per-channel mse:", per_channel_mse.cpu().numpy())

        # bfloat16 -> float32 比较（检验精度影响）
        z1_bf = z_x.clone().to(torch.float32)
        z2_bf = z_x_pca_inverse.clone().to(torch.float32)
        print("mse (float32):", torch.mean((z1_bf.to(torch.float32)-z2_bf.to(torch.float32))**2).item())

        
        return x_recon
    
    def pca_reconstruction(self, x, generator=None, n_components=3, n_channels=16, do_normalize=False):
        z_x = self._encode_vae_image(x, generator)  # [batch, 16, H/8, W/8]
        z_x_pca = self.pca_transform_batch(z_x,n_components=n_components, n_channels=n_channels)  # [batch, 3, H/8, W/8]
        z_x_pca_inverse = self.pca_inverse_transform_batch(z_x_pca, n_components=n_components, n_channels=n_channels)  # [batch, 16, H/8, W/8]
        x_recon = self._decode_vae_latents(z_x_pca_inverse, do_normalize)  # [batch, 3, H, W]
        return x_recon
    def latent_reconstruction(self, x, generator=None):
        """生成重建图像"""
        z_x = self._encode_vae_image(x, generator)
        x_recon = self._decode_vae_latents(z_x)
        return x_recon
        
    def save(self, path):
        """保存模型"""
        torch.save({
            'pca_predictor_state_dict': self.pca_predictor.state_dict(),
            'residual_detail_predictor_state_dict': self.residual_detail_predictor.state_dict() if self.residual_detail else None,
            'pca_model_state_dict': {
                'pca_components': self.pca_model.pca_components,
                'pca_mean': self.pca_model.pca_mean,
                'pca_components_delta': self.pca_model.pca_components_delta
            }
        }, path)
        
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.pca_predictor.load_state_dict(checkpoint['pca_predictor_state_dict'])
        if self.residual_detail:
            self.residual_detail_predictor.load_state_dict(checkpoint['residual_detail_predictor_state_dict'])
        else:
            self.residual_detail_predictor = None
        pca_state_dict = checkpoint['pca_model_state_dict']
        self.pca_model.pca_components = pca_state_dict['pca_components'].to(self.device)
        self.pca_model.pca_mean = pca_state_dict['pca_mean'].to(self.device)
        self.pca_model.pca_components_delta = pca_state_dict['pca_components_delta'].to(self.device)
    
