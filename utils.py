import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def pca_visualize_and_save(
    feature_map: torch.Tensor, save_path: str, cluster_size: int, n_components=2
):
    # 确保输入是torch.Tensor类型
    if not isinstance(feature_map, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")

    # 将torch.Tensor转换为numpy.ndarray
    feature_map_np = feature_map.to(torch.float32).detach().cpu().numpy()

    # 展开特征图为一个二维数组 (batch_size * height * width, channels)
    # 这里假设batch_size为1，如果不是1则需要额外处理
    if feature_map_np.shape[0] != 1:
        raise ValueError("Batch size must be 1 for this function")

    flat_feature_map = feature_map_np[0].reshape(-1, feature_map_np.shape[1])

    # 执行PCA降维到2维（方便可视化）
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(flat_feature_map)

    # 使用KMeans进行聚类（例如，聚成5类）
    kmeans = KMeans(n_clusters=cluster_size, random_state=0)
    labels = kmeans.fit_predict(reduced_data)

    # 可视化聚类结果并保存图像
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis"
    )
    plt.colorbar(scatter)
    plt.title("PCA and KMeans Clustering of Feature Map")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(save_path + "/pca_clustering.png")  # 保存散点图到指定路径
    plt.close()  # 关闭图形窗口，因为我们不需要显示它

    # 如果需要保存聚类的标签图像（height x width），这里我们简单地将标签映射回原空间大小
    # 并生成一个伪彩色图像（注意：这只是一个简单的示例）
    height, width = feature_map_np.shape[2], feature_map_np.shape[3]
    label_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            index = i * width + j
            label_image[i, j] = labels[index]

    # 使用colormap将标签映射为颜色图像
    color_map = plt.cm.viridis(
        label_image / np.max(label_image, initial=0, where=np.isfinite)
    )  # 防止除以0的错误
    color_image = (color_map[:, :, :3] * 255).astype(np.uint8)

    # 将numpy数组转换为PIL图像并保存
    from PIL import Image
    img = Image.fromarray(color_image)
    img.save(save_path + "/cluster_label_image.png")


# 示例用法
# 假设feature_map是一个形状为(1, 256, 64, 64)的torch.Tensor特征图
# np.random.seed(0)  # 如果需要可重复的结果，可以取消注释这行代码
# feature_map = torch.rand((1, 256, 64, 64))  # 创建一个随机特征图作为示例
# pca_visualize_and_save(feature_map, 'path/to/save')  # 调用函数并指定保存路径

def visualize_spectrum(image):
    """可视化图像频谱"""
    import matplotlib.pyplot as plt
    with torch.no_grad():
        # 计算频谱
        image_fft = torch.fft.fft2(image.float(), dim=(-2, -1))
        
        # shift频谱中心
        image_fft = torch.fft.fftshift(image_fft, dim=(-2, -1))
        
        # 计算幅度谱
        image_mag = torch.abs(image_fft).mean(dim=1)  # 平均所有通道
        
        # 可视化
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(np.log(image_mag[0].cpu().numpy() + 1e-8), cmap='viridis')
        ax.set_title('Image Spectrum')
        plt.colorbar(im, ax=ax)
        
        return fig

def visualize_spectrum_comparison(original, vae, pca_model, n_components=16, n_channels=16):
    """可视化频谱对比"""
    import matplotlib.pyplot as plt
    with torch.no_grad():
        # 获取潜变量
        z_original = vae.encode(original).latent_dist.sample()
        b, _, h, w = z_original.shape
        
        z_pca = pca_model.transform(
            z_original.permute(0, 2, 3, 1).reshape(-1, n_channels)
        )
        z_pca_recon = pca_model.inverse_transform(z_pca)
        z_pca_recon = z_pca_recon.reshape(b, h, w, n_channels).permute(0, 3, 1, 2)
        
        # 计算频谱
        z_original_fft = torch.fft.fft2(z_original.float(), dim=(-2, -1))
        z_pca_fft = torch.fft.fft2(z_pca_recon.float(), dim=(-2, -1))
        
        # shift频谱中心
        z_original_fft = torch.fft.fftshift(z_original_fft, dim=(-2, -1))
        z_pca_fft = torch.fft.fftshift(z_pca_fft, dim=(-2, -1))
        
        z_original_mag = torch.abs(z_original_fft).mean(dim=1)  # 平均所有通道
        z_pca_mag = torch.abs(z_pca_fft).mean(dim=1)
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始频谱
        im1 = axes[0, 0].imshow(np.log(z_original_mag[0].cpu().numpy() + 1e-8), cmap='viridis')
        axes[0, 0].set_title('Original Latent Spectrum')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # PCA重建频谱
        im2 = axes[0, 1].imshow(np.log(z_pca_mag[0].cpu().numpy() + 1e-8), cmap='viridis')
        axes[0, 1].set_title('PCA Reconstructed Spectrum')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # pdb.set_trace()
        # 差异
        diff = z_pca_mag[0] - z_original_mag[0]
        im3 = axes[1, 0].imshow(diff.cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Spectral Difference')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 径向平均
        radial_avg_original = radial_profile(z_original_mag[0])
        radial_avg_pca = radial_profile(z_pca_mag[0])
        
        axes[1, 1].plot(radial_avg_original, label='Original')
        axes[1, 1].plot(radial_avg_pca, label='PCA')
        axes[1, 1].set_title('Radial Average')
        axes[1, 1].legend()
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].set_ylabel('Magnitude')
        
        plt.tight_layout()
        
        # plt.savefig('spectrum_comparison.png')
        
        return fig
    
# 计算RGB图色彩变化梯度
def rgb_grad_map(original_images):
    import cv2 
    # 转换为numpy数组
    original_np = original_images.cpu().numpy().transpose(0, 2, 3, 1)
    grad_maps = []
    
    # 检查并调整数值范围
    if original_np.max() <= 1.0:
        original_np = (original_np * 255).astype(np.uint8)
    else:
        original_np = original_np.astype(np.uint8)
        
    for img in original_np:
        # 计算梯度
        grad = cv2.Laplacian(img, cv2.CV_64F)
        grad_maps.append(grad)
    
    grad_maps = np.array(grad_maps)
    grad_maps = torch.tensor(grad_maps).permute(0, 3, 1, 2)  # 转换回torch张量并调整维度
    
    from matplotlib import pyplot as plt
    if len(grad_maps) == 1:
        ncols = 2
    else:
        ncols = len(grad_maps)
    
    # 可视化梯度图
    fig, axes = plt.subplots(1, ncols, figsize=(15, 5))
    # import pdb;pdb.set_trace()
    for i, grad in enumerate(grad_maps):
        axes[i].imshow(grad.permute(1, 2, 0).cpu().numpy(), vmax=grad.max(), vmin=grad.min(), cmap='coolwarm')
        axes[i].set_title(f'Gradient Map {i+1}')
        plt.colorbar(axes[i].images[0], ax=axes[i], fraction=0.046, pad=0.04)
        axes[i].axis('off')
    plt.tight_layout()
    
    return grad_maps, fig

def rgb_color_gradient(images, method='sobel', return_magnitude=True, normalized=True, dtype=torch.bfloat16):
    """
    PyTorch实现的RGB图像颜色梯度提取函数
    
    Args:
        images: 输入图像 tensor [B, C, H, W] 或 [C, H, W]，值范围[0,1]或[0,255]
        method: 梯度计算方法 'sobel' | 'scharr' | 'laplacian'
        return_magnitude: 是否返回梯度幅值，否则返回x,y方向梯度
        normalized: 是否归一化到[0,1]范围
    
    Returns:
        梯度图 tensor [B, C, H, W] 或 [B, 1, H, W] (如果return_magnitude=True)
    """

    # 确保输入是4D tensor [B, C, H, W]
    if images.dim() == 3:
        images = images.unsqueeze(0)
    
    batch_size, channels, height, width = images.shape
    
    # 确保图像在[0,1]范围
    if images.max() > 1.0:
        images = images / 255.0
    
    # 创建梯度滤波器
    if method == 'sobel':
        # Sobel算子
        kernel_x = torch.tensor([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]], dtype=dtype)
        kernel_y = torch.tensor([[-1, -2, -1], 
                                [0, 0, 0], 
                                [1, 2, 1]], dtype=dtype)
    elif method == 'scharr':
        # Scharr算子（对噪声更敏感）
        kernel_x = torch.tensor([[-3, 0, 3], 
                                [-10, 0, 10], 
                                [-3, 0, 3]], dtype=dtype)
        kernel_y = torch.tensor([[-3, -10, -3], 
                                [0, 0, 0], 
                                [3, 10, 3]], dtype=dtype)
    elif method == 'laplacian':
        # Laplacian算子（二阶导数）
        kernel = torch.tensor([[0, 1, 0], 
                              [1, -4, 1], 
                              [0, 1, 0]], dtype=dtype)
    else:
        raise ValueError("Method must be 'sobel', 'scharr', or 'laplacian'")
    
    # 将滤波器移动到设备上并调整为4D [out_ch, in_ch, H, W]
    device = images.device
    
    if method == 'laplacian':
        kernel = kernel.to(device).view(1, 1, 3, 3)
        kernel = kernel.repeat(channels, 1, 1, 1)  # [C, 1, 3, 3]
    else:
        kernel_x = kernel_x.to(device).view(1, 1, 3, 3)
        kernel_y = kernel_y.to(device).view(1, 1, 3, 3)
        kernel_x = kernel_x.repeat(channels, 1, 1, 1)  # [C, 1, 3, 3]
        kernel_y = kernel_y.repeat(channels, 1, 1, 1)  # [C, 1, 3, 3]
    
    # 填充设置
    padding = 1
    
    if method == 'laplacian':
        # Laplacian直接计算
        gradients = F.conv2d(images, kernel, padding=padding, groups=channels)
        if return_magnitude:
            gradients = torch.abs(gradients)
    else:
        # Sobel/Scharr计算x,y方向梯度
        grad_x = F.conv2d(images, kernel_x, padding=padding, groups=channels)
        grad_y = F.conv2d(images, kernel_y, padding=padding, groups=channels)
        
        if return_magnitude:
            # 计算梯度幅值
            gradients = torch.sqrt(grad_x**2 + grad_y**2)
        else:
            # 返回x,y方向梯度
            gradients = torch.stack([grad_x, grad_y], dim=1)  # [B, 2, C, H, W]
            gradients = gradients.view(batch_size, 2 * channels, height, width)
    
    # 归一化到[0,1]范围
    if normalized and return_magnitude:
        # 对每个样本单独归一化
        normalized_gradients = []
        for i in range(batch_size):
            sample_grad = gradients[i]
            if sample_grad.max() > sample_grad.min():
                norm_sample = (sample_grad - sample_grad.min()) / (sample_grad.max() - sample_grad.min())
            else:
                norm_sample = torch.zeros_like(sample_grad)
            normalized_gradients.append(norm_sample)
        gradients = torch.stack(normalized_gradients)
    
    return gradients

# 比较RGB图像的色彩变化梯度与（重建图与原图之差）,
def rgb_grad_comparison(grad_maps, minus_maps):
    import matplotlib.pyplot as plt
    
    # 转换为numpy数组
    grad_np = grad_maps.cpu().numpy().transpose(0, 2, 3, 1)
    minus_np = minus_maps.cpu().numpy().transpose(0, 2, 3, 1)
    
    # 相关性分析
    correlation = np.corrcoef(grad_np.reshape(len(grad_np), -1), minus_np.reshape(len(minus_np), -1))[0, 1]
    print(f'Correlation between gradient maps and minus maps: {correlation:.4f}')
    
    if len(grad_np) == 1:
        ncols = 2
    else:
        ncols = len(grad_np)
    
    fig, axes = plt.subplots(2, ncols, figsize=(15, 6))
    
    for i in range(len(grad_np)):
        # import pdb;pdb.set_trace()
        axes[0, i].imshow(grad_np[i], vmax=grad_np[i].max(), vmin=grad_np[i].min(), cmap='coolwarm')
        axes[0, i].set_title(f'Gradient Map {i+1}')
        plt.colorbar(axes[0, i].images[0], ax=axes[0, i], fraction=0.046, pad=0.04)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(minus_np[i], vmax=minus_np[i].max(), vmin=minus_np[i].min(), cmap='coolwarm')
        axes[1, i].set_title(f'Minus Map {i+1}')
        plt.colorbar(axes[1, i].images[0], ax=axes[1, i], fraction=0.046, pad=0.04)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    return fig


    


def rgb_edge_analysis(original, reconstructed):
    """计算RGB图像的色彩变化梯度"""
    import cv2
    import matplotlib.pyplot as plt
    
    # 转换为numpy数组
    original_np = original.cpu().numpy().transpose(1, 2, 0)
    reconstructed_np = reconstructed.cpu().numpy().transpose(1, 2, 0)
    
    # 计算梯度
    grad_original = cv2.Laplacian(original_np, cv2.CV_64F)
    grad_reconstructed = cv2.Laplacian(reconstructed_np, cv2.CV_64F)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image')
    
    axes[1].imshow(reconstructed_np)
    axes[1].set_title('Reconstructed Image')
    
    axes[2].imshow(np.abs(grad_original - grad_reconstructed), cmap='gray')
    axes[2].set_title('Gradient Difference')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    
    return fig

def radial_profile(data):
    """计算径向平均"""
    data = data.cpu().numpy()  # 确保数据在CPU上
    center = np.array(data.shape) // 2
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def log_model_parameters(writer, model, epoch):
    """记录模型参数统计"""
    for name, param in model.named_parameters():
        writer.add_histogram(f'Parameters/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

def log_training_progress(writer, images, reconstructions, epoch):
    """记录训练进度图像"""
    # 将图像网格化并记录
    writer.add_images('Training/Input_Images', images[:8], epoch)
    writer.add_images('Training/Reconstructed_Images', reconstructions[:8], epoch)
    
def plot_image(image, title=None, save_path=None):
    """
    绘制单张图像
    Args:
        image: 输入图像 tensor [C, H, W] 或 [B, C, H, W]
        title: 图像标题
        save_path: 保存路径，如果为None则显示图像
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)  # 添加batch维度
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
    if title:
        plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
    
def save_image(save_path, image):
    """保存单张图像"""
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)
    return image

def plot_images(images, titles=None, ncols=4, save_path=None):
    nrows = (len(images) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axs = axs.flatten()
    
    for i, img in enumerate(images):
        axs[i].imshow(img.permute(1, 2, 0).cpu().numpy())
        if titles:
            axs[i].set_title(titles[i])
        axs[i].axis('off')
    
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
def plot_each_channel_figure_list(tensor_list, ncols=4 ,save_path=None):
    """
    输入一个列表，列表中的每个元素是一个张量，张量的维度为[3, H, W]
    对于每个张量，分别绘制三个通道的图像，带有标题和色值条
    ncols: 图像的列数
    """
    tensor_list = [tensor.cpu() for tensor in tensor_list]
    n = len(tensor_list)
    fig, axs = plt.subplots(nrows=(n + ncols - 1) // ncols, ncols=ncols * 3, figsize=(ncols*15, 5 * ((n + ncols - 1) // ncols)))
    axs = axs.flatten()
    
    for i, tensor in enumerate(tensor_list):
        for j in range(3):
            axs[i * 3 + j].imshow(tensor[j].cpu().numpy())
            axs[i * 3 + j].set_title(f'Individual {i}, Channel {j+1}')
            axs[i * 3 + j].axis('off')
            
            # 添加色值条
            cbar = plt.colorbar(axs[i * 3 + j].images[0], ax=axs[i * 3 + j], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.set_title('Value', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path)
    
def plot_figure(tensor, save_path=None):
    # tensor: [B, C, H, W]
    plt.figure(figsize=(10, 10))
    plt.imshow(tensor.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def analyze_edge_reconstruction(original_img, reconstructed_img, save_path=None):
    """
    分析原图与重建图在边缘信息上的差异
    
    Args:
        original_img: 原始图像 tensor [C, H, W] 或 [B, C, H, W]
        reconstructed_img: 重建图像 tensor [C, H, W] 或 [B, C, H, W]
        save_path: 结果保存路径，如果为None则显示图像
    
    Returns:
        dict: 包含各种分析结果的字典
    """
    
    # 确保图像是tensor且维度正确
    if original_img.dim() == 3:
        original_img = original_img.unsqueeze(0)
        reconstructed_img = reconstructed_img.unsqueeze(0)
    
    batch_size = original_img.shape[0]
    
    results = {}
    
    for i in range(batch_size):
        # 获取当前样本
        orig = original_img[i].detach().cpu()
        recon = reconstructed_img[i].detach().cpu()
        
        # 转换为numpy用于OpenCV处理
        if orig.shape[0] == 1:  # 灰度图
            orig_np = orig.squeeze().numpy()
            recon_np = recon.squeeze().numpy()
        else:  # RGB图
            orig_np = orig.permute(1, 2, 0).numpy()
            recon_np = recon.permute(1, 2, 0).numpy()
            # 转换为灰度
            orig_np = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
            recon_np = cv2.cvtColor(recon_np, cv2.COLOR_RGB2GRAY)
        
        # 归一化到0-255
        orig_np = (orig_np * 255).astype(np.uint8)
        recon_np = (recon_np * 255).astype(np.uint8)
        
        # 1. 计算重建误差图
        error_map = np.abs(orig_np.astype(float) - recon_np.astype(float))
        
        # 2. 提取原图的边缘（使用Canny边缘检测）
        edges_original = cv2.Canny(orig_np, 50, 150)
        
        # 3. 提取重建图的边缘
        edges_reconstructed = cv2.Canny(recon_np, 50, 150)
        
        # 4. 计算边缘差异
        edges_diff = np.abs(edges_original.astype(float) - edges_reconstructed.astype(float))
        
        # 5. 相关性分析
        # 将误差图缩放到与边缘图相同的尺度进行比较
        error_map_normalized = (error_map / error_map.max() * 255).astype(np.uint8)
        
        # 计算误差图与边缘图的相关性
        correlation = np.corrcoef(error_map.flatten(), edges_original.flatten())[0, 1]
        
        # 6. 可视化
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # 第一行：原图相关
        axes[0, 0].imshow(orig_np, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(edges_original, cmap='gray')
        axes[0, 1].set_title('Original Edges')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(recon_np, cmap='gray')
        axes[0, 2].set_title('Reconstructed Image')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(edges_reconstructed, cmap='gray')
        axes[0, 3].set_title('Reconstructed Edges')
        axes[0, 3].axis('off')
        
        # 第二行：分析相关
        im1 = axes[1, 0].imshow(error_map, cmap='hot')
        axes[1, 0].set_title('Reconstruction Error')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0])
        
        axes[1, 1].imshow(edges_diff, cmap='hot')
        axes[1, 1].set_title('Edge Difference')
        axes[1, 1].axis('off')
        
        # 重叠显示：边缘图上的误差
        axes[1, 2].imshow(edges_original, cmap='gray')
        overlap = axes[1, 2].imshow(error_map, cmap='hot', alpha=0.5)
        axes[1, 2].set_title('Edges + Error Overlay')
        axes[1, 2].axis('off')
        plt.colorbar(overlap, ax=axes[1, 2])
        
        # 统计信息
        axes[1, 3].axis('off')
        stats_text = f"""
        Statistical Analysis:
        - Max Error: {error_map.max():.2f}
        - Mean Error: {error_map.mean():.2f}
        - Edge-Error Correlation: {correlation:.3f}
        - Edge Preservation Rate: {(1 - edges_diff.sum() / edges_original.sum()):.3f}
        """
        axes[1, 3].text(0.1, 0.9, stats_text, transform=axes[1, 3].transAxes, 
                       fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_sample_{i}.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
        # 存储结果
        results[f'sample_{i}'] = {
            'error_map': error_map,
            'edges_original': edges_original,
            'edges_reconstructed': edges_reconstructed,
            'edges_diff': edges_diff,
            'correlation': correlation,
            'max_error': error_map.max(),
            'mean_error': error_map.mean()
        }
    
    return results

def batch_edge_analysis(original_imgs, reconstructed_imgs):
    """
    批量分析多张图像的边缘重建效果
    """
    all_correlations = []
    all_edge_preservation = []
    
    for i in range(len(original_imgs)):
        orig = original_imgs[i]
        recon = reconstructed_imgs[i]
        
        # 使用上面的函数进行分析
        result = analyze_edge_reconstruction(orig, recon, save_path=f"batch_analysis_{i}")
        
        # 收集统计信息
        sample_result = result[f'sample_{0}']  # 因为每张图单独处理
        all_correlations.append(sample_result['correlation'])
        edge_preservation = 1 - (sample_result['edges_diff'].sum() / 
                               sample_result['edges_original'].sum())
        all_edge_preservation.append(edge_preservation)
    
    # 打印总体统计
    print("=== Batch Edge Analysis Summary ===")
    print(f"Average Edge-Error Correlation: {np.mean(all_correlations):.3f} ± {np.std(all_correlations):.3f}")
    print(f"Average Edge Preservation Rate: {np.mean(all_edge_preservation):.3f} ± {np.std(all_edge_preservation):.3f}")
    print(f"Correlation Range: [{np.min(all_correlations):.3f}, {np.max(all_correlations):.3f}]")
    
    return {
        'correlations': all_correlations,
        'edge_preservation': all_edge_preservation
    }


def rgb_edge_detection_separate(rgb_image, low_threshold=50, high_threshold=150):
    """
    分别对RGB三个通道进行边缘检测，然后合并
    """
    if isinstance(rgb_image, torch.Tensor):
        rgb_image = rgb_image.permute(1, 2, 0).numpy()
    
    if rgb_image.max() <= 1.0:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # 分离通道
    b, g, r = cv2.split(rgb_image)
    
    # 分别检测边缘
    edges_b = cv2.Canny(b, low_threshold, high_threshold)
    edges_g = cv2.Canny(g, low_threshold, high_threshold)  
    edges_r = cv2.Canny(r, low_threshold, high_threshold)
    
    # 合并边缘（逻辑或）
    combined_edges = np.logical_or.reduce([edges_b, edges_g, edges_r]).astype(np.uint8) * 255
    
    return combined_edges, (edges_b, edges_g, edges_r)


def rgb_edge_detection_grayscale(rgb_image, low_threshold=50, high_threshold=150):
    """
    转换为灰度图后进行边缘检测
    """
    if isinstance(rgb_image, torch.Tensor):
        rgb_image = rgb_image.permute(1, 2, 0).numpy()
    
    if rgb_image.max() <= 1.0:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    return edges, gray


def color_gradient_edges(rgb_image):
    """
    计算颜色空间中的梯度来检测颜色变化
    """
    if isinstance(rgb_image, torch.Tensor):
        rgb_image = rgb_image.permute(1, 2, 0).numpy()
    
    if rgb_image.max() <= 1.0:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # 转换为Lab颜色空间，对亮度变化更敏感
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 计算每个通道的梯度
    grad_x_l = cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_l = cv2.Sobel(l, cv2.CV_64F, 0, 1, ksize=3)
    grad_l = np.sqrt(grad_x_l**2 + grad_y_l**2)
    
    grad_x_a = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_a = cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3)
    grad_a = np.sqrt(grad_x_a**2 + grad_y_a**2)
    
    grad_x_b = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_b = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)
    grad_b = np.sqrt(grad_x_b**2 + grad_y_b**2)
    
    # 合并颜色梯度
    color_gradient = (grad_l + grad_a + grad_b) / 3.0
    color_gradient = (color_gradient / color_gradient.max() * 255).astype(np.uint8)
    
    return color_gradient, (grad_l, grad_a, grad_b)


def color_difference_edges(rgb_image, threshold=30):
    """
    通过计算相邻像素的颜色差异来检测颜色边界
    """
    if isinstance(rgb_image, torch.Tensor):
        rgb_image = rgb_image.permute(1, 2, 0).numpy()
    
    if rgb_image.max() <= 1.0:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # 计算RGB空间中相邻像素的欧氏距离
    diff_x = np.sqrt(np.sum((rgb_image[1:, :, :] - rgb_image[:-1, :, :]) ** 2, axis=2))
    diff_y = np.sqrt(np.sum((rgb_image[:, 1:, :] - rgb_image[:, :-1, :]) ** 2, axis=2))
    
    # 填充以保持原始尺寸
    diff_x = np.pad(diff_x, ((0, 1), (0, 0)), mode='constant')
    diff_y = np.pad(diff_y, ((0, 0), (0, 1)), mode='constant')
    
    # 合并方向差异
    color_diff = np.sqrt(diff_x**2 + diff_y**2)
    
    # 二值化
    color_edges = (color_diff > threshold).astype(np.uint8) * 255
    
    return color_edges, color_diff

def comprehensive_rgb_edge_analysis(original_rgb, reconstructed_rgb, save_path=None):
    """
    对RGB图像进行全面的边缘和颜色变化分析
    """
    # 确保是numpy数组
    if isinstance(original_rgb, torch.Tensor):
        original_rgb = original_rgb.permute(1, 2, 0).detach().cpu().numpy()
    if isinstance(reconstructed_rgb, torch.Tensor):
        reconstructed_rgb = reconstructed_rgb.permute(1, 2, 0).detach().cpu().numpy()
    
    # 归一化到0-255
    if original_rgb.max() <= 1.0:
        original_rgb = (original_rgb * 255).astype(np.uint8)
    if reconstructed_rgb.max() <= 1.0:
        reconstructed_rgb = (reconstructed_rgb * 255).astype(np.uint8)
    
    # 1. 多通道边缘检测
    edges_orig_combined, edges_orig_channels = rgb_edge_detection_separate(original_rgb)
    edges_recon_combined, edges_recon_channels = rgb_edge_detection_separate(reconstructed_rgb)
    
    # 2. 灰度边缘检测
    edges_orig_gray, gray_orig = rgb_edge_detection_grayscale(original_rgb)
    edges_recon_gray, gray_recon = rgb_edge_detection_grayscale(reconstructed_rgb)
    
    # 3. 颜色梯度检测
    color_grad_orig, color_grad_channels_orig = color_gradient_edges(original_rgb)
    color_grad_recon, color_grad_channels_recon = color_gradient_edges(reconstructed_rgb)
    
    # 4. 色差边缘检测
    color_diff_edges_orig, color_diff_orig = color_difference_edges(original_rgb)
    color_diff_edges_recon, color_diff_recon = color_difference_edges(reconstructed_rgb)
    
    # 5. 计算重建误差（RGB空间）
    error_map_rgb = np.abs(original_rgb.astype(float) - reconstructed_rgb.astype(float))
    error_map_total = np.mean(error_map_rgb, axis=2)  # 平均三个通道
    
    # 可视化
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    
    # 第一行：原始图像和重建图像
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(reconstructed_rgb)
    axes[0, 1].set_title('Reconstructed RGB')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(error_map_total, cmap='hot')
    axes[0, 2].set_title('RGB Error Map')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(gray_orig, cmap='gray')
    axes[0, 3].set_title('Original Grayscale')
    axes[0, 3].axis('off')
    
    # 第二行：多通道边缘检测
    axes[1, 0].imshow(edges_orig_combined, cmap='gray')
    axes[1, 0].set_title('Orig: Multi-channel Edges')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edges_recon_combined, cmap='gray')
    axes[1, 1].set_title('Recon: Multi-channel Edges')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.abs(edges_orig_combined - edges_recon_combined), cmap='hot')
    axes[1, 2].set_title('Multi-channel Edge Diff')
    axes[1, 2].axis('off')
    
    # 显示单个通道的边缘
    axes[1, 3].imshow(edges_orig_channels[0], cmap='Reds', alpha=0.5)
    axes[1, 3].imshow(edges_orig_channels[1], cmap='Greens', alpha=0.5)
    axes[1, 3].imshow(edges_orig_channels[2], cmap='Blues', alpha=0.5)
    axes[1, 3].set_title('RGB Channels Edges')
    axes[1, 3].axis('off')
    
    # 第三行：颜色梯度检测
    axes[2, 0].imshow(color_grad_orig, cmap='viridis')
    axes[2, 0].set_title('Orig: Color Gradient')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(color_grad_recon, cmap='viridis')
    axes[2, 1].set_title('Recon: Color Gradient')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(np.abs(color_grad_orig - color_grad_recon), cmap='hot')
    axes[2, 2].set_title('Color Gradient Diff')
    axes[2, 2].axis('off')
    
    # 色差边缘
    axes[2, 3].imshow(color_diff_edges_orig, cmap='gray')
    axes[2, 3].set_title('Orig: Color Difference Edges')
    axes[2, 3].axis('off')
    
    # 第四行：相关性分析
    # 计算各种边缘与误差的相关性
    edge_methods = {
        'Multi-channel': edges_orig_combined,
        'Grayscale': edges_orig_gray,
        'Color Gradient': color_grad_orig,
        'Color Difference': color_diff_edges_orig
    }
    
    correlations = {}
    for name, edge_map in edge_methods.items():
        corr = np.corrcoef(error_map_total.flatten(), edge_map.flatten())[0, 1]
        correlations[name] = corr
    
    # 绘制相关性条形图
    methods = list(correlations.keys())
    corr_values = list(correlations.values())
    
    bars = axes[3, 0].bar(methods, corr_values, color=['red', 'blue', 'green', 'purple'])
    axes[3, 0].set_title('Edge-Error Correlations')
    axes[3, 0].set_ylabel('Correlation Coefficient')
    axes[3, 0].tick_params(axis='x', rotation=45)
    
    # 在条形上添加数值
    for bar, value in zip(bars, corr_values):
        axes[3, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
    
    # 统计信息
    stats_text = f"""
    Statistical Summary:
    - RGB Error Mean: {error_map_total.mean():.2f}
    - RGB Error Std: {error_map_total.std():.2f}
    - Best Edge Correlation: {max(corr_values):.3f}
    - Color Preservation: {1 - np.mean(error_map_rgb) / 255:.3f}
    """
    axes[3, 1].axis('off')
    axes[3, 1].text(0.1, 0.9, stats_text, transform=axes[3, 1].transAxes, 
                   fontsize=12, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 重叠显示：颜色梯度 + 误差
    axes[3, 2].imshow(color_grad_orig, cmap='viridis', alpha=0.7)
    axes[3, 2].imshow(error_map_total, cmap='hot', alpha=0.3)
    axes[3, 2].set_title('Color Gradient + Error Overlay')
    axes[3, 2].axis('off')
    
    # 色差图
    im = axes[3, 3].imshow(color_diff_orig, cmap='hot')
    axes[3, 3].set_title('Color Difference Magnitude')
    axes[3, 3].axis('off')
    plt.colorbar(im, ax=axes[3, 3])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
    
    return {
        'correlations': correlations,
        'error_map_rgb': error_map_rgb,
        'error_map_total': error_map_total,
        'multi_channel_edges_orig': edges_orig_combined,
        'multi_channel_edges_recon': edges_recon_combined,
        'color_gradient_orig': color_grad_orig,
        'color_gradient_recon': color_grad_recon,
        'color_diff_edges_orig': color_diff_edges_orig,
        'color_diff_edges_recon': color_diff_edges_recon
    }
    
def rgb_color_gradient_vis(images, method='sobel', return_magnitude=True, normalized=True):
    """
    PyTorch实现的RGB图像颜色梯度提取函数
    
    Args:
        images: 输入图像 tensor [B, C, H, W] 或 [C, H, W]，值范围[0,1]或[0,255]
        method: 梯度计算方法 'sobel' | 'scharr' | 'laplacian'
        return_magnitude: 是否返回梯度幅值，否则返回x,y方向梯度
        normalized: 是否归一化到[0,1]范围
    
    Returns:
        梯度图 tensor [B, C, H, W] 或 [B, 1, H, W] (如果return_magnitude=True)
    """
    
    # 确保输入是4D tensor [B, C, H, W]
    if images.dim() == 3:
        images = images.unsqueeze(0)
    
    batch_size, channels, height, width = images.shape
    
    # 确保图像在[0,1]范围
    if images.max() > 1.0:
        images = images / 255.0
    
    # 创建梯度滤波器
    if method == 'sobel':
        # Sobel算子
        kernel_x = torch.tensor([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], 
                                [0, 0, 0], 
                                [1, 2, 1]], dtype=torch.float32)
    elif method == 'scharr':
        # Scharr算子（对噪声更敏感）
        kernel_x = torch.tensor([[-3, 0, 3], 
                                [-10, 0, 10], 
                                [-3, 0, 3]], dtype=torch.float32)
        kernel_y = torch.tensor([[-3, -10, -3], 
                                [0, 0, 0], 
                                [3, 10, 3]], dtype=torch.float32)
    elif method == 'laplacian':
        # Laplacian算子（二阶导数）
        kernel = torch.tensor([[0, 1, 0], 
                              [1, -4, 1], 
                              [0, 1, 0]], dtype=torch.float32)
    else:
        raise ValueError("Method must be 'sobel', 'scharr', or 'laplacian'")
    
    # 将滤波器移动到设备上并调整为4D [out_ch, in_ch, H, W]
    device = images.device
    
    if method == 'laplacian':
        kernel = kernel.to(device).view(1, 1, 3, 3)
        kernel = kernel.repeat(channels, 1, 1, 1)  # [C, 1, 3, 3]
    else:
        kernel_x = kernel_x.to(device).view(1, 1, 3, 3)
        kernel_y = kernel_y.to(device).view(1, 1, 3, 3)
        kernel_x = kernel_x.repeat(channels, 1, 1, 1)  # [C, 1, 3, 3]
        kernel_y = kernel_y.repeat(channels, 1, 1, 1)  # [C, 1, 3, 3]
    
    # 填充设置
    padding = 1
    
    if method == 'laplacian':
        # Laplacian直接计算
        gradients = F.conv2d(images, kernel, padding=padding, groups=channels)
        if return_magnitude:
            gradients = torch.abs(gradients)
    else:
        # Sobel/Scharr计算x,y方向梯度
        grad_x = F.conv2d(images, kernel_x, padding=padding, groups=channels)
        grad_y = F.conv2d(images, kernel_y, padding=padding, groups=channels)
        
        if return_magnitude:
            # 计算梯度幅值
            gradients = torch.sqrt(grad_x**2 + grad_y**2)
        else:
            # 返回x,y方向梯度
            gradients = torch.stack([grad_x, grad_y], dim=1)  # [B, 2, C, H, W]
            gradients = gradients.view(batch_size, 2 * channels, height, width)
    
    # 归一化到[0,1]范围
    if normalized and return_magnitude:
        # 对每个样本单独归一化
        normalized_gradients = []
        for i in range(batch_size):
            sample_grad = gradients[i]
            if sample_grad.max() > sample_grad.min():
                norm_sample = (sample_grad - sample_grad.min()) / (sample_grad.max() - sample_grad.min())
            else:
                norm_sample = torch.zeros_like(sample_grad)
            normalized_gradients.append(norm_sample)
        gradients = torch.stack(normalized_gradients)
    
    return gradients