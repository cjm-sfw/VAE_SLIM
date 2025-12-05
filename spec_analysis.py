import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import os
import matplotlib

# 设置中文字体和支持负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体，如果找不到用DejaVu Sans
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['pdf.fonttype'] = 42  # 确保保存PDF时字体嵌入

class FrequencyDomainAnalyzer:
    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size
        self.results = {}
        
    def load_and_preprocess_images(self, image_paths):
        """Load and preprocess all images"""
        images = []
        
        for i, path in enumerate(image_paths):
            try:
                # Read image
                img = cv2.imread(path)
                if img is None:
                    print(f"Warning: Cannot read image {path}")
                    continue
                    
                # Convert to grayscale
                if len(img.shape) > 2:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray = img
                    
                # Resize
                img_resized = cv2.resize(img_gray, self.image_size)
                images.append(img_resized)
                print(f"Successfully loaded: {path}")
                
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
            
        return images
    
    def compute_spectral_features(self, img):
        """Compute frequency domain features for an image"""
        # Fourier transform
        dft = np.fft.fft2(img.astype(float))
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.abs(dft_shift)
        
        # Log magnitude spectrum (better for visualization)
        log_magnitude = np.log(magnitude + 1e-10)
        
        # Calculate radial power spectrum
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        r = np.sqrt((x - ccol)**2 + (y - crow)**2)
        r = r.astype(int)
        
        # Band energy statistics
        band_energies = []
        band_radii = [10, 30, 50, 80, 120, 160]  # Define frequency band radii
        
        prev_r = 0
        for max_r in band_radii:
            mask = (r >= prev_r) & (r < max_r)
            if np.any(mask):
                band_energy = np.sum(magnitude[mask])
                band_energies.append(band_energy)
            prev_r = max_r
        
        # Total energy normalization
        total_energy = np.sum(magnitude)
        band_energies = np.array(band_energies) / total_energy
        
        # Statistical features
        features = {
            'magnitude': magnitude,
            'log_magnitude': log_magnitude,
            'total_energy': total_energy,
            'band_energies': band_energies,
            'spectral_centroid': np.sum(r * magnitude) / total_energy,
            'spectral_spread': np.sqrt(np.sum((r - np.sum(r * magnitude) / total_energy)**2 * magnitude) / total_energy),
            'spectral_entropy': -np.sum((magnitude / total_energy) * np.log(magnitude / total_energy + 1e-10))
        }
        
        return features
    
    def analyze_images(self, image_paths, image_names=None):
        """Analyze all images"""
        if image_names is None:
            image_names = [f'Img_{i+1:02d}' for i in range(len(image_paths))]
            
        images = self.load_and_preprocess_images(image_paths)
        
        if len(images) != len(image_paths):
            print(f"Warning: Only {len(images)} images loaded successfully")
            # Adjust image_names to match actually loaded images
            image_names = image_names[:len(images)]
            
        all_features = []
        for i, (img, name) in enumerate(zip(images, image_names)):
            print(f"Analyzing {name}...")
            features = self.compute_spectral_features(img)
            features['name'] = name
            features['image'] = img
            self.results[name] = features
            all_features.append(features)
            
        print(f"Analysis completed for {len(all_features)} images")
        return all_features
    
    def visualize_spectral_comparison(self):
        """Visualize frequency domain comparison results"""
        if not self.results:
            print("Please run analyze_images first")
            return
            
        names = list(self.results.keys())
        n_images = len(names)
        
        print(f"Generating visualizations for {n_images} images...")
        
        # 1. Magnitude spectrum grid comparison
        n_cols = 5
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, name in enumerate(names):
            row = i // n_cols
            col = i % n_cols
            log_mag = self.results[name]['log_magnitude']
            axes[row, col].imshow(log_mag, cmap='hot')
            axes[row, col].set_title(f'{name}\nSpectrum', fontsize=10)
            axes[row, col].axis('off')
        
        # Remove empty subplots
        for i in range(len(names), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.suptitle('Magnitude Spectrum Comparison of Images', fontsize=16, y=1.02)
        plt.show()
        
        # 2. Frequency band energy radar chart
        self._plot_radar_chart()
        
        # 3. Frequency band energy bar chart
        self._plot_energy_bars()
        
        # 4. PCA analysis of spectral features
        self._plot_pca_analysis()
        
        # 5. Spectral similarity heatmap
        self._plot_spectral_similarity()
    
    def _plot_radar_chart(self):
        """增强版雷达图 - 解决颜色不够用问题"""
        names = list(self.results.keys())
        band_energies = np.array([self.results[name]['band_energies'] for name in names])
        
        # 频带标签
        band_labels = ['VLF(0-10)', 'LF(10-30)', 'MF(30-50)', 
                    'HF(50-80)', 'VHF(80-120)', 'UHF(120-160)']
        
        angles = np.linspace(0, 2*np.pi, len(band_labels), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 方案1: 使用扩展的颜色映射
        n_images = len(names)
        
        # 创建扩展的颜色方案
        colors = plt.cm.tab20(np.linspace(0, 1, n_images))
        
        fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(projection='polar'))
        
        for i, name in enumerate(names):
            values = band_energies[i].tolist()
            values += values[:1]  # 闭合图形
            
            # 使用唯一颜色
            color = colors[i]
            ax.plot(angles, values, 'o-', linewidth=2, label=name, 
                    markersize=4, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(band_labels)
        ax.set_title('Frequency Band Energy Distribution Radar Chart\n(Extended Color Scheme)', 
                    size=14, pad=20)
        
        # 改进图例显示
        ax.legend(bbox_to_anchor=(1.3, 1.0), fontsize=8, 
                ncol=2 if n_images > 10 else 1)
        plt.tight_layout()
        plt.show()
    
    def _plot_energy_bars(self):
        """Plot frequency band energy bar chart"""
        names = list(self.results.keys())
        band_energies = np.array([self.results[name]['band_energies'] for name in names])
        band_labels = ['VLF', 'LF', 'MF', 'HF', 'VHF', 'UHF']
        
        x = np.arange(len(band_labels))
        width = 0.8 / len(names)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, name in enumerate(names):
            offset = width * i
            ax.bar(x + offset, band_energies[i], width, label=name, alpha=0.7)
        
        ax.set_xlabel('Frequency Bands')
        ax.set_ylabel('Energy Ratio')
        ax.set_title('Energy Distribution Across Frequency Bands')
        ax.set_xticks(x + width * len(names) / 2)
        ax.set_xticklabels(band_labels)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.show()
    
    def enhanced_pca_visualization(self, color_strategy='spectral_centroid'):
        """
        增强的PCA可视化，提供多种颜色映射策略
        
        Parameters:
        color_strategy: 
            'sequence' - 按图像序列顺序
            'spectral_centroid' - 按频谱质心
            'entropy' - 按频谱熵
            'energy' - 按总能量
            'cluster' - 按聚类结果
        """
        names = list(self.results.keys())
        
        # 构建特征矩阵
        features_list = []
        for name in names:
            feat = self.results[name]
            feature_vector = np.concatenate([
                feat['band_energies'],
                [feat['spectral_centroid']],
                [feat['spectral_spread']], 
                [feat['spectral_entropy']]
            ])
            features_list.append(feature_vector)
        
        X = np.array(features_list)
        
        # 标准化并执行PCA
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # 根据策略选择颜色数据
        if color_strategy == 'sequence':
            color_data = range(len(X_pca))
            color_label = 'Sequence Index'
            cmap = 'viridis'
            
        elif color_strategy == 'spectral_centroid':
            color_data = [self.results[name]['spectral_centroid'] for name in names]
            color_label = 'Spectral Centroid'
            cmap = 'plasma'
            
        elif color_strategy == 'entropy':
            color_data = [self.results[name]['spectral_entropy'] for name in names]
            color_label = 'Spectral Entropy'
            cmap = 'cool'
            
        elif color_strategy == 'energy':
            color_data = [self.results[name]['total_energy'] for name in names]
            color_label = 'Total Energy'
            cmap = 'hot'
            
        elif color_strategy == 'cluster':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            color_data = clusters
            color_label = 'Cluster'
            cmap = 'Set1'
        
        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图: 带颜色映射的散点图
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], s=100, alpha=0.7, 
                            c=color_data, cmap=cmap, edgecolors='black', linewidth=0.5)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label(color_label)
        
        # 添加标签
        for i, name in enumerate(names):
            ax1.annotate(name, (X_pca[i, 0], X_pca[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title(f'PCA with {color_strategy.replace("_", " ").title()} Coloring')
        ax1.grid(True, alpha=0.3)
        
        # 右图: 特征贡献度分析
        feature_names = [f'Band{i+1}' for i in range(6)] + ['Centroid', 'Spread', 'Entropy']
        pca_components = pca.components_
        
        # 计算每个特征对PC1和PC2的贡献度
        contribution_pc1 = np.abs(pca_components[0])
        contribution_pc2 = np.abs(pca_components[1])
        
        x = np.arange(len(feature_names))
        width = 0.35
        
        ax2.bar(x - width/2, contribution_pc1, width, label='PC1 Contribution', alpha=0.7)
        ax2.bar(x + width/2, contribution_pc2, width, label='PC2 Contribution', alpha=0.7)
        
        ax2.set_xlabel('Spectral Features')
        ax2.set_ylabel('Absolute Contribution')
        ax2.set_title('Feature Contributions to Principal Components')
        ax2.set_xticks(x)
        ax2.set_xticklabels(feature_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"PCA Cumulative Explained Variance: {np.sum(pca.explained_variance_ratio_):.2%}")
        print(f"PC1 Explained Variance: {pca.explained_variance_ratio_[0]:.2%}")
        print(f"PC2 Explained Variance: {pca.explained_variance_ratio_[1]:.2%}")
        
        # 显示最重要的特征
        top_pc1_features = np.argsort(np.abs(pca_components[0]))[-3:][::-1]
        top_pc2_features = np.argsort(np.abs(pca_components[1]))[-3:][::-1]
        
        print(f"\nMost important features for PC1:")
        for idx in top_pc1_features:
            print(f"  {feature_names[idx]}: {pca_components[0, idx]:.3f}")
        
        print(f"\nMost important features for PC2:")
        for idx in top_pc2_features:
            print(f"  {feature_names[idx]}: {pca_components[1, idx]:.3f}")
    def _plot_pca_analysis(self):
        """PCA分析频域特征 - 修复版本"""
        names = list(self.results.keys())
        
        # 构建特征矩阵
        features_list = []
        for name in names:
            feat = self.results[name]
            feature_vector = np.concatenate([
                feat['band_energies'],
                [feat['spectral_centroid']],
                [feat['spectral_spread']], 
                [feat['spectral_entropy']]
            ])
            features_list.append(feature_vector)
        
        X = np.array(features_list)
        
        # 标准化并执行PCA
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # 绘制PCA结果 - 修复颜色映射
        plt.figure(figsize=(12, 8))
        
        # 方法1: 使用序列索引作为颜色值
        colors = range(len(X_pca))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], s=100, alpha=0.7, 
                            c=colors, cmap='viridis')
        plt.colorbar(scatter, label='Image Sequence Index')
        
        # 添加标签
        for i, name in enumerate(names):
            plt.annotate(name, (X_pca[i, 0], X_pca[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Analysis of Spectral Features')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"PCA Cumulative Explained Variance: {np.sum(pca.explained_variance_ratio_):.2%}")
        
    def _plot_spectral_similarity(self):
        """Plot spectral similarity heatmap"""
        names = list(self.results.keys())
        n = len(names)
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # Use spectral correlation as similarity measure
                spec1 = self.results[names[i]]['log_magnitude'].flatten()
                spec2 = self.results[names[j]]['log_magnitude'].flatten()
                correlation = np.corrcoef(spec1, spec2)[0, 1]
                similarity_matrix[i, j] = correlation if not np.isnan(correlation) else 0
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   xticklabels=names, 
                   yticklabels=names,
                   annot=True, fmt='.3f', 
                   cmap='coolwarm', 
                   center=0,
                   cbar_kws={'label': 'Spectral Correlation'})
        plt.title('Spectral Similarity Heatmap Between Images')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def generate_quantitative_report(self):
        """Generate quantitative analysis report"""
        names = list(self.results.keys())
        
        print("=" * 70)
        print("           QUANTITATIVE SPECTRAL ANALYSIS REPORT")
        print("=" * 70)
        
        # Create feature table
        features_data = []
        for name in names:
            feat = self.results[name]
            features_data.append([
                name,
                f"{feat['spectral_centroid']:.1f}",
                f"{feat['spectral_spread']:.1f}", 
                f"{feat['spectral_entropy']:.3f}",
                f"{feat['total_energy']:.2e}"
            ])
        
        # Print table
        print(f"{'Image':<15} {'Spectral':<12} {'Spectral':<12} {'Spectral':<10} {'Total':<15}")
        print(f"{'Name':<15} {'Centroid':<12} {'Spread':<12} {'Entropy':<10} {'Energy':<15}")
        print("-" * 70)
        for row in features_data:
            print(f"{row[0]:<15} {row[1]:<12} {row[2]:<12} {row[3]:<10} {row[4]:<15}")
        
        # Find most similar and most different image pairs
        max_similarity = -1
        min_similarity = 2
        max_pair = min_pair = (None, None)
        
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                spec1 = self.results[names[i]]['log_magnitude'].flatten()
                spec2 = self.results[names[j]]['log_magnitude'].flatten()
                correlation = np.corrcoef(spec1, spec2)[0, 1]
                if np.isnan(correlation):
                    continue
                    
                if correlation > max_similarity:
                    max_similarity = correlation
                    max_pair = (names[i], names[j])
                if correlation < min_similarity:
                    min_similarity = correlation  
                    min_pair = (names[i], names[j])
        
        print("\n" + "=" * 70)
        print("KEY FINDINGS:")
        print(f"Most similar pair: {max_pair[0]} and {max_pair[1]} (similarity: {max_similarity:.3f})")
        print(f"Most different pair: {min_pair[0]} and {min_pair[1]} (similarity: {min_similarity:.3f})")
        
        # Additional statistics
        spectral_centroids = [self.results[name]['spectral_centroid'] for name in names]
        print(f"Spectral centroid range: {min(spectral_centroids):.1f} - {max(spectral_centroids):.1f}")
        print("=" * 70)


def advanced_sequence_analysis(analyzer):
    """针对图像序列的进阶分析"""
    names = list(analyzer.results.keys())
    
    # 提取序列编号
    indices = [int(name.split('_')[-1].split('.')[0]) for name in names]
    sorted_indices = np.argsort(indices)
    sorted_names = [names[i] for i in sorted_indices]
    
    # 时序趋势分析
    spectral_centroids = [analyzer.results[name]['spectral_centroid'] for name in sorted_names]
    spectral_entropies = [analyzer.results[name]['spectral_entropy'] for name in sorted_names]
    total_energies = [analyzer.results[name]['total_energy'] for name in sorted_names]
    
    # 创建趋势图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 频谱质心趋势
    ax1.plot(range(len(sorted_names)), spectral_centroids, 'o-', linewidth=2, markersize=6)
    ax1.set_ylabel('Spectral Centroid')
    ax1.set_title('Temporal Evolution of Spectral Properties')
    ax1.grid(True, alpha=0.3)
    
    # 频谱熵趋势
    ax2.plot(range(len(sorted_names)), spectral_entropies, 's-', linewidth=2, markersize=6, color='orange')
    ax2.set_ylabel('Spectral Entropy')
    ax2.grid(True, alpha=0.3)
    
    # 总能量趋势
    ax3.plot(range(len(sorted_names)), total_energies, '^-', linewidth=2, markersize=6, color='green')
    ax3.set_ylabel('Total Energy')
    ax3.set_xlabel('Image Sequence')
    ax3.grid(True, alpha=0.3)
    
    # 设置x轴标签为图像编号
    image_numbers = [name.split('_')[-1].split('.')[0] for name in sorted_names]
    ax3.set_xticks(range(len(sorted_names)))
    ax3.set_xticklabels(image_numbers, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 计算变化率
    centroid_changes = np.diff(spectral_centroids)
    entropy_changes = np.diff(spectral_entropies)
    
    print("=== SEQUENCE CHANGE ANALYSIS ===")
    print(f"Max spectral centroid increase: {np.max(centroid_changes):.2f} (between images)")
    print(f"Max entropy increase: {np.max(entropy_changes):.3f}")
    print(f"Average centroid change: {np.mean(np.abs(centroid_changes)):.2f}")
    
    # 识别关键转折点
    large_jumps = np.where(np.abs(centroid_changes) > np.mean(np.abs(centroid_changes)) + np.std(np.abs(centroid_changes)))[0]
    if len(large_jumps) > 0:
        print(f"Key transitions at sequence jumps: {large_jumps}")

def domain_specific_metrics(analyzer):
    """针对差异图像的专用度量"""
    names = list(analyzer.results.keys())
    
    print("=== DOMAIN-SPECIFIC ANALYSIS ===")
    
    for name in names:
        features = analyzer.results[name]
        
        # 高频细节指标 (针对差异图像)
        high_freq_ratio = np.sum(features['band_energies'][-2:])  # 最后两个高频带
        detail_complexity = features['spectral_entropy'] * high_freq_ratio
        
        # 运动/变化强度估计
        change_intensity = features['total_energy'] / 1e7  # 归一化
        
        print(f"{name:<25} | HF Ratio: {high_freq_ratio:.3f} | "
              f"Detail Complexity: {detail_complexity:.2f} | "
              f"Change Intensity: {change_intensity:.2f}")


# Usage example
def main():
    # Replace with your actual image paths
    image_paths = [
        f'image_{i:02d}.jpg' for i in range(1, 16)  # Adjust based on your actual filenames
    ]
    
    # Or use specific paths like:
    # image_paths = ['path/to/your/image1.jpg', 'path/to/your/image2.jpg', ...]
    
    # Image names (optional)
    image_names = [f'Img_{i:02d}' for i in range(1, 16)]
    
    # Create analyzer and perform analysis
    analyzer = FrequencyDomainAnalyzer(image_size=(256, 256))
    
    # Check if images exist
    existing_paths = []
    existing_names = []
    for path, name in zip(image_paths, image_names):
        if os.path.exists(path):
            existing_paths.append(path)
            existing_names.append(name)
        else:
            print(f"File not found: {path}")
    
    if not existing_paths:
        print("No valid image files found!")
        return
    
    analyzer.analyze_images(existing_paths, existing_names)
    
    # Generate all visualizations
    analyzer.visualize_spectral_comparison()
    
    # Generate quantitative report
    analyzer.generate_quantitative_report()

if __name__ == "__main__":
    main()