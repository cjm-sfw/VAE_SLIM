import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torchvision import models

class WeightedL1Loss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(WeightedL1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, weight=None, mask=None):
        if weight is not None:
            loss = torch.abs(input - target)
            loss = loss * weight
            loss = loss[mask]
            return torch.mean(loss)
        else:
            return torch.mean(torch.abs(input - target))


class PerceptualLoss(_Loss):
    """Perceptual Loss based on VGG16 features"""
    
    def __init__(self, feature_layers=None, weights=None, reduction='mean', use_normalization=True, device='cuda', dtype=torch.bfloat16):
        """
        Args:
            feature_layers: List of VGG16 layer indices to extract features from
            weights: Weight for each feature layer
            reduction: 'mean' or 'sum'
            use_normalization: Whether to normalize input images to ImageNet stats
            device: Device to place VGG model on ('cuda' or 'cpu')
            dtype: Data type for VGG model (e.g., torch.float32, torch.bfloat16)
        """
        super(PerceptualLoss, self).__init__(reduction=reduction)
        
        if feature_layers is None:
            feature_layers = [2, 7, 12, 21, 30]  # relu1_2, relu2_2, relu3_3, relu4_3
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
            
        self.feature_layers = feature_layers
        self.weights = weights
        self.use_normalization = use_normalization
        self.device = device
        self.dtype = dtype
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True)
        vgg.eval()
        
        # Extract feature layers
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:max(feature_layers)+1])
        
        # Move VGG to specified device and dtype
        self.feature_extractor = self.feature_extractor.to(device=device, dtype=dtype)
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # ImageNet normalization
        self.normalize = nn.Sequential(
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
        ).to(device=device, dtype=dtype)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
        
    def forward(self, input, target):
        """
        Args:
            input: Tensor of shape (B, C, H, W) in range [0, 1] or [-1, 1]
            target: Tensor of shape (B, C, H, W) in same range as input
        Returns:
            Perceptual loss value
        """
        # Ensure input is in range [0, 1]
        if input.min() < 0:
            input = (input + 1) / 2
            target = (target + 1) / 2
        
        # Ensure input and target are on the same device and dtype as VGG
        input = input.to(device=self.device, dtype=self.dtype)
        target = target.to(device=self.device, dtype=self.dtype)
            
        # Normalize to ImageNet stats if requested
        if self.use_normalization:
            input = self.normalize(input)
            target = self.normalize(target)
            
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std
        
        # Extract features
        input_features = []
        target_features = []
        
        x_input = input
        x_target = target
        
        for i, layer in enumerate(self.feature_extractor):
            x_input = layer(x_input)
            x_target = layer(x_target)
            
            if i in self.feature_layers:
                input_features.append(x_input)
                target_features.append(x_target)
        
        # Compute perceptual loss
        loss = 0
        for input_feat, target_feat, weight in zip(input_features, target_features, self.weights):
            layer_loss = F.l1_loss(input_feat, target_feat, reduction=self.reduction)
            loss += weight * layer_loss
            
        return loss


class lpips_loss(_Loss):
    """LPIPS Loss using pretrained LPIPS model"""
    
    def __init__(self, lpips_model, reduction='mean', device='cuda', dtype=torch.bfloat16):
        super(lpips_loss, self).__init__(reduction=reduction)
        
        self.lpips_model = lpips_model
        self.lpips_model.eval()
        self.device = device
        self.dtype = dtype

    def forward(self, input, target):
        """
        Args:
            input: Tensor of shape (B, C, H, W) in range [0, 1] or [-1, 1]
            target: Tensor of shape (B, C, H, W) in same range as input
        Returns:
            LPIPS loss value
        """
        # Ensure input and target are on the same device and dtype as LPIPS model
        input = input.to(device=self.device, dtype=self.dtype)
        target = target.to(device=self.device, dtype=self.dtype)
        
        # Compute LPIPS loss
        with torch.no_grad():
            lpips_val = self.lpips_model(input, target, normalize=True)
        
        return lpips_val.mean()


    


class HuberLoss(_Loss):
    """Huber Loss"""
    
    def __init__(self, delta=1.0, reduction='mean'):
        super(HuberLoss, self).__init__(reduction=reduction)
        self.delta = delta

    def forward(self, input, target):
        """
        Args:
            input: Tensor of shape (B, C, H, W)
            target: Tensor of shape (B, C, H, W)
        Returns:
            Huber loss value
        """
        diff = input - target
        abs_diff = torch.abs(diff)
        
        loss = torch.where(abs_diff < self.delta,
                           0.5 * diff ** 2,
                           self.delta * (abs_diff - 0.5 * self.delta))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class DWTLoss(_Loss):
    """DWT Loss"""
        
class DCTLoss(_Loss):
    """DCT Loss"""
    
    def __init__(self, reduction='mean'):
        super(DCTLoss, self).__init__(reduction=reduction)

    def forward(self, input, target):
        import torch_dct
        """
        Args:
            input: Tensor of shape (B, C, H, W)
            target: Tensor of shape (B, C, H, W)
        Returns:
            DCT loss value
        """
        # Ensure input and target are on the same device
        input = input.to(target.device)
        target = target.to(input.device)

        # Compute DCT for both input and target
        input_dct = torch_dct.dct_2d(input)
        target_dct = torch_dct.dct_2d(target)

        # Compute the DCT loss
        loss = torch.abs(input_dct - target_dct)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss