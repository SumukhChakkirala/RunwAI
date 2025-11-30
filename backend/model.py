"""
ResNet34-UNet Model Architecture for Runway Segmentation
Matches the architecture used to train runway_0.9243.pth (92.43% IoU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d -> BN -> ReLU) x 2"""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResNetUNet(nn.Module):
    """
    Classic UNet with ResNet34 encoder - with flexible size handling.
    
    Architecture:
    - Encoder: ResNet34 pretrained on ImageNet
    - Decoder: UNet-style decoder with skip connections
    - Output: n_classes channel prediction map
    """
    
    def __init__(self, n_classes=2):
        super().__init__()
        
        # ResNet34 encoder
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64
        self.pool1 = resnet.maxpool
        self.encoder2 = resnet.layer1  # 64
        self.encoder3 = resnet.layer2  # 128
        self.encoder4 = resnet.layer3  # 256
        self.encoder5 = resnet.layer4  # 512
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.upconv5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.decoder5 = DoubleConv(512 + 512, 512)
        
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder4 = DoubleConv(256 + 256, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder3 = DoubleConv(128 + 128, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = DoubleConv(64 + 64, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.decoder1 = DoubleConv(64 + 64, 64)
        
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        enc1 = self.encoder1(x)       # 64, H/2, W/2
        enc1_pool = self.pool1(enc1)  # 64, H/4, W/4
        enc2 = self.encoder2(enc1_pool)  # 64, H/4, W/4
        enc3 = self.encoder3(enc2)    # 128, H/8, W/8
        enc4 = self.encoder4(enc3)    # 256, H/16, W/16
        enc5 = self.encoder5(enc4)    # 512, H/32, W/32
        
        # Bottleneck
        bottleneck = self.bottleneck(enc5)  # 1024, H/32, W/32
        
        # Decoder with size matching
        dec5 = self.upconv5(bottleneck)
        dec5 = self._match_size(dec5, enc5)
        dec5 = torch.cat([dec5, enc5], dim=1)
        dec5 = self.decoder5(dec5)
        
        dec4 = self.upconv4(dec5)
        dec4 = self._match_size(dec4, enc4)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = self._match_size(dec3, enc3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = self._match_size(dec2, enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = self._match_size(dec1, enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        out = self.final(dec1)
        
        # Resize to original input size if needed
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out
    
    def _match_size(self, x, target):
        """Match spatial dimensions using interpolation"""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x


def load_model(model_path: str, device: torch.device) -> ResNetUNet:
    """
    Load the trained runway segmentation model.
    
    Args:
        model_path: Path to the .pth checkpoint file
        device: torch.device to load the model onto
        
    Returns:
        Loaded and ready-to-use model in eval mode
    """
    model = ResNetUNet(n_classes=2)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the dict itself is the state_dict
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model
