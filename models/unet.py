import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.architecture_config import get_architecture_config
from config.model_registry import ModelRegistry
from models.base import BaseModel

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """
    Double convolution block for U-Net.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block.
    """
    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After upsampling and concatenation with skip connection
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # After transposed convolution and concatenation with skip connection
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle different input sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BaseUNet(BaseModel):
    """
    Base U-Net architecture for classification.
    """
    def __init__(self, model_size, input_channels, num_classes):
        """
        Initialize U-Net model.
        """
        super().__init__('unet', model_size)
        config = get_architecture_config('unet', model_size)
        self.hidden_channels = config['hidden_channels']
        self.depth = config['depth']
        self.bilinear = config.get('bilinear', True)

        # Initial double convolution
        self.inc = DoubleConv(input_channels, self.hidden_channels[0])

        # Downsampling
        self.down_path = nn.ModuleList()
        for i in range(self.depth):
            if i < len(self.hidden_channels) - 1:
                self.down_path.append(Down(self.hidden_channels[i], self.hidden_channels[i + 1]))

        # Upsampling
        self.up_path = nn.ModuleList()
        for i in range(min(self.depth, len(self.hidden_channels) - 1)):
            in_channels = self.hidden_channels[-1 - i]
            out_channels = self.hidden_channels[-2 - i]
            self.up_path.append(Up(in_channels, out_channels, self.bilinear))

        # Output convolution for reconstruction
        self.outc = OutConv(self.hidden_channels[0], input_channels)

        # Calculate bottleneck feature size
        bottleneck_channels = self.hidden_channels[min(self.depth, len(self.hidden_channels) - 1)]

        # Global Average Pooling and Classifier
        if self.depth <= 2:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(bottleneck_channels, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes)
            )
        elif self.depth <= 3:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(bottleneck_channels, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(inplace=True),
                nn.Linear(768, 384),
                nn.BatchNorm1d(384),
                nn.ReLU(inplace=True),
                nn.Linear(384, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(bottleneck_channels, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        x1 = self.inc(x)

        # Store features for skip connections
        features = [x1]

        # Downsampling
        x = x1
        for i, down in enumerate(self.down_path):
            x = down(x)
            features.append(x)

        # Bottleneck features
        bottleneck = x

        # Upsampling
        for i, up in enumerate(self.up_path):
            x = up(x, features[-(i + 2)])

        # Final reconstruction (not used)
        x_recon = self.outc(x)

        # Classification
        logits = self.classifier(bottleneck)

        return logits


class SmallUNet(BaseUNet):
    """
    Small U-Net model with ~1M parameters.
    """
    def __init__(self, input_channels, num_classes):
        super().__init__('small', input_channels, num_classes)


class MediumUNet(BaseUNet):
    """
    Medium U-Net model with ~3M parameters.
    """
    def __init__(self, input_channels, num_classes):
        super().__init__('medium', input_channels, num_classes)


class LargeUNet(BaseUNet):
    """
    Large U-Net model with ~9M parameters.
    """
    def __init__(self, input_channels, num_classes):
        super().__init__('large', input_channels, num_classes)


# Register models
ModelRegistry.register_model_class('unet', 'small', SmallUNet)
ModelRegistry.register_model_class('unet', 'medium', MediumUNet)
ModelRegistry.register_model_class('unet', 'large', LargeUNet)


def create_unet_model(model_size, input_channels, input_size, num_classes, **kwargs):
    """
    Factory function for creating U-Net models.
    """
    if model_size == 'small':
        return SmallUNet(input_channels, num_classes)
    elif model_size == 'medium':
        return MediumUNet(input_channels, num_classes)
    elif model_size == 'large':
        return LargeUNet(input_channels, num_classes)
    else:
        raise ValueError(f"Invalid model size: {model_size}")


# Register functions
ModelRegistry.register_model_factory('unet', create_unet_model)
