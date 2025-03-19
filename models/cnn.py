import logging

import torch
import torch.nn as nn

from config.architecture_config import get_architecture_config
from config.model_registry import ModelRegistry
from models.base import BaseModel

logger = logging.getLogger('models.cnn')

class ConvBlock(nn.Module):
    """
    A convolutional block with two conv layers, batch norm, and activation.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class BaseCNN(BaseModel):
    """
    Base CNN architecture for comparison.
    """

    def __init__(self, model_size, input_channels, num_classes):
        """
        Initialize the CNN model.
        """
        super().__init__('cnn', model_size)

        # Get architecture configuration
        config = get_architecture_config('cnn', model_size)
        self.channels = config['channels']

        # Create convolutional blocks
        self.blocks = nn.ModuleList()
        in_channels = input_channels

        for out_channels in self.channels:
            self.blocks.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        input_size = 28 if input_channels == 1 else 32


        self.max_pool_ops = min(len(self.channels), int(torch.log2(torch.tensor(input_size)).item()))
        logger.info(f"{model_size} CNN model will apply {self.max_pool_ops} pooling operations")

        feature_size = input_size // (2 ** self.max_pool_ops)

        if model_size == 'large':
            self.use_global_avg = True
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            feature_size = 1
            classifier_input_size = self.channels[-1]
        else:
            self.use_global_avg = False
            classifier_input_size = self.channels[-1] * feature_size * feature_size

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < self.max_pool_ops:
                x = self.pool(x)

        # Apply global average pooling for large models
        if self.use_global_avg:
            x = self.global_avg_pool(x)

        # Classifier
        x = self.classifier(x)
        return x


class SmallCNN(BaseCNN):
    """
    Small CNN model with ~1M parameters.
    """
    def __init__(self, input_channels, num_classes):
        super().__init__('small', input_channels, num_classes)


class MediumCNN(BaseCNN):
    """
    Medium CNN model with ~3M parameters.
    """
    def __init__(self, input_channels, num_classes):
        super().__init__('medium', input_channels, num_classes)


class LargeCNN(BaseCNN):
    """
    Large CNN model with ~9M parameters.
    """
    def __init__(self, input_channels, num_classes):
        super().__init__('large', input_channels, num_classes)


# Register models
ModelRegistry.register_model_class('cnn', 'small', SmallCNN)
ModelRegistry.register_model_class('cnn', 'medium', MediumCNN)
ModelRegistry.register_model_class('cnn', 'large', LargeCNN)


def create_cnn_model(model_size, input_channels, num_classes, **kwargs):
    """
    Factory function for creating CNN models.
    """
    if model_size == 'small':
        return SmallCNN(input_channels, num_classes)
    elif model_size == 'medium':
        return MediumCNN(input_channels, num_classes)
    elif model_size == 'large':
        return LargeCNN(input_channels, num_classes)
    else:
        raise ValueError(f"Invalid model size: {model_size}")


# Register functions
ModelRegistry.register_model_factory('cnn', create_cnn_model)