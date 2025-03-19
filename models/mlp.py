import logging

import torch.nn as nn

from config.architecture_config import get_architecture_config
from config.model_registry import ModelRegistry
from models.base import BaseModel

logger = logging.getLogger(__name__)


class BaseMLP(BaseModel):
    """
    Base MLP architecture for comparison.
    """
    def __init__(self, model_size, input_size, num_classes):
        """
        Initialize the MLP model.
        """
        super().__init__('mlp', model_size)

        config = get_architecture_config('mlp', model_size)
        self.hidden_dims =  config['hidden_dims']

        self.flatten = nn.Flatten()

        layers = []
        in_dim = input_size

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(self.hidden_dims[-1], num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x


class SmallMLP(BaseMLP):
    """
    Small MLP model with ~1M parameters.
    """
    def __init__(self, input_size, num_classes):
        super().__init__('small', input_size, num_classes)


class MediumMLP(BaseMLP):
    """
    Medium MLP model with ~3M parameters.
    """
    def __init__(self, input_size, num_classes):
        super().__init__('medium', input_size, num_classes)


class LargeMLP(BaseMLP):
    """
    Large MLP model with ~9M parameters.
    """
    def __init__(self, input_size, num_classes):
        super().__init__('large', input_size, num_classes)


# Register models
ModelRegistry.register_model_class('mlp', 'small', SmallMLP)
ModelRegistry.register_model_class('mlp', 'medium', MediumMLP)
ModelRegistry.register_model_class('mlp', 'large', LargeMLP)


def create_mlp_model(model_size, input_size, num_classes, **kwargs):
    """
    Factory function for creating MLP models.
    """
    if model_size == 'small':
        return SmallMLP(input_size, num_classes)
    elif model_size == 'medium':
        return MediumMLP(input_size, num_classes)
    elif model_size == 'large':
        return LargeMLP(input_size, num_classes)
    else:
        raise ValueError(f"Invalid model size: {model_size}")


# Register functions
ModelRegistry.register_model_factory('mlp', create_mlp_model)