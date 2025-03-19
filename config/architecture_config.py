import logging
from typing import Dict, Any

logger = logging.getLogger('config')

# Lists for experiment configurations
EXPERIMENT_MODEL_TYPES = [
    'mlp',
    'cnn',
    'transformer',
    'vit',
    'vae',
    'unet'
]

EXPERIMENT_MODEL_SIZES = [
    'small',
    'medium',
    'large'
]

EXPERIMENT_AUGMENTATIONS = [
    'gaussian_noise',
    'rotation',
    'translation',
    'cutout',
    'random_erasing',
    'horizontal_flip',
    'mixup',
    'cutmix',
    'augmix'
]

EXPERIMENT_CONFIG = {
    'dataset': 'cifar10',  # Options: 'mnist'or 'cifar10'
    'batch_size': 512,
    'num_epochs': 5,
    'learning_rate': 0.0004,
    'weight_decay': 1e-5,
    'optimizer': 'adamw',  # Options: 'adam', 'adamw', 'sgd'
    'scheduler': 'none',   # Options: 'step', 'cosine', 'none'
    'num_runs': 1,
    'random_seed': 0,
    'model_types': EXPERIMENT_MODEL_TYPES,  # Options include: 'mlp', 'cnn', 'transformer', 'vit', 'vae', 'unet'
    'model_sizes': EXPERIMENT_MODEL_SIZES,  # Options: 'small', 'medium', 'large'
    'augmentations': EXPERIMENT_AUGMENTATIONS,  # Data augmentation techniques like 'gaussian_noise', 'rotation', etc.
}

# Target parameter counts
PARAMETER_TARGETS = {
    'small': 1_000_000,  # ~1M parameters
    'medium': 3_000_000,  # ~3M parameters
    'large': 9_000_000,  # ~9M parameters
}

# MLP architecture configurations
MLP_ARCHITECTURE = {
    'small': {
        'hidden_dims': [512, 384, 384, 128],
        'target_params': PARAMETER_TARGETS['small'],
    },
    'medium': {
        'hidden_dims': [1024, 896, 640, 640, 512, 384],
        'target_params': PARAMETER_TARGETS['medium'],
    },
    'large': {
        'hidden_dims': [1536, 1280, 1280, 1152, 896, 768, 768, 512],
        'target_params': PARAMETER_TARGETS['large'],
    }
}

# CNN architecture configurations
CNN_ARCHITECTURE = {
    'small': {
        'channels': [64, 96, 128],
        'blocks': 2,
        'target_params': PARAMETER_TARGETS['small'],
    },
    'medium': {
        'channels': [64, 160, 224, 320],
        'blocks': 3,
        'target_params': PARAMETER_TARGETS['medium'],
    },
    'large': {
        'channels': [64, 128, 256, 384, 608],
        'blocks': 7,
        'target_params': PARAMETER_TARGETS['large'],
    }
}


# Transformer architecture configurations
TRANSFORMER_ARCHITECTURE = {
    'small': {
        'embed_dim': 160,
        'depth': 3,
        'heads': 1,
        'mlp_ratio': 2.0,
        'patch_size': 4,
        'target_params': PARAMETER_TARGETS['small'],
    },
    'medium': {
        'embed_dim': 216,
        'depth': 5,
        'heads': 2,
        'mlp_ratio': 4.0,
        'patch_size': 4,
        'target_params': PARAMETER_TARGETS['medium'],
    },
    'large': {
        'embed_dim': 320,
        'depth': 7,
        'heads': 4,
        'mlp_ratio': 8.0,
        'patch_size': 4,
        'target_params': PARAMETER_TARGETS['large'],
    }
}


# Vision Transformer configurations
VIT_ARCHITECTURE = {
    'small': {
        'embed_dim': 160,
        'depth': 3,
        'heads': 1,
        'mlp_ratio': 2.0,
        'patch_size': 4,
        'target_params': PARAMETER_TARGETS['small'],
    },
    'medium': {
        'embed_dim': 216,
        'depth': 5,
        'heads': 2,
        'mlp_ratio': 4.0,
        'patch_size': 4,
        'target_params': PARAMETER_TARGETS['medium'],
    },
    'large': {
        'embed_dim': 320,
        'depth': 7,
        'heads': 4,
        'mlp_ratio': 8.0,
        'patch_size': 4,
        'target_params': PARAMETER_TARGETS['large'],
    }
}


# VAE architecture configurations
VAE_ARCHITECTURE = {
    'small': {
        'hidden_dims': [64, 96, 128, 160],
        'latent_dim': 160,
        'target_params': PARAMETER_TARGETS['small'],
    },
    'medium': {
        'hidden_dims': [64, 128, 192, 256, 320],
        'latent_dim': 256,
        'target_params': PARAMETER_TARGETS['medium'],
    },
    'large': {
        'hidden_dims': [64, 128, 192, 256, 320, 384, 448],
        'latent_dim': 512,
        'target_params': PARAMETER_TARGETS['large'],
    }
}


# UNet architecture configurations
UNET_ARCHITECTURE = {
    'small': {
        'hidden_channels': [48, 96, 128],
        'depth': 2,
        'bilinear': True,
        'target_params': PARAMETER_TARGETS['small'],
    },
    'medium': {
        'hidden_channels': [64, 128, 256, 384],
        'depth': 3,
        'bilinear': True,
        'target_params': PARAMETER_TARGETS['medium'],
    },
    'large': {
        'hidden_channels': [64, 128, 192, 256, 384],
        'depth': 7,
        'bilinear': False,
        'target_params': PARAMETER_TARGETS['large'],
    }
}


# Dataset configuration
DATASET_CONFIG = {
    'mnist': {
        'input_channels': 1,
        'input_size': 28,
        'num_classes': 10,
        'mean': (0.1307,),
        'std': (0.3081,),
    },
    'cifar10': {
        'input_channels': 3,
        'input_size': 32,
        'num_classes': 10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2470, 0.2435, 0.2616),
    }
}

# Augmentation configuration
AUGMENTATION_CONFIG = {
    'standard_augmentations': [
        'gaussian_noise',
        'rotation',
        'translation',
        'cutout',
        'random_erasing',
        'horizontal_flip',
    ],
    'advanced_augmentations': [
        'mixup',
        'cutmix',
        'augmix',
    ],
    'intensities': [0.1, 0.3, 0.5],
    'max_combination_size': 3,
    'mixup_alpha': 0.2,
    'cutmix_alpha': 1.0,
    'augmix_severity': 3,
    'augmix_width': 3,
    'augmix_depth': 2,
}

# All architecture configurations
ARCHITECTURE_CONFIGS = {
    'cnn': CNN_ARCHITECTURE,
    'mlp': MLP_ARCHITECTURE,
    'transformer': TRANSFORMER_ARCHITECTURE,
    'vit': VIT_ARCHITECTURE,
    'vae': VAE_ARCHITECTURE,
    'unet': UNET_ARCHITECTURE,
}

def get_architecture_config(model_type: str, model_size: str) -> Dict[str, Any]:
    """
    Get the architecture configuration for a specific model type and size.
    """
    if model_type not in ARCHITECTURE_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")
    if model_size not in ['small', 'medium', 'large']:
        raise ValueError(f"Unsupported model size: {model_size}")
    return ARCHITECTURE_CONFIGS[model_type][model_size]


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get configuration for the specified dataset.
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return DATASET_CONFIG[dataset_name]


class ParameterEstimator:
    """
    Advanced parameter estimation with architecture-specific calculations.
    """

    @staticmethod
    def calculate_mlp_params(input_dim, hidden_dims, output_dim):
        """
        Calculate exact MLP parameter count.
        """
        params = 0
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            params += prev_dim * hidden_dim + hidden_dim  # weights + biases
            prev_dim = hidden_dim
        params += prev_dim * output_dim + output_dim  # output layer
        return params

    @staticmethod
    def calculate_cnn_params(input_channels, channels, kernel_size=3, use_batchnorm=True):
        """
        Calculate CNN parameter count with BatchNorm consideration.
        """
        params = 0
        prev_channels = input_channels
        for out_channels in channels:
            params += kernel_size * kernel_size * prev_channels * out_channels + out_channels
            if use_batchnorm:
                params += 2 * out_channels
            prev_channels = out_channels
        feature_dim = 256
        params += prev_channels * feature_dim + feature_dim
        params += feature_dim * 10 + 10  # classifier

        return params

    @staticmethod
    def calculate_transformer_params(input_dim, embed_dim, num_heads, num_layers, mlp_ratio, output_dim):
        """
        Calculate Transformer parameter count with attention head consideration.
        """
        params = 0
        params += input_dim * embed_dim

        # Positional embedding
        seq_length = 64
        params += seq_length * embed_dim

        # Transformer blocks
        for _ in range(num_layers):
            params += 2 * embed_dim
            params += 3 * embed_dim * embed_dim

            # Self-attention: Output projection
            params += embed_dim * embed_dim
            params += 2 * embed_dim

            # MLP block
            mlp_dim = int(embed_dim * mlp_ratio)
            params += embed_dim * mlp_dim + mlp_dim
            params += mlp_dim * embed_dim + embed_dim

        # Output projection
        params += embed_dim * output_dim + output_dim

        return params