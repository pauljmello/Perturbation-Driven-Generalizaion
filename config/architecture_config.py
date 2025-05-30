import copy
import logging
from typing import Dict, Any, List

logger = logging.getLogger('config')


# Target parameter
PARAMETER_TARGETS = {
    'small': 1_000_000,  # ~1M parameters
    'medium': 3_000_000,  # ~3M parameters
    'large': 9_000_000,  # ~9M parameters
}

# Define active experiments (uncomment to enable)
EXPERIMENT_MODEL_TYPES = [
    'mlp',
    'cnn',
    'transformer',
    'vit',
    'vae',
#    'unet'  # Least Optimized (remove for speed)
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
#    'cutout',
    'random_erasing',
#    'horizontal_flip',
#    'vertical_flip',
    'salt_pepper',
    'gaussian_blur',
    'frequency_domain',
    'scale',
#    'mixup',
#    'cutmix',
#    'augmix',
#    'adversarial'
]







# Define all available model types, sizes and augmentations
ALL_MODEL_TYPES = [
    'mlp',
    'cnn',
    'transformer',
    'vit',
    'vae',
    'unet'
]

ALL_MODEL_SIZES = [
    'small',
    'medium',
    'large'
]

ALL_STANDARD_AUGMENTATIONS = [
    'gaussian_noise',
    'rotation',
    'translation',
    'cutout',
    'random_erasing',
    'horizontal_flip',
    'vertical_flip',
    'salt_pepper',
    'gaussian_blur',
    'frequency_domain',
    'scale'
]

ALL_ADVANCED_AUGMENTATIONS = [
    'mixup',
    'cutmix',
    'augmix',
    'adversarial'
]

# Function to filter active standard and advanced augmentations
def get_active_standard_augmentations() -> List[str]:
    return [aug for aug in ALL_STANDARD_AUGMENTATIONS if aug in EXPERIMENT_AUGMENTATIONS]

def get_active_advanced_augmentations() -> List[str]:
    return [aug for aug in ALL_ADVANCED_AUGMENTATIONS if aug in EXPERIMENT_AUGMENTATIONS]

EXPERIMENT_CONFIG = {
    'dataset': 'cifar10',  # Options: 'mnist' or 'cifar10'
    'batch_size': 512,
    'num_epochs': 3,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'optimizer': 'adamw',  # Options: 'adam', 'adamw', 'sgd'
    'scheduler': 'none',  # Options: 'step', 'cosine', 'none'
    'num_runs': 1,
    'random_seed': 0,  # Useless (code constantly resets new seeds)
    'model_types': EXPERIMENT_MODEL_TYPES,  # Options include: 'mlp', 'cnn', 'transformer', 'vit', 'vae', 'unet'
    'model_sizes': EXPERIMENT_MODEL_SIZES,  # Options: 'small', 'medium', 'large'
    'augmentations': EXPERIMENT_AUGMENTATIONS,  # Data augmentation techniques
    'precision': 'bfp16',  # Options: 'fp32', 'fp16', 'bfp16'  Best: bfp16
    'metrics_precision': 'fp8',  # Options: 'fp32', 'fp8'
}

# Augmentation
AUGMENTATION_CONFIG = {
    'standard_augmentations': get_active_standard_augmentations(),
    'advanced_augmentations': get_active_advanced_augmentations(),
    'intensities': [0.1, 0.3, 0.5],
    'max_combination_size': 2,  # n >= 3 is too large for our resources

    # Params selected according to respective papers
    'mixup_alpha': 0.2,
    'cutmix_alpha': 1.0,
    'augmix_severity': 3,
    'augmix_width': 3,
    'augmix_depth': 2,
}

# Rest of the file remains unchanged
# MLP architecture
MLP_ARCHITECTURE = {
    'small': {
        'hidden_dims': [280, 280, 280],
        'target_params': PARAMETER_TARGETS['small'],
    },
    'medium': {
        'hidden_dims': [520, 520, 520, 520, 520, 520],
        'target_params': PARAMETER_TARGETS['medium'],
    },
    'large': {
        'hidden_dims': [1000, 1000, 1000, 1000, 1000, 1000, 1000],
        'target_params': PARAMETER_TARGETS['large'],
    }
}

# CNN architecture
CNN_ARCHITECTURE = {
    'small': {
        'channels': [112, 112, 112],
        'target_params': PARAMETER_TARGETS['small'],
    },
    'medium': {
        'channels': [190, 190, 190, 190, 190],
        'target_params': PARAMETER_TARGETS['medium'],
    },
    'large': {
        'channels': [256, 256, 256, 256, 256, 256, 256, 256],
        'target_params': PARAMETER_TARGETS['large'],
    }
}

# Transformer architecture
TRANSFORMER_ARCHITECTURE = {
    'small': {
        'embed_dim': 248,
        'depth': 3,
        'heads': 2,
        'mlp_ratio': 2.0,
        'target_params': PARAMETER_TARGETS['small'],
    },
    'medium': {
        'embed_dim': 352,
        'depth': 5,
        'heads': 4,
        'mlp_ratio': 2.0,
        'target_params': PARAMETER_TARGETS['medium'],
    },
    'large': {
        'embed_dim': 496,
        'depth': 8,
        'heads': 8,
        'mlp_ratio': 3.0,
        'target_params': PARAMETER_TARGETS['large'],
    }
}

# Vision Transformer (ViT)
VIT_ARCHITECTURE = {
    'small': {
        'embed_dim': 162,
        'depth': 3,
        'heads': 2,
        'mlp_ratio': 1.0,
        'patch_size': 4,
        'target_params': PARAMETER_TARGETS['small'],
    },
    'medium': {
        'embed_dim': 248,
        'depth': 4,
        'heads': 4,
        'mlp_ratio': 2.0,
        'patch_size': 4,
        'target_params': PARAMETER_TARGETS['medium'],
    },
    'large': {
        'embed_dim': 296,
        'depth': 8,
        'heads': 8,
        'mlp_ratio': 3.0,
        'patch_size': 4,
        'target_params': PARAMETER_TARGETS['large'],
    }
}

# VAE architecture
VAE_ARCHITECTURE = {
    'small': {
        'hidden_dims': [64, 96, 128, 160],
        'latent_dim': 64,
        'target_params': PARAMETER_TARGETS['small'],
    },
    'medium': {
        'hidden_dims': [64, 128, 180, 240, 292],
        'latent_dim': 128,
        'target_params': PARAMETER_TARGETS['medium'],
    },
    'large': {
        'hidden_dims': [96, 160, 224, 320, 412, 512],
        'latent_dim': 256,
        'target_params': PARAMETER_TARGETS['large'],
    }
}

# UNet architecture
UNET_ARCHITECTURE = {
    'small': {
        'hidden_channels': [64, 96, 128],
        'depth': 2,
        'bilinear': True,
        'target_params': PARAMETER_TARGETS['small'],
    },
    'medium': {
        'hidden_channels': [64, 80, 160, 232],
        'depth': 3,
        'bilinear': True,
        'target_params': PARAMETER_TARGETS['medium'],
    },
    'large': {
        'hidden_channels': [64, 112, 192, 272, 374],
        'depth': 6,
        'bilinear': True,
        'target_params': PARAMETER_TARGETS['large'],
    }
}

ARCHITECTURE_CONFIGS = {
    'cnn': CNN_ARCHITECTURE,
    'mlp': MLP_ARCHITECTURE,
    'transformer': TRANSFORMER_ARCHITECTURE,
    'vit': VIT_ARCHITECTURE,
    'vae': VAE_ARCHITECTURE,
    'unet': UNET_ARCHITECTURE,
}

# Dataset
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

def get_architecture_config(model_type: str, model_size: str, dataset: str = None) -> Dict[str, Any]:
    """
    Get the architecture configuration for a specific model type and size, with dataset-aware parameter scaling.
    """
    if model_type not in ARCHITECTURE_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")
    if model_size not in ['small', 'medium', 'large']:
        raise ValueError(f"Unsupported model size: {model_size}")
    config = copy.deepcopy(ARCHITECTURE_CONFIGS[model_type][model_size])
    return config


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get configuration for the specified dataset.
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return DATASET_CONFIG[dataset_name]