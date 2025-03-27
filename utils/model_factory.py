import logging

import torch
import torch.nn as nn

from config.architecture_config import get_dataset_config
from config.model_registry import ModelRegistry

logger = logging.getLogger('model_factory')


def create_model(model_type: str, model_size: str, dataset: str, device: torch.device) -> nn.Module:
    """
    Create a model with improved error handling and performance.
    """
    # Validate models
    if not ModelRegistry.is_model_available(model_type, model_size):
        available_models = [f"{t}_{s}" for t in ModelRegistry.get_supported_types()
                            for s in ModelRegistry.get_supported_sizes()
                            if ModelRegistry.is_model_available(t, s)]
        raise ValueError(f"Model {model_type} {model_size} not available. Available models: {', '.join(available_models)}")

    # Get dataset configs
    dataset_config = get_dataset_config(dataset)
    input_channels = dataset_config['input_channels']
    input_size = dataset_config['input_size']
    num_classes = dataset_config['num_classes']

    # Create models
    try:
        factory = ModelRegistry.get_model_factory(model_type)
        if model_type == 'mlp':
            flat_size = input_size * input_size * input_channels
            model = factory(model_size, flat_size, num_classes)
        elif model_type == 'cnn':
            model = factory(model_size, input_channels, num_classes)
        else:
            model = factory(model_size, input_channels, input_size, num_classes)
        return model.to(device)

    except Exception as e:
        logger.exception(f"Failed to create {model_type} {model_size} model: {str(e)}")
        raise
