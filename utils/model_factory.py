import logging

import torch
import torch.nn as nn

from config.architecture_config import get_dataset_config
from config.model_registry import ModelRegistry
from models.cnn import BaseCNN
from models.mlp import BaseMLP
from utils.validators import validate_model_parameters

logger = logging.getLogger('model_factory')

def create_model(model_type: str, model_size: str, dataset: str, device: torch.device, validate: bool = True, retry_on_error: bool = True) -> nn.Module:
    """
    Create a model for the specified type, size, and dataset with improved error handling.
    """
    if not ModelRegistry.is_model_available(model_type, model_size):
        available_models = [f"{t}_{s}" for t in ModelRegistry.get_supported_types()
                            for s in ModelRegistry.get_supported_sizes()
                            if ModelRegistry.is_model_available(t, s)]
        error_msg = f"Model {model_type} {model_size} not available. Available models: {', '.join(available_models)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        factory = ModelRegistry.get_model_factory(model_type)
    except Exception as e:
        logger.error(f"Error getting model factory for {model_type}: {str(e)}")
        raise RuntimeError(f"Failed to get model factory: {str(e)}")

    # Get dataset config
    try:
        dataset_config = get_dataset_config(dataset)
        input_channels = dataset_config['input_channels']
        input_size = dataset_config['input_size']
        num_classes = dataset_config['num_classes']
    except Exception as e:
        logger.error(f"Error getting dataset config for {dataset}: {str(e)}")
        raise ValueError(f"Invalid dataset '{dataset}': {str(e)}")

    model = None
    try:
        if model_type == 'mlp':
            # MLP needs flattened input
            flat_size = input_size * input_size * input_channels
            model = factory(model_size, flat_size, num_classes)
        elif model_type == 'cnn':
            # CNN needs input_channels and num_classes
            model = factory(model_size, input_channels, num_classes)
        else:
            # Other models need input_size
            model = factory(model_size, input_channels, input_size, num_classes)
    except Exception as e:
        logger.error(f"Error creating {model_type} {model_size} model: {str(e)}")
        if retry_on_error:
            logger.info(f"Retrying model creation with safer defaults")
            try:
                if model_type == 'mlp':
                    flat_size = input_size * input_size * input_channels
                    model = BaseMLP(model_size, flat_size, num_classes)
                elif model_type == 'cnn':
                    model = BaseCNN(model_size, input_channels, num_classes)
                else:
                    if model_size != 'small':
                        logger.info(f"Falling back to small {model_type} model")
                        factory = ModelRegistry.get_model_factory(model_type)
                        model = factory('small', input_channels, input_size, num_classes)
                    else:
                        raise ValueError(f"Cannot create model {model_type} {model_size}")
            except Exception as retry_error:
                logger.error(f"Retry failed with error: {str(retry_error)}")
                raise ValueError(f"Failed to create {model_type} {model_size} model: {str(e)}")
        else:
            raise ValueError(f"Failed to create {model_type} {model_size} model: {str(e)}")

    try:
        model = model.to(device)
    except Exception as e:
        logger.error(f"Error moving model to device {device}: {str(e)}")
        if str(device) != 'cpu':
            logger.info(f"Falling back to CPU")
            try:
                model = model.to('cpu')
            except Exception as cpu_error:
                logger.error(f"Error moving model to CPU: {str(cpu_error)}")
                raise RuntimeError(f"Cannot place model on any device: {str(e)}")
    return model
