import logging
import time
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from config.architecture_config import PARAMETER_TARGETS

logger = logging.getLogger('validators')


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {'total': total_params, 'trainable': trainable_params, 'non_trainable': non_trainable_params}


def validate_model_parameters(model: nn.Module, model_type: str, model_size: str, dataset: Optional[str] = None,  input_shape: Optional[Tuple[int, ...]] = None,  tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Validate model parameters against target parameter count and compute efficiency metrics.
    """
    # Get number
    target_params = PARAMETER_TARGETS.get(model_size)
    if target_params is None:
        raise ValueError(f"Unknown model size: {model_size}")

    param_counts = count_parameters(model)

    min_params = int(target_params * (1 - tolerance))
    max_params = int(target_params * (1 + tolerance))

    is_within_range = min_params <= param_counts['total'] <= max_params

    percent_of_target = (param_counts['total'] / target_params) * 100

    # Prepare result
    result = {
        'model_type': model_type,
        'model_size': model_size,
        'total_params': param_counts['total'],
        'trainable_params': param_counts['trainable'],
        'non_trainable_params': param_counts['non_trainable'],
        'target_params': target_params,
        'min_params': min_params,
        'max_params': max_params,
        'is_within_range': is_within_range,
        'percent_of_target': percent_of_target,
        'validation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'parameter_efficiency': {
            'params_per_layer': param_counts['total'] / len(list(model.modules())) if len(list(model.modules())) > 0 else 0,
            'percent_trainable': (param_counts['trainable'] / param_counts['total'] * 100) if param_counts['total'] > 0 else 0,
        }
    }

    log_message = (f"Model {model_type} {model_size}: {param_counts['total']:,} parameters ({percent_of_target:.1f}% of target {target_params:,})")

    if is_within_range:
        logger.info(f"✓ {log_message}")
    else:
        logger.warning(f"✗ {log_message} - outside tolerance range")

    # Test forward pass
    if input_shape is not None:
        try:
            # Generate input
            device = next(model.parameters()).device
            dummy_input = torch.randn(*input_shape, device=device)

            # Record inference time
            start_time = time.time()
            with torch.no_grad():
                output = model(dummy_input)
            inference_time = time.time() - start_time

            # Get output shape
            if isinstance(output, tuple):
                output_shape = tuple(o.shape for o in output)
            else:
                output_shape = tuple(output.shape)

            # Calculate throughput
            samples_per_second = input_shape[0] / inference_time

            result['forward_pass'] = {'success': True, 'input_shape': input_shape, 'output_shape': output_shape, 'inference_time': inference_time, 'samples_per_second': samples_per_second, 'ms_per_sample': (inference_time * 1000) / input_shape[0]}
            logger.info(f"Forward pass successful for {model_type} {model_size}: {samples_per_second:.1f} samples/sec, {(inference_time * 1000) / input_shape[0]:.2f} ms/sample")

        except Exception as e:
            result['forward_pass'] = {'success': False, 'input_shape': input_shape, 'error': str(e)}
            logger.error(f"Forward pass failed for {model_type} {model_size}: {str(e)}")

    return result