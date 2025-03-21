import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

logger = logging.getLogger('models')


class BaseModel(nn.Module, ABC):
    """
    Base class for all model architectures.
    Provides common functionality.
    """

    def __init__(self, model_type: str, model_size: str):
        """
        Initialize the base model.
        """
        super().__init__()
        self.model_type = model_type
        self.model_size = model_size
        self._param_count: Optional[Dict[str, int]] = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        """
        pass

    def count_parameters(self) -> Dict[str, int]:
        """
        Count the number of parameters in the model.
        """
        if self._param_count is not None:
            return self._param_count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        self._param_count = {'total': total_params, 'trainable': trainable_params, 'non_trainable': non_trainable_params}
        return self._param_count

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        """
        params = self.count_parameters()
        return {'model_type': self.model_type, 'model_size': self.model_size, 'total_params': params['total'], 'trainable_params': params['trainable'], 'non_trainable_params': params['non_trainable']}

    def test_forward_pass(self, input_shape: Tuple[int, ...], device: torch.device) -> Dict[str, Any]:
        """
        Test a forward pass through the model.
        """
        try:
            x = torch.randn(*input_shape, device=device)

            # Record inference time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            with torch.no_grad():
                output = self(x)
            end_time.record()

            # Synchronize CUDA
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000

            # Get output shape
            if isinstance(output, tuple):
                output_shape = tuple(o.shape for o in output)
            else:
                output_shape = tuple(output.shape)

            return {'success': True, 'input_shape': input_shape, 'output_shape': output_shape, 'inference_time': inference_time }
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            return { 'success': False, 'input_shape': input_shape, 'error': str(e)}

    def reset_parameters(self):
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()