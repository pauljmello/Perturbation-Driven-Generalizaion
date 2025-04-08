import logging
import random
from abc import ABC, abstractmethod
from typing import List, Callable, Tuple, Union

import torch

from utils import logging_utils

logging_utils.setup_logging(level=logging.INFO)
logger = logging.getLogger('augmentations')


class AugmentationBase(ABC):
    """
    Base class for all augmentation techniques.
    """
    def __init__(self, intensity: float = 0.5):
        """
        Initialize the augmentation.
        """
        self.intensity = intensity

    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the augmentation to a tensor.
        """
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the augmentation to a tensor.
        """
        return self.apply(x)

    @property
    def name(self) -> str:
        """
        Get the name of the augmentation.
        """
        return self.__class__.__name__


class BatchAugmentationBase(ABC):
    """
    Base class for augmentations that operate on batches (e.g., MixUp, CutMix).
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initialize the batch augmentation.
        """
        self.alpha = alpha

    @abstractmethod
    def apply_batch(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply the augmentation to a batch of images and targets.
        """
        pass

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply the augmentation to a batch.
        """
        return self.apply_batch(images, targets)

    @property
    def name(self) -> str:
        """
        Get the name of the augmentation.
        """
        return self.__class__.__name__


class AugmentationPipeline:
    """
    Pipeline for applying multiple augmentations sequentially.
    """
    def __init__(self, augmentations: List[Union[AugmentationBase, Callable]] = None):
        """
        Initialize the augmentation pipeline.
        """
        self.augmentations = augmentations or []

    def add(self, augmentation: Union[AugmentationBase, Callable]) -> None:
        """
        Add an augmentation to the pipeline.
        """
        self.augmentations.append(augmentation)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations in the pipeline with probability based on intensity.
        """
        result = x
        applied_augs = []
        for aug in self.augmentations:
            if random.random() <= 0.5:
                result = aug(result)
                applied_augs.append(aug.name)
        if applied_augs:
            logger.debug(f"Applied augmentations: {applied_augs}")
        return result
