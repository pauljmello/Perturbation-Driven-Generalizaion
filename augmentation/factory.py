import logging
from typing import List, Optional, Union

import torchvision.transforms as transforms

from augmentation.advanced import MixUp, CutMix, AugMix, AdversarialNoise
from augmentation.base import AugmentationBase, BatchAugmentationBase, AugmentationPipeline
from augmentation.standard import GaussianNoise, SaltPepperNoise, Cutout, RandomErasing, Rotation, Translation, Scaling, HorizontalFlip, VerticalFlip, GaussianBlur, FrequencyDomainTransform
from config.architecture_config import AUGMENTATION_CONFIG

logger = logging.getLogger('augmentation')


class AugmentationFactory:
    """
    Factory for creating augmentation pipelines.
    """
    _standard_augs = {
        'gaussian_noise': GaussianNoise,
        'salt_pepper': SaltPepperNoise,
        'cutout': Cutout,
        'random_erasing': RandomErasing,
        'rotation': Rotation,
        'translation': Translation,
        'scale': Scaling,
        'horizontal_flip': HorizontalFlip,
        'vertical_flip': VerticalFlip,
        'gaussian_blur': GaussianBlur,
        'frequency_domain': FrequencyDomainTransform,
    }

    _batch_augs = {'mixup': MixUp, 'cutmix': CutMix, 'augmix': AugMix, 'adversarial': AdversarialNoise}

    @classmethod
    def create_augmentation(cls, name: str, intensity: float = 0.5, **kwargs) -> Union[AugmentationBase, BatchAugmentationBase]:
        """
        Create an augmentation by name.
        """
        if name in cls._standard_augs:
            return cls._standard_augs[name](intensity=intensity, **kwargs)
        elif name in cls._batch_augs:
            if name == 'mixup':
                return MixUp(alpha=AUGMENTATION_CONFIG['mixup_alpha'])
            elif name == 'cutmix':
                return CutMix(alpha=AUGMENTATION_CONFIG['cutmix_alpha'])
            elif name == 'augmix':
                return AugMix(severity=AUGMENTATION_CONFIG['augmix_severity'], width=AUGMENTATION_CONFIG['augmix_width'], depth=AUGMENTATION_CONFIG['augmix_depth'])
            elif name == 'adversarial':
                return AdversarialNoise()
        else:
            raise ValueError(f"Unsupported augmentation: {name}")

    @classmethod
    def create_pipeline(cls, techniques: List[str], intensities: List[float]) -> AugmentationPipeline:
        """
        Create an augmentation pipeline from a list of techniques and intensities.
        """
        if not techniques:
            return AugmentationPipeline([])

        # Validate lengths
        if len(techniques) != len(intensities):
            error_msg = f"Mismatch between techniques ({len(techniques)}) and intensities ({len(intensities)}) lengths"
            logger.error(error_msg)
            raise ValueError(error_msg)

        standard_techniques = []
        standard_intensities = []

        for tech, intensity in zip(techniques, intensities):
            if tech in cls._standard_augs:
                standard_techniques.append(tech)
                standard_intensities.append(intensity)

        pipeline = AugmentationPipeline()

        for tech, intensity in zip(standard_techniques, standard_intensities):
            aug = cls.create_augmentation(tech, intensity)
            pipeline.add(aug)
            logger.debug(f"Added {tech} with intensity {intensity} to pipeline")

        return pipeline

    @classmethod
    def get_batch_augmentations(cls, techniques: List[str]) -> List[BatchAugmentationBase]:
        """
        Get batch augmentations from a list of techniques.
        """
        batch_augs = []

        for tech in techniques:
            if tech in cls._batch_augs:
                aug = cls.create_augmentation(tech)
                batch_augs.append(aug)

        return batch_augs

    @classmethod
    def get_transform_from_pipeline(cls, pipeline: AugmentationPipeline, dataset: str) -> transforms.Compose:
        """
        Convert an augmentation pipeline to a torchvision transform.
        """
        if dataset == 'mnist':
            mean, std = (0.1307,), (0.3081,) # Standard normalization numbers
        elif dataset == 'cifar10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616) # Standard normalization numbers
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(pipeline), transforms.Normalize(mean, std)])
        return transform

    @classmethod
    def create_transform(cls, techniques: Optional[List[str]], intensities: Optional[List[float]], dataset: str) -> transforms.Compose:
        """
        Create a torchvision transform from augmentation techniques.
        """
        if not techniques or not intensities:
            if dataset == 'mnist':
                return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            else:
                return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        # Create pipeline
        pipeline = cls.create_pipeline(techniques, intensities)
        return cls.get_transform_from_pipeline(pipeline, dataset)