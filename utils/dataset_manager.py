import gc
import logging
from collections import OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from augmentation.factory import AugmentationFactory
from config.architecture_config import get_dataset_config

logger = logging.getLogger('dataset_manager')


class LRUCache(OrderedDict):
    """
    Least Recently Used (LRU) cache with limited size.
    """
    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


class TransformSubset(torch.utils.data.Dataset):
    """
    Dataset that applies transform to a subset of indices.
    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


class DatasetManager:
    """
    Optimized dataset manager with better caching and resource handling.
    """
    def __init__(self, dataset_name, batch_size, seed, max_cache_size):
        """
        Initialize DatasetManager with LRU cache for dataloaders.
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_config = get_dataset_config(dataset_name)
        self.dataloader_cache = LRUCache(maxsize=max_cache_size)
        self._prepare_datasets_and_transforms(seed)

        if hasattr(torch, 'multiprocessing'):
            torch.multiprocessing.set_sharing_strategy('file_system')

    def _prepare_datasets_and_transforms(self, seed):
        """
        Load datasets and prepare train/val split with transforms.
        """
        if self.dataset_name == 'mnist':
            self.train_dataset_base = torchvision.datasets.MNIST('./data', train=True, download=True, transform=None)
            self.test_dataset_base = torchvision.datasets.MNIST('./data', train=False, download=True, transform=None)
            self.base_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*self.dataset_config['mean'], *self.dataset_config['std'])])
        else:
            self.train_dataset_base = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=None)
            self.test_dataset_base = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=None)
            self.base_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.dataset_config['mean'], self.dataset_config['std'])])

        # Create train/val split
        generator = torch.Generator().manual_seed(seed)
        train_size = int(0.8 * len(self.train_dataset_base))
        indices = torch.randperm(len(self.train_dataset_base), generator=generator).tolist()
        self.train_indices = indices[:train_size]
        self.val_indices = indices[train_size:]

    def get_dataloaders(self, aug_techniques=None, aug_intensities=None):
        """
        Get dataloaders with efficient LRU caching.
        """
        cache_key = str((aug_techniques, aug_intensities))

        # Check cache
        if cache_key in self.dataloader_cache:
            return self.dataloader_cache[cache_key]

        # Create transforms and dataloaders
        train_transform = (AugmentationFactory.create_transform(aug_techniques, aug_intensities, self.dataset_name) if aug_techniques and aug_intensities else self.base_transform)

        train_dataset = TransformSubset(self.train_dataset_base, self.train_indices, train_transform)
        val_dataset = TransformSubset(self.train_dataset_base, self.val_indices, self.base_transform)
        test_dataset = TransformSubset(self.test_dataset_base, range(len(self.test_dataset_base)), self.base_transform)

        # Significant gains in using an optimized number of workers with prefetch factor in train specifically. Best found 8work @ 8pre, else 2work @ 2pre
        train_loader = self._create_dataloader(train_dataset, shuffle=True, workers=8)
        val_loader = self._create_dataloader(val_dataset, shuffle=False, workers=2)
        test_loader = self._create_dataloader(test_dataset, shuffle=False, workers=2)

        # cache
        self.dataloader_cache[cache_key] = (train_loader, val_loader, test_loader)
        return self.dataloader_cache[cache_key]

    def _create_dataloader(self, dataset, shuffle, workers):
        """
        Create optimized dataloader with memory-efficient settings.
        """
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=workers, persistent_workers=workers > 0, pin_memory=True, prefetch_factor=8 if workers >= 8 else 2)

    def cleanup_between_iterations(self, model=None, optimizer=None, tag=None):
        """
        Clean up resources between experiment iterations to prevent memory leaks.
        """
        if tag:
            logger.info(f"Starting cleanup for {tag}")
        self.cleanup_cache()

        if model is not None:
            if next(model.parameters()).is_cuda:
                model_device = next(model.parameters()).device
                model.to('cpu')

                # Zero out gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = None
                model.to(model_device)

        # Optimizer cleanup
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        log_tag = f"post_cleanup_{tag}" if tag else "post_cleanup"
        try:
            from utils.utils import log_memory_usage
            log_memory_usage(log_tag)
        except ImportError:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                logger.info(f"MEMORY [{log_tag}]: CUDA memory: {gpu_mem:.2f} MB")

        logger.info(f"Cleanup complete for {tag if tag else 'iteration'}")

    def cleanup_cache(self):
        """
        Explicitly clean up cached dataloaders.
        """
        logger.info(f"Cleaning up {len(self.dataloader_cache)} cached dataloaders")

        for loaders in list(self.dataloader_cache.values()):
            for loader in loaders:
                if hasattr(loader, '_iterator') and loader._iterator is not None:
                    try:
                        loader._iterator._shutdown_workers()
                        logger.debug(f"Shut down workers for dataloader")
                    except Exception as e:
                        logger.warning(f"Error shutting down dataloader workers: {e}")

        # Clear cache dictionary
        self.dataloader_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Dataloader cache cleaned up")

    def dataloader_warmup(self, device, aug_techniques=None, aug_intensities=None):
        aug_msg = "with augmentations" if aug_techniques else "without augmentations"
        logger.info(f"Warming up dataloaders {aug_msg}")

        try:
            # Important: Use the same augmentations that will be used in training
            loaders = self.get_dataloaders(aug_techniques, aug_intensities)

            for i, loader in enumerate(loaders):
                loader_name = ["train", "val", "test"][i]
                try:
                    batch_count = 0
                    max_batches = 3

                    for batch in loader:
                        sample_batch = [b.to(device, non_blocking=True) for b in batch]
                        batch_count += 1
                        # Clear references to free memory
                        del sample_batch
                        if batch_count >= max_batches:
                            break

                    logger.debug(f"Successfully warmed up {loader_name} loader with {batch_count} batches")
                except StopIteration:
                    logger.warning(f"Empty {loader_name} dataloader")

            logger.info("Dataloader warmup complete")
            return True
        except Exception as e:
            logger.error(f"Error during dataloader warmup: {e}")
            return False
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
