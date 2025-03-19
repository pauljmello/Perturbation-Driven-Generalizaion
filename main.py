import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import gc
import logging
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from itertools import combinations, product

from config.architecture_config import EXPERIMENT_CONFIG, get_dataset_config, ARCHITECTURE_CONFIGS, AUGMENTATION_CONFIG
from augmentation.factory import AugmentationFactory
from config.model_registry import ModelRegistry
from utils.model_factory import create_model
from training.trainer import ModelTrainer
from visualization.plots import PlotGenerator
from visualization.report import ReportGenerator
from utils import checkpoint, logging_utils


logging_utils.setup_logging(level=logging.INFO)
logger = logging.getLogger('main')


def parse_args():
    parser = argparse.ArgumentParser(description="Architecture Comparison Framework")

    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], help='Dataset to use (default: from config)')
    parser.add_argument('--model_type', type=str, help='Model type(s) to use, comma-separated (default: all from config)')
    parser.add_argument('--model_size', type=str, help='Model size(s) to use, comma-separated (default: all from config)')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--seed', type=int, help='Random seed')

    # Augs
    parser.add_argument('--aug_techniques', type=str, nargs='+', help='Augmentation techniques to use (default: all from config)')
    parser.add_argument('--aug_intensities', type=float, nargs='+', help='Intensities for augmentation techniques')

    if len(sys.argv) > 1:
        return parser.parse_args()
    else:
        logger.info("No arguments provided, using defaults")
        return parser.parse_args([])


def safe_cleanup():
    """
    Thorough cleanup of resources between experiments.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()


def create_experiment_directory(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"run_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'models').mkdir(exist_ok=True)
    (exp_dir / 'plots').mkdir(exist_ok=True)
    (exp_dir / 'reports').mkdir(exist_ok=True)
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir

def reset_training_environment(base_seed):
    unique_seed = base_seed + int(time.time() * 1000) % 10000
    random.seed(unique_seed)
    np.random.seed(unique_seed)
    torch.manual_seed(unique_seed)
    torch.cuda.manual_seed_all(unique_seed)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info(f"Training environment reset with new seed: {unique_seed}")
    return unique_seed

def safe_cleanup_dataloaders(loaders):
    for loader in loaders:
        if hasattr(loader, '_iterator') and loader._iterator is not None:
            try:
                loader._iterator._shutdown_workers()
            except Exception as e:
                logger.warning(f"Error shutting down dataloader workers: {str(e)}")
        if hasattr(loader, '_iterator'):
            del loader._iterator


def cleanup_experiment_resources(model=None, trainer=None):
    """
    Clean up experiment resources to prevent memory leaks.
    """
    if model is not None:
        try:
            model.to('cpu')
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None  # Clear gradients
        except Exception as e:
            logger.warning(f"Error during model cleanup: {str(e)}")
        finally:
            del model
    if trainer is not None:
        try:
            if hasattr(trainer, 'optimizer'):
                del trainer.optimizer
            if hasattr(trainer, 'scheduler'):
                del trainer.scheduler
        except Exception as e:
            logger.warning(f"Error during trainer cleanup: {str(e)}")
        finally:
            del trainer

    safe_cleanup()
    logger.info("Experiment resources cleaned up")

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


def warmup_pytorch(device):
    """
    Initialize PyTorch systems before experiments to ensure fair timing.
    """
    if torch.cuda.is_available():
        dummy_tensor = torch.ones(1, device=device)
        torch.matmul(dummy_tensor, dummy_tensor)
        torch.cuda.synchronize()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logger.info("PyTorch systems warmed up")


def silent_dataloader_warmup(dm, dev):
    """
    Silently initialize dataloaders with minimal overhead.
    """
    [next(iter(l))[0].to(dev) for l in dm.get_dataloaders()]
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

class DatasetManager:
    """
    Manages dataset loading and creation of dataloaders.
    """
    def __init__(self, dataset_name, batch_size, seed):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_config = get_dataset_config(dataset_name)

        # Load data
        self._load_datasets()

        # Create train/val split
        generator = torch.Generator().manual_seed(seed)
        train_size = int(0.8 * len(self.train_dataset_base))
        indices = torch.randperm(len(self.train_dataset_base), generator=generator).tolist()
        self.train_indices = indices[:train_size]
        self.val_indices = indices[train_size:]

        # Create  transforms
        if dataset_name == 'mnist':
            self.base_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*self.dataset_config['mean'], *self.dataset_config['std'])
            ])
        else:
            self.base_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.dataset_config['mean'], self.dataset_config['std'])
            ])

        # Cache augmentation configurations
        self.dataloader_cache = {}

    def _load_datasets(self):
        """
        Load base datasets.
        """
        logger.info(f"Loading {self.dataset_name} datasets...")
        if self.dataset_name == 'mnist':
            self.train_dataset_base = torchvision.datasets.MNIST('./data', train=True, download=True, transform=None)
            self.test_dataset_base = torchvision.datasets.MNIST('./data', train=False, download=True, transform=None)
        else:
            self.train_dataset_base = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=None)
            self.test_dataset_base = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=None)
        logger.info(f"Datasets loaded")

    def get_dataloaders(self, aug_techniques=None, aug_intensities=None):
        """
        Get dataloaders with specified augmentation technique.
        """
        # cache key
        key = str((aug_techniques, aug_intensities))
        if key in self.dataloader_cache:
            return self.dataloader_cache[key]

        if aug_techniques and aug_intensities:
            train_transform = AugmentationFactory.create_transform(aug_techniques, aug_intensities, self.dataset_name)
        else:
            train_transform = self.base_transform

        train_dataset = TransformSubset(self.train_dataset_base, self.train_indices, train_transform)
        val_dataset = TransformSubset(self.train_dataset_base, self.val_indices, self.base_transform)
        test_dataset = TransformSubset(self.test_dataset_base, range(len(self.test_dataset_base)), self.base_transform)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12, prefetch_factor=6, pin_memory=True, persistent_workers=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, prefetch_factor=2,  pin_memory=True, persistent_workers=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, prefetch_factor=2,  pin_memory=True, persistent_workers=True, drop_last=True)

        # Cache loaders
        self.dataloader_cache[key] = (train_loader, val_loader, test_loader)

        return train_loader, val_loader, test_loader

    def cleanup(self):
        """
        Clean up all dataloaders.
        """
        for loaders in self.dataloader_cache.values():
            safe_cleanup_dataloaders(loaders)
        self.dataloader_cache = {}


def main():
    args = parse_args()
    base_random_seed = EXPERIMENT_CONFIG['random_seed']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    dataset_name = EXPERIMENT_CONFIG['dataset']
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = create_experiment_directory(base_output_dir)
    checkpoint_manager = checkpoint.CheckpointManager(exp_dir)
    model_dir = exp_dir / 'models'
    plots_dir = exp_dir / 'plots'
    report_dir = exp_dir / 'reports'
    model_types = args.model_type.split(',') if args.model_type else EXPERIMENT_CONFIG['model_types']
    model_sizes = args.model_size.split(',') if args.model_size else EXPERIMENT_CONFIG['model_sizes']

    # Get augmentation configuration
    standard_augmentations = AUGMENTATION_CONFIG['standard_augmentations']
    advanced_augmentations = AUGMENTATION_CONFIG['advanced_augmentations']
    all_augmentation_techniques = standard_augmentations + advanced_augmentations
    intensities = AUGMENTATION_CONFIG['intensities']
    max_combination_size = AUGMENTATION_CONFIG.get('max_combination_size', 3)

    # experiment counter
    valid_models = [(model_type, model_size) for model_type in model_types
                    for model_size in model_sizes
                    if ModelRegistry.is_model_available(model_type, model_size)]
    baseline_count = len(valid_models)

    total_experiments = baseline_count
    experiment_breakdown = {'baseline': baseline_count}

    for combo_size in range(1, max_combination_size + 1):
        aug_combos = list(combinations(all_augmentation_techniques, combo_size))
        intensity_combinations = list(product(intensities, repeat=combo_size))
        combo_experiments = len(aug_combos) * len(intensity_combinations) * baseline_count
        experiment_breakdown[f'{combo_size}_augmentations'] = combo_experiments
        total_experiments += combo_experiments

    current_experiment = 0
    logger.info(f"Starting experiments: {total_experiments} total combinations to run")
    logger.info(f"Experiment breakdown: {experiment_breakdown}")

    if args.aug_techniques:
        logger.info("Generating visualizations of augmentation techniques...")
        plot_generator = PlotGenerator(plots_dir)
        try:
            plot_generator.visualize_augmentations(dataset_name, args.aug_techniques)
        except Exception as e:
            logger.error(f"Error generating augmentation visualizations: {str(e)}")

    all_results = []

    # Initialize systems
    warmup_pytorch(device)

    # Create dataset manager
    dataset_manager = DatasetManager(dataset_name, EXPERIMENT_CONFIG['batch_size'], base_random_seed)

    # Silent warmup for dataloader
    logger.info("Silent dataloader warmup process...")
    silent_dataloader_warmup(dataset_manager, device)

    logger.info("Running baseline experiments (no augmentations)...")
    train_loader, val_loader, test_loader = dataset_manager.get_dataloaders()

    for model_type, model_size in valid_models:
        current_experiment += 1
        exp_id = f"{model_type}_{model_size}_baseline"
        reset_training_environment(base_random_seed)

        logger.info(f"Starting experiment {current_experiment}/{total_experiments}: {exp_id}")

        try:
            model = create_model(model_type=model_type, model_size=model_size, dataset=dataset_name, device=device)
            logger.info(f"Created fresh model for {exp_id}")
            trainer = ModelTrainer(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, device=device, model_dir=model_dir, exp_id=exp_id, augmentations=None, save_intermediate=False)

            logger.info(f"Starting training for {exp_id}")
            metrics = trainer.train()
            result = {'model_type': model_type, 'model_size': model_size, 'augmentation_techniques': 'None', 'augmentation_intensities': 'None', 'augmentation_count': 0, 'random_seed': torch.initial_seed(), **metrics}

            all_results.append(result)
            checkpoint_manager.save_checkpoint(result)
        except Exception as e:
            logger.error(f"Error during training for {exp_id}: {str(e)}")
        finally:
            cleanup_experiment_resources(model if 'model' in locals() else None, trainer if 'trainer' in locals() else None)

    for combo_size in range(1, max_combination_size + 1):
        aug_combos = list(combinations(all_augmentation_techniques, combo_size))
        logger.info(f"Running experiments with {combo_size} augmentation techniques ({len(aug_combos)} combinations)...")
        intensity_combinations = list(product(intensities, repeat=combo_size))

        for aug_combo in aug_combos:
            for intensity_combo in intensity_combinations:
                train_loader, val_loader, test_loader = dataset_manager.get_dataloaders(list(aug_combo), list(intensity_combo))
                for model_type, model_size in valid_models:
                    current_experiment += 1
                    intensity_str = '-'.join([f"{i:.1f}" for i in intensity_combo])
                    exp_id = f"{model_type}_{model_size}_{'_'.join(aug_combo)}_{intensity_str}"
                    reset_training_environment(base_random_seed)

                    logger.info(f"Starting experiment {current_experiment}/{total_experiments}: {exp_id}")

                    try:
                        model = create_model(model_type=model_type, model_size=model_size, dataset=dataset_name, device=device)
                        logger.info(f"Created fresh model for {exp_id}")
                        trainer = ModelTrainer(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, device=device,
                                               model_dir=model_dir, exp_id=exp_id, augmentations=list(aug_combo), augmentation_intensities=list(intensity_combo), save_intermediate=False)

                        logger.info(f"Starting training for {exp_id}")
                        metrics = trainer.train()
                        result = {'model_type': model_type, 'model_size': model_size, 'augmentation_techniques': ','.join(aug_combo),
                                  'augmentation_intensities': ','.join([f"{i:.1f}" for i in intensity_combo]), 'augmentation_count': len(aug_combo), 'random_seed': torch.initial_seed(), **metrics}

                        all_results.append(result)
                        checkpoint_manager.save_checkpoint(result)
                    except Exception as e:
                        logger.error(f"Error during training for {exp_id}: {str(e)}")
                    finally:
                        cleanup_experiment_resources(model if 'model' in locals() else None, trainer if 'trainer' in locals() else None)

    # Clean up dataloaders
    dataset_manager.cleanup()

    # Save results
    results_df = pd.DataFrame(all_results)

    # CSV
    for col in ['augmentation_techniques', 'augmentation_intensities']:
        if col in results_df.columns and results_df[col].dtype == 'object':
            results_df[col] = results_df[col].apply(lambda x: str(x) if isinstance(x, (list, tuple)) else x)

    results_df.to_csv(exp_dir / 'results.csv', index=False)

    # Generate plots and reports
    if results_df.empty:
        logger.warning("No results available to generate comprehensive plots or reports.")
    else:
        try:
            plot_generator = PlotGenerator(plots_dir)
            plot_generator.generate_standard_plots(results_df)
        except Exception as e:
            logger.error(f"Error generating comprehensive plots: {str(e)}")

        report_generator = ReportGenerator(report_dir)
        try:
            report_generator.generate_full_report(results_df, {})
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")

    logger.info(f"All experiments completed. Results saved to {exp_dir}")
    logger.info(f"Experiment summary: {experiment_breakdown}")
    logger.info(f"Total experiments run: {current_experiment} out of {total_experiments}")


if __name__ == "__main__":
    main()
