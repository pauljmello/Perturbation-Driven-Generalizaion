import gc
import logging
import random
import time
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from torch import amp, nn

from utils.checkpoint import CheckpointManager
from config.architecture_config import AUGMENTATION_CONFIG
from config.model_registry import ModelRegistry
from training.trainer import ModelTrainer
from utils.model_factory import create_model
from visualization.report import ReportGenerator

logger = logging.getLogger('utils')


# Memory Management


def log_memory_usage(tag=None):
    """
    Log current system and GPU memory usage.
    """
    try:
        # Get memory
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / (1024 * 1024)
        system = psutil.virtual_memory()
        system_used_percent = system.percent
        system_used_mb = (system.total - system.available) / (1024 * 1024)
        system_total_mb = system.total / (1024 * 1024)

        log_prefix = f"MEMORY [{tag}]" if tag else "MEMORY"
        memory_msg = f"{log_prefix}: System {system_used_percent:.1f}% ({system_used_mb:.0f}/{system_total_mb:.0f}MB), Process {process_memory_mb:.0f}MB"

        # Add GPU info
        for i in range(torch.cuda.device_count()):
            gpu_allocated_mb = torch.cuda.memory_allocated(i) / (1024 * 1024)
            max_memory_mb = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
            memory_msg += f", GPU{i}: {gpu_allocated_mb:.0f}/{max_memory_mb:.0f}MB"

        logger.info(memory_msg)
    except Exception as e:
        logger.error(f"Error tracking memory: {e}")


# Training Environment


def reset_training_environment(base_seed):
    """
    Reset training environment with a unique seed.
    """
    unique_seed = base_seed + int(time.time() * 1000) % 10000
    random.seed(unique_seed)
    np.random.seed(unique_seed)
    torch.manual_seed(unique_seed)
    torch.cuda.manual_seed_all(unique_seed)
    logger.info(f"Training environment reset with new seed: {unique_seed}")
    return unique_seed


def cleanup_experiment_resources(model=None, trainer=None):
    """
    Clean up experiment resources to prevent memory leaks.
    """
    try:
        # Clean model
        if model is not None:
            model.to('cpu')
            for param in model.parameters():
                param.grad = None
            getattr(model, 'reset_model_state', lambda: None)()
            del model

        # Clean trainer
        if trainer is not None:
            if hasattr(trainer, 'optimizer'):
                trainer.optimizer.zero_grad(set_to_none=True)
                del trainer.optimizer
            if hasattr(trainer, 'scheduler'):
                del trainer.scheduler
            for loader_attr in ['train_loader', 'val_loader', 'test_loader']:
                setattr(trainer, loader_attr, None)
            del trainer

        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            getattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)()

        logger.info("Resources cleaned up")
    except Exception as e:
        logger.warning(f"Error during cleanup: {str(e)}")


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


# Directory Management


def create_experiment_directory(base_dir: Path) -> Path:
    """
    Create a timestamped experiment directory with required subdirectories.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"run_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'models').mkdir(exist_ok=True)
    (exp_dir / 'reports').mkdir(exist_ok=True)
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


# Experiment Setup and Configuration


def setup_environment(base_random_seed, output_dir):
    """
    Set up the environment for experiments.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    log_memory_usage("startup")

    base_output_dir = Path(output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = create_experiment_directory(base_output_dir)
    checkpoint_manager = CheckpointManager(exp_dir)

    return device, exp_dir, checkpoint_manager


def prepare_experiment_config(args, config):
    """
    Prepare the experiment configuration.
    """
    dataset_name = args.dataset or config['dataset']
    model_types = args.model_type.split(',') if args.model_type else config['model_types']
    model_sizes = args.model_size.split(',') if args.model_size else config['model_sizes']

    aug_configs = prepare_augmentation_configs(args, AUGMENTATION_CONFIG)
    experiment_counter = calculate_experiment_count(model_types, model_sizes, aug_configs)
    logger.info(f"Starting experiments: {experiment_counter['total']} total combinations")

    return dataset_name, model_types, model_sizes, aug_configs, experiment_counter


# Experiment Configuration and Execution


def calculate_experiment_count(model_types, model_sizes, aug_configs):
    """
    Calculate the total number of experiments and breakdown by category.
    """
    valid_models = [(model_type, model_size)
                    for model_type in model_types
                    for model_size in model_sizes
                    if ModelRegistry.is_model_available(model_type, model_size)]

    baseline_count = len(valid_models)
    total_experiments = baseline_count
    experiment_breakdown = {'baseline': baseline_count}

    for combo_size, aug_combos, intensity_combos in aug_configs:
        combo_experiments = len(aug_combos) * len(intensity_combos) * baseline_count
        experiment_breakdown[f'{combo_size}_augmentations'] = combo_experiments
        total_experiments += combo_experiments

    experiment_breakdown['total'] = total_experiments
    logger.info(f"Experiment breakdown: {experiment_breakdown}")
    return experiment_breakdown


def valid_model_combinations(model_types, model_sizes):
    """
    Generate valid model type and size combinations.
    """
    for model_type in model_types:
        for model_size in model_sizes:
            if ModelRegistry.is_model_available(model_type, model_size):
                yield model_type, model_size


def run_single_experiment(model_type, model_size, dataset_name, device, train_loader, val_loader, test_loader, exp_dir, current_experiment,
                          total_experiments, base_random_seed, checkpoint_manager, all_results, aug_combo=None, intensity_combo=None):
    """
    Run a single experiment with the given configuration.
    """
    if aug_combo is None:
        exp_id = f"{model_type}_{model_size}_baseline"
        aug_description = "None"
        intensity_description = "None"
        aug_count = 0
    else:
        intensity_str = '-'.join([f"{i:.1f}" for i in intensity_combo])
        exp_id = f"{model_type}_{model_size}_{'_'.join(aug_combo)}_{intensity_str}"
        aug_description = ','.join(aug_combo)
        intensity_description = ','.join([f"{i:.1f}" for i in intensity_combo])
        aug_count = len(aug_combo)

    logger.info(f"Starting experiment {current_experiment}/{total_experiments}: {exp_id}")

    # Reset environment
    seed = reset_training_environment(base_random_seed)

    model = None
    trainer = None

    try:
        model = create_model(model_type=model_type, model_size=model_size, dataset=dataset_name, device=device)
        logger.info(f"Created fresh model for {exp_id}")

        model_dir = exp_dir / 'models'
        trainer = ModelTrainer(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, device=device, model_dir=model_dir, exp_id=exp_id,
                               augmentations=list(aug_combo) if aug_combo else None, augmentation_intensities=list(intensity_combo) if intensity_combo else None, save_intermediate=False)

        logger.info(f"Starting training for {exp_id}")
        metrics = trainer.train()
        result = {'model_type': model_type, 'model_size': model_size, 'augmentation_techniques': aug_description, 'augmentation_intensities': intensity_description, 'augmentation_count': aug_count, 'random_seed': seed, **metrics}

        all_results.append(result)
        checkpoint_manager.save_checkpoint(result)

    except Exception as e:
        logger.error(f"Error during training for {exp_id}: {str(e)}")

    cleanup_experiment_resources(model, trainer)


def prepare_augmentation_configs(args, aug_config):
    """
    Prepare augmentation configurations from command-line arguments or defaults.
    """
    if args.aug_techniques and args.aug_intensities:
        augmentations = args.aug_techniques
        intensities = args.aug_intensities
        return [(1, [tuple(augmentations)], [tuple(intensities)])]
    else:
        standard_augs = aug_config['standard_augmentations']
        advanced_augs = aug_config['advanced_augmentations']
        all_augs = standard_augs + advanced_augs
        intensities = aug_config['intensities']
        max_combo_size = aug_config.get('max_combination_size', 3)

        result = []
        for combo_size in range(1, max_combo_size + 1):
            aug_combos = list(combinations(all_augs, combo_size))
            intensity_combos = list(product(intensities, repeat=combo_size))
            result.append((combo_size, aug_combos, intensity_combos))

    return result


# Experiments


def run_baseline_experiments(model_types, model_sizes, dataset_name, device, exp_dir, experiment_counter, base_random_seed, checkpoint_manager, dataset_manager):
    """
    Run baseline experiments without augmentations.
    """
    logger.info("Running baseline experiments (no augmentations)...")
    baseline_results = []
    current_experiment = 0
    current_model_type = None

    train_loader, val_loader, test_loader = dataset_manager.get_dataloaders()

    for model_type, model_size in valid_model_combinations(model_types, model_sizes):
        if current_model_type is not None and current_model_type != model_type:
            train_loader, val_loader, test_loader = dataset_manager.get_dataloaders()
        current_model_type = model_type

        current_experiment += 1
        log_memory_usage(f"exp_{current_experiment}")
        run_single_experiment(model_type, model_size, dataset_name, device, train_loader, val_loader, test_loader, exp_dir, current_experiment,
                              experiment_counter['total'], base_random_seed, checkpoint_manager, baseline_results, None, None)

    dataset_manager.cleanup_between_iterations()

    return baseline_results, current_experiment


def prepare_augmentation_experiment_list(combo_size, aug_combinations, intensity_combinations, model_types, model_sizes):
    """
    Prepare a list of augmentation experiments to run.
    """
    combo_type = {1: 'single', 2: 'double', 3: 'triple'}.get(combo_size, f'combo_{combo_size}')
    logger.info(f"Running experiments with {combo_size} augmentation technique(s) ({len(aug_combinations)} augmentation combinations × {len(intensity_combinations)} intensity combinations)")
    all_experiments = []
    for aug_combo in aug_combinations:
        for intensity_combo in intensity_combinations:
            for model_type, model_size in valid_model_combinations(model_types, model_sizes):
                all_experiments.append({'aug_combo': aug_combo, 'intensity_combo': intensity_combo, 'model_type': model_type, 'model_size': model_size})
    all_experiments.sort(key=lambda x: (x['aug_combo'], x['intensity_combo']))
    return all_experiments



def run_single_augmentation_experiment(exp, dataset_manager, device, dataset_name, exp_dir, current_experiment, experiment_counter, base_random_seed, checkpoint_manager, combo_results, current_dataloader_key, model_count):
    """
    Run a single augmentation experiment.
    """
    aug_combo = exp['aug_combo']
    intensity_combo = exp['intensity_combo']
    model_type = exp['model_type']
    model_size = exp['model_size']

    dataloader_key = str((list(aug_combo), list(intensity_combo)))

    if current_dataloader_key != dataloader_key:
        if current_dataloader_key is not None:
            dataset_manager.cleanup_between_iterations(tag=f"{current_dataloader_key}_complete")

        # Get dataloaders with the specific augmentations
        train_loader, val_loader, test_loader = dataset_manager.get_dataloaders(list(aug_combo), list(intensity_combo))
        logger.info(f"Warming up dataloaders with augmentations: {aug_combo}")
        dataset_manager.dataloader_warmup(device, list(aug_combo), list(intensity_combo))

        current_dataloader_key = dataloader_key
        model_count = 0
    else:
        # Refresh dataloader reference even when reused
        train_loader, val_loader, test_loader = dataset_manager.get_dataloaders(list(aug_combo), list(intensity_combo))

    current_experiment += 1
    model_count += 1
    run_single_experiment(model_type, model_size, dataset_name, device, train_loader, val_loader, test_loader, exp_dir, current_experiment,
                          experiment_counter['total'], base_random_seed, checkpoint_manager, combo_results, aug_combo, intensity_combo)

    return current_experiment, model_count, current_dataloader_key, model_count


def run_augmentation_experiments(aug_configs, model_types, model_sizes, dataset_name, device, exp_dir, current_experiment, experiment_counter, base_random_seed, checkpoint_manager, dataset_manager):
    """
    Run experiments with augmentations.
    """
    augmentation_results = []

    for combo_size, aug_combinations, intensity_combinations in aug_configs:
        combo_results = []
        all_experiments = prepare_augmentation_experiment_list(
            combo_size, aug_combinations, intensity_combinations, model_types, model_sizes)

        current_dataloader_key = None
        model_count = 0

        for exp in all_experiments:
            current_experiment, model_count, current_dataloader_key, _ = run_single_augmentation_experiment(
                exp, dataset_manager, device, dataset_name, exp_dir, current_experiment,
                experiment_counter, base_random_seed, checkpoint_manager,
                combo_results, current_dataloader_key, model_count)

        # Final cleanup
        if current_dataloader_key is not None:
            dataset_manager.cleanup_between_iterations(tag=f"combo_size_{combo_size}_last_dataloader")

        # Process results
        augmentation_results.extend(combo_results)
        gc.collect()
        dataset_manager.cleanup_between_iterations(tag=f"combo_size_{combo_size}_complete")

    return augmentation_results, current_experiment


# Reporting and Visualization


def generate_final_reports(all_results, exp_dir):
    """
    Generate final reports and visualizations from experiment results.
    """
    if not all_results:
        logger.warning("No results available to generate reports or plots")
        return
    results_df = pd.DataFrame(all_results)

    for col in ['augmentation_techniques', 'augmentation_intensities']:
        if col in results_df.columns and results_df[col].dtype == 'object':
            results_df[col] = results_df[col].apply(lambda x: str(x) if isinstance(x, (list, tuple)) else x)
    csv_path = exp_dir / 'results.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

    # Generate reports
    try:
        report_dir = exp_dir / 'reports'
        report_generator = ReportGenerator(report_dir)
        report_generator.generate_full_report(results_df, {})
        logger.info(f"Generated reports in {report_dir}")
    except Exception as e:
        logger.error(f"Error generating reports: {str(e)}")

    gc.collect()


def finalize_experiments(all_results, exp_dir, current_experiment, experiment_counter):
    """
    Finalize experiments and generate reports.
    """
    log_memory_usage("before_final_reports")
    generate_final_reports(all_results, exp_dir)
    log_memory_usage("end")

    logger.info(f"All experiments completed. Results saved to {exp_dir}")
    logger.info(f"Total experiments run: {current_experiment} out of {experiment_counter['total']}")


# Tools


def setup_mixed_precision(model, precision='fp32'):
    """
    Efficiently configure model for mixed precision training with CUDA.
    """
    if precision == 'fp32' or not torch.cuda.is_available():
        return model, None

    if precision == 'fp16':
        scaler = amp.GradScaler()
        return model, scaler
    elif precision == 'bfp16':
        if not hasattr(torch, 'bfloat16'):
            logger.warning("bfp16 not supported in this PyTorch version, falling back to fp32")
            return model, None
        return model, None
    elif precision == 'fp8':
        if hasattr(torch.cuda, 'is_fp8_available') and torch.cuda.is_fp8_available():
            scaler = amp.GradScaler()  # FP8 needs scaling
            return model, scaler
        else:
            logger.warning("FP8 not supported in this PyTorch version, falling back to bfp16")
            return setup_mixed_precision(model, 'bfp16')

    else:
        logger.warning(f"Unknown precision format: {precision}, falling back to fp32")
        return model, None