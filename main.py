import argparse
import gc
import logging
import os
import sys

import torch

# Remove potentially problematic environment variables
for env_var in ['KMP_DUPLICATE_LIB_OK', 'KMP_INIT_AT_FORK', 'OMP_NUM_THREADS']:
    os.environ.pop(env_var, None)

torch.set_num_threads(torch.get_num_threads())

from config.architecture_config import EXPERIMENT_CONFIG
from utils import logging_utils
from utils.utils import warmup_pytorch, setup_environment, prepare_experiment_config, run_baseline_experiments, run_augmentation_experiments, finalize_experiments
from utils.dataset_manager import DatasetManager

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


def main():
    args = parse_args()
    base_random_seed = args.seed if args.seed is not None else EXPERIMENT_CONFIG['random_seed']

    # Setup environment
    device, exp_dir, checkpoint_manager = setup_environment(base_random_seed, args.output_dir)
    dataset_name, model_types, model_sizes, aug_configs, experiment_counter = prepare_experiment_config(args, EXPERIMENT_CONFIG)

    warmup_pytorch(device)
    dataset_manager = DatasetManager(dataset_name, EXPERIMENT_CONFIG['batch_size'], base_random_seed, max_cache_size=3)
    DatasetManager.dataloader_warmup(dataset_manager, device)

    # Run baselines
    baseline_results, current_experiment = run_baseline_experiments(model_types, model_sizes, dataset_name, device, exp_dir, experiment_counter, base_random_seed, checkpoint_manager, dataset_manager)

    all_results = list(baseline_results)
    gc.collect()
    warmup_pytorch(device)

    # Run augmentations
    augmentation_results, current_experiment = run_augmentation_experiments(aug_configs, model_types, model_sizes, dataset_name, device, exp_dir,
                                                                            current_experiment, experiment_counter, base_random_seed, checkpoint_manager, dataset_manager)

    all_results.extend(augmentation_results)

    # Finalize
    finalize_experiments(all_results, exp_dir, current_experiment, experiment_counter)


if __name__ == "__main__":
    main()