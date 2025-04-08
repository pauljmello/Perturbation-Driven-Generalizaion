import argparse
import logging
import sys

import pandas as pd
import torch
from tabulate import tabulate

# Force models to register in experiments
from models.base import BaseModel # Crucial, do not remove

sys.path.append('.')


from config.architecture_config import PARAMETER_TARGETS,get_dataset_config
from config.model_registry import ModelRegistry
from utils.validators import count_parameters
from utils import logging_utils

logging_utils.setup_logging(level=logging.INFO)
logger = logging.getLogger('check_params')

STATUS_SYMBOLS = {"pass": "PASS", "warning": "WARN", "fail": "FAIL"}

def create_model_safely(model_type, model_size, input_channels, input_size, num_classes, device):
    """
    Create model with proper handling for each model type's specific requirements.
    """
    try:
        factory = ModelRegistry.get_model_factory(model_type)

        if model_type == 'mlp':
            # MLP needs flattened input size
            flat_size = input_size * input_size * input_channels
            model = factory(model_size, flat_size, num_classes)
        elif model_type == 'cnn':
            # CNN just needs channels and classes
            model = factory(model_size, input_channels, num_classes)
        elif model_type == 'unet':
            # Special handling for UNet - try both argument patterns
            try:
                model = factory(model_size, input_channels, num_classes)
            except TypeError:
                try:
                    model = factory(model_size, input_channels, input_size, num_classes)
                except Exception as e2:
                    logger.error(f"Failed to create UNet with both argument patterns: {str(e2)}")
                    return None
        else:
            # Other models need all parameters
            model = factory(model_size, input_channels, input_size, num_classes)

        return model.to(device)

    except Exception as e:
        logger.error(f"Error creating {model_type} {model_size}: {str(e)}")
        return None


def check_model_parameters(datasets=None, full_analysis=True):
    """
    Check parameter counts of models against target values and output results to terminal.
    """
    # Set defaults
    if datasets is None:
        datasets = ['mnist', 'cifar10']

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Check available models
    available_models = []
    for model_type in ModelRegistry.get_supported_types():
        for model_size in ModelRegistry.get_supported_sizes():
            if ModelRegistry.is_model_available(model_type, model_size):
                available_models.append(f"{model_type}_{model_size}")

    logger.info(f"Available models: {', '.join(available_models) if available_models else 'None'}")

    if not available_models:
        logger.error("No models available. Check model imports and registration.")
        print("\nError: No models registered with ModelRegistry!")
        print("Please ensure all model modules are properly imported.")
        return None

    all_results = {}

    for dataset_name in datasets:
        logger.info(f"Checking models with {dataset_name} dataset")
        results = []

        dataset_config = get_dataset_config(dataset_name)
        input_channels = dataset_config['input_channels']
        input_size = dataset_config['input_size']
        num_classes = dataset_config['num_classes']

        model_types = list(ModelRegistry.get_supported_types())
        model_sizes = list(ModelRegistry.get_supported_sizes())
        input_shape = (1, input_channels, input_size, input_size)

        # Check each model
        for model_type in model_types:
            for model_size in model_sizes:
                if not ModelRegistry.is_model_available(model_type, model_size):
                    logger.warning(f"Model {model_type} {model_size} not available")
                    continue

                try:
                    model = create_model_safely(model_type, model_size, input_channels, input_size, num_classes, device)

                    if model is None:
                        continue

                    params = count_parameters(model)
                    target = PARAMETER_TARGETS[model_size]
                    percent = (params['total'] / target) * 100

                    # Add to results
                    results.append({
                        'model_type': model_type,
                        'model_size': model_size,
                        'dataset': dataset_name,
                        'input_channels': input_channels,
                        'input_size': input_size,
                        'total_params': params['total'],
                        'trainable_params': params['trainable'],
                        'non_trainable_params': params['non_trainable'],
                        'target_params': target,
                        'percent_of_target': percent,
                        'within_range': 95 <= percent <= 105,
                    })

                    logger.info(f"{dataset_name} - {model_type} {model_size}: {params['total']:,} parameters ({percent:.1f}% of target {target:,})")

                except Exception as e:
                    logger.error(f"Error checking {model_type} {model_size} with {dataset_name}: {str(e)}")

        # Store results
        all_results[dataset_name] = results

    # Process results
    combined_results = []
    for dataset_name, results in all_results.items():
        combined_results.extend(results)

    df = pd.DataFrame(combined_results)

    # Create summary
    for dataset_name in datasets:
        dataset_results = df[df['dataset'] == dataset_name].copy()

        # Format for display (using .loc to avoid SettingWithCopyWarning)
        dataset_results.loc[:, 'Status'] = dataset_results['percent_of_target'].apply(lambda x: STATUS_SYMBOLS["pass"]
            if 95 <= x <= 105 else(STATUS_SYMBOLS["warning"] if 90 <= x <= 110 else STATUS_SYMBOLS["fail"]))
        dataset_results.loc[:, 'Parameters'] = dataset_results['total_params'].apply(lambda x: f"{x:,}")
        dataset_results.loc[:, 'Target'] = dataset_results['target_params'].apply(lambda x: f"{x:,}")
        dataset_results.loc[:, '% of Target'] = dataset_results['percent_of_target'].apply(lambda x: f"{x:.1f}%")

        # Sort by model type and size
        dataset_results = dataset_results.sort_values(['model_type', 'model_size'])
        display_df = dataset_results[['Status', 'model_type', 'model_size', 'Parameters', 'Target', '% of Target']]

        # Generate table
        table = tabulate(display_df, headers=['', 'Model Type', 'Size', 'Parameters', 'Target', '% of Target'], tablefmt='pipe')

        # Print dataset summary
        print(f"\nModel Parameter Count Summary - {dataset_name.upper()}")
        print("=" * (30 + len(dataset_name)))
        print(table)

        # Summary statistics
        within_range = dataset_results[dataset_results['Status'] == STATUS_SYMBOLS["pass"]].shape[0]
        warning_range = dataset_results[dataset_results['Status'] == STATUS_SYMBOLS["warning"]].shape[0]
        out_of_range = dataset_results[dataset_results['Status'] == STATUS_SYMBOLS["fail"]].shape[0]

        print(f"\nTotal models checked for {dataset_name}: {len(dataset_results)}")
        print(f"Models within target range (90-110%): {within_range}")
        print(f"Models in warning range (80-90% or 110-120%): {warning_range}")
        print(f"Models out of target range: {out_of_range}")
    logger.info("Parameter analysis complete!")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check parameter counts for neural network models")
    parser.add_argument("--datasets", nargs="+", choices=["mnist", "cifar10"], default=["mnist", "cifar10"], help="Datasets to check models with")
    parser.add_argument("--simple", action="store_false", dest="full_analysis", help="Perform simple analysis only")
    args = parser.parse_args()
    check_model_parameters(datasets=args.datasets, full_analysis=args.full_analysis)
