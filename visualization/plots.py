import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger('visualization')

class PlotGenerator:
    COMPARISON = "comparisons"
    LEARNING = "learning"
    AUGMENTATION = "augmentation"
    HEATMAP = "heatmaps"

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._init_directories()
        self.plot_configs = self._default_configs()

    def _init_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for category in [self.COMPARISON, self.LEARNING, self.AUGMENTATION, self.HEATMAP]:
            (self.output_dir / category).mkdir(exist_ok=True)

    def _default_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "general": {
                "dpi": 300,
                "title_fontsize": 14,
                "label_fontsize": 12,
                "legend_fontsize": 10,
                "grid_alpha": 0.3,
            },
            "comparison": {
                "figsize": (12, 8),
                "palette": "muted",
            },
            "learning": {
                "figsize": (10, 6),
                "line_width": 2,
                "marker_size": 4,
            },
            "augmentation": {
                "figsize": (14, 8),
                "bar_width": 0.7,
                "error_cap_size": 5,
            },
            "heatmap": {
                "figsize": (16, 10),
                "cmap": "RdYlGn",
                "annot_format": ".2f",
            }
        }

    def _get_path(self, category: str, filename: str) -> Path:
        return self.output_dir / category / f"{filename}.png"

    def _safe_plot(self, plot_func: Callable, *args, **kwargs) -> Optional[Figure]:
        try:
            return plot_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")
            return None

    def model_comparison(self,results: pd.DataFrame, metric: str = 'test_acc', filename: str = 'model_comparison') -> Optional[Figure]:
        def _plot():
            if results.empty or metric not in results.columns:
                raise ValueError(f"Missing required data for metric: {metric}")

            results[metric] = pd.to_numeric(results[metric], errors='coerce')

            cfg = self.plot_configs
            fig, ax = plt.subplots(figsize=cfg["comparison"]["figsize"])

            model_perf = results.groupby(['model_type', 'model_size'])[metric].mean().reset_index()
            sns.barplot(x='model_type', y=metric, hue='model_size', data=model_perf,palette=cfg["comparison"]["palette"], ax=ax)

            # Customize appearance
            ax.set_title(f'Model Performance Comparison ({metric})', fontsize=cfg["general"]["title_fontsize"])
            ax.set_xlabel('Model Architecture', fontsize=cfg["general"]["label_fontsize"])
            ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', fontsize=cfg["general"]["label_fontsize"])
            ax.grid(axis='y', linestyle='--', alpha=cfg["general"]["grid_alpha"])
            ax.legend(title='Model Size', fontsize=cfg["general"]["legend_fontsize"])

            plt.tight_layout()
            plt.savefig(self._get_path(self.COMPARISON, filename), dpi=cfg["general"]["dpi"])
            plt.close()

            return fig

        return self._safe_plot(_plot)

    def learning_curves(self,data: Union[pd.DataFrame, Dict], group_by: str = 'model_size', metric: str = 'train_acc', filename: str = 'learning_curves') -> Optional[Figure]:
        """
        Plot learning curves grouped by model.
        """
        def _plot():
            cfg = self.plot_configs
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

            if df.empty or group_by not in df.columns:
                raise ValueError(f"Missing required grouping column: {group_by}")

            group_values = sorted(df[group_by].unique())
            figure_objects = []

            for group_val in group_values:
                fig, ax = plt.subplots(figsize=cfg["learning"]["figsize"])
                group_data = df[df[group_by] == group_val]

                # For each item in the group, plot learning curve
                for idx, row in group_data.iterrows():
                    if metric in row and isinstance(row[metric], (list, tuple)) and len(row[metric]) > 0:
                        epochs = range(1, len(row[metric]) + 1)
                        label = f"{row.get('model_type', '')}" if group_by == 'model_size' else f"{row.get('model_size', '')}"
                        ax.plot(epochs, row[metric], marker='o', linewidth=cfg["learning"]["line_width"], markersize=cfg["learning"]["marker_size"], label=label)

                if len(ax.get_lines()) > 0:
                    ax.set_title(f"Learning Curves for {group_val.capitalize()} Models", fontsize=cfg["general"]["title_fontsize"])
                    ax.set_xlabel("Epoch", fontsize=cfg["general"]["label_fontsize"])
                    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)", fontsize=cfg["general"]["label_fontsize"])
                    ax.grid(True, alpha=cfg["general"]["grid_alpha"])
                    ax.legend(fontsize=cfg["general"]["legend_fontsize"])

                    output_path = self._get_path(self.LEARNING, f"{filename}_{group_val}")
                    plt.savefig(output_path, dpi=cfg["general"]["dpi"])
                    figure_objects.append(fig)

                plt.close()

            return figure_objects

        return self._safe_plot(_plot)

    def augmentation_effects(self,results: pd.DataFrame, metric: str = 'test_acc', filename: str = 'augmentation_effects') -> Optional[Figure]:
        """
        Visualize augmentation techniques results.
        """
        def _plot():
            if results.empty or 'augmentation_techniques' not in results.columns or metric not in results.columns:
                raise ValueError("Missing required columns for augmentation effects plot")

            results[metric] = pd.to_numeric(results[metric], errors='coerce')

            cfg = self.plot_configs
            fig, ax = plt.subplots(figsize=cfg["augmentation"]["figsize"])

            # baseline
            baseline = results[results['augmentation_techniques'] == 'None'][metric]
            if len(baseline) == 0:
                raise ValueError("No baseline (no augmentation) data available")
            baseline_mean = baseline.mean()

            # aug
            aug_data = results[results['augmentation_techniques'] != 'None']
            if len(aug_data) == 0:
                raise ValueError("No augmentation data available besides baseline")

            # Add baseline
            ax.axhline(y=baseline_mean, color='r', linestyle='--', label=f'Baseline (no augmentation): {baseline_mean:.2f}%')

            # Customize appearance
            ax.set_title(f'Effect of Augmentation Techniques on {metric.replace("_", " ").title()}', fontsize=cfg["general"]["title_fontsize"])
            ax.set_xlabel('Augmentation Technique', fontsize=cfg["general"]["label_fontsize"])
            ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', fontsize=cfg["general"]["label_fontsize"])
            ax.tick_params(axis='x', rotation=45, labelsize=cfg["general"]["label_fontsize"])
            ax.grid(axis='y', linestyle='--', alpha=cfg["general"]["grid_alpha"])
            ax.legend(fontsize=cfg["general"]["legend_fontsize"])

            plt.tight_layout()
            plt.savefig(self._get_path(self.AUGMENTATION, filename), dpi=cfg["general"]["dpi"])
            plt.close()

            return fig

        return self._safe_plot(_plot)

    def augmentation_heatmap(self,results: pd.DataFrame, metric: str = 'test_acc', filename: str = 'augmentation_heatmap') -> Optional[Figure]:
        """
        Create heatmap of augmentation effectiveness by model type.
        """
        def _plot():
            required_cols = ['model_type', 'augmentation_techniques', metric]
            if results.empty or not all(col in results.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")

            results[metric] = pd.to_numeric(results[metric], errors='coerce')

            cfg = self.plot_configs
            fig, ax = plt.subplots(figsize=cfg["heatmap"]["figsize"])

            baseline_df = results[results['augmentation_techniques'] == 'None']
            if len(baseline_df) == 0:
                raise ValueError("No baseline (no augmentation) data available")

            baseline_by_model = baseline_df.groupby('model_type')[metric].mean().reset_index()
            baseline_by_model.columns = ['model_type', 'baseline']

            aug_df = results[results['augmentation_techniques'] != 'None'].copy()
            if len(aug_df) == 0:
                raise ValueError("No augmentation data available besides baseline")

            # Merge baseline
            aug_df = aug_df.merge(baseline_by_model, on='model_type', how='left')
            aug_df['improvement'] = aug_df[metric] - aug_df['baseline']

            # Group heatmap
            avg_improvement = aug_df.groupby(['model_type', 'augmentation_techniques'])['improvement'].mean().reset_index()
            heatmap_data = avg_improvement.pivot(index='model_type', columns='augmentation_techniques', values='improvement')

            # Create visualization
            sns.heatmap(heatmap_data, annot=True, fmt=cfg["heatmap"]["annot_format"], cmap=cfg["heatmap"]["cmap"], center=0, ax=ax, cbar_kws={'label': 'Improvement over Baseline (%)'})
            ax.set_title('Improvement in Performance by Augmentation Technique and Model Type',fontsize=cfg["general"]["title_fontsize"])
            ax.set_ylabel('Model Type', fontsize=cfg["general"]["label_fontsize"])
            ax.set_xlabel('Augmentation Technique', fontsize=cfg["general"]["label_fontsize"])

            plt.tight_layout()
            plt.savefig(self._get_path(self.HEATMAP, filename), dpi=cfg["general"]["dpi"])
            plt.close()

            return fig

        return self._safe_plot(_plot)

    def visualize_augmentations(self, dataset_name: str, techniques: List[str], num_samples: int = 4, intensities: List[float] = None) -> Figure | None:
        """
        Visualize effect of augmentation techniques on sample images.
        """
        def _plot():
            import torch
            import torchvision
            from augmentation.factory import AugmentationFactory

            cfg = self.plot_configs

            # Create directory
            aug_dir = self.output_dir / 'augmentation_samples'
            aug_dir.mkdir(exist_ok=True)

            # Load samples
            try:
                if dataset_name == 'mnist':
                    dataset = torchvision.datasets.MNIST('./data', train=True, download=True)
                else:  # cifar10
                    dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset: {str(e)}")

            sample_indices = list(range(0, num_samples * 10, 10))[:num_samples]
            samples = [dataset[i][0] for i in sample_indices]

            figure_objects = []

            for technique in techniques:
                if technique in ['mixup', 'cutmix', 'adversarial']:
                    continue

                fig, axs = plt.subplots(len(samples), len(intensities), figsize=(3*len(intensities), 3*len(samples)))
                fig.suptitle(f"Effect of {technique} Augmentation", fontsize=16)

                for i, intensity in enumerate(intensities):
                    aug = AugmentationFactory.create_augmentation(technique, intensity)

                    for j, sample in enumerate(samples):
                        ax = axs[j, i] if len(samples) > 1 else axs[i]

                        if not isinstance(sample, torch.Tensor):
                            to_tensor = torchvision.transforms.ToTensor()
                            sample = to_tensor(sample)

                        # Apply augmentation
                        augmented = aug(sample.clone())

                        # Display image
                        if dataset_name == 'mnist':
                            ax.imshow(augmented.squeeze().numpy(), cmap='gray')
                        else:
                            ax.imshow(augmented.permute(1, 2, 0).numpy())

                        ax.set_title(f"Intensity: {intensity}", fontsize=10)
                        ax.axis('off')

                plt.tight_layout()
                plt.savefig(aug_dir / f"{technique}_samples.png", dpi=cfg["general"]["dpi"])
                figure_objects.append(fig)
                plt.close()

            return figure_objects

        return self._safe_plot(_plot)

    def generate_standard_plots(self, results_df: pd.DataFrame) -> None:
        """
        Generate all standard plots from experimental results with support for combination experiments.
        """
        if results_df.empty:
            logger.warning("Cannot generate plots: empty results DataFrame")
            return

        for col in ['test_acc', 'val_acc', 'train_acc', 'test_loss', 'val_loss', 'train_loss']:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

        if 'augmentation_count' not in results_df.columns:
            results_df['augmentation_count'] = results_df['augmentation_techniques'].apply(
                lambda x: 0 if x == 'None' or pd.isna(x) else len(str(x).split(',')))

        # Create comparison plots by model
        self.model_comparison(results_df, 'test_acc', 'model_comparison_test')
        self.model_comparison(results_df, 'val_acc', 'model_comparison_val')

        # Create augmentation count effect plots
        self._plot_augmentation_count_effect(results_df, 'test_acc', 'augmentation_count_effect_test')
        self._plot_augmentation_count_effect(results_df, 'val_acc', 'augmentation_count_effect_val')

        # Plot learning curves by model
        self.learning_curves(results_df, 'model_size', 'train_acc', 'learning_curves_by_size')
        self.learning_curves(results_df, 'model_type', 'train_acc', 'learning_curves_by_type')

        single_aug_df = results_df[results_df['augmentation_count'] == 1].copy()
        if not single_aug_df.empty:
            self.augmentation_effects(single_aug_df, 'test_acc', 'single_augmentation_effects_test')
            self.augmentation_effects(single_aug_df, 'val_acc', 'single_augmentation_effects_val')
            self.augmentation_heatmap(single_aug_df, 'test_acc', 'single_augmentation_heatmap_test')

            if 'augmentation_intensities' in single_aug_df.columns:
                single_aug_df['intensity'] = single_aug_df['augmentation_intensities'].apply(
                    lambda x: float(x) if not pd.isna(x) else 0.0)
                self.intensity_effect_plot(single_aug_df, 'test_acc', 'single_intensity_effect_test')

        self._plot_combination_effect_by_model(results_df, 'test_acc', 'combination_effect_by_model_test')

        logger.info(f"Generated standard plots in {self.output_dir}")

    def _plot_augmentation_count_effect(self, results_df, metric, filename):
        """
        Plot effect of augmentation count on performance.
        """
        try:
            count_perf = results_df.groupby('augmentation_count')[metric].agg(['mean', 'std']).reset_index()

            cfg = self.plot_configs
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.bar(count_perf['augmentation_count'], count_perf['mean'],
                   yerr=count_perf['std'], capsize=5, alpha=0.7)

            for count, mean, std in zip(count_perf['augmentation_count'], count_perf['mean'], count_perf['std']):
                ax.text(count, mean + std + 0.5, f"{mean:.2f}%", ha='center', va='bottom', fontsize=cfg["general"]["label_fontsize"] - 2)

            ax.set_title(f'Effect of Augmentation Combination Size on {metric.replace("_", " ").title()}', fontsize=cfg["general"]["title_fontsize"])
            ax.set_xlabel('Number of Combined Augmentations', fontsize=cfg["general"]["label_fontsize"])
            ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', fontsize=cfg["general"]["label_fontsize"])
            ax.grid(axis='y', alpha=cfg["general"]["grid_alpha"])
            ax.set_xticks(count_perf['augmentation_count'])

            plt.tight_layout()
            plt.savefig(self._get_path(self.COMPARISON, filename), dpi=cfg["general"]["dpi"])
            plt.close()

            return fig
        except Exception as e:
            logger.error(f"Error generating augmentation count effect plot: {str(e)}")
            return None

    def _plot_combination_effect_by_model(self, results_df, metric, filename):
        """
        Plot effect of augmentation combinations by model type.
        """
        try:
            if results_df.empty or 'model_type' not in results_df.columns or 'augmentation_count' not in results_df.columns:
                return None

            baseline_df = results_df[results_df['augmentation_count'] == 0]

            if baseline_df.empty:
                return None

            baseline_by_model = baseline_df.groupby('model_type')[metric].mean().reset_index()
            baseline_by_model.columns = ['model_type', 'baseline']

            aug_df = results_df[results_df['augmentation_count'] > 0].copy()

            if aug_df.empty:
                return None

            aug_df = aug_df.merge(baseline_by_model, on='model_type', how='left')
            aug_df['improvement'] = aug_df[metric] - aug_df['baseline']

            model_count_perf = aug_df.groupby(['model_type', 'augmentation_count'])['improvement'].mean().reset_index()

            cfg = self.plot_configs
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='model_type', y='improvement', hue='augmentation_count', data=model_count_perf, ax=ax)
            ax.set_title(f'Effect of Augmentation Combinations by Model Type ({metric.replace("_", " ").title()})', fontsize=cfg["general"]["title_fontsize"])
            ax.set_xlabel('Model Type', fontsize=cfg["general"]["label_fontsize"])
            ax.set_ylabel(f'Improvement over Baseline (%)', fontsize=cfg["general"]["label_fontsize"])
            ax.grid(axis='y', alpha=cfg["general"]["grid_alpha"])
            ax.legend(title='Augmentation Count', fontsize=cfg["general"]["legend_fontsize"])

            ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(self._get_path(self.COMPARISON, filename), dpi=cfg["general"]["dpi"])
            plt.close()

            return fig
        except Exception as e:
            logger.error(f"Error generating combination effect by model plot: {str(e)}")
            return None

    def intensity_effect_plot(self, results: pd.DataFrame, metric: str = 'test_acc', filename: str = 'intensity_effect') -> Optional[Figure]:
        """
        Visualize effect of different intensities for each augmentation technique.
        """
        def _plot():
            if results.empty or 'augmentation_technique' not in results.columns or 'augmentation_intensity' not in results.columns or metric not in results.columns:
                raise ValueError("Missing required columns for intensity effect plot")

            results[metric] = pd.to_numeric(results[metric], errors='coerce')

            cfg = self.plot_configs
            fig, ax = plt.subplots(figsize=cfg["learning"]["figsize"])
            grouped_data = results.groupby(['augmentation_technique', 'augmentation_intensity'])[metric].mean().reset_index()

            # Plot intensity  vs.  performance for each augmentation
            for technique, data in grouped_data.groupby('augmentation_technique'):
                ax.plot(data['augmentation_intensity'], data[metric], marker='o', linewidth=cfg["learning"]["line_width"], markersize=cfg["learning"]["marker_size"], label=technique)

            if 'None' in results['augmentation_technique'].values:
                baseline = results[results['augmentation_technique'] == 'None'][metric].mean()
                ax.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline (no augmentation): {baseline:.2f}%')

            ax.set_title(f'Effect of Augmentation Intensity on {metric.replace("_", " ").title()}', fontsize=cfg["general"]["title_fontsize"])
            ax.set_xlabel('Augmentation Intensity', fontsize=cfg["general"]["label_fontsize"])
            ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', fontsize=cfg["general"]["label_fontsize"])
            ax.grid(axis='y', linestyle='--', alpha=cfg["general"]["grid_alpha"])
            ax.legend(fontsize=cfg["general"]["legend_fontsize"])

            plt.tight_layout()
            plt.savefig(self._get_path(self.AUGMENTATION, filename), dpi=cfg["general"]["dpi"])
            plt.close()

            return fig

        return self._safe_plot(_plot)