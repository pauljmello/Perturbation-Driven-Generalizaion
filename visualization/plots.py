import ast
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ArchitectureAnalyzer:
    """
    Modular analyzer for neural architecture experiments with noise augmentation focus.
    """
    def __init__(self, csv_file, output_dir="../z.analysis_plots"):
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.df = None
        self.baseline_performances = {}
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """
        Load and preprocess the experimental data.
        """
        print(f"Reading data from {self.csv_file}")
        header_lines = self._count_header_lines()

        expected_columns = ['model_type', 'model_size', 'augmentation_techniques', 'augmentation_intensities', 'augmentation_count', 'random_seed',
                            'train_loss', 'train_acc', 'val_loss', 'val_acc', 'epoch_times', 'timing', 'throughput', 'test_loss', 'test_acc']

        try:
            self.df = pd.read_csv(self.csv_file, skiprows=header_lines)
            print(f"Successfully loaded data with {len(self.df)} rows")
            if 'model_type' not in self.df.columns and len(self.df.columns) == len(expected_columns):
                self.df.columns = expected_columns
                print("Fixed column names based on expected schema")
            self._process_dataframe()
            return True
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return False

    def _count_header_lines(self):
        """
        Count header comment lines in the CSV file.
        """
        header_lines = 0
        with open(self.csv_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    header_lines += 1
                else:
                    break
        return header_lines

    def _process_dataframe(self):
        """
        Process the dataframe to prepare for analysis.
        """
        # Process list and JSON columns
        for col in ['train_acc', 'val_acc', 'train_loss', 'val_loss', 'epoch_times']:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._parse_list)

        for col in ['timing', 'throughput']:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._parse_json)

        # Extract metrics
        for col in ['train_acc', 'val_acc']:
            if col in self.df.columns:
                self.df[f'final_{col}'] = self.df[col].apply(
                    lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        # Create augmentation combo
        if all(c in self.df.columns for c in ['augmentation_techniques', 'augmentation_intensities']):
            self.df['aug_combo'] = self.df.apply(lambda row: self._combine_aug_info(row['augmentation_techniques'], row['augmentation_intensities']), axis=1)
            self.df['has_augmentation'] = self.df['augmentation_techniques'].apply(lambda x: not (pd.isna(x) or str(x).lower() in ['none', 'nan', '']))

            self._calculate_baseline_performances()
            self._calculate_augmentation_effect()

        print("Data processing complete")

    def _parse_list(self, value):
        """
        Parse string representation of lists into actual Python lists.
        """
        if pd.isna(value) or not isinstance(value, str):
            return [] if pd.isna(value) else value
        try:
            if '[' in value and ']' in value:
                return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        return []

    def _parse_json(self, value):
        """
        Parse JSON strings into dictionaries.
        """
        if pd.isna(value) or not isinstance(value, str):
            return {} if pd.isna(value) else value
        try:
            if value.startswith('{') and value.endswith('}'):
                return json.loads(value)
        except (ValueError, SyntaxError):
            pass
        return {}

    def _combine_aug_info(self, techniques, intensities):
        """
        Combine augmentation techniques and intensities into a readable format.
        """
        if pd.isna(techniques) or techniques == 'None' or techniques == 'none':
            return 'No Augmentation'

        if pd.isna(intensities):
            return str(techniques)
        try:
            tech_list = ast.literal_eval(techniques) if isinstance(techniques, str) else techniques
            int_list = ast.literal_eval(intensities) if isinstance(intensities, str) else intensities
            combos = [f"{tech_list[i]}({int_list[i]})" for i in range(min(len(tech_list), len(int_list)))]
            return '+'.join(combos)
        except:
            return f"{techniques}@{intensities}"

    def _calculate_baseline_performances(self):
        """
        Calculate baseline performances for each model type and size.
        """
        baseline_df = self.df[~self.df['has_augmentation']]
        for _, row in baseline_df.iterrows():
            model_key = (row['model_type'], row['model_size'])
            if 'final_val_acc' in row and not pd.isna(row['final_val_acc']):
                self.baseline_performances[model_key] = row['final_val_acc']

    def _calculate_augmentation_effect(self):
        """
        Calculate the effect of augmentations relative to baseline performance.
        """
        self.df['aug_improvement'] = np.nan
        aug_df = self.df[self.df['has_augmentation']]
        for idx, row in aug_df.iterrows():
            model_key = (row['model_type'], row['model_size'])
            if model_key in self.baseline_performances and 'final_val_acc' in row and not pd.isna(row['final_val_acc']):
                baseline_acc = self.baseline_performances[model_key]
                aug_effect = row['final_val_acc'] - baseline_acc
                self.df.at[idx, 'aug_improvement'] = aug_effect

    def generate_plots(self):
        """
        Generate all analysis plots.
        """
        plots = {"noise_augmentation_effectiveness": self.plot_noise_augmentation, "top_augmented_runs": self.plot_top_augmented_runs, "augmentation_size_comparison": self.plot_augmentation_size_comparison,
                 "comprehensive_comparison": self.plot_comprehensive_comparison, "training_efficiency": self.plot_training_efficiency}
        for plot_name, plot_func in plots.items():
            try:
                plot_func()
            except Exception as e:
                print(f"Error generating {plot_name} plot: {e}")

    def plot_noise_augmentation(self):
        """
        Create a heatmap showing the effectiveness of noise augmentations across intensities.
        """
        if 'augmentation_techniques' not in self.df.columns or 'aug_improvement' not in self.df.columns:
            print("Missing required columns for noise augmentation analysis")
            return

        # Filter to noise-based augmentations
        noise_augs = ['gaussian_noise', 'salt_pepper', 'adversarial']
        noise_df = self.df[self.df['augmentation_techniques'].apply(lambda x: any(aug in str(x) for aug in noise_augs) if not pd.isna(x) else False)].copy()

        if len(noise_df) == 0:
            print("No noise augmentation data found")
            return

        # Extract augmentation and intensity
        noise_df['noise_aug_info'] = noise_df.apply(lambda row: self._extract_noise_augmentation(row, noise_augs), axis=1)
        noise_df = noise_df.dropna(subset=['noise_aug_info'])

        if len(noise_df) == 0:
            print("Could not extract noise augmentation information")
            return

        # Split augmentation and intensity
        noise_df['noise_aug'] = noise_df['noise_aug_info'].apply(lambda x: x[0] if x else None)
        noise_df['noise_intensity'] = noise_df['noise_aug_info'].apply(lambda x: x[1] if x and len(x) > 1 else None)

        grouped = noise_df.groupby(['model_type', 'noise_aug', 'noise_intensity'])['aug_improvement'].mean().reset_index()
        pivot_df = grouped.pivot_table(values='aug_improvement', index=['model_type', 'noise_aug'], columns='noise_intensity', aggfunc='mean')

        self._plot_heatmap(pivot_df,"Noise Augmentation Effectiveness by Intensity","noise_augmentation_effectiveness.png", cmap='RdBu_r', center=0, cbar_label='Validation Accuracy Improvement (pp)')

    def _extract_noise_augmentation(self, row, noise_augs):
        """
        Extract noise augmentation type and intensity from a row.
        """
        if pd.isna(row['augmentation_techniques']):
            return None

        try:
            augs = ast.literal_eval(row['augmentation_techniques']) if isinstance(row['augmentation_techniques'], str) else row['augmentation_techniques']
            ints = ast.literal_eval(row['augmentation_intensities']) if isinstance(row['augmentation_intensities'], str) else row['augmentation_intensities']

            for i, aug in enumerate(augs):
                if any(noise in aug for noise in noise_augs):
                    return (aug, ints[i] if i < len(ints) else None)
        except:
            pass
        return None

    def plot_top_augmented_runs(self, top_n=20):
        """
        Plot the top N training runs with augmentations, showing their accuracy curves.
        """
        if 'train_acc' not in self.df.columns or 'aug_combo' not in self.df.columns:
            print("Missing required columns for top augmented runs analysis")
            return

        aug_df = self.df[self.df['has_augmentation']].copy()
        if len(aug_df) == 0:
            print("No augmented runs found in the dataset")
            return

        # Sort by validation acc
        top_runs = aug_df.sort_values('final_val_acc', ascending=False).head(top_n)
        plt.figure(figsize=(16, 10))

        n_colors = len(top_runs['model_type'].unique())
        color_palette = plt.cm.viridis(np.linspace(0, 0.9, n_colors))
        color_map = dict(zip(sorted(top_runs['model_type'].unique()), color_palette))
        line_styles = {'small': '-', 'medium': '--', 'large': '-.'}

        # Plot training acc
        for idx, row in top_runs.iterrows():
            if isinstance(row['train_acc'], list) and len(row['train_acc']) > 0:
                epochs = range(1, len(row['train_acc']) + 1)
                label = f"{row['model_type']}_{row['model_size']} ({row['aug_combo']})"

                plt.plot(epochs, row['train_acc'], label=label, color=color_map.get(row['model_type'], 'black'), linestyle=line_styles.get(row['model_size'], '-'), marker='o', markersize=4)

        plt.title("Top 20 Augmented Training Runs: Training Accuracy Over Time", fontsize=16)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Training Accuracy (%)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()

        filename = os.path.join(self.output_dir, "top_augmented_runs.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()

    def plot_augmentation_size_comparison(self):
        """
        Compare how the same augmentations perform across different model sizes.
        """
        if 'aug_combo' not in self.df.columns or 'final_val_acc' not in self.df.columns:
            print("Missing required columns for augmentation-size comparison")
            return

        # Get aug combos
        aug_counts = self.df['aug_combo'].value_counts()
        common_augs = aug_counts[aug_counts >= 3].index.tolist()

        if len(common_augs) == 0:
            print("No common augmentation combinations found")
            return

        # Limiter
        top_augs = common_augs[:min(6, len(common_augs))]
        plot_df = self.df[self.df['aug_combo'].isin(top_augs)].copy()

        if len(plot_df) == 0:
            print("No data available for common augmentations")
            return

        # Create grid
        n_augs = len(top_augs)
        n_cols = min(3, n_augs)
        n_rows = (n_augs + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(15, 4 * n_rows))

        for i, aug in enumerate(top_augs):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)

            aug_data = plot_df[plot_df['aug_combo'] == aug]

            pivoted = aug_data.pivot_table(values='final_val_acc', index='model_type', columns='model_size', aggfunc='mean')
            pivoted.plot(kind='bar', ax=ax)

            ax.set_title(f"Augmentation: {aug}", fontsize=12)
            ax.set_ylabel("Validation Accuracy (%)", fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)

            # Add labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', fontsize=8)

        plt.suptitle("Performance Comparison Across Model Sizes with Same Augmentations", fontsize=16)
        plt.tight_layout()

        filename = os.path.join(self.output_dir, "augmentation_size_comparison.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()

    def plot_comprehensive_comparison(self):
        """
        Create a comprehensive comparison matrix across model types, sizes and augmentations.
        """
        if not {'model_type', 'model_size', 'aug_combo', 'final_val_acc'}.issubset(self.df.columns):
            print("Missing required columns for comprehensive comparison matrix")
            return

        pivot_df = self.df.pivot_table(values='final_val_acc', index=['model_type', 'model_size'], columns='aug_combo', aggfunc='mean')

        # Handle large matrices
        if pivot_df.shape[1] > 8:
            col_means = pivot_df.mean()
            baseline_col = 'No Augmentation' if 'No Augmentation' in pivot_df.columns else None

            # Select top augs
            top_cols = col_means.sort_values(ascending=False).head(7).index.tolist()
            if baseline_col and baseline_col not in top_cols:
                top_cols.append(baseline_col)

            pivot_df = pivot_df[top_cols]

        height_per_row = 0.5
        fig_height = max(8, len(pivot_df) * height_per_row)

        self._plot_heatmap(pivot_df,"Comprehensive Performance Matrix: Architecture × Size × Augmentation",
                           "comprehensive_comparison.png", figsize=(14, fig_height), cmap='viridis', cbar_label='Validation Accuracy (%)')

    def plot_training_efficiency(self):
        """
        Analyze training speed vs. validation accuracy.
        """
        if 'epoch_times' not in self.df.columns or 'final_val_acc' not in self.df.columns:
            print("Missing required columns for training efficiency analysis")
            return

        self.df['avg_epoch_time'] = self.df['epoch_times'].apply(
            lambda x: sum(x) / len(x) if isinstance(x, list) and len(x) > 0 else np.nan)

        plot_df = self.df.dropna(subset=['avg_epoch_time', 'final_val_acc'])

        if len(plot_df) == 0:
            print("No valid data for training efficiency plot")
            return

        plt.figure(figsize=(12, 8))

        # Create scatter plot
        model_types = plot_df['model_type'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))
        markers = {'small': 'o', 'medium': 's', 'large': '^'}

        for i, model_type in enumerate(model_types):
            model_data = plot_df[plot_df['model_type'] == model_type]
            for size in model_data['model_size'].unique():
                size_data = model_data[model_data['model_size'] == size]
                plt.scatter(size_data['avg_epoch_time'], size_data['final_val_acc'], label=f"{model_type.upper()} {size}", color=colors[i], marker=markers.get(size, 'o'), s=100, alpha=0.7)

        plt.title("Training Efficiency: Speed vs. Accuracy", fontsize=14)
        plt.xlabel("Average Time per Epoch (seconds)", fontsize=12)
        plt.ylabel("Final Validation Accuracy (%)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add efficiency boundary lines
        plt.axhline(y=np.percentile(plot_df['final_val_acc'], 75), color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=np.percentile(plot_df['avg_epoch_time'], 25), color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()

        filename = os.path.join(self.output_dir, "training_efficiency.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.close()

    def _plot_heatmap(self, data, title, filename, figsize=None, cmap='viridis', center=None, cbar_label=None):
        """
        Helper method for creating heatmap plots.
        """
        plt.figure(figsize=figsize or (12, 8))
        ax = sns.heatmap(data, annot=True, fmt=".1f" if 'acc' in cbar_label.lower() else ".2f", cmap=cmap, center=center, linewidths=.5, cbar_kws={'label': cbar_label})

        plt.title(title, fontsize=16)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {output_path}")
        plt.close()

    def run_all_analyses(self):
        """
        Run all analysis plots.
        """
        if self.load_data():
            self.generate_plots()
            print(f"All analyses complete! Results saved to {self.output_dir}/")
        else:
            print("Failed to load data. Cannot generate analyses.")


def main():
    """
    Main function to run all analyses.
    """
    csv_file_path = "../results/run_20250322_161910/experiment_checkpoints.csv"
    analyzer = ArchitectureAnalyzer(csv_file_path)
    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()
