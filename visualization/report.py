import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from tabulate import tabulate

logger = logging.getLogger('report')


class ReportGenerator:
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_experiment_summary(self, results_df: pd.DataFrame, filename: str = 'summary.md') -> None:
        # Convert to numeric
        for col in ['test_acc', 'val_acc', 'train_acc', 'test_loss', 'val_loss', 'train_loss']:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

        # Ensure accuracy columns exist
        for acc_col in ['train_acc', 'val_acc', 'test_acc']:
            if acc_col not in results_df.columns or results_df[acc_col].isna().all():
                logger.warning(f"Metric {acc_col} not available or all null, using placeholder values")
                results_df[acc_col] = 0.0

        model_summary = results_df.groupby('model_type')[['train_acc', 'val_acc', 'test_acc']].agg(['mean', 'std']).reset_index()

        # Flatten the multi-index columns
        flat_cols = []
        for col in model_summary.columns:
            if col[0] == 'model_type':
                flat_cols.append('Model Type')
            else:
                metric_name = col[0].split('_')[0].title()
                agg_name = col[1].title()
                flat_cols.append(f'{metric_name} Acc {agg_name} (%)')
        model_summary.columns = flat_cols

        # Compile metrics
        model_count = results_df.groupby('model_type').size().reset_index(name='Count')
        model_summary = model_summary.merge(model_count, on='Model Type')
        model_summary = model_summary.sort_values(by='Train Acc Mean (%)', ascending=True)
        size_summary = results_df.groupby('model_size')[['train_acc', 'val_acc', 'test_acc']].agg(['mean', 'std']).reset_index()

        # Flatten the Multi-Index Columns
        flat_cols = []
        for col in size_summary.columns:
            if col[0] == 'model_size':
                flat_cols.append('Model Size')
            else:
                metric_name = col[0].split('_')[0].title()
                agg_name = col[1].title()
                flat_cols.append(f'{metric_name} Acc {agg_name} (%)')
        size_summary.columns = flat_cols

        # Compile metrics
        size_count = results_df.groupby('model_size').size().reset_index(name='Count')
        size_summary = size_summary.merge(size_count, on='Model Size')
        size_summary = size_summary.sort_values(by='Train Acc Mean (%)', ascending=True)
        aug_summary = results_df.groupby('augmentation_techniques')[['train_acc', 'val_acc', 'test_acc']].agg(['mean', 'std']).reset_index()

        # Flatten the Multi-index Columns
        flat_cols = []
        for col in aug_summary.columns:
            if col[0] == 'augmentation_techniques':
                flat_cols.append('Augmentation')
            else:
                metric_name = col[0].split('_')[0].title()
                agg_name = col[1].title()
                flat_cols.append(f'{metric_name} Acc {agg_name} (%)')
        aug_summary.columns = flat_cols

        # Compile Metrics
        aug_count = results_df.groupby('augmentation_techniques').size().reset_index(name='Count')
        aug_summary = aug_summary.merge(aug_count, on='Augmentation')
        aug_summary = aug_summary.sort_values(by='Train Acc Mean (%)', ascending=True)
        best_models = results_df.sort_values('train_acc', ascending=True).head(25)[['model_type', 'model_size', 'augmentation_techniques', 'train_acc', 'val_acc', 'test_acc']]
        best_models.columns = ['Model Type', 'Size', 'Augmentation', 'Train Acc (%)', 'Val Acc (%)', 'Test Acc (%)']

        model_table = tabulate(model_summary, headers='keys', tablefmt='pipe', floatfmt='.2f')
        size_table = tabulate(size_summary, headers='keys', tablefmt='pipe', floatfmt='.2f')
        aug_table = tabulate(aug_summary, headers='keys', tablefmt='pipe', floatfmt='.2f')
        best_table = tabulate(best_models, headers='keys', tablefmt='pipe', floatfmt='.2f')

        with open(self.report_dir / filename, 'w') as f:
            f.write('# Experiment Summary\n\n')
            f.write('## Performance by Model Type\n\n' + model_table + '\n\n')
            f.write('## Performance by Model Size\n\n' + size_table + '\n\n')
            f.write('## Performance by Augmentation Technique\n\n' + aug_table + '\n\n')
            f.write('## Top Performing Models (Sorted by Ascending Training Accuracy)\n\n' + best_table + '\n\n')
            f.write('## Summary\n\n')
            f.write(f"Total experiments: {len(results_df)}\n")
            f.write(f"Average train accuracy: {results_df['train_acc'].mean():.2f}%\n")
            f.write(f"Average validation accuracy: {results_df['val_acc'].mean():.2f}%\n")
            f.write(f"Average test accuracy: {results_df['test_acc'].mean():.2f}%\n")
            f.write(f"Best train accuracy: {results_df['train_acc'].max():.2f}%\n")
            f.write(f"Best validation accuracy: {results_df['val_acc'].max():.2f}%\n")
            f.write(f"Best test accuracy: {results_df['test_acc'].max():.2f}%\n")

        logger.info(f"Generated experiment summary at {self.report_dir / filename}")

    def generate_model_analysis(self, results_df: pd.DataFrame, model_params: Dict[str, Dict[str, int]] = None, filename: str = 'model_analysis.md') -> None:
        if not model_params:
            logger.warning("No model parameter data provided, generating simplified report")
            with open(self.report_dir / filename, 'w') as f:
                f.write('# Model Architecture Analysis\n\n')
                f.write('## Note\n\n')
                f.write('Parameter data not available for this report.\n')

                if not results_df.empty and 'model_type' in results_df.columns and 'model_size' in results_df.columns:
                    f.write('\n## Model Types and Sizes\n\n')
                    model_info = results_df[['model_type', 'model_size']].drop_duplicates()
                    f.write(tabulate(model_info, headers='keys', tablefmt='pipe'))
            return

        # Convert to numeric
        for col in ['test_acc', 'val_acc', 'train_acc']:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
            else:
                results_df[col] = 0.0

        param_data = []
        for model_key, params in model_params.items():
            model_type, model_size = model_key
            param_data.append({'Model Type': model_type, 'Size': model_size, 'Total Parameters': params.get('total', 0), 'Trainable Parameters': params.get('trainable', 0)})

        param_df = pd.DataFrame(param_data)
        param_df = param_df.sort_values('Total Parameters', ascending=True)
        param_table = tabulate(param_df, headers='keys', tablefmt='pipe', floatfmt=',d')

        # Performance vs parameters analysis
        perf_params = []
        for model_key, params in model_params.items():
            model_type, model_size = model_key
            model_results = results_df[(results_df['model_type'] == model_type) & (results_df['model_size'] == model_size)]
            train_acc = model_results['train_acc'].mean() if not model_results['train_acc'].isna().all() else 0
            val_acc = model_results['val_acc'].mean() if not model_results['val_acc'].isna().all() else 0
            test_acc = model_results['test_acc'].mean() if not model_results['test_acc'].isna().all() else 0
            perf_params.append({'Model Type': model_type, 'Size': model_size, 'Parameters': params.get('total', 0), 'Train Acc (%)': train_acc, 'Val Acc (%)': val_acc, 'Test Acc (%)': test_acc})

        perf_df = pd.DataFrame(perf_params)
        perf_df['Parameters (M)'] = perf_df['Parameters'] / 1_000_000
        perf_df['Train Efficiency'] = perf_df['Train Acc (%)'] / perf_df['Parameters (M)']
        perf_df['Val Efficiency'] = perf_df['Val Acc (%)'] / perf_df['Parameters (M)']
        perf_df['Test Efficiency'] = perf_df['Test Acc (%)'] / perf_df['Parameters (M)']

        perf_df = perf_df.sort_values('Train Acc (%)', ascending=True)
        efficiency_table = tabulate(perf_df[['Model Type', 'Size', 'Parameters (M)', 'Train Acc (%)', 'Val Acc (%)', 'Test Acc (%)', 'Train Efficiency', 'Val Efficiency', 'Test Efficiency']], headers='keys', tablefmt='pipe', floatfmt='.2f')

        with open(self.report_dir / filename, 'w') as f:
            f.write('# Model Architecture Analysis\n\n')
            f.write('## Model Parameter Counts\n\n')
            f.write(param_table + '\n\n')
            f.write('## Efficiency Analysis (Sorted by Ascending Training Accuracy)\n\n')
            f.write(efficiency_table + '\n\n')
            f.write('## Observations\n\n')
            f.write('- Models are ranked by ascending training accuracy\n')
            f.write('- Efficiency is calculated as accuracy (%) per million parameters\n')
            f.write('- Higher efficiency indicates better parameter utilization\n')
            f.write('- Training, validation, and test metrics provide a complete picture of model performance\n')

        logger.info(f"Generated model analysis report at {self.report_dir / filename}")

    def generate_augmentation_analysis(self, results_df: pd.DataFrame, filename: str = 'augmentation_analysis.md') -> None:
        """
        Analyze the effect of augmentation techniques and their combinations.
        """
        for col in ['test_acc', 'val_acc', 'train_acc']:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
            else:
                results_df[col] = 0.0

        # get aug count
        if 'augmentation_count' not in results_df.columns:
            results_df['augmentation_count'] = results_df['augmentation_techniques'].apply(
                lambda x: 0 if x == 'None' else len(str(x).split(',')))

        # Baseline
        baseline_df = results_df[results_df['augmentation_count'] == 0]
        if baseline_df.empty:
            logger.warning("No baseline (no augmentation) data available for comparison")
            with open(self.report_dir / filename, 'w') as f:
                f.write('# Augmentation Technique Analysis\n\n')
                f.write('## Note\n\n')
                f.write('No baseline (no augmentation) data available for comparison.\n')
            return

        baseline_by_model = baseline_df.groupby(['model_type', 'model_size'])[['train_acc', 'val_acc', 'test_acc']].mean().reset_index()

        # augments
        aug_df = results_df[results_df['augmentation_count'] > 0].copy()
        if aug_df.empty:
            logger.warning("No augmentation data available besides baseline")
            with open(self.report_dir / filename, 'w') as f:
                f.write('# Augmentation Technique Analysis\n\n')
                f.write('## Note\n\n')
                f.write('No augmentation data available besides baseline.\n')
            return

        # concat baseline
        aug_df = aug_df.merge(baseline_by_model, on=['model_type', 'model_size'], suffixes=('', '_baseline'))
        aug_df['train_improvement'] = aug_df['train_acc'] - aug_df['train_acc_baseline']
        aug_df['val_improvement'] = aug_df['val_acc'] - aug_df['val_acc_baseline']
        aug_df['test_improvement'] = aug_df['test_acc'] - aug_df['test_acc_baseline']

        count_analysis = aug_df.groupby('augmentation_count')[['train_improvement', 'val_improvement', 'test_improvement']].agg(['mean', 'std']).reset_index()

        flat_cols = []
        for col in count_analysis.columns:
            if col[0] == 'augmentation_count':
                flat_cols.append('Augmentation Count')
            else:
                metric_name = col[0].split('_')[0].title()
                agg_name = col[1].title()
                flat_cols.append(f'{metric_name} Improvement {agg_name} (%)')
        count_analysis.columns = flat_cols

        single_aug_df = aug_df[aug_df['augmentation_count'] == 1]
        if not single_aug_df.empty:
            tech_improvement = single_aug_df.groupby('augmentation_techniques')[['train_improvement', 'val_improvement', 'test_improvement']].agg(['mean', 'std']).reset_index()

            flat_cols = []
            for col in tech_improvement.columns:
                if col[0] == 'augmentation_techniques':
                    flat_cols.append('Technique')
                else:
                    metric_name = col[0].split('_')[0].title()
                    agg_name = col[1].title()
                    flat_cols.append(f'{metric_name} Improvement {agg_name} (%)')
            tech_improvement.columns = flat_cols

            tech_count = single_aug_df.groupby('augmentation_techniques').size().reset_index(name='Count')
            tech_improvement = tech_improvement.merge(tech_count, on='Technique')
            tech_improvement = tech_improvement.sort_values('Train Improvement Mean (%)', ascending=False)
            single_tech_table = tabulate(tech_improvement, headers='keys', tablefmt='pipe', floatfmt='.2f')
        else:
            single_tech_table = "No data available for single augmentation techniques."

        # Find top combinations
        combo_tables = {}
        max_combo = aug_df['augmentation_count'].max()
        for combo_size in range(2, max_combo + 1):
            combo_df = aug_df[aug_df['augmentation_count'] == combo_size]
            if not combo_df.empty:
                top_combos = combo_df.groupby('augmentation_techniques')[['train_improvement', 'val_improvement', 'test_improvement']].mean().reset_index()
                top_combos.columns = ['Technique Combination', 'Train Improvement (%)', 'Val Improvement (%)', 'Test Improvement (%)']
                top_combos = top_combos.sort_values('Test Improvement (%)', ascending=False).head(10)
                combo_tables[combo_size] = tabulate(top_combos, headers='keys', tablefmt='pipe', floatfmt='.2f')
            else:
                combo_tables[combo_size] = f"No data available for {combo_size}-technique combinations."

        # Find best augmentation by model type
        best_by_model_test = aug_df.groupby('model_type')['test_improvement'].idxmax()
        best_combinations = aug_df.loc[best_by_model_test][['model_type', 'model_size', 'augmentation_techniques', 'augmentation_count', 'train_improvement', 'val_improvement', 'test_improvement']]
        best_combinations.columns = ['Model Type', 'Size', 'Augmentation', 'Aug Count', 'Train Improvement (%)', 'Val Improvement (%)', 'Test Improvement (%)']
        best_combinations = best_combinations.sort_values('Test Improvement (%)', ascending=False)
        best_table = tabulate(best_combinations, headers='keys', tablefmt='pipe', floatfmt='.2f')

        # Analyze intensity effects
        if 'augmentation_intensities' in aug_df.columns and not single_aug_df.empty:
            single_aug_df['intensity'] = single_aug_df['augmentation_intensities'].apply(float)
            intensity_analysis = single_aug_df.groupby(['augmentation_techniques', 'intensity'])[['train_improvement', 'val_improvement', 'test_improvement']].mean().reset_index()
            best_intensities = intensity_analysis.sort_values('test_improvement', ascending=False).groupby('augmentation_techniques').head(1)
            best_intensities.columns = ['Technique', 'Best Intensity', 'Train Improvement (%)', 'Val Improvement (%)', 'Test Improvement (%)']
            best_intensities = best_intensities.sort_values('Test Improvement (%)', ascending=False)
            intensity_table = tabulate(best_intensities, headers='keys', tablefmt='pipe', floatfmt='.2f')
        else:
            intensity_table = "Intensity data not available for single augmentation techniques."

        with open(self.report_dir / filename, 'w') as f:
            f.write('# Augmentation Technique Analysis\n\n')

            f.write('## Effect of Augmentation Count\n\n')
            count_table = tabulate(count_analysis, headers='keys', tablefmt='pipe', floatfmt='.2f')
            f.write(count_table + '\n\n')

            f.write('## Single Augmentation Technique Effectiveness\n\n')
            f.write(single_tech_table + '\n\n')

            for combo_size, combo_table in combo_tables.items():
                f.write(f'## Top {combo_size}-Technique Combinations\n\n')
                f.write(combo_table + '\n\n')

            f.write('## Best Augmentation by Model Type\n\n')
            f.write(best_table + '\n\n')

            f.write('## Optimal Intensities for Single Techniques\n\n')
            f.write(intensity_table + '\n\n')

            f.write('## Observations\n\n')
            f.write('- Improvement is measured as percentage points above baseline (no augmentation)\n')
            f.write('- Different model architectures benefit from different augmentation techniques\n')
            f.write('- The effect of combining multiple augmentation techniques varies by combination\n')
            f.write('- Training, validation, and test improvements may vary for the same technique\n')
            f.write('- Optimal intensity might differ depending on the performance metric of interest\n')

        logger.info(f"Generated augmentation analysis report at {self.report_dir / filename}")

    def generate_full_report(self, results_df: pd.DataFrame, model_params: Dict[str, Dict[str, int]] = None) -> None:
        self.generate_experiment_summary(results_df)
        self.generate_model_analysis(results_df, model_params)
        self.generate_augmentation_analysis(results_df)
