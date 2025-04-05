import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any

from config.architecture_config import EXPERIMENT_CONFIG

logger = logging.getLogger('checkpoint')


class CheckpointManager:
    """
    Manages incremental saving of experiment data with checkpoints after each experiment.
    """
    def __init__(self, exp_dir: Path, filename: str = "experiment_checkpoints.csv"):
        """
        Initialize checkpoint manager.
        """
        self.exp_dir = exp_dir
        self.checkpoint_path = exp_dir / filename
        self.fieldnames = None
        self._initialize_checkpoint_file()

    def _initialize_checkpoint_file(self):
        """
        Create checkpoint file with metadata headers.
        """
        if not self.checkpoint_path.exists():
            logger.info(f"Initializing checkpoint file at {self.checkpoint_path}")
            with open(self.checkpoint_path, 'w') as f:
                f.write("# Experiment Checkpoints\n")
                f.write("# Automatically updated after each experiment completes\n")
                f.write("# Created: {}\n\n".format(
                    Path(self.checkpoint_path).parent.name))

    def _serialize_complex_values(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert complex data types (lists, dicts) to JSON strings for CSV compatibility.
        """
        serialized = {}
        for key, value in result.items():
            if isinstance(value, (list, dict, tuple)) and key not in ('random_seed'):
                serialized[key] = json.dumps(value)
            else:
                serialized[key] = value
        return serialized

    def save_checkpoint(self, result: Dict[str, Any]):
        """
        Save a single experiment result to the checkpoint file.
        """
        try:
            if EXPERIMENT_CONFIG.get('metrics_precision', 'fp32') == 'fp8':
                result_to_save = result.copy()
                for field in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'epoch_times']:
                    if field in result_to_save:
                        result_to_save[field] = compress_to_fp8(result_to_save[field])
            else:
                result_to_save = result

            serialized_result = self._serialize_complex_values(result_to_save)
            write_header = not self.checkpoint_path.exists() or self.checkpoint_path.stat().st_size == 0

            if not self.fieldnames:
                self.fieldnames = list(serialized_result.keys())
            else:
                for key in serialized_result.keys():
                    if key not in self.fieldnames:
                        self.fieldnames.append(key)

            for field in self.fieldnames:
                if field not in serialized_result:
                    serialized_result[field] = None

            with open(self.checkpoint_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(serialized_result)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")

    def load_checkpoints(self):
        """
        Load all saved checkpoints from file.
        """
        if not self.checkpoint_path.exists():
            return []

        try:
            results = []
            with open(self.checkpoint_path, 'r', newline='') as f:
                # Skip comments
                for line in f:
                    if not line.startswith('#'):
                        f.seek(f.tell() - len(line))
                        break

                reader = csv.DictReader(f)
                self.fieldnames = reader.fieldnames
                for row in reader:
                    # Deserialize data
                    for key, value in row.items():
                        try:
                            if value and (value.startswith('[') or value.startswith('{')):
                                row[key] = json.loads(value)
                        except (json.JSONDecodeError, AttributeError):
                            pass
                    results.append(row)

            logger.info(f"Loaded {len(results)} checkpoints from {self.checkpoint_path}")
            return results
        except Exception as e:
            logger.error(f"Error loading checkpoints: {str(e)}")
            return []


def compress_to_fp8(value):
    """
    FP8 precision for storing data.
    """
    if isinstance(value, (list, tuple)):
        return type(value)(compress_to_fp8(x) for x in value)
    elif isinstance(value, dict):
        return {k: compress_to_fp8(v) for k, v in value.items()}
    elif isinstance(value, float):
        return 0.0 if abs(value) < 1e-10 else float('{:.2g}'.format(value))
    return value