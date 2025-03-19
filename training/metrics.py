import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

logger = logging.getLogger('metrics')


class MetricsTracker:
    """
    Track and compute metrics during training and evaluation.
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize metrics tracker.
        """
        self.metrics = metrics or ['accuracy', 'loss']
        self.history = {metric: [] for metric in self.metrics}
        self.epoch_metrics = {metric: [] for metric in self.metrics}

        # Register metrics
        self.metric_functions = {'accuracy': self.accuracy, 'loss': self.loss, 'precision': self.precision, 'recall': self.recall, 'f1': self.f1}

    def reset(self) -> None:
        """
        Reset epoch metrics.
        """
        self.epoch_metrics = {metric: [] for metric in self.metrics}

    def update(self, outputs: torch.Tensor, targets: torch.Tensor, loss: Optional[float] = None) -> None:
        """
        Update metrics with batch results.
        """
        for metric in self.metrics:
            if metric == 'loss' and loss is not None:
                self.epoch_metrics[metric].append(loss)
            elif metric in self.metric_functions:
                result = self.metric_functions[metric](outputs, targets)
                self.epoch_metrics[metric].append(result)

    def compute_epoch_metrics(self) -> Dict[str, float]:
        """
        Compute average metrics.
        """
        results = {}

        for metric in self.metrics:
            if len(self.epoch_metrics[metric]) > 0:
                results[metric] = np.mean(self.epoch_metrics[metric])
                self.history[metric].append(results[metric])
            else:
                results[metric] = 0.0

        return results

    def get_history(self) -> Dict[str, List[float]]:
        """
        Get history of tracked metrics.
        """
        return self.history

    @staticmethod
    def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute accuracy.
        """
        _, predicted = outputs.max(1)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()
        return 100.0 * correct / total

    @staticmethod
    def loss(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute cross-entropy loss.
        """
        return F.cross_entropy(outputs, targets).item()

    @staticmethod
    def precision(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute precision
        """
        _, predicted = outputs.max(1)
        precision, _, _, _ = precision_recall_fscore_support(targets.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        return precision

    @staticmethod
    def recall(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute recall
        """
        _, predicted = outputs.max(1)
        _, recall, _, _ = precision_recall_fscore_support(targets.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        return recall

    @staticmethod
    def f1(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute F1 score
        """
        _, predicted = outputs.max(1)
        _, _, f1, _ = precision_recall_fscore_support(targets.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        return f1


def compute_confusion_matrix(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix.
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))