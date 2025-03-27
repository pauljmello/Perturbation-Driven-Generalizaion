import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger('callbacks')


class Callback:
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of training.
        """
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of training.
        """
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of an epoch.
        """
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of a batch.
        """
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of a batch.
        """
        pass


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints during training.
    """
    def __init__(self, filepath: Path, monitor: str = 'val_acc', mode: str = 'max', save_best_only: bool = True, save_weights_only: bool = False):
        """
        Initialize model checkpoint callback.
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.best = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Save checkpoint at the end of each epoch.
        """
        if logs is None or self.monitor not in logs:
            return

        current = logs[self.monitor]

        is_best = False
        if self.mode == 'min':
            is_best = current < self.best
        else:
            is_best = current > self.best

        if is_best:
            self.best = current

        if is_best or not self.save_best_only:
            model = getattr(self.trainer, 'model', None)
            if model is None:
                return

            # Create files name
            filename = f"{self.filepath.stem}_epoch{epoch + 1}"
            if is_best:
                filename += "_best"
            filename += ".pth"
            filepath = self.filepath.parent / filename

            if self.save_weights_only:
                torch.save(model.state_dict(), filepath)
            else:
                torch.save(model, filepath)

            logger.info(f"Saved checkpoint to {filepath}")


class LearningRateScheduler(Callback):
    """
    Callback to adjust learning rate during training.
    """
    def __init__(self, scheduler, monitor: str = 'val_loss'):
        """
        Initialize learning rate scheduler callback.
        """
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Adjust learning rate at the end of each epoch.
        """
        if hasattr(self.scheduler, 'step'):
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if logs is not None and self.monitor in logs:
                    self.scheduler.step(logs[self.monitor])
                else:
                    self.scheduler.step()
            else:
                self.scheduler.step()
