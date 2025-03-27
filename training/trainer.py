import logging
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from augmentation.advanced import MixUp, CutMix, AdversarialNoise
from config.architecture_config import EXPERIMENT_CONFIG
from models.base import BaseModel

logger = logging.getLogger('trainer')


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class ModelTrainer:
    """
    Unified trainer.
    """
    def __init__(self, model: BaseModel, train_loader, val_loader, test_loader, device: torch.device, model_dir, exp_id: str, augmentations=None, augmentation_intensities=None, callbacks=None, save_intermediate: bool = False):
        """
        Initialize the model trainer with parameters for training.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model_dir = model_dir
        self.exp_id = exp_id
        self.augmentations = augmentations or []
        self.augmentation_intensities = augmentation_intensities or []
        self.callbacks = callbacks or []
        self.save_intermediate = save_intermediate

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_components()

        self.metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'epoch_times': [], }
        self.best_val_acc = 0
        self.best_epoch = 0
        self.counter = 0

    def _initialize_components(self):
        """
        Initialize the training components including optimizer, scheduler, and augmentations.
        """
        config = EXPERIMENT_CONFIG
        if config['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=config['weight_decay'])
        else:
            raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

        if config['scheduler'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif config['scheduler'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['num_epochs'])
        elif config['scheduler'] == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {config['scheduler']}")

        self.criterion = nn.CrossEntropyLoss()

        self.batch_augs = []
        self.is_mixup = False
        self.is_cutmix = False
        self.is_adversarial = False

        # Handle batch augmentations
        if self.augmentations:
            logger.info(f"Initializing augmentations: {self.augmentations}")
            for i, aug in enumerate(self.augmentations):
                intensity = self.augmentation_intensities[i]
                if aug == 'mixup':
                    self.batch_augs.append(MixUp(alpha=intensity))
                    self.is_mixup = True
                    logger.info(f"Added MixUp augmentation (alpha={intensity})")
                elif aug == 'cutmix':
                    self.batch_augs.append(CutMix(alpha=intensity))
                    self.is_cutmix = True
                    logger.info(f"Added CutMix augmentation (alpha={intensity})")
                elif aug == 'adversarial':
                    adv_aug = AdversarialNoise(epsilon=intensity, alpha=intensity / 3 if intensity > 0 else 0.01, iterations=1)
                    adv_aug.set_model(self.model)
                    self.batch_augs.append(adv_aug)
                    self.is_adversarial = True
                    logger.info(f"Added Adversarial augmentation (epsilon={intensity})")

        if self.batch_augs:
            if len(self.batch_augs) > 1:
                logger.info(f"Using {len(self.batch_augs)} batch augmentations: {[type(aug).__name__ for aug in self.batch_augs]}")
            else:
                logger.info(f"Using batch augmentation: {type(self.batch_augs[0]).__name__}")
        else:
            logger.info("No batch augmentations configured")

    def calculate_loss(self, outputs, inputs=None, targets=None, targets_a=None, targets_b=None, lam=1.0):
        """
        Simplified and fair loss function that focuses on classification performance.
        """
        if isinstance(outputs, tuple):
            # For tuple outputs (like VAE)
            y_pred = outputs[0]
            if len(outputs) >= 4 and hasattr(self, 'track_auxiliary_losses') and self.track_auxiliary_losses:
                self._track_vae_losses(outputs, inputs)
        else:
            y_pred = outputs
        if targets_a is not None and targets_b is not None and lam != 1.0:
            # Mixup/Cutmix
            cls_loss = lam * self.criterion(y_pred, targets_a) + (1 - lam) * self.criterion(y_pred, targets_b)
        else:
            # classification
            cls_loss = self.criterion(y_pred, targets)

        return cls_loss

    def _track_vae_losses(self, outputs, inputs):
        """
        Track VAE auxiliary losses for monitoring.
        """
        _, x_recon, mu, log_var = outputs[:4]

        if inputs is not None and x_recon is not None:
            if x_recon.shape[-2:] != inputs.shape[-2:]:
                x_recon = F.interpolate(x_recon, size=inputs.shape[2:], mode='bilinear', align_corners=False)
            recon_loss = F.mse_loss(x_recon, inputs).item()
            self.metrics.setdefault('recon_loss', []).append(recon_loss)

        # KL divergence
        if mu is not None and log_var is not None:
            kl_loss = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)).item()
            self.metrics.setdefault('kl_loss', []).append(kl_loss)

    def train_epoch(self, epoch: int):
        """
        Training loop with reduced memory usage.
        """
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        correct = torch.tensor(0, device=self.device)
        total = torch.tensor(0, device=self.device)
        start_time = time.time()
        iterator = self.train_loader

        for inputs, targets in iterator:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if self.batch_augs:
                batch_aug = random.choice(self.batch_augs)
                inputs, targets_a, targets_b, lam = batch_aug.apply_batch(inputs, targets)
            else:
                targets_a = targets_b = targets
                lam = 1.0

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.calculate_loss(outputs=outputs, inputs=inputs, targets=targets, targets_a=targets_a, targets_b=targets_b, lam=lam)

            loss.backward()
            self.optimizer.step()

            total_loss += loss

            if isinstance(outputs, tuple) and len(outputs) >= 1:
                pred = outputs[0]
            else:
                pred = outputs

            with torch.no_grad():
                _, predicted = pred.max(1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        # Calculate metrics
        epoch_time = time.time() - start_time
        batch_count = torch.tensor(len(self.train_loader), device=self.device)
        avg_loss = (total_loss / batch_count).item()
        accuracy = (100.0 * correct / total).item()

        return {'loss': avg_loss, 'acc': accuracy, 'epoch_time': epoch_time}

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        correct = torch.tensor(0, device=self.device)
        total = torch.tensor(0, device=self.device)
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                if isinstance(outputs, tuple) and len(outputs) == 4:
                    y_pred, x_recon, mu, log_var = outputs
                    loss = self.criterion(y_pred, targets)
                    pred = y_pred
                else:
                    loss = self.criterion(outputs, targets)
                    pred = outputs

                total_loss += loss.item()
                _, predicted = pred.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        batch_count = torch.tensor(len(self.train_loader), device=self.device)
        avg_loss = (total_loss / batch_count).item()
        accuracy = (100.0 * correct / total).item()

        return {'loss': avg_loss, 'acc': accuracy}

    def save_checkpoint(self, is_best: bool = False, is_final: bool = False) -> None:
        """
        Save a model checkpoint with additional training dynamics information.
        """
        # Skip intermediate checkpoints if save_intermediate is False
        if not self.save_intermediate and is_final:
            return

        checkpoint = {'epoch': len(self.metrics['train_loss']), 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict() if not is_final else None, 'metrics': self.metrics,
                      'best_val_acc': self.best_val_acc, 'best_epoch': self.best_epoch, 'parameter_count': self.model.count_parameters(), 'flops_per_sample': getattr(self.model, 'flops_per_sample', None)}

        try:
            if is_final:
                # Save final model
                filepath = self.model_dir / f"{self.exp_id}_final.pth"
                torch.save(checkpoint, filepath)
                logger.info(f"Saved final model to {filepath}")
            elif is_best:
                # Save best model
                if self.save_intermediate:
                    filepath = self.model_dir / f"{self.exp_id}_best.pth"
                    torch.save(checkpoint, filepath)
                    logger.info(f"Saved best model to {filepath}")
            else:
                # Save latest model
                if self.save_intermediate:
                    filepath = self.model_dir / f"{self.exp_id}_latest.pth"
                    torch.save(checkpoint, filepath)
                    logger.info(f"Saved latest model to {filepath}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            try:
                simplified_checkpoint = {'epoch': len(self.metrics['train_loss']), 'model_state_dict': self.model.state_dict(), 'best_val_acc': self.best_val_acc}
                filepath = self.model_dir / f"{self.exp_id}_simplified.pth"
                torch.save(simplified_checkpoint, filepath)
                logger.info(f"Saved simplified checkpoint after error: {filepath}")
            except Exception as nested_e:
                logger.error(f"Could not save even simplified checkpoint: {str(nested_e)}")

    def train(self):
        # Call callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin()

        self.metrics['timing'] = {'initial_validation_time': 0.0, 'training_time': 0.0, 'validation_time': 0.0, 'test_time': 0.0, 'total_time': 0.0, 'per_epoch': []}

        logger.info(f"Evaluating initial model performance for {self.exp_id}")
        init_val_start = time.time()
        init_val_metrics = self.evaluate(self.val_loader)
        init_val_time = time.time() - init_val_start
        self.metrics['timing']['initial_validation_time'] = init_val_time
        logger.info(f"Initial validation - Loss: {init_val_metrics['loss']:.4f}, Accuracy: {init_val_metrics['acc']:.2f}%")

        # measure throughput
        try:
            sample_batch, _ = next(iter(self.val_loader))
            batch_size = sample_batch.size(0)

            # Warm up
            for _ in range(3):
                with torch.no_grad():
                    self.model(sample_batch.to(self.device))

            # Time forward pass
            iterations = 5
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            with torch.no_grad():
                for _ in range(iterations):
                    self.model(sample_batch.to(self.device))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            # Calculate metrics
            inference_time = (end_time - start_time) / iterations
            samples_per_second = batch_size / inference_time

            # Log metrics
            logger.info(f"Initial forward pass throughput: {samples_per_second:.1f} samples/sec")
            self.metrics['initial_throughput'] = {'samples_per_second': samples_per_second,}
        except Exception as e:
            logger.error(f"Error measuring initial throughput: {str(e)}")

        total_start_time = time.time()

        # Main training loop
        for epoch in range(EXPERIMENT_CONFIG['num_epochs']):
            train_start = time.time()
            epoch_metrics = self.train_epoch(epoch)
            train_time = time.time() - train_start

            val_metrics = self.evaluate(self.val_loader)

            epoch_timing = {'epoch': epoch + 1, 'train_time': train_time, 'total_epoch_time': train_time}

            self.metrics['timing']['per_epoch'].append(epoch_timing)
            self.metrics['timing']['training_time'] += train_time
            self.metrics['train_loss'].append(epoch_metrics['loss'])
            self.metrics['train_acc'].append(epoch_metrics['acc'])
            self.metrics['val_loss'].append(val_metrics['loss'])
            self.metrics['val_acc'].append(val_metrics['acc'])
            self.metrics['epoch_times'].append(epoch_timing['total_epoch_time'])

            logger.info(f"Epoch {epoch + 1}/{EXPERIMENT_CONFIG['num_epochs']} - Train Loss: {epoch_metrics['loss']:.4f}, Train Acc: {epoch_metrics['acc']:.2f}%, "
                        f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.2f}%, Train Time: {train_time:.2f}s")

            # Call callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(epoch, {'loss': epoch_metrics['loss'], 'acc': epoch_metrics['acc']})

            # Step scheduler if needed
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception as e:
                    logger.error(f"Error updating scheduler: {str(e)}")

        test_metrics = self.evaluate(self.test_loader)

        # Record test metrics
        self.metrics['test_loss'] = test_metrics['loss']
        self.metrics['test_acc'] = test_metrics['acc']

        # Calculate total time
        total_time = time.time() - total_start_time
        self.metrics['timing']['total_time'] = total_time

        # Generate timing summary
        logger.info(f"Time breakdown - Initial validation: {self.metrics['timing']['initial_validation_time']:.2f}s, Training: {self.metrics['timing']['training_time']:.2f}s, "
                    f"Validation: {self.metrics['timing']['validation_time']:.2f}s, Testing: {self.metrics['timing']['test_time']:.2f}s")

        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end({"metrics": self.metrics})

        return self.metrics
