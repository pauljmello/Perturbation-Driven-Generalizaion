import numpy as np
import torch
from torchvision.transforms import RandomAdjustSharpness, RandomAutocontrast, RandomEqualize, GaussianBlur, RandomAffine
from augmentation.base import BatchAugmentationBase, AugmentationBase


class MixUp(BatchAugmentationBase):
    """
    MixUp augmentation implementation.

    Reference: "mixup: Beyond Empirical Risk Minimization"
    https://arxiv.org/abs/1710.09412
    """
    def apply_batch(self, images, targets):
        batch_size = images.size(0)
        device = images.device

        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample([batch_size]).to(device)
        else:
            lam = torch.ones(batch_size, device=device)

        lam = torch.from_numpy(lam).float().to(device)
        lam_expanded = lam.view(-1, 1, 1, 1)

        # Permute batch for and mix samples
        perm_indices = torch.randperm(batch_size, device=device)
        mixed_images = lam_expanded * images + (1 - lam_expanded) * images[perm_indices]

        return mixed_images, targets, targets[perm_indices], lam


class CutMix(BatchAugmentationBase):
    """
    CutMix augmentation implementation.

    Reference: "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    https://arxiv.org/abs/1905.04899
    """
    def apply_batch(self, images, targets):
        batch_size, _, height, width = images.shape
        device = images.device

        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().to(device)
        else:
            lam = torch.tensor(1.0, device=device)

        cut_ratio = torch.sqrt(1.0 - lam)
        cut_w = int((width * cut_ratio).item())
        cut_h = int((height * cut_ratio).item())

        cx = torch.randint(width, (1,), device=device).item()
        cy = torch.randint(height, (1,), device=device).item()

        bbx1 = max(cx - cut_w // 2, 0)
        bby1 = max(cy - cut_h // 2, 0)
        bbx2 = min(cx + cut_w // 2, width)
        bby2 = min(cy + cut_h // 2, height)

        # Process images
        perm_indices = torch.randperm(batch_size, device=device)
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[perm_indices, :, bby1:bby2, bbx1:bbx2]

        lam_adjusted = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (width * height))
        lam_tensor = torch.tensor(lam_adjusted, device=device)

        return mixed_images, targets, targets[perm_indices], lam_tensor


class AugMix(AugmentationBase):
    """
    AugMix augmentation implementation.

    Reference: "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty"
    https://arxiv.org/abs/1912.02781
    """
    def __init__(self, severity=3, width=3, depth=2):
        """
        Initialize AugMix.
        """
        super().__init__()
        self.severity = severity
        self.width = width
        self.depth = depth

        self.aug_ops = [RandomAdjustSharpness(severity, p=1.0), RandomAutocontrast(p=1.0), RandomEqualize(p=1.0),
                        GaussianBlur(3, sigma=(0.1, 2.0)), RandomAffine(degrees=severity * 10, translate=(0.1 * severity, 0.1 * severity)), ]

    def apply(self, x):
        """
        Apply AugMix augmentation.
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input image must be a torch.Tensor")

        original_image = x.clone()
        mixed = torch.zeros_like(x)

        for _ in range(self.width):
            aug_image = x.clone()
            chain_depth = self.depth if self.depth > 0 else np.random.randint(1, 4)

            for _ in range(chain_depth):
                op = np.random.choice(self.aug_ops)
                aug_image = op(aug_image)

            mixed += aug_image

        # Normalize
        mixed = mixed / self.width
        weight = torch.tensor(np.random.beta(1.0, 1.0)).float()
        augmented = weight * original_image + (1 - weight) * mixed

        return augmented.clamp(0, 1)


class AdversarialNoise(BatchAugmentationBase):
    """
    Adversarial noise augmentation (similar to FGSM).
    """
    def __init__(self, epsilon=0.03, alpha=0.01, iterations=1):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.model = None

    def set_model(self, model):
        """
        Set the model for adversarial noise generation.
        """
        self.model = model
        return self

    def apply_batch(self, images, targets):
        """
        Apply adversarial noise augmentation.
        """
        if self.model is None:
            lam = torch.ones(images.size(0), device=images.device)
            return images, targets, targets, lam

        # Clone images
        perturbed_images = images.clone().detach().requires_grad_(True)

        for _ in range(self.iterations):
            outputs = self.model(perturbed_images)
            loss = torch.nn.functional.cross_entropy(outputs, targets)

            self.model.zero_grad()
            loss.backward()

            # Add gradient-based noise
            with torch.no_grad():
                grad_sign = perturbed_images.grad.sign()
                perturbed_images.data = perturbed_images.data + self.alpha * grad_sign

                # Project back to noise hyper-sphere and valid image range
                delta = torch.clamp(perturbed_images.data - images.data, -self.epsilon, self.epsilon)
                perturbed_images.data = torch.clamp(images.data + delta, 0, 1)

            # Reset gradients
            if self.iterations > 1:
                perturbed_images.grad.zero_()

        # No label mixing
        lam = torch.ones(images.size(0), device=images.device)

        return perturbed_images.detach(), targets, targets, lam
