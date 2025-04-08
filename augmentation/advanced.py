import torch
from torchvision.transforms import RandomAdjustSharpness, RandomAutocontrast, RandomEqualize, GaussianBlur, RandomAffine
from augmentation.base import BatchAugmentationBase, AugmentationBase


class MixUp(BatchAugmentationBase):
    """
    MixUp augmentation implementation.
    Reference: "mixup: Beyond Empirical Risk Minimization" https://arxiv.org/abs/1710.09412
    """

    def apply_batch(self, images, targets):
        batch_size = images.size(0)
        device = images.device

        # Sample lambda directly using PyTorch (no need for NumPy conversion)
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample([batch_size]).to(device)
        else:
            lam = torch.ones(batch_size, device=device)

        lam_expanded = lam.view(-1, 1, 1, 1)
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

        # Compute cutout dimensions based on lam (keep the computation on CPU)
        cut_ratio = (1.0 - lam).sqrt().item()
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        cx = torch.randint(width, (1,), device=device).item()
        cy = torch.randint(height, (1,), device=device).item()
        bbx1 = max(cx - cut_w // 2, 0)
        bby1 = max(cy - cut_h // 2, 0)
        bbx2 = min(cx + cut_w // 2, width)
        bby2 = min(cy + cut_h // 2, height)

        perm_indices = torch.randperm(batch_size, device=device)
        mixed_images = images.clone()  # cloning only once
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
        super().__init__()
        self.severity = severity
        self.width = width
        self.depth = depth
        # Define augmentation operators using torchvision transforms.
        self.aug_ops = [
            RandomAdjustSharpness(severity, p=1.0),
            RandomAutocontrast(p=1.0),
            RandomEqualize(p=1.0),
            GaussianBlur(3, sigma=(0.1, 2.0)),
            RandomAffine(degrees=severity * 10, translate=(0.1 * severity, 0.1 * severity)),
        ]

    def apply(self, x):
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input image must be a torch.Tensor")

        original_image = x.clone()
        mixed = torch.zeros_like(x)

        # Process width number of augmentation chains
        for _ in range(self.width):
            aug_image = x.clone()
            # Use provided depth or choose a random chain depth if self.depth==0
            chain_depth = self.depth if self.depth > 0 else torch.randint(1, 4, (1,)).item()
            for _ in range(chain_depth):
                # Randomly choose an augmentation operator
                op = torch.utils.data._utils.collate.default_convert(
                    (self.aug_ops))[0]  # This workaround ensures op is selected each iteration.
                # Alternatively, you can simply use Python's random.choice:
                import random
                op = random.choice(self.aug_ops)
                aug_image = op(aug_image)
            mixed += aug_image

        mixed /= self.width
        # Use torch Beta to sample weight on the same device
        weight = torch.distributions.Beta(1.0, 1.0).sample().to(x.device)
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
        self.model = model
        return self

    def apply_batch(self, images, targets):
        device = images.device
        if self.model is None:
            lam = torch.ones(images.size(0), device=device)
            return images, targets, targets, lam

        # Begin adversarial attack
        perturbed_images = images.clone().detach().requires_grad_(True)
        for _ in range(self.iterations):
            outputs = self.model(perturbed_images)
            model_output = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = torch.nn.functional.cross_entropy(model_output, targets)

            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad_sign = perturbed_images.grad.sign()
                # In-place add for efficiency
                perturbed_images.add_(self.alpha * grad_sign)
                delta = torch.clamp(perturbed_images - images, -self.epsilon, self.epsilon)
                perturbed_images.copy_(torch.clamp(images + delta, 0, 1))
            # Reset gradients unconditionally (since iterations are low)
            perturbed_images.grad.zero_()

        lam = torch.ones(images.size(0), device=device)
        return perturbed_images.detach(), targets, targets, lam
