import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

from augmentation.base import AugmentationBase


class GaussianNoise(AugmentationBase):
    """
    Add Gaussian noise to an image.
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian noise augmentation.
        """
        noise = torch.randn_like(x) * self.intensity
        return torch.clamp(x + noise, 0, 1)


class SaltPepperNoise(AugmentationBase):
    """
    Add salt and pepper noise to an image.
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply salt and pepper noise augmentation.
        """
        mask = torch.rand_like(x)
        result = x.clone()
        salt_mask = mask < (self.intensity / 2)
        result[salt_mask] = 1.0
        pepper_mask = mask > (1 - self.intensity / 2)
        result[pepper_mask] = 0.0
        return result


class Cutout(AugmentationBase):
    """
    Apply Cutout augmentation (random square masking).
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Cutout augmentation.
        """
        h, w = x.shape[-2:]
        size = int(min(h, w) * self.intensity)
        y = torch.randint(0, h - size + 1, (1,)).item()
        x_pos = torch.randint(0, w - size + 1, (1,)).item()
        result = x.clone()
        result[:, y:y + size, x_pos:x_pos + size] = 0
        return result


class RandomErasing(AugmentationBase):
    def __init__(self, intensity):
        super().__init__(intensity)
        self._transform = None

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self._transform is None:
            self._transform = transforms.RandomErasing(p=1.0, scale=(0.02, self.intensity), ratio=(0.3, 3.3))
        return self._transform(x)


class Rotation(AugmentationBase):
    """
    Apply random rotation augmentation with full tensor operations.
    """

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation
        """
        max_angle = self.intensity * 90  # Scale intensity to rotation angle
        angle = torch.rand(1).item() * 2 * max_angle - max_angle  # Random angle
        return TF.rotate(x, angle)


class Translation(AugmentationBase):
    """
    Apply random translation augmentation.
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply translation using grid sampling.
        """
        original_ndim = x.ndim
        if original_ndim == 3:
            x = x.unsqueeze(0)

        # Get device and sizes
        device = x.device
        batch_size, _, height, width = x.shape

        theta = torch.eye(2, 3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Generate random shifts on proper device
        max_shift = self.intensity
        rand_x = torch.rand(batch_size, device=device) * 2 * max_shift - max_shift
        rand_y = torch.rand(batch_size, device=device) * 2 * max_shift - max_shift
        theta[:, 0, 2] = rand_x  # x translation
        theta[:, 1, 2] = rand_y  # y translation

        # Apply the affine transformation
        grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)
        transformed = torch.nn.functional.grid_sample(x, grid, align_corners=False)

        if original_ndim == 3:
            transformed = transformed.squeeze(0)

        return transformed

class Scaling(AugmentationBase):
    """
    Apply random scaling augmentation.
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply scaling augmentation.
        """
        scale_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.intensity  # Random scale
        return TF.affine(x, angle=0, translate=[0, 0], scale=scale_factor, shear=0)


class HorizontalFlip(AugmentationBase):
    """
    Apply horizontal flip augmentation.
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply horizontal flip augmentation.
        """
        if torch.rand(1).item() < self.intensity:
            return TF.hflip(x)
        return x


class VerticalFlip(AugmentationBase):
    """
    Apply vertical flip augmentation.
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply vertical flip augmentation.
        """
        if torch.rand(1).item() < self.intensity:
            return TF.vflip(x)
        return x


class GaussianBlur(AugmentationBase):
    """
    Apply Gaussian blur augmentation.
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur augmentation.
        """
        h, w = x.shape[-2:]
        min_dim = min(h, w)
        # Scale kernel size based on image dimensions and intensity
        kernel_size = int(3 + (min_dim / 32) * 8 * self.intensity)
        kernel_size = max(3, min(kernel_size, 11))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        sigma = self.intensity * 2
        return TF.gaussian_blur(x, kernel_size, [sigma, sigma])


class FrequencyDomainTransform(AugmentationBase):
    """
    Apply frequency domain transformations.
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency domain transformation.
        """
        # Apply FFT
        fft = torch.fft.fft2(x)

        # Directly modify the FFT result with random scaling
        # This avoids separate amplitude/phase computation
        fft_mod = fft * (1 + self.intensity * torch.randn_like(fft.real))

        # Inverse FFT and take absolute value
        result = torch.abs(torch.fft.ifft2(fft_mod))

        # Normalize using a faster method
        result = result / (torch.amax(result, dim=(-2, -1), keepdim=True) + 1e-8)
        return torch.clamp(result, 0, 1)
