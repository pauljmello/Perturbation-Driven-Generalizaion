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
        return x + noise


class SaltPepperNoise(AugmentationBase):
    """
    Add salt and pepper noise to an image.
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply salt and pepper noise augmentation.
        """
        noise_mask = torch.rand_like(x)
        salt = (noise_mask < self.intensity / 2).float()
        pepper = (noise_mask > (1 - self.intensity / 2)).float()
        return x * (1 - salt - pepper) + salt


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

        # Random coordinates
        y = torch.randint(0, h - size + 1, (1,)).item()
        x_pos = torch.randint(0, w - size + 1, (1,)).item()

        # Apply cutout
        result = x.clone()
        result[:, y:y + size, x_pos:x_pos + size] = 0

        return result


class RandomErasing(AugmentationBase):
    """
    Apply random erasing augmentation.
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random erasing augmentation.
        """
        transform = transforms.RandomErasing(p=1.0, scale=(0.02, self.intensity), ratio=(0.3, 3.3))
        return transform(x)


class Rotation(AugmentationBase):
    """
    Apply random rotation augmentation.
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation augmentation.
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
        Apply translation augmentation.
        """
        h, w = x.shape[-2:]
        max_shift_h = int(h * self.intensity)
        max_shift_w = int(w * self.intensity)
        shift_h = torch.randint(-max_shift_h, max_shift_h + 1, (1,)).item()
        shift_w = torch.randint(-max_shift_w, max_shift_w + 1, (1,)).item()

        # Apply affine transform
        return TF.affine(x, angle=0, translate=[shift_w, shift_h], scale=1.0, shear=0)


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

        # Modify amplitude
        amplitude = torch.abs(fft)
        phase = torch.angle(fft)

        # Apply random modifications to amplitude
        amplitude *= (1 + self.intensity * torch.randn_like(amplitude))

        # Reconstruct signal
        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)
        modified_fft = torch.complex(real, imag)

        # Inverse FFT
        ifft = torch.fft.ifft2(modified_fft)
        result = torch.abs(ifft)

        # Normalize
        result = result / result.max()
        return torch.clamp(result, 0, 1)
