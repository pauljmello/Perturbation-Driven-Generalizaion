import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from pathlib import Path
from torchvision import transforms

from augmentation.standard import (
    GaussianNoise, SaltPepperNoise, Cutout, RandomErasing,
    Rotation, Translation, Scaling, HorizontalFlip,
    VerticalFlip, GaussianBlur, FrequencyDomainTransform
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def visualize_augmentations(output_dir='z.analysis_plots'):
    """
    Create grid visualization of augmentations applied to CIFAR-10 classes
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # Define augmentations to visualize
    augmentations = {
        'Original': None,
        'Gaussian Noise': GaussianNoise(intensity=0.5),
        'Salt & Pepper': SaltPepperNoise(intensity=0.5),
        'Cutout': Cutout(intensity=0.5),
        'Random Erasing': RandomErasing(intensity=0.5),
        'Rotation': Rotation(intensity=0.5),
        'Translation': Translation(intensity=0.5),
        'Scaling': Scaling(intensity=0.5),
        'Horizontal Flip': HorizontalFlip(intensity=1.0),
        'Vertical Flip': VerticalFlip(intensity=1.0),
        'Gaussian Blur': GaussianBlur(intensity=0.5),
        'Frequency Domain': FrequencyDomainTransform(intensity=0.5)
    }

    # Load CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

    # Get sample images for each class
    samples = {}
    for class_idx in range(len(class_names)):
        indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
        if indices:
            samples[class_idx] = dataset[indices[0]][0]

    # Create figure
    fig, axes = plt.subplots(
        len(class_names),
        len(augmentations),
        figsize=(len(augmentations) * 1.5, len(class_names) * 1.5)
    )

    # Set title and column headers
    fig.suptitle('CIFAR-10 Classes with Different Augmentations', fontsize=20)
    for i, aug_name in enumerate(augmentations.keys()):
        axes[0, i].set_title(aug_name, fontsize=10)

    # Create visualization grid
    for row, class_idx in enumerate(range(len(class_names))):
        # Set row labels
        axes[row, 0].set_ylabel(class_names[class_idx], rotation=90, va='center')

        # Original image
        img = samples[class_idx]

        # Apply each augmentation
        for col, (_, aug) in enumerate(augmentations.items()):
            ax = axes[row, col]

            try:
                # Apply augmentation or keep original
                result = img if aug is None else aug(img)

                # Convert to numpy for display
                display_img = result.permute(1, 2, 0).numpy()
                display_img = np.clip(display_img, 0, 1)

                # Display image
                ax.imshow(display_img)
            except Exception:
                ax.text(0.5, 0.5, "Error", ha='center', va='center', color='red')

            # Clean up axes
            ax.set_xticks([])
            ax.set_yticks([])

    # Save the visualization
    plt.tight_layout()
    output_file = output_path / "augmentation_examples.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_file}")
    return str(output_file)


if __name__ == "__main__":
    visualize_augmentations()