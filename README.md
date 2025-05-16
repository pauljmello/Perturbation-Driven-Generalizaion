# Perturbation-Driven Generalization

## Research Direction

This repository investigates how different types of data perturbations (augmentations) affect neural network generalization across model architectures and sizes.

## Key Components

- **Systematic Evaluation**: Analysis of the effects of augmentations across CNNs, MLPs, VAEs, and Vision Transformers at three parameter scales (1M, 3M, 9M)
- **Augmentation Strategies**: 
  - Geometric: (translation, rotation, scaling)
  - Pixel-value: (Gaussian noise, salt-pepper noise, blurring)
  - Frequency domain: transformations

## Main Findings

- **Geometric Findings**: Transformations that preserve semantic content while introducing variance (especially translation) consistently outperform other perturbation types
- **Combination Effects**: Augmentation pairs (like translation with scale) deliver superior generalization compared to single augmentations
- **Architecture Impact**: CNNs benefit the most from augmentations, while benefits vary across other architectures (likely from its inductive biases)
- **Manifold Preservation**: Augmentations that keep data on or near the manifold yield better generalization (expanding the manifold with orthogonal noise increases the search space and decreases model accuracy)

## Our Code Features

- A very modular design to allow for easy experimentation and expansion of experiments across model types and augmentation strategies
- We provide a way to validate model parameters for fair comparisons at scale (1M, 3M, and 9M parameter versions)
- Comprehensive analytics pipeline for evaluation and visualization

## Strengths and Limitations

This work is particularly useful as we cover a comprehensive range of architectures and data augmentations. In this way, we provide a significant boost to selecting the best data augmentation techinques for any given approach.
We have also designed the codebase to be easily expanded to more approaches. The main limitations of this work are in our lack of computational resources to run the full codebase as with each added configuration the number of experiments incresed.
Additionally, one direction which we did not initially consider, which would be a core addition to this work, would be to test different data types and across a range of data sizes. Through our experiments we believe the larger models were starved of data given their parameter counts when compared to the small or medium sized model.


## Applications

With this work, we hope to provide insights and understanding regarding the core benefits of augmentations. We illustrate the underlying patterns of what makes a "good" augmentations to be semantic preservation of the input information and transformations which respects the initial data manifold's geometry.
With our findings we hope our research helps others train models with more efficiency aimed at generalizing beyond the training set.