import logging

import torch
import torch.nn as nn

from config.architecture_config import get_architecture_config
from config.model_registry import ModelRegistry
from models.base import BaseModel


logger = logging.getLogger(__name__)

class BaseVAE(BaseModel):
    """
    Base Variational Autoencoder (VAE) architecture.
    """

    def __init__(self, model_size, input_channels, input_size, num_classes):
        """
        Initialize VAE model.
        """
        super().__init__('vae', model_size)

        config = get_architecture_config('vae', model_size)
        self.latent_dim = config['latent_dim']
        self.hidden_dims = config['hidden_dims']

        self.input_channels = input_channels
        self.input_size = input_size

        # Encoder
        encoder_modules = []
        in_channels = input_channels

        for h_dim in self.hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_modules)

        # Calculate output size after encoder
        encoder_output_size = input_size
        for i in range(len(self.hidden_dims)):
            encoder_output_size = (encoder_output_size - 1) // 2 + 1

        self.encoder_output_dim = self.hidden_dims[-1] * encoder_output_size * encoder_output_size

        # Latent space projections
        self.fc_mu = nn.Linear(self.encoder_output_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_dim, self.latent_dim)

        # Decoder Input
        self.decoder_input = nn.Linear(self.latent_dim, self.encoder_output_dim)

        # Decoder
        decoder_modules = []
        hidden_dims_reversed = self.hidden_dims.copy()
        hidden_dims_reversed.reverse()

        self.decoder_reshape = nn.Unflatten(1, (hidden_dims_reversed[0], encoder_output_size, encoder_output_size)) # Ugly fix

        # Decoder CNN layers
        for i in range(len(hidden_dims_reversed) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims_reversed[i], hidden_dims_reversed[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims_reversed[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        # Final layer
        decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims_reversed[-1], input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Tanh()
            )
        )

        self.decoder = nn.Sequential(*decoder_modules)

        # Classification head from latent space
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes)
        )

    def encode(self, x):
        """
        Encode input to latent space.
        """
        # Encoder
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def decode(self, z):
        """
        Decode latent vector to image.
        """
        # Project to decoder
        x = self.decoder_input(z)

        x = self.decoder_reshape(x)

        # Decoder
        x = self.decoder(x)

        return x

    # Praise be to reparamaterization
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for VAE.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through VAE and classifier.
        """
        # Encoder input
        mu, log_var = self.encode(x)

        # Sample latent
        z = self.reparameterize(mu, log_var)

        # Classify
        y_pred = self.classifier(z)

        # Decode
        x_recon = self.decode(z)

        return y_pred, x_recon, mu, log_var


class SmallVAE(BaseVAE):
    """
    Small VAE model with ~1M parameters.
    """
    def __init__(self, input_channels, input_size, num_classes):
        super().__init__('small', input_channels, input_size, num_classes)


class MediumVAE(BaseVAE):
    """
    Medium VAE model with ~3M parameters.
    """
    def __init__(self, input_channels, input_size, num_classes):
        super().__init__('medium', input_channels, input_size, num_classes)


class LargeVAE(BaseVAE):
    """
    Large VAE model with ~9M parameters.
    """
    def __init__(self, input_channels, input_size, num_classes):
        super().__init__('large', input_channels, input_size, num_classes)


# Register models
ModelRegistry.register_model_class('vae', 'small', SmallVAE)
ModelRegistry.register_model_class('vae', 'medium', MediumVAE)
ModelRegistry.register_model_class('vae', 'large', LargeVAE)


def create_vae_model(model_size, input_channels, input_size, num_classes, **kwargs):
    """
    Factory function for creating VAE models.
    """
    if model_size == 'small':
        return SmallVAE(input_channels, input_size, num_classes)
    elif model_size == 'medium':
        return MediumVAE(input_channels, input_size, num_classes)
    elif model_size == 'large':
        return LargeVAE(input_channels, input_size, num_classes)
    else:
        raise ValueError(f"Invalid model size: {model_size}")


# Register functions
ModelRegistry.register_model_factory('vae', create_vae_model)