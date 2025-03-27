import logging
import math

import torch
import torch.nn as nn

from config.architecture_config import get_architecture_config
from config.model_registry import ModelRegistry
from models.base import BaseModel

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and MLP, supporting KV caching.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=0.0, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

        # Cache for keys and values
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, use_cache=False):
        x_norm = self.norm1(x)

        # Handle KV caching
        if use_cache and self.k_cache is not None and self.v_cache is not None:
            # Only process the current input tokens with cached history
            attn_out, _ = self.attn(query=x_norm, key=torch.cat([self.k_cache, x_norm], dim=1), value=torch.cat([self.v_cache, x_norm], dim=1))

            # Update cache with current input
            self.k_cache = torch.cat([self.k_cache, x_norm], dim=1)
            self.v_cache = torch.cat([self.v_cache, x_norm], dim=1)
        else:
            # Standard attention computation
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)

            # Initialize cache if using caching
            if use_cache:
                self.k_cache = x_norm
                self.v_cache = x_norm

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class BaseTransformer(BaseModel):
    """
    Base class for transformer models with KV caching.
    """
    def __init__(self, model_size, input_channels, input_size, num_classes):
        """
        Initialize transformer model.
        """
        super().__init__('transformer', model_size)

        # Get architecture config
        config = get_architecture_config('transformer', model_size)
        self.embedding_dim = config['embed_dim']
        self.num_heads = config['heads']
        self.num_layers = config['depth']
        self.patch_size = config.get('patch_size', 4)

        self.input_channels = input_channels
        self.input_size = input_size

        if self.patch_size > 1:
            # Patching
            self.embedding = nn.Conv2d(input_channels, self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size)
            # calculate sequence length with patching
            self.seq_length = (input_size // self.patch_size) ** 2
        else:
            # no patching
            self.embedding = nn.Linear(input_channels, self.embedding_dim)
            # Calculate sequence length (each pixel is a token)
            self.seq_length = input_size * input_size

        self.positional_encoding = PositionalEncoding(self.embedding_dim)

        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([TransformerBlock(dim=self.embedding_dim, n_heads=self.num_heads, mlp_ratio=4.0) for _ in range(self.num_layers)])

        self.norm = nn.LayerNorm(self.embedding_dim)

        # Classifier Heads
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, num_classes)
        )

    def reset_kv_cache(self):
        """
        Reset KV cache for all transformer blocks.
        """
        for module in self.modules():
            if hasattr(module, 'k_cache'):
                module.k_cache = None
            if hasattr(module, 'v_cache'):
                module.v_cache = None

    def forward(self, x, use_cache=False):
        batch_size = x.size(0)

        # Reset cache at the start of a new sequence
        if not use_cache:
            self.reset_kv_cache()

        if self.patch_size > 1:
            # Patched embedding
            x = self.embedding(x)  # (B, C, H, W) -> (B, embed_dim, H', W')
            x = x.flatten(2).transpose(1, 2)  # (B, embed_dim, H'*W') -> (B, H'*W', embed_dim)
        else:
            # pixel-level
            x = x.view(batch_size, self.input_channels, -1).permute(0, 2, 1)
            x = self.embedding(x)

        x = self.positional_encoding(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, use_cache=use_cache)

        x = self.norm(x) # norm
        x = x.mean(dim=1) # pool avg
        x = self.classifier(x) # classifier

        return x


class SmallTransformer(BaseTransformer):
    """
    Small transformer model with ~1M parameters.
    """
    def __init__(self, input_channels, input_size, num_classes):
        super().__init__('small', input_channels, input_size, num_classes)


class MediumTransformer(BaseTransformer):
    """
    Medium transformer model with ~3M parameters.
    """
    def __init__(self, input_channels, input_size, num_classes):
        super().__init__('medium', input_channels, input_size, num_classes)


class LargeTransformer(BaseTransformer):
    """
    Large transformer model with ~9M parameters.
    """
    def __init__(self, input_channels, input_size, num_classes):
        super().__init__('large', input_channels, input_size, num_classes)


# Register models
ModelRegistry.register_model_class('transformer', 'small', SmallTransformer)
ModelRegistry.register_model_class('transformer', 'medium', MediumTransformer)
ModelRegistry.register_model_class('transformer', 'large', LargeTransformer)


def create_transformer_model(model_size, input_channels, input_size, num_classes, **kwargs):
    """
    Factory function for creating transformer models.
    """
    if model_size == 'small':
        return SmallTransformer(input_channels, input_size, num_classes)
    elif model_size == 'medium':
        return MediumTransformer(input_channels, input_size, num_classes)
    elif model_size == 'large':
        return LargeTransformer(input_channels, input_size, num_classes)
    else:
        raise ValueError(f"Invalid model size: {model_size}")


# Register functions
ModelRegistry.register_model_factory('transformer', create_transformer_model)
