import logging
import math

import torch
import torch.nn as nn

import torch.nn.functional as F
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


class Attention(nn.Module):
    def __init__(self, dim, n_heads, qkv_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # Combine Q, K, V projections into one for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)  # output projection

    def forward(self, x):
        B, N, C = x.shape  # batch, seq_len, embed_dim
        # Compute Q, K, V in a single linear projection
        qkv = self.qkv(x)  # shape (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # shape (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each shape (B, heads, N, head_dim)

        # Scaled dot-product attention (uses fused FlashAttention if available)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        # Merge heads back together
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(attn_out)


class Block(nn.Module):
    """
    Transformer block with attention and MLP.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, n_heads=n_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x, use_cache=False):
        x = x + self.attn(self.norm1(x))
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
        self.transformer_blocks = nn.ModuleList([Attention(dim=self.embedding_dim, n_heads=self.num_heads, qkv_bias=True) for _ in range(self.num_layers)])

        self.norm = nn.LayerNorm(self.embedding_dim)

        # Classifier Heads
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

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
            x = block(x)

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
