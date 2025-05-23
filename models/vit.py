import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.architecture_config import get_architecture_config
from config.model_registry import ModelRegistry
from models.base import BaseModel

logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W) -> (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, E, N)
        return x


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

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BaseViT(BaseModel):
    """
    Base Vision Transformer model with KV caching support.
    """
    def __init__(self, model_size, input_channels, input_size, num_classes):
        super().__init__('vit', model_size)
        config = get_architecture_config('vit', model_size)
        self.embed_dim = config['embed_dim']
        self.num_heads = config['heads']
        self.num_layers = config['depth']
        self.patch_size = config.get('patch_size', 4)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Position and Patch Embedding
        self.patch_embed = PatchEmbed(img_size=input_size, patch_size=self.patch_size, in_channels=input_channels, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(dim=self.embed_dim, n_heads=self.num_heads, mlp_ratio=4.0) for _ in range(self.num_layers)])

        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the model with optional KV caching.
        """
        batch_size = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification from cls tokens
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)

        return x


class SmallViT(BaseViT):
    """
    Small Vision Transformer model.
    """
    def __init__(self, input_channels, input_size, num_classes):
        super().__init__(model_size='small', input_channels=input_channels, input_size=input_size, num_classes=num_classes)


class MediumViT(BaseViT):
    """
    Medium Vision Transformer model.
    """
    def __init__(self, input_channels, input_size, num_classes):
        super().__init__(model_size='medium', input_channels=input_channels, input_size=input_size, num_classes=num_classes)


class LargeViT(BaseViT):
    """
    Large Vision Transformer model.
    """
    def __init__(self, input_channels, input_size, num_classes):
        super().__init__(model_size='large', input_channels=input_channels, input_size=input_size, num_classes=num_classes)


# Register models
ModelRegistry.register_model_class('vit', 'small', SmallViT)
ModelRegistry.register_model_class('vit', 'medium', MediumViT)
ModelRegistry.register_model_class('vit', 'large', LargeViT)


def create_vit_model(model_size, input_channels, input_size, num_classes, **kwargs):
    """
    Factory function for creating ViT models.
    """
    if model_size == 'small':
        return SmallViT(input_channels, input_size, num_classes)
    elif model_size == 'medium':
        return MediumViT(input_channels, input_size, num_classes)
    elif model_size == 'large':
        return LargeViT(input_channels, input_size, num_classes)
    else:
        raise ValueError(f"Invalid model size: {model_size}")


# Register functions
ModelRegistry.register_model_factory('vit', create_vit_model)
