"""
Shared encoder architectures for climate data.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from .film_layer import FiLMLayer, LeadTimeEmbedding
import math


class ResidualBlock(nn.Module):
    """Residual block with optional FiLM conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_film: bool = True,
        lead_embed_dim: int = 128,
    ):
        super().__init__()

        self.use_film = use_film

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

        # FiLM layer
        if use_film:
            self.film = FiLMLayer(out_channels, lead_embed_dim)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, lead_embed: torch.Tensor = None) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply FiLM conditioning if enabled
        if self.use_film and lead_embed is not None:
            out = self.film(out, lead_embed)

        out += identity
        out = self.relu(out)

        return out


class CNNEncoder(nn.Module):
    """
    CNN-based shared encoder.

    Processes static and dynamic features separately, then fuses them.
    Incorporates FiLM layers for lead-time conditioning.
    """

    def __init__(
        self,
        static_channels: int = 3,  # dem, rho, phi
        dynamic_channels: int = 15,  # pr, tas, tasmax, hurs, sin_time, cos_time, cci_agg (flattened)
        base_channels: int = 64,
        num_blocks: int = 5,
        output_dim: int = 512,
        use_film: bool = True,
        num_leads: int = 11,
        lead_embed_dim: int = 128,
    ):
        """
        Initialize CNN encoder.

        Args:
            static_channels: Number of static input channels
            dynamic_channels: Number of dynamic input channels
            base_channels: Base number of feature channels
            num_blocks: Number of residual blocks
            output_dim: Output feature dimension
            use_film: Whether to use FiLM conditioning
            num_leads: Number of lead time values
            lead_embed_dim: Lead time embedding dimension
        """
        super().__init__()

        self.use_film = use_film

        # Lead time embedding
        if use_film:
            self.lead_embedding = LeadTimeEmbedding(num_leads, lead_embed_dim)

        # Static feature encoder
        self.static_encoder = nn.Sequential(
            nn.Conv2d(static_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Dynamic feature encoder (with FiLM)
        dynamic_layers = [
            nn.Conv2d(dynamic_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Residual blocks with increasing channels
        channel_progression = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 4,
            output_dim,
        ]

        self.dynamic_blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_ch = channel_progression[i]
            out_ch = (
                channel_progression[i + 1]
                if i < num_blocks - 1
                else channel_progression[i]
            )

            self.dynamic_blocks.append(
                ResidualBlock(in_ch, out_ch, use_film, lead_embed_dim)
            )

        self.dynamic_init = nn.Sequential(*dynamic_layers)

        # Fusion layer
        fusion_in_channels = base_channels * 2 + output_dim  # static + dynamic
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, output_dim, 1),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(
        self, static: torch.Tensor, dynamic: torch.Tensor, lead_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            static: Static features [batch, static_channels, H, W]
            dynamic: Dynamic features [batch, dynamic_channels, H, W]
            lead_indices: Lead time indices [batch] (values 0-10)

        Returns:
            Encoded features [batch, output_dim, H, W]
        """
        # Embed lead time
        lead_embed = None
        if self.use_film:
            lead_embed = self.lead_embedding(lead_indices)

        # Encode static features
        static_features = self.static_encoder(static)

        # Encode dynamic features with FiLM conditioning
        dynamic_features = self.dynamic_init(dynamic)

        for block in self.dynamic_blocks:
            dynamic_features = block(dynamic_features, lead_embed)

        # Fuse static and dynamic features
        fused = torch.cat([static_features, dynamic_features], dim=1)
        output = self.fusion(fused)

        return output


class PatchEmbedding(nn.Module):
    """
    Convert image into sequence of patches with learned embeddings.
    """

    def __init__(
        self,
        in_channels: int = 19,
        embed_dim: int = 512,
        image_size: Tuple[int, int] = (35, 77),
        patch_size: int = 7,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size

        # Calculate number of patches
        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patch embedding via convolution
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Layer norm AFTER projection (critical for training stability)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            patches: [B, N, D] where N = num_patches
        """
        B, C, H, W = x.shape

        # Project to patches: [B, D, H', W']
        x = self.projection(x)

        # Reshape to sequence: [B, D, H', W'] -> [B, D, N] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)

        # Apply layer norm
        x = self.norm(x)

        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with Pre-LN (LayerNorm before attention/MLP).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        # Pre-LN: LayerNorm before attention
        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,  # Important: use batch_first=True
        )

        # Pre-LN: LayerNorm before MLP
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),  # GELU is standard for ViT
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            x: [B, N, D]
        """
        # Pre-LN: normalize before attention
        x_norm = self.norm1(x)

        # Self-attention with residual
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-LN: normalize before MLP
        x_norm = self.norm2(x)

        # MLP with residual
        x = x + self.mlp(x_norm)

        return x


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer encoder for climate data.
    """

    def __init__(
        self,
        in_channels: int = 19,
        embed_dim: int = 512,
        image_size: Tuple[int, int] = (35, 77),
        patch_size: int = 7,
        num_blocks: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
        )

        num_patches = self.patch_embed.num_patches

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings (learnable) - includes CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Positional dropout
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # Final layer norm (important for stability)
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize patch embedding projection
        w = self.patch_embed.projection.weight.data
        nn.init.trunc_normal_(w, std=0.02)

        # Initialize transformer blocks
        for block in self.blocks:
            # Initialize attention weights
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            nn.init.zeros_(block.attn.in_proj_bias)
            nn.init.zeros_(block.attn.out_proj.bias)

            # Initialize MLP weights
            for module in block.mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            features: [B, D, H', W'] - spatial feature map for decoder
        """
        B = x.shape[0]

        # Patch embedding: [B, C, H, W] -> [B, N, D]
        x = self.patch_embed(x)

        # Add CLS token: [B, N, D] -> [B, N+1, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Remove CLS token and reshape to spatial format
        x = x[:, 1:, :]  # Remove CLS token: [B, N+1, D] -> [B, N, D]

        # Reshape to 2D feature map: [B, N, D] -> [B, D, H', W']
        H = self.patch_embed.num_patches_h
        W = self.patch_embed.num_patches_w
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        return x
