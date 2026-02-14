"""
Complete 1EMD architecture for climate bias correction.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .encoder import CNNEncoder, VisionTransformerEncoder
from .decoder import MultiDecoder
from .film_layer import FiLMLayer, LeadTimeEmbedding


class ClimateNet(nn.Module):
    """Single-Encoder Multi-Decoder (1EMD) architecture."""

    def __init__(
        self,
        static_channels: int = 3,
        dynamic_channels: int = 16,
        image_size: Tuple[int, int] = (35, 77),
        encoder_type: str = "cnn",
        encoder_dim: int = 512,
        encoder_blocks: int = 5,
        vit_patch_size: int = 7,
        vit_num_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.1,
        vit_attention_dropout: float = 0.1,
        decoder_hidden_dims: List[int] = [512, 256, 128, 64],
        target_vars: List[str] = ["tasERA", "tasmaxERA", "tpERA", "rhERA"],
        use_film: bool = True,
        num_leads: int = 11,
        lead_embed_dim: int = 128,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.use_film = use_film
        self.encoder_dim = encoder_dim

        total_channels = static_channels + dynamic_channels

        # Create encoder
        if encoder_type == "vit":
            self.encoder = VisionTransformerEncoder(
                in_channels=total_channels,
                embed_dim=encoder_dim,
                image_size=image_size,
                patch_size=vit_patch_size,
                num_blocks=encoder_blocks,
                num_heads=vit_num_heads,
                mlp_ratio=vit_mlp_ratio,
                dropout=vit_dropout,
                attention_dropout=vit_attention_dropout,
            )
            # DON'T create FiLM here - wait until after weight init

        elif encoder_type == "cnn":
            self.encoder = CNNEncoder(
                static_channels=static_channels,
                dynamic_channels=dynamic_channels,
                base_channels=64,
                num_blocks=encoder_blocks,
                output_dim=encoder_dim,
                use_film=use_film,
                num_leads=num_leads,
                lead_embed_dim=lead_embed_dim,
            )

        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Create decoder
        self.decoder = MultiDecoder(
            input_dim=encoder_dim,
            target_vars=target_vars,
            hidden_dims=decoder_hidden_dims,
            output_size=image_size,
        )

        # Initialize weights for encoder and decoder
        self.apply(self._init_weights)

        # Create FiLM AFTER weight initialization (so it doesn't get overwritten)
        if encoder_type == "vit" and use_film:
            print("[ClimateNet] Creating FiLM layer after weight init...")
            self.lead_embedding = LeadTimeEmbedding(
                num_leads=num_leads, embed_dim=lead_embed_dim
            )
            self.film_layer = FiLMLayer(
                num_features=encoder_dim, cond_dim=lead_embed_dim
            )

    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self, static: torch.Tensor, dynamic: torch.Tensor, lead_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x = torch.cat([static, dynamic], dim=1)

        if self.encoder_type == "cnn":
            encoded = self.encoder(static, dynamic, lead_indices)

        elif self.encoder_type == "vit":
            encoded = self.encoder(x)

            if self.use_film:
                lead_embed = self.lead_embedding(lead_indices)
                encoded = self.film_layer(encoded, lead_embed)

        outputs = self.decoder(encoded)
        return outputs

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        film_params = 0
        if self.encoder_type == "vit" and self.use_film:
            film_params += sum(p.numel() for p in self.lead_embedding.parameters())
            film_params += sum(p.numel() for p in self.film_layer.parameters())

        total_params = encoder_params + decoder_params + film_params

        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "film": film_params,
            "total": total_params,
            "total_millions": total_params / 1e6,
        }
