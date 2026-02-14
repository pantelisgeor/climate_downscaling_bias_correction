"""
Feature-wise Linear Modulation (FiLM) layer for lead-time conditioning.
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    Applies affine transformation to feature maps conditioned on lead time:
        output = gamma * input + beta

    where gamma and beta are learned functions of the lead time embedding.
    """

    def __init__(self, num_features: int, lead_embed_dim: int = 128):
        """
        Initialize FiLM layer.

        Args:
            num_features: Number of feature channels to modulate
            lead_embed_dim: Dimension of lead time embedding
        """
        super().__init__()

        self.num_features = num_features

        # MLPs to generate scale (gamma) and shift (beta) parameters
        self.gamma_fc = nn.Sequential(
            nn.Linear(lead_embed_dim, lead_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(lead_embed_dim, num_features),
        )

        self.beta_fc = nn.Sequential(
            nn.Linear(lead_embed_dim, lead_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(lead_embed_dim, num_features),
        )

        # Initialize to identity transformation
        self.gamma_fc[-1].weight.data.zero_()
        self.gamma_fc[-1].bias.data.fill_(1.0)
        self.beta_fc[-1].weight.data.zero_()
        self.beta_fc[-1].bias.data.zero_()

    def forward(self, x: torch.Tensor, lead_embed: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            x: Input features [batch, channels, height, width]
            lead_embed: Lead time embedding [batch, embed_dim]

        Returns:
            Modulated features [batch, channels, height, width]
        """
        # Generate scale and shift parameters
        gamma = self.gamma_fc(lead_embed)  # [batch, channels]
        beta = self.beta_fc(lead_embed)  # [batch, channels]

        # Reshape for broadcasting: [batch, channels, 1, 1]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Apply affine transformation
        return gamma * x + beta


class LeadTimeEmbedding(nn.Module):
    """Learnable embedding for lead time values."""

    def __init__(self, num_leads: int = 11, embed_dim: int = 128):
        """
        Initialize lead time embedding.

        Args:
            num_leads: Number of distinct lead time values (0-10 = 11)
            embed_dim: Embedding dimension
        """
        super().__init__()

        self.embedding = nn.Embedding(num_leads, embed_dim)

        # Initialize with small random values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, lead_indices: torch.Tensor) -> torch.Tensor:
        """
        Embed lead time indices.

        Args:
            lead_indices: Lead time indices [batch] (values 0-10)

        Returns:
            Embeddings [batch, embed_dim]
        """
        return self.embedding(lead_indices)
