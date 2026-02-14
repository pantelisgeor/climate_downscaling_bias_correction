"""
Independent decoders for each target variable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class Decoder(nn.Module):
    """
    Independent decoder for a single target variable.

    Maps encoded features to target variable prediction with upsampling.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [512, 256, 128, 64],
        output_size: tuple = (35, 77),  # Target spatial size
        output_activation: str = "none",  # 'none', 'sigmoid', 'relu'
    ):
        """
        Initialize decoder.

        Args:
            input_dim: Input feature dimension from encoder
            hidden_dims: List of hidden layer dimensions
            output_size: Target output spatial size (H, W)
            output_activation: Output activation function
        """
        super().__init__()

        self.output_size = output_size

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_dim = hidden_dim

        # Final layer to single channel
        layers.append(nn.Conv2d(in_dim, 1, 1))

        self.decoder = nn.Sequential(*layers)

        # Output activation
        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "relu":
            self.output_activation = nn.ReLU()
        else:
            self.output_activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Encoded features [batch, input_dim, H_enc, W_enc]

        Returns:
            Prediction [batch, 1, H_target, W_target]
        """
        # Decode
        out = self.decoder(x)  # [B, 1, H_enc, W_enc]

        # Upsample to target size if needed
        if out.shape[2:] != self.output_size:
            out = F.interpolate(
                out, size=self.output_size, mode="bilinear", align_corners=False
            )

        # Apply output activation
        out = self.output_activation(out)

        return out


class MultiDecoder(nn.Module):
    """
    Multiple independent decoders for all target variables.
    """

    def __init__(
        self,
        input_dim: int = 512,
        target_vars: List[str] = ["tasERA", "tasmaxERA", "tpERA", "rhERA"],
        hidden_dims: List[int] = [512, 256, 128, 64],
        output_size: tuple = (35, 77),  # Add target size
    ):
        """
        Initialize multi-decoder.

        Args:
            input_dim: Input feature dimension from encoder
            target_vars: List of target variable names
            hidden_dims: Hidden dimensions for each decoder
            output_size: Target output spatial size (H, W)
        """
        super().__init__()

        self.target_vars = target_vars
        self.num_targets = len(target_vars)

        # Create independent decoder for each target
        self.decoders = nn.ModuleDict()

        for var in target_vars:
            # Use ReLU for precipitation (non-negative)
            if var == "tpERA":
                activation = "relu"
            elif var == "rhERA":
                activation = "none"  # Will apply bounds in loss
            else:
                activation = "none"

            self.decoders[var] = Decoder(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_size=output_size,
                output_activation=activation,
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all decoders.

        Args:
            x: Encoded features [batch, input_dim, H_enc, W_enc]

        Returns:
            Dictionary mapping variable names to predictions [batch, 1, H_target, W_target]
        """
        outputs = {}
        for var in self.target_vars:
            outputs[var] = self.decoders[var](x)

        return outputs
