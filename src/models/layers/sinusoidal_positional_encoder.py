"""This module contains  the positional encoder."""

import math

import torch
import torch.nn


class SinusoidalPositionalEncoder(torch.nn.Module):
    """The sinusoidal positinal encoder as described in the attention is all your need paper.

    Attributes:
        encodings:
            The positional encodings.
        dropout:
            The drop out layer.
    """

    encodings: torch.Tensor
    dropout: torch.nn.Dropout

    def __init__(self, max_sequence_length: int, in_features: int, dropout_rate: float) -> None:
        """Initialises the positional encoder.

        Args:
            max_sequence_length:
                The maximum sequence length.
            in_features:
                The number of input features.
            dropout_rate:
                The dropout rate.
        """
        super().__init__()

        encodings = torch.zeros(max_sequence_length, in_features)
        positions = torch.arange(0, max_sequence_length).unsqueeze(1)
        angular_frequencies = torch.exp(torch.arange(0, in_features, 2) * -math.log(10_000) / in_features)

        encodings[:, ::2] = torch.sin(positions * angular_frequencies)
        encodings[:, 1::2] = torch.cos(positions * angular_frequencies)

        self.register_buffer("encodings", encodings)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the positional encoding.

        Args:
            x: The 2d or 3d input tensor.

        Returns:
            The input tensor with positional encoding applied.
        """
        sequence_length = x.shape[-2]
        x = x + self.encodings[:sequence_length]
        return self.dropout(x)
