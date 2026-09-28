"""This module contains the encoder block."""

import torch

from models.layers.attention import MultiHeadAttention


class EncoderBlock(torch.nn.Module):
    """A single encoder block as per the "Attention is all you need" paper.

    Attributes:
        multihead_attention:
            The multi-head attention layer.
        multihead_attention_norm:
            The layer normalisation following the multi-head attention layer.
        feed_forward_network:
            The feed-forward network.
        feed_forward_network_norm:
            The layer normalisation following the feed-forward network.
    """

    multihead_attention: MultiHeadAttention
    multihead_attention_norm: torch.nn.LayerNorm
    feed_forward_network: torch.nn.Sequential
    feed_forward_network_norm: torch.nn.LayerNorm

    def __init__(self, in_features: int, attention_heads: int, expansion_dimensions: int) -> None:
        """Initialises an encoder block.

        Args:
            in_features:
                The input dimension.
            attention_heads:
                The number of attention heads.
            expansion_dimensions:
                The number of dimensions to expand to in the feed-forward network.
        """
        super().__init__()

        self.multihead_attention = MultiHeadAttention(in_features=in_features, num_heads=attention_heads)
        self.multihead_attention_norm = torch.nn.LayerNorm(in_features)
        self.feed_forward_network = torch.nn.Sequential(
            torch.nn.Linear(in_features, expansion_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(expansion_dimensions, in_features),
        )
        self.feed_forward_network_norm = torch.nn.LayerNorm(in_features)

    def reset_parameters(self) -> None:
        """Resets parameters."""
        layers = [
            self.multihead_attention,
            self.multihead_attention_norm,
            self.feed_forward_network_norm,
        ]
        for layer in layers:
            layer.reset_parameters()

        for layer in self.feed_forward_network:
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward passs.

        Args:
            x:
                The input tensor.

        Returns:
            The encoded tensor.
        """
        attention_output = self.multihead_attention(x, x, x)
        normalised_attention_output = self.multihead_attention_norm(x + attention_output)

        feed_forward_output = self.feed_forward_network(normalised_attention_output)

        return self.feed_forward_network_norm(normalised_attention_output + feed_forward_output)
