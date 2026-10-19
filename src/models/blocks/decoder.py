"""This module contains the decoder block."""

import torch

from models.layers.attention import MultiHeadAttention


class DecoderBlock(torch.nn.Module):
    """A single decoder block as per the "Attention is all you need" paper.

    Attributes:
        masked_self_attention:
            The masked self-attention layer.
        masked_self_attention_norm:
            The layer normalisation following the masked self-attention layer.
        cross_attention:
            The cross attention layer.
        cross_attention_norm:
            The layer normalisation following the cross attention layer.
        feed_forward_network:
            The feed-forward network.
        feed_forward_network_norm:
            The layer normalisation following the feed-forward network.
    """

    masked_self_attention: MultiHeadAttention
    masked_self_attention_norm: torch.nn.LayerNorm
    cross_attention: MultiHeadAttention
    cross_attention_norm: torch.nn.LayerNorm
    feed_forward_network: torch.nn.Sequential
    feed_forward_network_norm: torch.nn.LayerNorm

    def __init__(self, in_features: int, attention_heads: int, expansion_dimensions: int) -> None:
        """Initialises a decoder block.

        Args:
            in_features:
                The input dimension.
            attention_heads:
                The number of attention heads.
            expansion_dimensions:
                The number of dimensions to expand to in the feed-forward network.
        """
        super().__init__()

        self.masked_self_attention = MultiHeadAttention(in_features=in_features, num_heads=attention_heads)
        self.masked_self_attention_norm = torch.nn.LayerNorm(in_features)
        self.cross_attention = MultiHeadAttention(in_features=in_features, num_heads=attention_heads)
        self.cross_attention_norm = torch.nn.LayerNorm(in_features)
        self.feed_forward_network = torch.nn.Sequential(
            torch.nn.Linear(in_features, expansion_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(expansion_dimensions, in_features),
        )
        self.feed_forward_network_norm = torch.nn.LayerNorm(in_features)

    def reset_parameters(self) -> None:
        """Resets parameters."""
        layers = [
            self.masked_self_attention,
            self.masked_self_attention_norm,
            self.cross_attention,
            self.cross_attention_norm,
            self.feed_forward_network_norm,
        ]
        for layer in layers:
            layer.reset_parameters()

        for layer in self.feed_forward_network:
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass.

        Args:
            x:
                The input tensor.
            encoder_output:
                The encoder output.

        Returns:
            The encoded tensor.
        """
        self_attention_output = self.masked_self_attention(x, x, x, is_causal=True)
        normalised_self_attention_output = self.masked_self_attention_norm(x + self_attention_output)

        cross_attention_output = self.cross_attention(normalised_self_attention_output, encoder_output, encoder_output)
        cross_attention_output = self.cross_attention_norm(normalised_self_attention_output + cross_attention_output)

        feed_forward_output = self.feed_forward_network(cross_attention_output)

        return self.feed_forward_network_norm(cross_attention_output + feed_forward_output)
