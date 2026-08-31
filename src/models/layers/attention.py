"""This module contains the attention layer."""

import math

import torch


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Computes the scaled dot product attention.

    Args:
        query:
            The query tensor.
        key:
            The key tensor.
        value:
            The value tensor.

    Returns:
        The scaled dot product attention tensor.
    """
    scale_factor = 1 / math.sqrt(key.shape[-1])
    query_key_product = query @ key.transpose(-2, -1)
    value_weights = torch.softmax(query_key_product * scale_factor, dim=-1)

    return value_weights @ value
