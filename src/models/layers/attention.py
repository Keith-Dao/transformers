"""This module contains the attention layer."""

import math

import torch


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """Computes the scaled dot product attention.

    Args:
        query:
            The query tensor.
        key:
            The key tensor.
        value:
            The value tensor.

    Keyword Args:
        is_causal:
            If true, apply the causal mask of negative infinity to the upper right triangle of the query key
            product. Otherwise, masking is skipped.

    Returns:
        The scaled dot product attention tensor.
    """
    scale_factor = 1 / math.sqrt(key.shape[-1])
    query_key_product = query @ key.transpose(-2, -1) * scale_factor
    if is_causal:
        mask = torch.full((query.shape[-2], key.shape[-2]), -torch.inf, device=query.device).triu(diagonal=1)
        query_key_product += mask

    value_weights = torch.softmax(query_key_product, dim=-1)

    return value_weights @ value
