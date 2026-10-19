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


class MultiHeadAttention(torch.nn.Module):
    """The multihead attention layer.

    Attributes:
        num_heads:
            The number of heads.
        embed_dimension:
            The embedding dimension.
        query_projection:
            The query projection.
        key_projection:
            The key projection.
        value_projection:
            The value projection.
        output_projection:
            The output projection.
    """

    class DimensionError(ValueError):
        """Error for incorrect input dimensions values."""

    class Dimensions:
        """Dimension constants."""

        BATCH = 0
        SEQUENCE = 1
        FEATURE = 2

    embed_dimension: int
    query_projection: torch.nn.Linear
    key_projection: torch.nn.Linear
    value_projection: torch.nn.Linear
    output_projection: torch.nn.Linear

    def __init__(self, in_features: int, num_heads: int) -> None:
        """Initialises the multihead attention layer.

        Args:
            in_features:
                The input dimension.
            num_heads:
                The number of attention heads.
        """
        super().__init__()

        if in_features % num_heads != 0:
            error_message = (
                f"Input feature dimension ({in_features}) is not divisible by the number of heads ({num_heads})."
            )
            raise ValueError(error_message)

        self.num_heads = num_heads
        self.embed_dimension = in_features // num_heads

        self.query_projection = torch.nn.Linear(in_features, in_features)
        self.key_projection = torch.nn.Linear(in_features, in_features)
        self.value_projection = torch.nn.Linear(in_features, in_features)
        self.output_projection = torch.nn.Linear(in_features, in_features)

    def reset_parameters(self) -> None:
        """Resets parameters."""
        layers = [self.query_projection, self.key_projection, self.value_projection, self.output_projection]
        for layer in layers:
            layer.reset_parameters()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Performs the forward pass.

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
            The multihead attention output.

        Raises:
            DimensionError:
                If any of the query, key or value are not 2- or 3-dimensional.
        """

        def normalise_and_validate_tensor(tensor: torch.Tensor, tensor_name: str) -> torch.Tensor:
            """Ensures the tensor is 3-dimensional and validates its dimensions.

            If the input tensor is 2-dimensional, a batch dimension of 1 is added.

            Args:
                tensor:
                    The input tensor to normalise and validate.
                tensor_name:
                    The display name of the tensor for error messages.

            Returns:
                The 3-dimensional tensor.

            Raises:
                DimensionError:
                    If the tensor is not 2- or 3-dimensional.
            """
            expected_dimensions = 3
            if tensor.dim() == expected_dimensions - 1:
                tensor = tensor.unsqueeze(0)

            if tensor.dim() != expected_dimensions:
                error_message = f"{tensor_name} shape must be two or three dimensional, got {tensor.dim()}."
                raise self.DimensionError(error_message)

            return tensor

        query = normalise_and_validate_tensor(query, "Query")
        key = normalise_and_validate_tensor(key, "Key")
        value = normalise_and_validate_tensor(value, "Value")

        for dimension_name, dimension in [("batch", self.Dimensions.BATCH), ("feature", self.Dimensions.FEATURE)]:
            if not query.shape[dimension] == key.shape[dimension] == value.shape[dimension]:
                error_message = (
                    f"The {dimension_name} sizes for query ({query.shape[dimension]}), key ({key.shape[dimension]}) "
                    f"and/or value ({value.shape[dimension]}) do not match."
                )
                raise self.DimensionError(error_message)

        if key.shape[self.Dimensions.SEQUENCE] != value.shape[self.Dimensions.SEQUENCE]:
            error_message = (
                f"The sequence length for key ({key.shape[self.Dimensions.SEQUENCE]}) and value "
                f"({value.shape[self.Dimensions.SEQUENCE]}) do not match."
            )
            raise self.DimensionError(error_message)

        batch_size, query_sequence_length, in_features = query.shape
        _, kv_sequence_length, _ = key.shape

        query_projection_shape = (batch_size, query_sequence_length, self.num_heads, self.embed_dimension)
        kv_projection_shape = (batch_size, kv_sequence_length, self.num_heads, self.embed_dimension)

        query_embedding = self.query_projection(query).view(*query_projection_shape).transpose(1, 2)
        key_embedding = self.key_projection(key).view(*kv_projection_shape).transpose(1, 2)
        value_embedding = self.value_projection(value).view(*kv_projection_shape).transpose(1, 2)

        attention_output = scaled_dot_product_attention(
            query=query_embedding,
            key=key_embedding,
            value=value_embedding,
            is_causal=is_causal,
        )

        attention_output = (
            attention_output.transpose(1, 2).contiguous().view(batch_size, query_sequence_length, in_features)
        )
        return self.output_projection(attention_output)
