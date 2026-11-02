"""This module contains tests for the attention layer."""

import pytest
import torch
import torch.testing

from models.layers.attention import MultiHeadAttention, scaled_dot_product_attention
from tests import constants


@pytest.mark.parametrize("device", constants.DEVICES)
@pytest.mark.parametrize("is_causal", [pytest.param(True, id="Causal"), pytest.param(False, id="Non-causal")])
@pytest.mark.usefixtures("seed")
class TestScaledDotProductAttention:
    """Tests scaled_dot_product_attention."""

    @pytest.mark.parametrize(
        ("batch_size", "num_heads", "seq_q", "seq_k", "d_k", "d_v"),
        [
            pytest.param(1, 1, 4, 8, 6, 16, id="2d"),
            pytest.param(1, 2, 4, 6, 8, 16, id="3d"),
            pytest.param(2, 4, 8, 10, 16, 32, id="4d"),
        ],
    )
    def test_scaled_dot_product_attention_output(
        self,
        device: str,
        is_causal: bool,
        batch_size: int,
        num_heads: int,
        seq_q: int,
        seq_k: int,
        d_k: int,
        d_v: int,
    ) -> None:
        """Test that output matches PyTorch built-in scaled_dot_product_attention."""
        query = torch.randn(batch_size, num_heads, seq_q, d_k, device=device).squeeze()
        key = torch.randn(batch_size, num_heads, seq_k, d_k, device=device).squeeze()
        value = torch.randn(batch_size, num_heads, seq_k, d_v, device=device).squeeze()

        result = scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        torch.testing.assert_close(result, expected), "Implementation does not match pytorch's."

    def test_scaled_dot_product_attention_uniform_weights(self, device: str, is_causal: bool) -> None:
        """Test that uniform attention scores result in average of value vectors."""
        seq_q, seq_k, d_k, d_v = 3, 5, 8, 4
        query = torch.zeros(seq_q, d_k, device=device)
        key = torch.zeros(seq_k, d_k, device=device)
        value = torch.randn(seq_k, d_v, device=device)

        output = scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        if is_causal:
            weights = torch.arange(1, seq_q + 1, device=value.device).unsqueeze(-1)
            expected = torch.cumsum(value, dim=0)[:seq_q] / weights
        else:
            expected = value.mean(dim=0).unsqueeze(0).expand(seq_q, -1)

        torch.testing.assert_close(output, expected)

    def test_scaled_dot_product_attention_constant_value(self, device: str, is_causal: bool) -> None:
        """Test that constant value matrix returns constant rows equal to that value vector."""
        query = torch.randn(4, 8, device=device)
        key = torch.randn(6, 8, device=device)
        constant_row = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        value = constant_row.unsqueeze(0).expand(6, 4)

        output = scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        expected = constant_row.unsqueeze(0).expand(4, 4)

        torch.testing.assert_close(output, expected)

    def test_scaled_dot_product_attention_sharp_attention(self, device: str, is_causal: bool) -> None:
        """Test that strong match with a single key selects its corresponding value vector."""
        query = torch.tensor([[100.0, 0.0, 0.0, 0.0]], device=device)
        key = torch.tensor(
            [
                [100.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            device=device,
        )
        value = torch.tensor(
            [
                [1.0, 2.0],
                [10.0, 20.0],
            ],
            device=device,
        )

        output = scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        expected = torch.tensor([[1.0, 2.0]], device=device)

        torch.testing.assert_close(output, expected)

    def test_scaled_dot_product_attention_backward_pass(self, device: str, is_causal: bool) -> None:
        """Test that gradients are correctly backpropagated to query, key, and value."""
        query = torch.randn(2, 4, 8, device=device, requires_grad=True)
        key = torch.randn(2, 6, 8, device=device, requires_grad=True)
        value = torch.randn(2, 6, 16, device=device, requires_grad=True)

        output = scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        assert not torch.isnan(query.grad).any()
        assert not torch.isnan(key.grad).any()
        assert not torch.isnan(value.grad).any()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_scaled_dot_product_attention_dtypes(self, device: str, is_causal: bool, dtype: torch.dtype) -> None:
        """Test that scaled dot product attention works with various precision."""
        query = torch.randn(2, 4, 8, dtype=dtype, device=device)
        key = torch.randn(2, 6, 8, dtype=dtype, device=device)
        value = torch.randn(2, 6, 16, dtype=dtype, device=device)

        output = scaled_dot_product_attention(query, key, value, is_causal=is_causal)

        assert output.dtype == dtype
        assert output.shape == (2, 4, 16)


@pytest.mark.parametrize("device", constants.DEVICES)
@pytest.mark.usefixtures("seed")
class TestCausalDotProductAttention:
    """Specific tests for causal masking behavior in scaled_dot_product_attention."""

    def test_scaled_dot_product_attention_causal_prevents_future_attention(self, device: str) -> None:
        """Test that causal masking prevents attending to future tokens even with strong match."""
        query = torch.tensor([[100.0, 0.0, 0.0, 0.0]], device=device)
        key = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0, 0.0],
            ],
            device=device,
        )
        value = torch.tensor(
            [
                [1.0, 2.0],
                [10.0, 20.0],
            ],
            device=device,
        )

        output_causal = scaled_dot_product_attention(query, key, value, is_causal=True)
        expected_causal = torch.tensor([[1.0, 2.0]], device=device)
        torch.testing.assert_close(output_causal, expected_causal)

        output_non_causal = scaled_dot_product_attention(query, key, value, is_causal=False)
        expected_non_causal = torch.tensor([[10.0, 20.0]], device=device)
        torch.testing.assert_close(output_non_causal, expected_non_causal)

    def test_scaled_dot_product_attention_causal_future_independence(self, device: str) -> None:
        """Test that modifying future key/value tokens does not affect present/past query outputs."""
        seq_q, seq_k, d_k, d_v = 4, 4, 8, 16
        future_index = 2
        query = torch.randn(seq_q, d_k, device=device)
        original_key = torch.randn(seq_k, d_k, device=device)
        original_value = torch.randn(seq_k, d_v, device=device)

        altered_key = original_key.clone()
        altered_value = original_value.clone()
        altered_key[future_index:] += torch.randn_like(altered_key[future_index:])
        altered_value[future_index:] += torch.randn_like(altered_value[future_index:])

        original_output = scaled_dot_product_attention(query, original_key, original_value, is_causal=True)
        altered_output = scaled_dot_product_attention(query, altered_key, altered_value, is_causal=True)

        torch.testing.assert_close(original_output[:future_index], altered_output[:future_index])


@pytest.fixture
def in_features() -> int:
    """Returns the number of input features.

    Returns:
        The number of input features.
    """
    return 16


@pytest.fixture
def num_heads() -> int:
    """Returns the number of attention heads.

    Returns:
        The number of attention heads.
    """
    return 4


@pytest.fixture
def sequence_length() -> int:
    """Returns the sequence length.

    Returns:
        The sequence length.
    """
    return 4


@pytest.fixture
def batch_size() -> int:
    """Returns the batch size.

    Returns:
        The batch size.
    """
    return 2


@pytest.fixture
def multi_head_attention(in_features: int, num_heads: int, device: str) -> MultiHeadAttention:
    """Returns a multi-head attention layer.

    Args:
        in_features:
            The number of input features.
        num_heads:
            The number of attention heads.
        device:
            The device to store the multi-head attention layer on.

    Retruns:
        The multi-head attention layer.
    """
    return MultiHeadAttention(in_features, num_heads).to(device)


@pytest.mark.parametrize("device", constants.DEVICES)
@pytest.mark.usefixtures("seed")
class TestMultiHeadAttention:
    """Tests MultiHeadAttention."""

    @pytest.mark.parametrize(
        ("in_features", "num_heads", "expected_embed_dimension"),
        [
            pytest.param(16, 4, 4, id="divisible_4"),
            pytest.param(32, 8, 4, id="divisible_8"),
        ],
    )
    def test_initialization_valid(
        self,
        multi_head_attention: MultiHeadAttention,
        num_heads: int,
        expected_embed_dimension: int,
    ) -> None:
        """Test initialization with valid dimensions."""
        assert multi_head_attention.num_heads == num_heads
        assert multi_head_attention.embed_dimension == expected_embed_dimension

    @pytest.mark.parametrize(
        ("in_features", "num_heads"),
        [
            pytest.param(15, 4, id="indivisible_4"),
            pytest.param(31, 8, id="indivisible_8"),
        ],
    )
    def test_initialization_invalid_divisibility(self, device: str, in_features: int, num_heads: int) -> None:
        """Test that initialization raises ValueError if in_features is not divisible by num_heads."""
        with pytest.raises(ValueError, match="is not divisible by the number of heads"):
            MultiHeadAttention(in_features, num_heads).to(device)

    def test_reset_parameters(self, multi_head_attention: MultiHeadAttention) -> None:
        """Test that resetting parameters reinitializes the projection weights and biases."""
        original_query_weight = multi_head_attention.query_projection.weight.clone()
        assert multi_head_attention.query_projection.bias is not None
        original_query_bias = multi_head_attention.query_projection.bias.clone()

        constant_value = 99.0
        with torch.no_grad():
            for parameter in multi_head_attention.parameters():
                parameter.fill_(constant_value)

        for layer in [
            multi_head_attention.query_projection,
            multi_head_attention.key_projection,
            multi_head_attention.value_projection,
            multi_head_attention.output_projection,
        ]:
            expected_weight = torch.full_like(layer.weight, constant_value)
            torch.testing.assert_close(layer.weight, expected_weight), "Weight fill failed."
            if layer.bias is not None:
                expected_bias = torch.full_like(layer.bias, constant_value)
                torch.testing.assert_close(layer.bias, expected_bias), "Bias fill failed."

        multi_head_attention.reset_parameters()

        for layer in [
            multi_head_attention.query_projection,
            multi_head_attention.key_projection,
            multi_head_attention.value_projection,
            multi_head_attention.output_projection,
        ]:
            expected_weight = torch.full_like(layer.weight, constant_value)
            assert not torch.allclose(layer.weight, expected_weight), "Weight reset failed."
            if layer.bias is not None:
                expected_bias = torch.full_like(layer.bias, constant_value)
                assert not torch.allclose(layer.bias, expected_bias), "Bias reset failed."

        assert not torch.allclose(multi_head_attention.query_projection.weight, original_query_weight)
        assert not torch.allclose(multi_head_attention.query_projection.bias, original_query_bias)

    @pytest.mark.parametrize(
        ("query_shape", "key_shape", "value_shape", "exception_message"),
        [
            pytest.param((16,), (16,), (16,), "Query shape must be two or three dimensional, got 1.", id="query_1d"),
            pytest.param((16, 2), (16,), (16,), "Key shape must be two or three dimensional, got 1.", id="key_1d"),
            pytest.param(
                (16, 2),
                (16, 2),
                (16,),
                "Value shape must be two or three dimensional, got 1.",
                id="value_1d",
            ),
            pytest.param(
                (2, 4, 16, 32),
                (2, 4, 16, 32),
                (2, 4, 16, 32),
                "Query shape must be two or three dimensional, got 4.",
                id="query_4d",
            ),
            pytest.param(
                (4, 16, 32),
                (2, 4, 16, 32),
                (2, 4, 16, 32),
                "Key shape must be two or three dimensional, got 4.",
                id="key_4d",
            ),
            pytest.param(
                (4, 16, 32),
                (4, 16, 32),
                (2, 4, 16, 32),
                "Value shape must be two or three dimensional, got 4.",
                id="value_4d",
            ),
            pytest.param(
                (3, 8, 16),
                (2, 8, 16),
                (2, 8, 16),
                "The batch sizes for query (3), key (2) and/or value (2) do not match.",
                id="query_mismatched_batch_size",
            ),
            pytest.param(
                (2, 8, 16),
                (3, 8, 16),
                (2, 8, 16),
                "The batch sizes for query (2), key (3) and/or value (2) do not match.",
                id="key_mismatched_batch_size",
            ),
            pytest.param(
                (2, 8, 16),
                (2, 8, 16),
                (3, 8, 16),
                "The batch sizes for query (2), key (2) and/or value (3) do not match.",
                id="value_mismatched_batch_size",
            ),
            pytest.param(
                (2, 8, 32),
                (2, 8, 16),
                (2, 8, 16),
                "The feature sizes for query (32), key (16) and/or value (16) do not match.",
                id="query_mismatched_feature_dim",
            ),
            pytest.param(
                (2, 8, 16),
                (2, 8, 32),
                (2, 8, 16),
                "The feature sizes for query (16), key (32) and/or value (16) do not match.",
                id="key_mismatched_feature_dim",
            ),
            pytest.param(
                (2, 8, 16),
                (2, 8, 16),
                (2, 8, 32),
                "The feature sizes for query (16), key (16) and/or value (32) do not match.",
                id="value_mismatched_feature_dim",
            ),
            pytest.param(
                (2, 10, 16),
                (2, 8, 16),
                (2, 9, 16),
                "The sequence length for key (8) and value (9) do not match.",
                id="mismatched_kv_seq_len",
            ),
        ],
    )
    def test_forward_invalid_input_shape(
        self,
        multi_head_attention: MultiHeadAttention,
        device: str,
        query_shape: tuple[int, ...],
        key_shape: tuple[int, ...],
        value_shape: tuple[int, ...],
        exception_message: str,
    ) -> None:
        """Test that forward raises ValueError for incorrect or mismatched input tensor dimensions."""
        query = torch.randn(*query_shape, device=device)
        key = torch.randn(*key_shape, device=device)
        value = torch.randn(*value_shape, device=device)
        with pytest.raises(MultiHeadAttention.DimensionError) as exception_info:
            multi_head_attention(query, key, value)

        assert str(exception_info.value) == exception_message

    @pytest.mark.parametrize("is_causal", [True, False])
    @pytest.mark.parametrize(
        ("input_shape", "expected_output_shape"),
        [
            pytest.param((8, 16), (1, 8, 16), id="2d"),
            pytest.param((2, 8, 16), (2, 8, 16), id="3d"),
        ],
    )
    def test_forward_output_shape(
        self,
        multi_head_attention: MultiHeadAttention,
        device: str,
        is_causal: bool,
        input_shape: tuple[int, ...],
        expected_output_shape: tuple[int, ...],
    ) -> None:
        """Test forward output shape for both 2D and 3D inputs."""
        query = torch.randn(*input_shape, device=device)
        key = torch.randn(*input_shape, device=device)
        value = torch.randn(*input_shape, device=device)

        output = multi_head_attention(query, key, value, is_causal=is_causal)
        assert output.shape == expected_output_shape

    def test_forward_causal_future_independence(
        self,
        multi_head_attention: MultiHeadAttention,
        device: str,
        in_features: int,
        sequence_length: int,
    ) -> None:
        """Test that causal masking in forward prevents future token leakage."""
        multi_head_attention.eval()

        query = torch.randn(2, sequence_length, in_features, device=device)
        original_key = torch.randn(2, sequence_length, in_features, device=device)
        original_value = torch.randn(2, sequence_length, in_features, device=device)

        modified_key = original_key.clone()
        modified_value = original_value.clone()
        change_index = 2
        modified_key[:, change_index:] += torch.randn_like(modified_key[:, change_index:])
        modified_value[:, change_index:] += torch.randn_like(modified_value[:, change_index:])

        original_output = multi_head_attention(query, original_key, original_value, is_causal=True)
        modified_output = multi_head_attention(query, modified_key, modified_value, is_causal=True)

        torch.testing.assert_close(original_output[:, :change_index], modified_output[:, :change_index])
        assert not torch.allclose(original_output[:, change_index:], modified_output[:, change_index:])

    def test_forward_gradients(
        self,
        multi_head_attention: MultiHeadAttention,
        device: str,
        sequence_length: int,
        in_features: int,
    ) -> None:
        """Test that gradients flow back to projections and inputs."""
        query = torch.randn(2, sequence_length, in_features, device=device, requires_grad=True)
        key = torch.randn(2, sequence_length, in_features, device=device, requires_grad=True)
        value = torch.randn(2, sequence_length, in_features, device=device, requires_grad=True)

        output = multi_head_attention(query, key, value, is_causal=True)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        assert not query.grad.isnan().any()
        assert not key.grad.isnan().any()
        assert not value.grad.isnan().any()

        for name, param in multi_head_attention.named_parameters():
            assert param.grad is not None, f"Gradient for {name} was not calculated."
            assert not param.grad.isnan().any()

    def test_forward_cross_attention_different_sequence_lengths(
        self,
        multi_head_attention: MultiHeadAttention,
        device: str,
        in_features: int,
    ) -> None:
        """Test that cross-attention works with different query and key/value sequence lengths."""
        query_sequence_length = 5
        kv_sequence_length = 9
        batch_size = 2

        query = torch.randn(batch_size, query_sequence_length, in_features, device=device)
        key = torch.randn(batch_size, kv_sequence_length, in_features, device=device)
        value = torch.randn(batch_size, kv_sequence_length, in_features, device=device)

        output = multi_head_attention(query, key, value)
        assert output.shape == (batch_size, query_sequence_length, in_features)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_forward_dtypes(
        self,
        multi_head_attention: MultiHeadAttention,
        device: str,
        sequence_length: int,
        in_features: int,
        dtype: torch.dtype,
    ) -> None:
        """Test MultiHeadAttention works with different floating point dtypes."""
        batch_size = 2
        multi_head_attention = multi_head_attention.to(dtype)

        query = torch.randn(batch_size, sequence_length, in_features, dtype=dtype, device=device)
        key = torch.randn(batch_size, sequence_length, in_features, dtype=dtype, device=device)
        value = torch.randn(batch_size, sequence_length, in_features, dtype=dtype, device=device)

        output = multi_head_attention(query, key, value)
        assert output.dtype == dtype
        assert output.shape == (batch_size, sequence_length, in_features)

    @pytest.mark.parametrize("is_causal", [True, False])
    def test_multi_head_attention_matches_pytorch(
        self,
        multi_head_attention: MultiHeadAttention,
        device: str,
        in_features: int,
        num_heads: int,
        batch_size: int,
        sequence_length: int,
        is_causal: bool,
    ) -> None:
        """Test that MultiHeadAttention output matches torch.nn.MultiheadAttention output."""
        torch_mha = torch.nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=num_heads,
            batch_first=True,
            bias=True,
            device=device,
        )

        with torch.no_grad():
            torch_mha.in_proj_weight.copy_(
                torch.cat(
                    [
                        multi_head_attention.query_projection.weight,
                        multi_head_attention.key_projection.weight,
                        multi_head_attention.value_projection.weight,
                    ],
                    dim=0,
                ),
            )
            assert multi_head_attention.query_projection.bias is not None
            assert multi_head_attention.key_projection.bias is not None
            assert multi_head_attention.value_projection.bias is not None

            torch_mha.in_proj_bias.copy_(
                torch.cat(
                    [
                        multi_head_attention.query_projection.bias,
                        multi_head_attention.key_projection.bias,
                        multi_head_attention.value_projection.bias,
                    ],
                    dim=0,
                ),
            )
            torch_mha.out_proj.weight.copy_(multi_head_attention.output_projection.weight)
            torch_mha.out_proj.bias.copy_(multi_head_attention.output_projection.bias)

        query = torch.randn(batch_size, sequence_length, in_features, device=device)
        key = torch.randn(batch_size, sequence_length, in_features, device=device)
        value = torch.randn(batch_size, sequence_length, in_features, device=device)

        custom_output = multi_head_attention(query, key, value, is_causal=is_causal)
        attn_mask = (
            torch.nn.Transformer.generate_square_subsequent_mask(sequence_length, device=torch.device(device))
            if is_causal
            else None
        )
        torch_output, _ = torch_mha(
            query,
            key,
            value,
            need_weights=False,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

        torch.testing.assert_close(custom_output, torch_output)
