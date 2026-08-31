"""This module contains tests for the attention layer."""

import constants
import pytest
import torch
import torch.testing

from models.layers.attention import scaled_dot_product_attention


@pytest.mark.parametrize("device", constants.DEVICES)
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

        result = scaled_dot_product_attention(query, key, value)
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        torch.testing.assert_close(result, expected), "Implementation does not match pytorch's."

    def test_scaled_dot_product_attention_uniform_weights(self, device: str) -> None:
        """Test that uniform attention scores result in average of value vectors."""
        seq_q, seq_k, d_k, d_v = 3, 5, 8, 4
        query = torch.zeros(seq_q, d_k, device=device)
        key = torch.zeros(seq_k, d_k, device=device)
        value = torch.randn(seq_k, d_v, device=device)

        output = scaled_dot_product_attention(query, key, value)
        expected_row = value.mean(dim=0)
        expected = expected_row.unsqueeze(0).expand(seq_q, d_v)

        torch.testing.assert_close(output, expected)

    def test_scaled_dot_product_attention_constant_value(self, device: str) -> None:
        """Test that constant value matrix returns constant rows equal to that value vector."""
        query = torch.randn(4, 8, device=device)
        key = torch.randn(6, 8, device=device)
        constant_row = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        value = constant_row.unsqueeze(0).expand(6, 4)

        output = scaled_dot_product_attention(query, key, value)
        expected = constant_row.unsqueeze(0).expand(4, 4)

        torch.testing.assert_close(output, expected)

    def test_scaled_dot_product_attention_sharp_attention(self, device: str) -> None:
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

        output = scaled_dot_product_attention(query, key, value)
        expected = torch.tensor([[1.0, 2.0]], device=device)

        torch.testing.assert_close(output, expected)

    def test_scaled_dot_product_attention_backward_pass(self, device: str) -> None:
        """Test that gradients are correctly backpropagated to query, key, and value."""
        query = torch.randn(2, 4, 8, device=device, requires_grad=True)
        key = torch.randn(2, 6, 8, device=device, requires_grad=True)
        value = torch.randn(2, 6, 16, device=device, requires_grad=True)

        output = scaled_dot_product_attention(query, key, value)
        loss = output.sum()
        loss.backward()

        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        assert not torch.isnan(query.grad).any()
        assert not torch.isnan(key.grad).any()
        assert not torch.isnan(value.grad).any()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_scaled_dot_product_attention_dtypes(self, dtype: torch.dtype, device: str) -> None:
        """Test that scaled dot product attention works with various precision."""
        query = torch.randn(2, 4, 8, dtype=dtype, device=device)
        key = torch.randn(2, 6, 8, dtype=dtype, device=device)
        value = torch.randn(2, 6, 16, dtype=dtype, device=device)

        output = scaled_dot_product_attention(query, key, value)

        assert output.dtype == dtype
        assert output.shape == (2, 4, 16)
