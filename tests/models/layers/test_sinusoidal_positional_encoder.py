"""This module contains tests for the sinusoidal positional encoder."""

import pytest
import torch
import torch.testing

from models.layers.sinusoidal_positional_encoder import SinusoidalPositionalEncoder
from tests import constants


@pytest.mark.parametrize("device", constants.DEVICES)
@pytest.mark.usefixtures("seed")
class TestSinusoidalPositionalEncoder:
    """Tests SinusoidalPositionalEncoder."""

    @pytest.mark.parametrize(
        ("max_sequence_length", "in_features", "seq_len", "batch_size"),
        [
            pytest.param(50, 16, 10, 1, id="2d-input"),
            pytest.param(100, 32, 20, 2, id="3d-input"),
        ],
    )
    def test_positional_encoder_output_shape(
        self,
        device: str,
        max_sequence_length: int,
        in_features: int,
        seq_len: int,
        batch_size: int,
    ) -> None:
        """Test that positional encoder produces the correct output shape."""
        encoder = SinusoidalPositionalEncoder(
            max_sequence_length=max_sequence_length,
            in_features=in_features,
            dropout_rate=0.0,
        ).to(device)

        if batch_size == 1:
            x = torch.randn(seq_len, in_features, device=device)
            expected_shape = (seq_len, in_features)
        else:
            x = torch.randn(batch_size, seq_len, in_features, device=device)
            expected_shape = (batch_size, seq_len, in_features)

        output = encoder(x)
        assert output.shape == expected_shape

    def test_positional_encoder_zero_input(self, device: str) -> None:
        """Test that with zero input and zero dropout, output equals the positional encodings."""
        max_sequence_length = 20
        in_features = 16
        seq_len = 8

        encoder = SinusoidalPositionalEncoder(
            max_sequence_length=max_sequence_length,
            in_features=in_features,
            dropout_rate=0.0,
        ).to(device)

        x = torch.zeros(seq_len, in_features, device=device)
        output = encoder(x)

        expected = encoder.encodings[:seq_len]
        torch.testing.assert_close(output, expected)

    def test_positional_encoder_backward_pass(self, device: str) -> None:
        """Test that gradients are correctly backpropagated through the positional encoder."""
        encoder = SinusoidalPositionalEncoder(
            max_sequence_length=50,
            in_features=16,
            dropout_rate=0.0,
        ).to(device)

        x = torch.randn(2, 10, 16, device=device, requires_grad=True)
        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
