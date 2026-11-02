"""This module contains tests for the sinusoidal positional encoder."""

import pytest
import torch
import torch.testing

from models.layers.sinusoidal_positional_encoder import SinusoidalPositionalEncoder
from tests import constants


@pytest.fixture
def max_sequence_length() -> int:
    """Returns the maximum sequence length.

    Returns:
        The maximum sequence length.
    """
    return 20


@pytest.fixture
def in_features() -> int:
    """Returns the number of input features.

    Returns:
        The number of input features.
    """
    return 16


@pytest.fixture
def sequence_length() -> int:
    """Returns the sequence length.

    Returns:
        The sequence length.
    """
    return 8


@pytest.fixture
def batch_size() -> int:
    """Returns the batch size.

    Returns:
        The batch size.
    """
    return 2


@pytest.fixture
def dropout_rate() -> float:
    """Returns the dropout rate.

    Returns:
        The dropout rate.
    """
    return 0.0


@pytest.fixture
def positional_encoder(
    max_sequence_length: int,
    in_features: int,
    dropout_rate: float,
    device: str,
) -> SinusoidalPositionalEncoder:
    """Returns a sinusoidal positional encoder.

    Args:
        max_sequence_length:
            The maximum sequence length.
        in_features:
            The number of input features.
        dropout_rate:
            The dropout rate.
        device:
            The device to store the positional encoder on.

    Returns:
        A sinusoidal positional encoder.
    """
    return SinusoidalPositionalEncoder(
        max_sequence_length=max_sequence_length,
        in_features=in_features,
        dropout_rate=dropout_rate,
    ).to(device)


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
        positional_encoder: SinusoidalPositionalEncoder,
        device: str,
        in_features: int,
        seq_len: int,
        batch_size: int,
    ) -> None:
        """Test that positional encoder produces the correct output shape."""
        if batch_size == 1:
            x = torch.randn(seq_len, in_features, device=device)
            expected_shape = (seq_len, in_features)
        else:
            x = torch.randn(batch_size, seq_len, in_features, device=device)
            expected_shape = (batch_size, seq_len, in_features)

        output = positional_encoder(x)
        assert output.shape == expected_shape

    def test_positional_encoder_zero_input(
        self,
        positional_encoder: SinusoidalPositionalEncoder,
        sequence_length: int,
        in_features: int,
        device: str,
    ) -> None:
        """Test that with zero input and zero dropout, output equals the positional encodings."""
        x = torch.zeros(sequence_length, in_features, device=device)
        output = positional_encoder(x)

        expected = positional_encoder.encodings[:sequence_length]
        torch.testing.assert_close(output, expected)

    def test_positional_encoder_backward_pass(
        self,
        positional_encoder: SinusoidalPositionalEncoder,
        batch_size: int,
        sequence_length: int,
        in_features: int,
        device: str,
    ) -> None:
        """Test that gradients are correctly backpropagated through the positional encoder."""
        x = torch.randn(batch_size, sequence_length, in_features, device=device, requires_grad=True)
        output = positional_encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
