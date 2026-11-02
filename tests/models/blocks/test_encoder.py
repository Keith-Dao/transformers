"""This module contains tests for the encoder block."""

import pytest
import torch
import torch.testing

from models.blocks.encoder import EncoderBlock
from tests import constants


@pytest.fixture
def in_features() -> int:
    """Returns the number of input features.

    Returns:
        The number of input features.
    """
    return 16


@pytest.fixture
def attention_heads() -> int:
    """Returns the number of attention heads.

    Returns:
        The number of attention heads.
    """
    return 2


@pytest.fixture
def expansion_dimensions() -> int:
    """Returns the number of expansion dimensions.

    Returns:
        The number of expansion dimensions.
    """
    return 32


@pytest.fixture
def encoder(in_features: int, attention_heads: int, expansion_dimensions: int, device: str) -> EncoderBlock:
    """Retruns a encoder block.

    Args:
        in_features:
            The number of input features.
        attention_heads:
            The number of attention heads in the block.
        expansion_dimensions:
            The number of expansion dimensions.
        device:
            The device to store the encoder on.

    Returns:
        The encoder block.
    """
    return EncoderBlock(
        in_features=in_features,
        attention_heads=attention_heads,
        expansion_dimensions=expansion_dimensions,
    ).to(device)


@pytest.mark.parametrize("device", constants.DEVICES)
@pytest.mark.usefixtures("seed")
class TestEncoderBlock:
    """Tests EncoderBlock."""

    @pytest.mark.parametrize(
        ("batch_size", "sequence_length", "in_features", "attention_heads", "expansion_dimensions"),
        [
            pytest.param(1, 4, 16, 2, 32, id="small"),
            pytest.param(2, 8, 32, 4, 64, id="medium"),
        ],
    )
    def test_encoder_block_output_shape(
        self,
        encoder: EncoderBlock,
        device: str,
        batch_size: int,
        sequence_length: int,
        in_features: int,
    ) -> None:
        """Test that the encoder block produces the correct output shape."""
        x = torch.randn(batch_size, sequence_length, in_features, device=device)
        output = encoder(x)

        assert output.shape == (batch_size, sequence_length, in_features)

    def test_encoder_block_backward_pass(self, encoder: EncoderBlock, in_features: int, device: str) -> None:
        """Test that gradients are correctly backpropagated through the encoder block."""
        x = torch.randn(2, 5, in_features, device=device, requires_grad=True)
        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        for param in encoder.parameters():
            assert param.grad is not None

    def test_reset_parameters(self, encoder: EncoderBlock) -> None:
        """Test that resetting parameters reinitializes weights and biases across all sublayers."""
        constant_value = 99.0
        with torch.no_grad():
            for parameter in encoder.parameters():
                parameter.fill_(constant_value)

        mha_layers = [
            encoder.multihead_attention.query_projection,
            encoder.multihead_attention.key_projection,
            encoder.multihead_attention.value_projection,
            encoder.multihead_attention.output_projection,
        ]
        ffn_layers = [layer for layer in encoder.feed_forward_network if isinstance(layer, torch.nn.Linear)]
        all_layers = mha_layers + ffn_layers

        for layer in all_layers:
            expected_weight = torch.full_like(layer.weight, constant_value)
            assert torch.allclose(layer.weight, expected_weight), "Weight fill failed."
            if layer.bias is not None:
                expected_bias = torch.full_like(layer.bias, constant_value)
                assert torch.allclose(layer.bias, expected_bias), "Bias fill failed."

        encoder.reset_parameters()

        for layer in all_layers:
            expected_weight = torch.full_like(layer.weight, constant_value)
            assert not torch.allclose(layer.weight, expected_weight), "Weight reset failed."
            if layer.bias is not None:
                expected_bias = torch.full_like(layer.bias, constant_value)
                assert not torch.allclose(layer.bias, expected_bias), "Bias reset failed."
