"""This module contains tests for the decoder block."""

import pytest
import torch
import torch.testing

from models.blocks.decoder import DecoderBlock
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
def decoder(in_features: int, attention_heads: int, expansion_dimensions: int, device: str) -> DecoderBlock:
    """Returns a decoder block.

    Args:
        in_features:
            The number of input features.
        attention_heads:
            The number of attention heads in the block.
        expansion_dimensions:
            The number of expansion dimensions.
        device:
            The device to store the decoder on.

    Returns:
        The decoder block.
    """
    return DecoderBlock(
        in_features=in_features,
        attention_heads=attention_heads,
        expansion_dimensions=expansion_dimensions,
    ).to(device)


@pytest.mark.parametrize("device", constants.DEVICES)
@pytest.mark.usefixtures("seed")
class TestDecoderBlock:
    """Tests DecoderBlock."""

    @pytest.mark.parametrize(
        (
            "batch_size",
            "target_sequence_length",
            "source_sequence_length",
            "in_features",
            "attention_heads",
            "expansion_dimensions",
        ),
        [
            pytest.param(1, 4, 6, 16, 2, 32, id="small"),
            pytest.param(2, 8, 10, 32, 4, 64, id="medium"),
        ],
    )
    def test_decoder_block_output_shape(
        self,
        decoder: DecoderBlock,
        device: str,
        batch_size: int,
        target_sequence_length: int,
        source_sequence_length: int,
        in_features: int,
    ) -> None:
        """Test that the decoder block produces the correct output shape."""
        x = torch.randn(batch_size, target_sequence_length, in_features, device=device)
        encoder_output = torch.randn(batch_size, source_sequence_length, in_features, device=device)
        output = decoder(x, encoder_output)

        assert output.shape == (batch_size, target_sequence_length, in_features)

    def test_decoder_block_backward_pass(self, decoder: DecoderBlock, in_features: int, device: str) -> None:
        """Test that gradients are correctly backpropagated through the decoder block."""
        x = torch.randn(2, 5, in_features, device=device, requires_grad=True)
        encoder_output = torch.randn(2, 7, in_features, device=device, requires_grad=True)
        output = decoder(x, encoder_output)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert encoder_output.grad is not None
        assert encoder_output.grad.shape == encoder_output.shape
        for param in decoder.parameters():
            assert param.grad is not None

    def test_reset_parameters(self, decoder: DecoderBlock) -> None:
        """Test that resetting parameters reinitializes weights and biases across all decoder sublayers."""
        constant_value = 99.0
        with torch.no_grad():
            for parameters in decoder.parameters():
                parameters.fill_(constant_value)

        mha_layers = [
            decoder.masked_self_attention.query_projection,
            decoder.masked_self_attention.key_projection,
            decoder.masked_self_attention.value_projection,
            decoder.masked_self_attention.output_projection,
            decoder.cross_attention.query_projection,
            decoder.cross_attention.key_projection,
            decoder.cross_attention.value_projection,
            decoder.cross_attention.output_projection,
        ]
        ffn_layers = [layer for layer in decoder.feed_forward_network if isinstance(layer, torch.nn.Linear)]
        all_layers = mha_layers + ffn_layers

        for layer in all_layers:
            expected_weight = torch.full_like(layer.weight, constant_value)
            assert torch.allclose(layer.weight, expected_weight), "Weight fill failed."
            if layer.bias is not None:
                expected_bias = torch.full_like(layer.bias, constant_value)
                assert torch.allclose(layer.bias, expected_bias), "Bias fill failed."

        decoder.reset_parameters()

        for layer in all_layers:
            expected_weight = torch.full_like(layer.weight, constant_value)
            assert not torch.allclose(layer.weight, expected_weight), "Weight reset failed."
            if layer.bias is not None:
                expected_bias = torch.full_like(layer.bias, constant_value)
                assert not torch.allclose(layer.bias, expected_bias), "Bias reset failed."
