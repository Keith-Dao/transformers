"""This module contains tests for the encoder block."""

import constants
import pytest
import torch
import torch.testing

from models.blocks.encoder import EncoderBlock


@pytest.mark.parametrize("device", constants.DEVICES)
@pytest.mark.usefixtures("seed")
class TestEncoderBlock:
    """Tests EncoderBlock."""

    @pytest.mark.parametrize(
        ("batch_size", "seq_len", "in_features", "attention_heads", "expansion_dimensions"),
        [
            pytest.param(1, 4, 16, 2, 32, id="small"),
            pytest.param(2, 8, 32, 4, 64, id="medium"),
        ],
    )
    def test_encoder_block_output_shape(
        self,
        device: str,
        batch_size: int,
        seq_len: int,
        in_features: int,
        attention_heads: int,
        expansion_dimensions: int,
    ) -> None:
        """Test that the encoder block produces the correct output shape."""
        block = EncoderBlock(
            in_features=in_features,
            attention_heads=attention_heads,
            expansion_dimensions=expansion_dimensions,
        ).to(device)

        x = torch.randn(batch_size, seq_len, in_features, device=device)
        output = block(x)

        assert output.shape == (batch_size, seq_len, in_features)

    def test_encoder_block_backward_pass(self, device: str) -> None:
        """Test that gradients are correctly backpropagated through the encoder block."""
        in_features = 16
        attention_heads = 2
        expansion_dimensions = 32

        block = EncoderBlock(
            in_features=in_features,
            attention_heads=attention_heads,
            expansion_dimensions=expansion_dimensions,
        ).to(device)

        x = torch.randn(2, 5, in_features, device=device, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        for param in block.parameters():
            assert param.grad is not None

    def test_reset_parameters(self, device: str) -> None:
        """Test that resetting parameters reinitializes weights and biases across all sublayers."""
        in_features = 16
        attention_heads = 2
        expansion_dimensions = 32

        block = EncoderBlock(
            in_features=in_features,
            attention_heads=attention_heads,
            expansion_dimensions=expansion_dimensions,
        ).to(device)

        constant_weight_value = 1.23
        constant_bias_value = 4.56

        mha_layers = [
            block.multihead_attention.query_projection,
            block.multihead_attention.key_projection,
            block.multihead_attention.value_projection,
            block.multihead_attention.output_projection,
        ]
        ffn_layers = [layer for layer in block.feed_forward_network if isinstance(layer, torch.nn.Linear)]
        all_layers = mha_layers + ffn_layers

        with torch.no_grad():
            for layer in all_layers:
                layer.weight.fill_(constant_weight_value)
                if layer.bias is not None:
                    layer.bias.fill_(constant_bias_value)

        block.reset_parameters()

        for layer in all_layers:
            expected_weight = torch.full_like(layer.weight, constant_weight_value)
            assert not torch.allclose(layer.weight, expected_weight)
            if layer.bias is not None:
                expected_bias = torch.full_like(layer.bias, constant_bias_value)
                assert not torch.allclose(layer.bias, expected_bias)
