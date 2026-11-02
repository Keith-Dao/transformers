"""This module contains tests for the Transformer encoder and decoder architectures."""

import pytest
import torch
import torch.testing

from models.architectures.attention_is_all_you_need import (
    Decoder,
    DecoderBlocksConfig,
    Encoder,
    EncoderBlocksConfig,
    PositionalEncoderConfig,
)
from tests import constants


@pytest.fixture
def vocab_size() -> int:
    """Returns the vocabulary size.

    Returns:
        The vocabulary size.
    """
    return 100


@pytest.fixture
def embedding_dimensions() -> int:
    """Returns the embedding dimensions.

    Returns:
        The embedding dimensions.
    """
    return 32


@pytest.fixture
def positional_encoder_config() -> PositionalEncoderConfig:
    """Returns positional encoder configuration.

    Returns:
        The positional encoder configuration.
    """
    return PositionalEncoderConfig(max_sequence_length=50, dropout_rate=0.0)


@pytest.fixture
def encoder_blocks_config() -> EncoderBlocksConfig:
    """Returns encoder blocks configuration.

    Returns:
        The encoder blocks configuration.
    """
    return EncoderBlocksConfig(n_blocks=2, attention_heads=4, expansion_dimensions=64)


@pytest.fixture
def encoder(
    device: str,
    vocab_size: int,
    embedding_dimensions: int,
    positional_encoder_config: PositionalEncoderConfig,
    encoder_blocks_config: EncoderBlocksConfig,
) -> Encoder:
    """Returns configured encoder.

    Args:
        device:
            The device to place the encoder on.
        vocab_size:
            The vocabulary size.
        embedding_dimensions:
            The embedding dimensions.
        positional_encoder_config:
            The positional encoder configuration.
        encoder_blocks_config:
            The encoder blocks configuration.

    Returns:
        The configured encoder instance.
    """
    return Encoder(
        vocab_size=vocab_size,
        embedding_dimensions=embedding_dimensions,
        positional_encoder_config=positional_encoder_config,
        encoder_blocks_config=encoder_blocks_config,
    ).to(device)


@pytest.fixture
def decoder_blocks_config() -> DecoderBlocksConfig:
    """Returns decoder blocks configuration.

    Returns:
        The decoder blocks configuration.
    """
    return DecoderBlocksConfig(n_blocks=2, attention_heads=4, expansion_dimensions=64)


@pytest.fixture
def decoder(
    device: str,
    vocab_size: int,
    embedding_dimensions: int,
    positional_encoder_config: PositionalEncoderConfig,
    decoder_blocks_config: DecoderBlocksConfig,
) -> Decoder:
    """Returns configured decoder.

    Args:
        device:
            The device to place the decoder on.
        vocab_size:
            The vocabulary size.
        embedding_dimensions:
            The embedding dimensions.
        positional_encoder_config:
            The positional encoder configuration.
        decoder_blocks_config:
            The decoder blocks configuration.

    Returns:
        The configured decoder instance.
    """
    return Decoder(
        vocab_size=vocab_size,
        embedding_dimensions=embedding_dimensions,
        positional_encoder_config=positional_encoder_config,
        decoder_blocks_config=decoder_blocks_config,
    ).to(device)


@pytest.mark.parametrize("device", constants.DEVICES)
@pytest.mark.usefixtures("seed")
class TestEncoderArchitecture:
    """Tests Encoder architecture."""

    def test_output_shape(
        self,
        encoder: Encoder,
        device: str,
        vocab_size: int,
        embedding_dimensions: int,
    ) -> None:
        """Test output is the correct shape."""
        batch_size = 2
        sequence_length = 10
        x = torch.randn(batch_size, sequence_length, vocab_size, device=device)
        output = encoder(x)

        assert output.shape == (batch_size, sequence_length, embedding_dimensions)

    def test_vocab_size(self, encoder: Encoder, vocab_size: int) -> None:
        """Test vocabulary size property."""
        assert encoder.vocab_size == vocab_size

    def test_embedding_dimensions(self, encoder: Encoder, embedding_dimensions: int) -> None:
        """Test embedding dimensions property."""
        assert encoder.embedding_dimensions == embedding_dimensions

    def test_backward_pass(self, encoder: Encoder, device: str, vocab_size: int) -> None:
        """Test that gradients are correctly backpropagated."""
        batch_size = 2
        sequence_length = 5
        x = torch.randn(batch_size, sequence_length, vocab_size, device=device, requires_grad=True)
        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        for param in encoder.parameters():
            assert param.grad is not None

    def test_reset_parameters(self, encoder: Encoder) -> None:
        """Test resetting parameters."""
        constant_value = 99.0

        with torch.no_grad():
            for parameter in encoder.parameters():
                parameter.fill_(constant_value)

        linear_layers = [encoder.embedding]
        encoder_blocks = encoder.encoder_blocks

        for layer in linear_layers:
            assert torch.allclose(layer.weight, torch.full_like(layer.weight, constant_value)), "Weight fill failed."
            if layer.bias is not None:
                assert torch.allclose(layer.bias, torch.full_like(layer.bias, constant_value)), "Bias fill failed."
        for parameter in encoder_blocks.parameters():
            assert torch.allclose(parameter, torch.full_like(parameter, constant_value)), "Parameter fill failed."

        encoder.reset_parameters()

        for layer in linear_layers:
            assert not torch.allclose(layer.weight, torch.full_like(layer.weight, constant_value)), (
                "Weight reset failed."
            )
            if layer.bias is not None:
                assert not torch.allclose(layer.bias, torch.full_like(layer.bias, constant_value)), "Bias reset failed."
        for parameter in encoder_blocks.parameters():
            assert not torch.allclose(parameter, torch.full_like(parameter, constant_value)), "Parameter reset failed."


@pytest.mark.parametrize("device", constants.DEVICES)
@pytest.mark.usefixtures("seed")
class TestDecoderArchitecture:
    """Tests Decoder architecture."""

    def test_output_shape(
        self,
        decoder: Decoder,
        device: str,
        vocab_size: int,
        embedding_dimensions: int,
    ) -> None:
        """Test output is the correct shape."""
        batch_size = 2
        target_sequence_length = 8
        source_sequence_length = 12
        x = torch.randn(batch_size, target_sequence_length, vocab_size, device=device)
        encoder_output = torch.randn(batch_size, source_sequence_length, embedding_dimensions, device=device)
        output = decoder(x, encoder_output)

        assert output.shape == (batch_size, target_sequence_length, vocab_size)

    def test_vocab_size(self, decoder: Decoder, vocab_size: int) -> None:
        """Test vocabulary size property."""
        assert decoder.vocab_size == vocab_size

    def test_embedding_dimensions(self, decoder: Decoder, embedding_dimensions: int) -> None:
        """Test embedding dimensions property."""
        assert decoder.embedding_dimensions == embedding_dimensions

    def test_backward_pass(
        self,
        decoder: Decoder,
        device: str,
        vocab_size: int,
        embedding_dimensions: int,
    ) -> None:
        """Test that gradients are correctly backpropagated."""
        batch_size = 2
        target_sequence_length = 5
        source_sequence_length = 6
        x = torch.randn(batch_size, target_sequence_length, vocab_size, device=device, requires_grad=True)
        encoder_output = torch.randn(
            batch_size,
            source_sequence_length,
            embedding_dimensions,
            device=device,
            requires_grad=True,
        )
        output = decoder(x, encoder_output)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert encoder_output.grad is not None
        assert encoder_output.grad.shape == encoder_output.shape
        for param in decoder.parameters():
            assert param.grad is not None

    def test_reset_parameters(self, decoder: Decoder) -> None:
        """Test resetting parameters."""
        constant_value = 99.0

        with torch.no_grad():
            for parameter in decoder.parameters():
                parameter.fill_(constant_value)

        linear_layers = [decoder.embedding, decoder.output_projection]
        decoder_blocks = decoder.decoder_blocks

        for layer in linear_layers:
            assert torch.allclose(layer.weight, torch.full_like(layer.weight, constant_value)), "Weight fill failed."
            if layer.bias is not None:
                assert torch.allclose(layer.bias, torch.full_like(layer.bias, constant_value)), "Bias fill failed."
        for parameter in decoder_blocks.parameters():
            assert torch.allclose(parameter, torch.full_like(parameter, constant_value)), (
                "Decoder block parameter fill failed."
            )

        decoder.reset_parameters()

        for layer in linear_layers:
            assert not torch.allclose(layer.weight, torch.full_like(layer.weight, constant_value)), (
                "Weight reset failed."
            )
            if layer.bias is not None:
                assert not torch.allclose(layer.bias, torch.full_like(layer.bias, constant_value)), "Bias reset failed."
        for parameter in decoder_blocks.parameters():
            assert not torch.allclose(parameter, torch.full_like(parameter, constant_value)), (
                "Decoder block parameter reset failed."
            )
