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
        device: str,
        vocab_size: int,
        embedding_dimensions: int,
        encoder: Encoder,
    ) -> None:
        """Test output is the correct shape."""
        batch_size = 2
        sequence_length = 10
        x = torch.randn(batch_size, sequence_length, vocab_size, device=device)
        output = encoder(x)

        assert output.shape == (batch_size, sequence_length, embedding_dimensions)

    def test_vocab_size(self, vocab_size: int, encoder: Encoder) -> None:
        """Test vocabulary size property."""
        assert encoder.vocab_size == vocab_size

    def test_embedding_dimensions(self, embedding_dimensions: int, encoder: Encoder) -> None:
        """Test embedding dimensions property."""
        assert encoder.embedding_dimensions == embedding_dimensions

    def test_backward_pass(self, device: str, vocab_size: int, encoder: Encoder) -> None:
        """Test that gradients are correctly backpropagated."""
        x = torch.randn(2, 5, vocab_size, device=device, requires_grad=True)
        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        for param in encoder.parameters():
            assert param.grad is not None

    def test_reset_parameters(self, encoder: Encoder) -> None:
        """Test resetting parameters."""
        with torch.no_grad():
            encoder.embedding.weight.fill_(99.0)
            if encoder.embedding.bias is not None:
                encoder.embedding.bias.fill_(99.0)
            for block in encoder.encoder_blocks:
                for param in block.parameters():
                    param.fill_(99.0)

        encoder.reset_parameters()

        assert not torch.allclose(encoder.embedding.weight, torch.full_like(encoder.embedding.weight, 99.0))
        if encoder.embedding.bias is not None:
            assert not torch.allclose(encoder.embedding.bias, torch.full_like(encoder.embedding.bias, 99.0))
        for block in encoder.encoder_blocks:
            for param in block.parameters():
                assert not torch.allclose(param, torch.full_like(param, 99.0))


@pytest.mark.parametrize("device", constants.DEVICES)
@pytest.mark.usefixtures("seed")
class TestDecoderArchitecture:
    """Tests Decoder architecture."""

    def test_output_shape(
        self,
        device: str,
        vocab_size: int,
        embedding_dimensions: int,
        decoder: Decoder,
    ) -> None:
        """Test output is the correct shape."""
        batch_size = 2
        sequence_length_target = 8
        sequence_length_source = 12
        x = torch.randn(batch_size, sequence_length_target, vocab_size, device=device)
        encoder_output = torch.randn(batch_size, sequence_length_source, embedding_dimensions, device=device)
        output = decoder(x, encoder_output)

        assert output.shape == (batch_size, sequence_length_target, vocab_size)

    def test_vocab_size(self, vocab_size: int, decoder: Decoder) -> None:
        """Test vocabulary size property."""
        assert decoder.vocab_size == vocab_size

    def test_embedding_dimensions(self, embedding_dimensions: int, decoder: Decoder) -> None:
        """Test embedding dimensions property."""
        assert decoder.embedding_dimensions == embedding_dimensions

    def test_backward_pass(
        self,
        device: str,
        vocab_size: int,
        embedding_dimensions: int,
        decoder: Decoder,
    ) -> None:
        """Test that gradients are correctly backpropagated."""
        x = torch.randn(2, 5, vocab_size, device=device, requires_grad=True)
        encoder_output = torch.randn(2, 6, embedding_dimensions, device=device, requires_grad=True)
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
        with torch.no_grad():
            decoder.embedding.weight.fill_(99.0)
            if decoder.embedding.bias is not None:
                decoder.embedding.bias.fill_(99.0)
            decoder.output_projection.weight.fill_(99.0)
            if decoder.output_projection.bias is not None:
                decoder.output_projection.bias.fill_(99.0)
            for block in decoder.decoder_blocks:
                for param in block.parameters():
                    param.fill_(99.0)

        decoder.reset_parameters()

        assert not torch.allclose(decoder.embedding.weight, torch.full_like(decoder.embedding.weight, 99.0))
        if decoder.embedding.bias is not None:
            assert not torch.allclose(decoder.embedding.bias, torch.full_like(decoder.embedding.bias, 99.0))
        assert not torch.allclose(
            decoder.output_projection.weight,
            torch.full_like(decoder.output_projection.weight, 99.0),
        )
        if decoder.output_projection.bias is not None:
            assert not torch.allclose(
                decoder.output_projection.bias,
                torch.full_like(decoder.output_projection.bias, 99.0),
            )
        for block in decoder.decoder_blocks:
            for param in block.parameters():
                assert not torch.allclose(param, torch.full_like(param, 99.0))
