"""This module contains the attention is all you need model architecture."""

import dataclasses

import torch
import torch.nn

import models.blocks
import models.layers


@dataclasses.dataclass(frozen=True)
class PositionalEncoderConfig:
    """Configuration for the positional encoder.

    Args:
        max_sequence_length:
            The maximum sequence length.
        dropout_rate:
            The dropout rate.
    """

    max_sequence_length: int = 512
    dropout_rate: float = 0.1


DEFAULT_POSITIONAL_ENCODER_CONFIG = PositionalEncoderConfig()


@dataclasses.dataclass(frozen=True)
class EncoderBlocksConfig:
    """Configuration for the encoder blocks.

    Args:
        n_blocks:
            The number of sequential encoder blocks.
        attention_heads:
            The number of attention heads for each attention later.
        expansion_dimensions:
            The number of dimensions to expand to in the feed-forward network.
    """

    n_blocks: int = 6
    attention_heads: int = 8
    expansion_dimensions: int = 2048


DEFAULT_ENCODER_BLOCKS_CONFIG = EncoderBlocksConfig()


@dataclasses.dataclass(frozen=True)
class DecoderBlocksConfig:
    """Configuration for the deccoder blocks.

    Args:
        n_blocks:
            The number of sequential decoder blocks.
        attention_heads:
            The number of attention heads for each attention later.
        expansion_dimensions:
            The number of dimensions to expand to in the feed-forward network.
    """

    n_blocks: int = 6
    attention_heads: int = 8
    expansion_dimensions: int = 2048


DEFAULT_DECODER_BLOCKS_CONFIG = DecoderBlocksConfig()


class Encoder(torch.nn.Module):
    """The attention is all you need encoder.

    Attributes:
        embedding:
            The embedding layer.
        positional_encoder:
            The positional encoder.
        encoder_blocks:
            The encoder blocks.
    """

    embedding: torch.nn.Linear
    positional_encoder: models.layers.SinusoidalPositionalEncoder
    encoder_blocks: torch.nn.ModuleList[models.blocks.EncoderBlock]

    def __init__(
        self,
        vocab_size: int,
        embedding_dimensions: int = 512,
        positional_encoder_config: PositionalEncoderConfig = DEFAULT_POSITIONAL_ENCODER_CONFIG,
        encoder_blocks_config: EncoderBlocksConfig = DEFAULT_ENCODER_BLOCKS_CONFIG,
    ) -> None:
        """Initialises the encoder."""
        super().__init__()
        self.embedding = torch.nn.Linear(in_features=vocab_size, out_features=embedding_dimensions)
        self.positional_encoder = models.layers.SinusoidalPositionalEncoder(
            in_features=embedding_dimensions,
            max_sequence_length=positional_encoder_config.max_sequence_length,
            dropout_rate=positional_encoder_config.dropout_rate,
        )
        self.encoder_blocks = torch.nn.ModuleList(
            [
                models.blocks.EncoderBlock(
                    in_features=embedding_dimensions,
                    attention_heads=encoder_blocks_config.attention_heads,
                    expansion_dimensions=encoder_blocks_config.expansion_dimensions,
                )
                for _ in range(encoder_blocks_config.n_blocks)
            ],
        )

    @property
    def vocab_size(self) -> int:
        """The vocabulary size."""
        return self.embedding.in_features

    @property
    def embedding_dimensions(self) -> int:
        """The number of dimensions from the embedding output."""
        return self.embedding.out_features

    def reset_parameters(self) -> None:
        """Resets the parameters."""
        self.embedding.reset_parameters()
        for encoder_block in self.encoder_blocks:
            encoder_block.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass.

        Args:
            x:
                The input tokens.

        Returns:
            The encoded sequence.
        """
        embedded_x = self.embedding(x)
        encoded_x = self.positional_encoder(embedded_x)

        for encoder_block in self.encoder_blocks:
            encoded_x = encoder_block(encoded_x)

        return encoded_x


class Decoder(torch.nn.Module):
    """The attention is all you need decoder.

    Attributes:
        embedding:
            The embedding layer.
        positional_encoder:
            The positional encoder.
        decoder_blocks:
            The decoder blocks.
    """

    embedding: torch.nn.Linear
    positional_encoder: models.layers.SinusoidalPositionalEncoder
    decoder_blocks: torch.nn.ModuleList[models.blocks.DecoderBlock]
    output_projection: torch.nn.Linear

    def __init__(
        self,
        vocab_size: int,
        embedding_dimensions: int = 512,
        positional_encoder_config: PositionalEncoderConfig = DEFAULT_POSITIONAL_ENCODER_CONFIG,
        decoder_blocks_config: DecoderBlocksConfig = DEFAULT_DECODER_BLOCKS_CONFIG,
    ) -> None:
        """Initialises the decoder."""
        super().__init__()
        self.embedding = torch.nn.Linear(in_features=vocab_size, out_features=embedding_dimensions)
        self.positional_encoder = models.layers.SinusoidalPositionalEncoder(
            in_features=embedding_dimensions,
            max_sequence_length=positional_encoder_config.max_sequence_length,
            dropout_rate=positional_encoder_config.dropout_rate,
        )
        self.decoder_blocks = torch.nn.ModuleList(
            [
                models.blocks.DecoderBlock(
                    in_features=embedding_dimensions,
                    attention_heads=decoder_blocks_config.attention_heads,
                    expansion_dimensions=decoder_blocks_config.expansion_dimensions,
                )
                for _ in range(decoder_blocks_config.n_blocks)
            ],
        )
        self.output_projection = torch.nn.Linear(in_features=embedding_dimensions, out_features=vocab_size)

    @property
    def vocab_size(self) -> int:
        """The vocabulary size."""
        return self.embedding.in_features

    @property
    def embedding_dimensions(self) -> int:
        """The number of dimensions from the embedding output."""
        return self.embedding.out_features

    def reset_parameters(self) -> None:
        """Resets the parameters."""
        self.embedding.reset_parameters()
        self.output_projection.reset_parameters()
        for decoder_block in self.decoder_blocks:
            decoder_block.reset_parameters()

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass.

        Args:
            x:
                The input tokens.
            encoder_output:
                The encoder outputs.

        Returns:
            The decoded sequence vocabulary logits.
        """
        embedded_x = self.embedding(x)
        encoded_x = self.positional_encoder(embedded_x)

        for decoder_block in self.decoder_blocks:
            encoded_x = decoder_block(encoded_x, encoder_output)

        return self.output_projection(encoded_x)
