# Transformers

A lightweight Python library built with PyTorch for constructing and training a simple transformer pipeline.

## Overview

This repository provides modular building blocks to construct, train, and evaluate Transformer models. It is designed to be readable, extensible, and suitable for learning or experimenting with attention mechanisms and sequence modeling.

## Features

- **Modular Layers**: Core transformer components such as scaled dot-product attention, multi-head attention, and positional encodings.
- **Model Architectures**: Flexible encoder and decoder transformer models.
- **Training & Data Pipeline**: Integration with PyTorch and Hugging Face `datasets` for data preprocessing, batching, and training loops.
- **Testing & Code Quality**: Comprehensive unit test suite using `pytest` and linting with `ruff`.

## Project Structure

```text
transformers/
├── src/
│   ├── models/
│   │   ├── architectures/  # Model definition files (Encoder, Decoder, Transformer)
│   │   └── layers/         # Sub-layers (Attention, FeedForward, LayerNorm)
│   └── transformers/       # Pipeline utilities and entry points
├── tests/                  # Unit and integration test suite
├── pyproject.toml          # Project configuration and dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.14+ (or compatible Python version)
- [uv](https://github.com/astral-sh/uv) (recommended package manager) or `pip`

### Installation

Clone the repository and install the dependencies:

```bash
# Using uv (recommended)
uv sync

# Or using standard pip
pip install -e .
```

## Usage

> _Usage instructions and code examples will be added as the model architectures and training pipelines are implemented._

## License

This project is open-source and available under the terms of the MIT License.
