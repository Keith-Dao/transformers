"""This module is the root config for pytest testing."""

import constants
import pytest
import torch


@pytest.fixture(params=constants.SEEDS)
def seed(request: pytest.FixtureRequest) -> None:
    """Sets the seed."""
    seed_ = request.param
    torch.manual_seed(seed_)
