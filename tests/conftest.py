"""This module is the root config for pytest testing."""

import pytest
import torch

from tests import constants


@pytest.fixture(params=constants.SEEDS)
def seed(request: pytest.FixtureRequest) -> None:
    """Sets the seed."""
    seed_ = request.param
    torch.manual_seed(seed_)
