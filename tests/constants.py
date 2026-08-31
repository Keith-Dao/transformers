"""This module contains constants for testing that cannot be in a fixture instead."""

import torch

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if torch.backends.mps.is_available():
    DEVICES.append("mps")
if torch.xpu.is_available():
    DEVICES.append("xpu")

SEEDS = [0, 42, 333]
