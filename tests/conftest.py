import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch

from src.config import load_config


def pytest_sessionstart(session):
    torch.set_num_threads(2)


@pytest.fixture
def config():
    return load_config("configs/experiments/shared_hard/efficientnet_b0.yaml")
