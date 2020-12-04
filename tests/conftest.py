import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")
