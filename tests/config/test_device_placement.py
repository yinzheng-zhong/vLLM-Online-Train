"""Resolving the device the head trains on.

The setting is a string an operator writes, so the two things that matter are that a
bare type never silently moves the training off the serving GPU, and that an index this
process cannot see fails at startup rather than at the first allocation.
"""

import pytest
import torch

from tests.conftest import device_placement

CUDA0 = torch.device("cuda", 0)


def test_no_setting_trains_on_the_engines_device():
    assert device_placement.resolve(None, CUDA0) == CUDA0


def test_a_bare_cuda_means_the_engines_own_index():
    """`torch.device("cuda") != torch.device("cuda:0")`, so an unindexed request has to
    resolve to the engine's index or every comparison downstream sees two devices."""
    resolved = device_placement.resolve("cuda", CUDA0)
    assert resolved == CUDA0
    assert resolved.index == 0


def test_an_explicit_index_is_kept():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second visible CUDA device")
    assert device_placement.resolve("cuda:1", CUDA0) == torch.device("cuda", 1)


def test_a_different_type_is_kept_without_an_index():
    assert device_placement.resolve("cpu", CUDA0) == torch.device("cpu")


def test_an_invisible_cuda_index_is_refused():
    """Unset `train_device` rather than have the first head allocation fail."""
    with pytest.raises(ValueError, match="CUDA device"):
        device_placement.resolve("cuda:99", CUDA0)


def test_a_string_that_names_no_device_is_refused():
    with pytest.raises(ValueError, match="does not name a device"):
        device_placement.resolve("gpu1", CUDA0)
