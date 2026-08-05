"""Pushing trained weights into a running drafter.

The hazard is aliasing, not restart: `_build_context_kv_buffers` reassigns
`torch.cat` snapshots at new addresses, and anything holding the old pointer keeps
reading the tensor that was replaced.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.conftest import (
    drafter_locator,
    head_factory,
    make_arch,
    weight_publisher,
)


class FakeInner(nn.Module):
    """Stands in for `DFlashQwen3Model`, including its derived buffers."""

    def __init__(self, rebuild_at_new_address: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.hidden_norm = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.rebuild_at_new_address = rebuild_at_new_address
        self._hidden_norm_weight = self.hidden_norm.weight.data
        self._fused_kv_weight = torch.zeros(4, 4, dtype=torch.bfloat16)
        self._k_norm_weights = torch.zeros(2, 4, dtype=torch.bfloat16)

    def rebuild(self) -> None:
        if self.rebuild_at_new_address:
            self._fused_kv_weight = torch.ones(4, 4, dtype=torch.bfloat16)
            self._k_norm_weights = torch.ones(2, 4, dtype=torch.bfloat16)


class FakeDrafter:
    """Stands in for `DFlashQwen3ForCausalLM`."""

    def __init__(self, inner: FakeInner) -> None:
        self.model = inner
        self.loaded: list[str] = []

    def load_weights(self, weights) -> None:
        self.loaded = [name for name, _ in weights]
        self.model.rebuild()


def test_locator_finds_a_drafter_that_can_load_weights():
    inner = FakeInner()
    drafter = FakeDrafter(inner)
    runner = SimpleNamespace(drafter=SimpleNamespace(model=drafter))
    assert drafter_locator.find(runner) is drafter


@pytest.mark.parametrize(
    "runner",
    [
        SimpleNamespace(),
        SimpleNamespace(drafter=None),
        SimpleNamespace(drafter=SimpleNamespace(model=None)),
        SimpleNamespace(drafter=SimpleNamespace(model=object())),
    ],
)
def test_locator_returns_none_without_a_loadable_drafter(runner):
    assert drafter_locator.find(runner) is None


def test_publish_sends_split_names_in_the_serving_dtype():
    arch = make_arch()
    head = head_factory.create(arch)
    drafter = FakeDrafter(FakeInner())
    weight_publisher.publish(drafter, head, arch)

    assert drafter.loaded, "nothing was published"
    assert any("q_proj" in name for name in drafter.loaded)
    assert not any("qkv_proj" in name for name in drafter.loaded)
    assert not any("lm_head" in name for name in drafter.loaded)


def test_publish_restores_derived_buffer_addresses():
    """A captured CUDA graph holds the old pointer, so the rebuilt values have to be
    copied back into the original storage rather than left at a new address."""
    arch = make_arch()
    head = head_factory.create(arch)
    inner = FakeInner()
    drafter = FakeDrafter(inner)

    before = inner._fused_kv_weight
    weight_publisher.publish(drafter, head, arch)

    assert inner._fused_kv_weight is before, "address was not restored"
    assert bool((inner._fused_kv_weight == 1).all()), "values were not updated"


def test_publish_leaves_in_place_aliases_alone():
    """A buffer that is an alias of a parameter is written in place by the load, so
    there is nothing to restore."""
    arch = make_arch()
    head = head_factory.create(arch)
    inner = FakeInner(rebuild_at_new_address=False)
    drafter = FakeDrafter(inner)
    before = inner._fused_kv_weight
    weight_publisher.publish(drafter, head, arch)
    assert inner._fused_kv_weight is before


def test_publish_rejects_a_buffer_that_changed_shape():
    """A shape change means the head and the serving module have diverged
    structurally, and a hot publish can no longer be trusted."""
    arch = make_arch()
    head = head_factory.create(arch)

    class Diverging(FakeInner):
        def rebuild(self) -> None:
            self._fused_kv_weight = torch.ones(8, 4, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="changed from"):
        weight_publisher.publish(FakeDrafter(Diverging()), head, arch)


def test_infer_dtype_reads_the_first_parameter():
    assert weight_publisher.infer_dtype(FakeInner()) == torch.bfloat16


def test_infer_dtype_falls_back_without_parameters():
    assert weight_publisher.infer_dtype(nn.Module()) == torch.bfloat16
