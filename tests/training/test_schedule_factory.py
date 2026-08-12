"""The learning-rate schedules, and which one the settings select."""

import pytest

from tests.conftest import make_settings, schedule_factory
from vllm_online_train.training.optim.cosine_warmup import CosineWithWarmup
from vllm_online_train.training.optim.warmup_only import WarmupOnly


def test_warmup_only_rises_then_holds():
    lr_schedule = WarmupOnly(10)
    assert lr_schedule(0) < lr_schedule(5)
    assert lr_schedule(9) == pytest.approx(1.0)
    assert lr_schedule(50) == 1.0


def test_warmup_only_never_exceeds_one():
    lr_schedule = WarmupOnly(10)
    assert max(lr_schedule(step) for step in range(40)) == 1.0


def test_cosine_peaks_at_the_end_of_warmup():
    lr_schedule = CosineWithWarmup(total_steps=100, warmup_ratio=0.05)
    assert lr_schedule(4) == pytest.approx(1.0)
    assert lr_schedule(3) < 1.0


def test_cosine_decays_to_zero_at_the_horizon():
    lr_schedule = CosineWithWarmup(total_steps=100, warmup_ratio=0.05)
    assert lr_schedule(100) < 1e-6


def test_cosine_is_clamped_past_the_horizon():
    """Overrunning the horizon must hold at zero rather than turn back upwards."""
    lr_schedule = CosineWithWarmup(total_steps=100, warmup_ratio=0.05)
    assert lr_schedule(200) == pytest.approx(lr_schedule(100))


def test_cosine_is_monotone_after_warmup():
    from itertools import pairwise

    lr_schedule = CosineWithWarmup(total_steps=50, warmup_ratio=0.1)
    values = [
        lr_schedule(step) for step in range(lr_schedule.warmup_steps, 50)
    ]
    assert all(earlier >= later for earlier, later in pairwise(values))


def test_a_horizon_selects_the_cosine():
    lr_schedule = schedule_factory.create(make_settings(total_steps=100).optim)
    assert isinstance(lr_schedule, CosineWithWarmup)
    assert lr_schedule.total_steps == 100


def test_no_horizon_selects_warmup_then_flat():
    """There is no final step to decay towards, and decaying against a guessed
    horizon would stop the head learning while traffic keeps arriving."""
    lr_schedule = schedule_factory.create(make_settings(total_steps=None).optim)
    assert isinstance(lr_schedule, WarmupOnly)


def test_open_ended_warmup_scales_with_the_ratio():
    default_steps = schedule_factory.warmup_steps(
        make_settings(warmup_ratio=0.05).optim
    )
    doubled_steps = schedule_factory.warmup_steps(
        make_settings(warmup_ratio=0.10).optim
    )
    assert default_steps == 100
    assert doubled_steps == 200


def test_open_ended_warmup_is_at_least_one_step():
    assert schedule_factory.warmup_steps(make_settings(warmup_ratio=0.0).optim) == 1
