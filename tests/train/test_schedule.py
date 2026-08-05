"""The learning-rate schedules, and which one the settings select."""

import pytest

from tests.conftest import make_settings, schedule_factory
from vllm_online_train.train.optim.cosine import CosineWithWarmup
from vllm_online_train.train.optim.flat import WarmupOnly


def test_warmup_only_rises_then_holds():
    schedule = WarmupOnly(10)
    assert schedule(0) < schedule(5)
    assert schedule(9) == pytest.approx(1.0)
    assert schedule(50) == 1.0


def test_warmup_only_never_exceeds_one():
    schedule = WarmupOnly(10)
    assert max(schedule(step) for step in range(40)) == 1.0


def test_cosine_peaks_at_the_end_of_warmup():
    schedule = CosineWithWarmup(total_steps=100, warmup_ratio=0.05)
    assert schedule(4) == pytest.approx(1.0)
    assert schedule(3) < 1.0


def test_cosine_decays_to_zero_at_the_horizon():
    schedule = CosineWithWarmup(total_steps=100, warmup_ratio=0.05)
    assert schedule(100) < 1e-6


def test_cosine_is_clamped_past_the_horizon():
    """Overrunning the horizon must hold at zero rather than turn back upwards."""
    schedule = CosineWithWarmup(total_steps=100, warmup_ratio=0.05)
    assert schedule(200) == pytest.approx(schedule(100))


def test_cosine_is_monotone_after_warmup():
    from itertools import pairwise

    schedule = CosineWithWarmup(total_steps=50, warmup_ratio=0.1)
    values = [schedule(step) for step in range(schedule.warmup_steps, 50)]
    assert all(a >= b for a, b in pairwise(values))


def test_a_horizon_selects_the_cosine():
    schedule = schedule_factory.create(make_settings(total_steps=100).optim)
    assert isinstance(schedule, CosineWithWarmup)
    assert schedule.total_steps == 100


def test_no_horizon_selects_warmup_then_flat():
    """There is no final step to decay towards, and decaying against a guessed
    horizon would stop the head learning while traffic keeps arriving."""
    schedule = schedule_factory.create(make_settings(total_steps=None).optim)
    assert isinstance(schedule, WarmupOnly)


def test_open_ended_warmup_scales_with_the_ratio():
    default = schedule_factory.warmup_steps(make_settings(warmup_ratio=0.05).optim)
    doubled = schedule_factory.warmup_steps(make_settings(warmup_ratio=0.10).optim)
    assert default == 100
    assert doubled == 200


def test_open_ended_warmup_is_at_least_one_step():
    assert schedule_factory.warmup_steps(make_settings(warmup_ratio=0.0).optim) == 1
