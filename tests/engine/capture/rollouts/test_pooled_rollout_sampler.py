"""How a training batch is drawn from the pool."""

import random

from tests.conftest import make_record, make_settings
from vllm_online_train.engine.capture.rollouts.pooled_rollout_sampler import (
    PooledRolloutSampler,
)


def make_records(count: int) -> list:
    return [make_record(40, 2, seed=index) for index in range(count)]


def test_fifo_takes_the_oldest_in_order():
    pooled_rollout_sampler = PooledRolloutSampler(
        make_settings(sample_strategy="fifo").buffer, random.Random(0)
    )
    pooled_rollouts = make_records(3)
    assert pooled_rollout_sampler.draw(pooled_rollouts, 2) == pooled_rollouts[:2]


def test_reservoir_draws_without_replacement():
    pooled_rollout_sampler = PooledRolloutSampler(
        make_settings(sample_strategy="reservoir").buffer, random.Random(0)
    )
    drawn_records = pooled_rollout_sampler.draw(make_records(5), 3)
    assert len(drawn_records) == 3
    assert len({id(record) for record in drawn_records}) == 3


def test_reservoir_is_seeded():
    """The draw has to be reproducible across runs, or a training curve cannot be
    compared against itself."""
    pooled_rollouts = make_records(6)
    first_draw = PooledRolloutSampler(make_settings().buffer, random.Random(1)).draw(
        pooled_rollouts, 3
    )
    second_draw = PooledRolloutSampler(make_settings().buffer, random.Random(1)).draw(
        pooled_rollouts, 3
    )
    assert [id(record) for record in first_draw] == [
        id(record) for record in second_draw
    ]


def test_drawing_more_than_the_pool_holds_is_clamped():
    pooled_rollout_sampler = PooledRolloutSampler(
        make_settings().buffer, random.Random(0)
    )
    assert len(pooled_rollout_sampler.draw(make_records(2), 5)) == 2


def test_an_empty_pool_draws_nothing():
    pooled_rollout_sampler = PooledRolloutSampler(
        make_settings().buffer, random.Random(0)
    )
    assert pooled_rollout_sampler.draw([], 3) == []


def test_drawing_counts_against_the_cap():
    """Pinned at the exact boundary: a rollout is still drawable on its
    `max_draws_per_rollout`-th draw and retired on the next one."""
    pooled_rollout_sampler = PooledRolloutSampler(
        make_settings(sample_strategy="fifo", max_draws_per_rollout=2).buffer,
        random.Random(0),
    )
    pooled_rollouts = make_records(1)

    assert pooled_rollout_sampler.draw(pooled_rollouts, 1) == pooled_rollouts
    assert pooled_rollouts[0].draws == 1
    assert pooled_rollout_sampler.draw(pooled_rollouts, 1) == pooled_rollouts
    assert pooled_rollouts[0].draws == 2
    assert pooled_rollout_sampler.draw(pooled_rollouts, 1) == []
    assert pooled_rollouts[0].draws == 2


def test_the_cap_shortens_a_draw_rather_than_reusing():
    """A batch is served from whatever is still eligible, not padded out with rollouts
    that have run out of draws."""
    pooled_rollout_sampler = PooledRolloutSampler(
        make_settings(sample_strategy="fifo", max_draws_per_rollout=1).buffer,
        random.Random(0),
    )
    pooled_rollouts = make_records(3)

    assert len(pooled_rollout_sampler.draw(pooled_rollouts, 2)) == 2
    assert len(pooled_rollout_sampler.draw(pooled_rollouts, 2)) == 1
    assert pooled_rollout_sampler.draw(pooled_rollouts, 2) == []
    assert [record.draws for record in pooled_rollouts] == [1, 1, 1]


def test_a_zero_cap_draws_the_same_rollout_forever():
    """The default: at low load, retiring rollouts would starve training exactly when it
    has the most idle capacity to spend."""
    pooled_rollout_sampler = PooledRolloutSampler(
        make_settings(sample_strategy="fifo", max_draws_per_rollout=0).buffer,
        random.Random(0),
    )
    pooled_rollouts = make_records(1)

    for _ in range(50):
        assert pooled_rollout_sampler.draw(pooled_rollouts, 1) == pooled_rollouts
    assert pooled_rollouts[0].draws == 50
