"""How a training batch is drawn from the pool."""

import random

from tests.conftest import make_record, make_settings
from vllm_online_train.engine.capture.pool_sampler import PoolSampler


def make_records(count: int) -> list:
    return [make_record(40, 2, seed=index) for index in range(count)]


def test_fifo_takes_the_oldest_in_order():
    pool_sampler = PoolSampler(
        make_settings(sample_strategy="fifo").buffer, random.Random(0)
    )
    pooled_rollouts = make_records(3)
    assert pool_sampler.draw(pooled_rollouts, 2) == pooled_rollouts[:2]


def test_reservoir_draws_without_replacement():
    pool_sampler = PoolSampler(
        make_settings(sample_strategy="reservoir").buffer, random.Random(0)
    )
    drawn_records = pool_sampler.draw(make_records(5), 3)
    assert len(drawn_records) == 3
    assert len({id(record) for record in drawn_records}) == 3


def test_reservoir_is_seeded():
    """The draw has to be reproducible across runs, or a training curve cannot be
    compared against itself."""
    pooled_rollouts = make_records(6)
    first_draw = PoolSampler(make_settings().buffer, random.Random(1)).draw(
        pooled_rollouts, 3
    )
    second_draw = PoolSampler(make_settings().buffer, random.Random(1)).draw(
        pooled_rollouts, 3
    )
    assert [id(record) for record in first_draw] == [
        id(record) for record in second_draw
    ]


def test_drawing_more_than_the_pool_holds_is_clamped():
    pool_sampler = PoolSampler(make_settings().buffer, random.Random(0))
    assert len(pool_sampler.draw(make_records(2), 5)) == 2


def test_an_empty_pool_draws_nothing():
    pool_sampler = PoolSampler(make_settings().buffer, random.Random(0))
    assert pool_sampler.draw([], 3) == []
