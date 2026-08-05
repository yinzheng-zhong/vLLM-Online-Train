"""How a training batch is drawn from the pool."""

import random

from tests.conftest import make_record, make_settings
from vllm_online_train.capture.pool import PoolSampler


def records(count: int) -> list:
    return [make_record(40, 2, seed=index) for index in range(count)]


def test_fifo_takes_the_oldest_in_order():
    sampler = PoolSampler(
        make_settings(sample_strategy="fifo").buffer, random.Random(0)
    )
    pool = records(3)
    assert sampler.draw(pool, 2) == pool[:2]


def test_reservoir_draws_without_replacement():
    sampler = PoolSampler(
        make_settings(sample_strategy="reservoir").buffer, random.Random(0)
    )
    drawn = sampler.draw(records(5), 3)
    assert len(drawn) == 3
    assert len({id(record) for record in drawn}) == 3


def test_reservoir_is_seeded():
    """The draw has to be reproducible across runs, or a training curve cannot be
    compared against itself."""
    pool = records(6)
    first = PoolSampler(make_settings().buffer, random.Random(1)).draw(pool, 3)
    second = PoolSampler(make_settings().buffer, random.Random(1)).draw(pool, 3)
    assert [id(r) for r in first] == [id(r) for r in second]


def test_drawing_more_than_the_pool_holds_is_clamped():
    sampler = PoolSampler(make_settings().buffer, random.Random(0))
    assert len(sampler.draw(records(2), 5)) == 2


def test_an_empty_pool_draws_nothing():
    sampler = PoolSampler(make_settings().buffer, random.Random(0))
    assert sampler.draw([], 3) == []
