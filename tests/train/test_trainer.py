"""The optimizer loop.

The overfit test is the one that matters: it is the cheapest evidence that the
replay, the objective and the parameter set are wired to each other at all.
"""

import torch

from tests.conftest import (
    make_buffer,
    make_config,
    make_settings,
    make_trainer,
)


def test_overfits_a_single_batch(head, objective, config, batch):
    """Loss must fall substantially on a fixed batch with a shared LM head."""
    trainer = make_trainer(config, head, objective)
    first = trainer.step(batch)["loss/total"]
    for _ in range(40):
        last = trainer.step(batch)["loss/total"]
    assert last < first, f"loss rose from {first} to {last}"
    assert last < 0.75 * first, f"loss barely moved: {first} -> {last}"
    assert trainer.step_count == 41


def test_gradient_accumulation_steps_once_per_cycle(head, objective, batch):
    """`grad_accum_steps` changes the optimizer cadence, not the micro cadence."""
    config = make_config(make_settings(grad_accum_steps=3))
    trainer = make_trainer(config, head, objective)

    stepped = [trainer.step(batch)["stepped"] for _ in range(6)]
    assert stepped == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert trainer.step_count == 2
    assert trainer.micro_step_count == 6


def test_gradients_are_cleared_after_an_optimizer_step(head, objective, batch):
    config = make_config(make_settings(grad_accum_steps=1))
    trainer = make_trainer(config, head, objective)
    trainer.step(batch)
    assert all(p.grad is None for p in head.trainable_parameters())


def test_state_dict_round_trips_the_optimizer_too(head, objective, config, batch):
    """Restoring weights alone restarts AdamW cold and rewinds the warmup, undoing
    part of what the head had learned."""
    trainer = make_trainer(config, head, objective)
    for _ in range(3):
        trainer.step(batch)
    state = trainer.state_dict()
    assert state["step_count"] == 3

    restored = make_trainer(config, head, objective)
    restored.load_state_dict(state)
    assert restored.step_count == 3
    assert restored.micro_step_count == trainer.micro_step_count
    assert restored.optimizer.state_dict()["state"], "optimizer moments were lost"
    # Freezing must survive a reload, or the borrowed tensors start training.
    assert not head.lm_head.weight.requires_grad


def test_restoring_keeps_this_runs_schedule_not_the_checkpoints(head, objective, batch):
    """`LambdaLR.state_dict()` saves the `__dict__` of a callable schedule class and
    `load_state_dict` writes it back.

    Without a rebuild, resuming a 10-step run inside a 10000-step one would restore
    `total_steps=10`, put the cosine past its horizon and pin the learning rate at
    zero -- with no error anywhere.
    """
    short = make_config(make_settings(total_steps=10))
    trainer = make_trainer(short, head, objective)
    for _ in range(3):
        trainer.step(batch)
    state = trainer.state_dict()

    long = make_config(make_settings(total_steps=10_000))
    resumed = make_trainer(long, head, objective)
    resumed.load_state_dict(state)

    assert resumed.scheduler.lr_lambdas[0].total_steps == 10_000
    assert resumed.scheduler.last_epoch == trainer.scheduler.last_epoch

    resumed.step(batch)
    assert resumed.optimizer.param_groups[0]["lr"] > 0.0, (
        "the checkpoint's horizon pinned the learning rate at zero"
    )


def test_restoring_keeps_the_step_position(head, objective, batch):
    short = make_config(make_settings(total_steps=10))
    trainer = make_trainer(short, head, objective)
    for _ in range(4):
        trainer.step(batch)

    resumed = make_trainer(make_config(make_settings(total_steps=10)), head, objective)
    resumed.load_state_dict(trainer.state_dict())
    assert resumed.scheduler.last_epoch == 4


def test_step_from_buffer_is_none_when_the_pool_is_thin(head, objective, records):
    config = make_config(make_settings(sequences_per_step=4))
    trainer = make_trainer(config, head, objective)
    buffer = make_buffer(config)
    assert trainer.step_from_buffer(buffer) is None

    for index, record in enumerate(records):
        buffer.begin(f"r{index}", prompt_len=record.prompt_len)
        buffer.add_chunk(
            f"r{index}",
            token_ids=record.token_ids,
            features=record.features,
            final_hidden=record.final_hidden,
        )
        buffer.finish(f"r{index}")
    assert not buffer.can_sample(4)
    assert trainer.step_from_buffer(buffer) is None


def test_step_from_buffer_trains_once_the_pool_fills(head, objective, records):
    config = make_config(make_settings(sequences_per_step=2))
    trainer = make_trainer(config, head, objective)
    buffer = make_buffer(config)
    for index, record in enumerate(records):
        buffer.begin(f"r{index}", prompt_len=record.prompt_len)
        buffer.add_chunk(
            f"r{index}",
            token_ids=record.token_ids,
            features=record.features,
            final_hidden=record.final_hidden,
        )
        buffer.finish(f"r{index}")

    metrics = trainer.step_from_buffer(buffer)
    assert metrics is not None
    assert metrics["count/tokens"] > 0
    # Sampling must not consume: at low load, consuming would starve training
    # exactly when it has the most idle capacity to use.
    assert buffer.num_rollouts == len(records)


def test_open_ended_runs_warm_up_then_hold(head, objective, batch):
    config = make_config(make_settings(total_steps=None))
    trainer = make_trainer(config, head, objective)
    rates = []
    for _ in range(12):
        trainer.step(batch)
        rates.append(trainer.optimizer.param_groups[0]["lr"])
    assert rates[-1] >= rates[0]
    assert rates[-1] == max(rates)


def test_metrics_carry_the_step_and_the_rate(head, objective, config, batch):
    trainer = make_trainer(config, head, objective)
    metrics = trainer.step(batch)
    assert metrics["step"] == 1.0
    assert metrics["lr"] > 0.0
    assert metrics["seconds"] >= 0.0
    assert "grad_norm" in metrics
    assert trainer.last_metrics is metrics


def test_gradients_are_clipped(head, objective, batch):
    config = make_config(make_settings(grad_clip=1e-6))
    trainer = make_trainer(config, head, objective)
    before = [p.detach().clone() for p in head.trainable_parameters()]
    trainer.step(batch)
    moved = max(
        float((after.detach() - prior).abs().max())
        for prior, after in zip(before, head.trainable_parameters(), strict=True)
    )
    assert moved < 1e-2, "clipping did not bound the update"


def test_the_trainer_draws_its_configured_batch_size(head, objective, records):
    config = make_config(make_settings(sequences_per_step=2))
    trainer = make_trainer(config, head, objective)
    assert trainer.sequences_per_step == 2
    assert torch.is_tensor(trainer.collator.collate(records).input_ids)
