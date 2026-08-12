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


def test_overfits_a_single_batch(
    draft_head, sft_objective, resolved_config, replay_batch
):
    """Loss must fall substantially on a fixed batch with a shared LM head."""
    online_trainer = make_trainer(resolved_config, draft_head, sft_objective)
    first = online_trainer.step(replay_batch)["loss/total"]
    for _ in range(40):
        last = online_trainer.step(replay_batch)["loss/total"]
    assert last < first, f"loss rose from {first} to {last}"
    assert last < 0.75 * first, f"loss barely moved: {first} -> {last}"
    assert online_trainer.step_count == 41


def test_gradient_accumulation_steps_once_per_cycle(
    draft_head, sft_objective, replay_batch
):
    """`grad_accum_steps` changes the optimizer cadence, not the micro cadence."""
    resolved_config = make_config(make_settings(grad_accum_steps=3))
    online_trainer = make_trainer(resolved_config, draft_head, sft_objective)

    stepped = [online_trainer.step(replay_batch)["stepped"] for _ in range(6)]
    assert stepped == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert online_trainer.step_count == 2
    assert online_trainer.micro_step_count == 6


def test_gradients_are_cleared_after_an_optimizer_step(
    draft_head, sft_objective, replay_batch
):
    resolved_config = make_config(make_settings(grad_accum_steps=1))
    online_trainer = make_trainer(resolved_config, draft_head, sft_objective)
    online_trainer.step(replay_batch)
    assert all(
        parameter.grad is None
        for parameter in draft_head.trainable_parameters()
    )


def test_state_dict_round_trips_the_optimizer_too(
    draft_head, sft_objective, resolved_config, replay_batch
):
    """Restoring weights alone restarts AdamW cold and rewinds the warmup, undoing
    part of what the head had learned."""
    online_trainer = make_trainer(resolved_config, draft_head, sft_objective)
    for _ in range(3):
        online_trainer.step(replay_batch)
    state = online_trainer.state_dict()
    assert state["step_count"] == 3

    restored_trainer = make_trainer(resolved_config, draft_head, sft_objective)
    restored_trainer.load_state_dict(state)
    assert restored_trainer.step_count == 3
    assert restored_trainer.micro_step_count == online_trainer.micro_step_count
    assert restored_trainer.optimizer.state_dict()["state"], (
        "optimizer moments were lost"
    )
    # Freezing must survive a reload, or the borrowed tensors start training.
    assert not draft_head.lm_head.weight.requires_grad


def test_restoring_keeps_this_runs_schedule_not_the_checkpoints(
    draft_head, sft_objective, replay_batch
):
    """`LambdaLR.state_dict()` saves the `__dict__` of a callable schedule class and
    `load_state_dict` writes it back.

    Without a rebuild, resuming a 10-step run inside a 10000-step one would restore
    `total_steps=10`, put the cosine past its horizon and pin the learning rate at
    zero -- with no error anywhere.
    """
    short_run_config = make_config(make_settings(total_steps=10))
    online_trainer = make_trainer(short_run_config, draft_head, sft_objective)
    for _ in range(3):
        online_trainer.step(replay_batch)
    state = online_trainer.state_dict()

    long_run_config = make_config(make_settings(total_steps=10_000))
    resumed_trainer = make_trainer(long_run_config, draft_head, sft_objective)
    resumed_trainer.load_state_dict(state)

    assert resumed_trainer.lr_scheduler.lr_lambdas[0].total_steps == 10_000
    assert (
        resumed_trainer.lr_scheduler.last_epoch
        == online_trainer.lr_scheduler.last_epoch
    )

    resumed_trainer.step(replay_batch)
    assert resumed_trainer.optimizer.param_groups[0]["lr"] > 0.0, (
        "the checkpoint's horizon pinned the learning rate at zero"
    )


def test_restoring_keeps_the_step_position(
    draft_head, sft_objective, replay_batch
):
    short_run_config = make_config(make_settings(total_steps=10))
    online_trainer = make_trainer(short_run_config, draft_head, sft_objective)
    for _ in range(4):
        online_trainer.step(replay_batch)

    resumed_trainer = make_trainer(
        make_config(make_settings(total_steps=10)), draft_head, sft_objective
    )
    resumed_trainer.load_state_dict(online_trainer.state_dict())
    assert resumed_trainer.lr_scheduler.last_epoch == 4


def test_step_from_buffer_is_none_when_the_pool_is_thin(
    draft_head, sft_objective, rollout_records
):
    resolved_config = make_config(make_settings(sequences_per_step=4))
    online_trainer = make_trainer(resolved_config, draft_head, sft_objective)
    rollout_buffer = make_buffer(resolved_config)
    assert online_trainer.step_from_buffer(rollout_buffer) is None

    for index, rollout_record in enumerate(rollout_records):
        rollout_buffer.begin(f"r{index}", prompt_len=rollout_record.prompt_len)
        rollout_buffer.add_chunk(
            f"r{index}",
            token_ids=rollout_record.token_ids,
            features=rollout_record.features,
            final_hidden=rollout_record.final_hidden,
        )
        rollout_buffer.finish(f"r{index}")
    assert not rollout_buffer.can_sample(4)
    assert online_trainer.step_from_buffer(rollout_buffer) is None


def test_step_from_buffer_trains_once_the_pool_fills(
    draft_head, sft_objective, rollout_records
):
    resolved_config = make_config(make_settings(sequences_per_step=2))
    online_trainer = make_trainer(resolved_config, draft_head, sft_objective)
    rollout_buffer = make_buffer(resolved_config)
    for index, rollout_record in enumerate(rollout_records):
        rollout_buffer.begin(f"r{index}", prompt_len=rollout_record.prompt_len)
        rollout_buffer.add_chunk(
            f"r{index}",
            token_ids=rollout_record.token_ids,
            features=rollout_record.features,
            final_hidden=rollout_record.final_hidden,
        )
        rollout_buffer.finish(f"r{index}")

    metrics = online_trainer.step_from_buffer(rollout_buffer)
    assert metrics is not None
    assert metrics["count/tokens"] > 0
    # Sampling must not consume: at low load, consuming would starve training
    # exactly when it has the most idle capacity to use.
    assert rollout_buffer.num_rollouts == len(rollout_records)


def test_open_ended_runs_warm_up_then_hold(
    draft_head, sft_objective, replay_batch
):
    resolved_config = make_config(make_settings(total_steps=None))
    online_trainer = make_trainer(resolved_config, draft_head, sft_objective)
    rates = []
    for _ in range(12):
        online_trainer.step(replay_batch)
        rates.append(online_trainer.optimizer.param_groups[0]["lr"])
    assert rates[-1] >= rates[0]
    assert rates[-1] == max(rates)


def test_metrics_carry_the_step_and_the_rate(
    draft_head, sft_objective, resolved_config, replay_batch
):
    online_trainer = make_trainer(resolved_config, draft_head, sft_objective)
    metrics = online_trainer.step(replay_batch)
    assert metrics["step"] == 1.0
    assert metrics["lr"] > 0.0
    assert metrics["seconds"] >= 0.0
    assert "grad_norm" in metrics
    assert online_trainer.last_metrics is metrics


def test_gradients_are_clipped(
    draft_head, sft_objective, replay_batch
):
    resolved_config = make_config(make_settings(grad_clip=1e-6))
    online_trainer = make_trainer(resolved_config, draft_head, sft_objective)
    before = [
        parameter.detach().clone()
        for parameter in draft_head.trainable_parameters()
    ]
    online_trainer.step(replay_batch)
    largest_move = max(
        float((after.detach() - prior).abs().max())
        for prior, after in zip(
            before, draft_head.trainable_parameters(), strict=True
        )
    )
    assert largest_move < 1e-2, "clipping did not bound the update"


def test_the_trainer_draws_its_configured_batch_size(
    draft_head, sft_objective, rollout_records
):
    resolved_config = make_config(make_settings(sequences_per_step=2))
    online_trainer = make_trainer(resolved_config, draft_head, sft_objective)
    assert online_trainer.sequences_per_step == 2
    collated = online_trainer.rollout_collator.collate(rollout_records)
    assert torch.is_tensor(collated.input_ids)
