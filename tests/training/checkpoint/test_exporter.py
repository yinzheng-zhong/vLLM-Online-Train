"""Writing a checkpoint directory a restarted engine can serve."""

import json

import torch

from tests.conftest import checkpoint_writer, head_factory, head_weights, make_arch
from vllm_online_train.training.checkpoint.exporter import CheckpointExporter


def test_the_served_config_is_copied_verbatim(tmp_path):
    """Restart-to-adopt cannot be subtly wrong: the exported directory must differ
    from the running head in no field except the weights."""
    source_dir = tmp_path / "served"
    source_dir.mkdir()
    draft_config = {
        "model_type": "qwen3",
        "num_hidden_layers": 3,
        "marker": "original",
    }
    (source_dir / checkpoint_writer.CONFIG_FILENAME).write_text(
        json.dumps(draft_config)
    )

    out_dir = tmp_path / "exported"
    checkpoint_exporter = CheckpointExporter(
        checkpoint_writer, head_weights, source_dir
    )
    written = checkpoint_exporter.export(
        head_factory.create(make_arch()), out_dir, torch.bfloat16
    )

    assert (out_dir / checkpoint_writer.WEIGHTS_FILENAME).is_file()
    assert written.endswith(checkpoint_writer.WEIGHTS_FILENAME)
    copied = json.loads((out_dir / checkpoint_writer.CONFIG_FILENAME).read_text())
    assert copied == draft_config


def test_a_missing_source_config_still_writes_weights(tmp_path):
    """The operator can copy one in; losing the weights would be worse."""
    checkpoint_exporter = CheckpointExporter(
        checkpoint_writer, head_weights, tmp_path / "absent"
    )
    out_dir = tmp_path / "exported"
    checkpoint_exporter.export(
        head_factory.create(make_arch()), out_dir, torch.bfloat16
    )
    assert (out_dir / checkpoint_writer.WEIGHTS_FILENAME).is_file()
    assert not (out_dir / checkpoint_writer.CONFIG_FILENAME).exists()


def test_weights_go_out_at_the_serving_dtype(tmp_path):
    """The trainable head is fp32; a checkpoint at that precision is twice the size
    it needs to be."""
    from safetensors.torch import load_file

    checkpoint_exporter = CheckpointExporter(
        checkpoint_writer, head_weights, tmp_path / "absent"
    )
    written = checkpoint_exporter.export(
        head_factory.create(make_arch()), tmp_path / "exported", torch.bfloat16
    )
    loaded = load_file(written)
    assert loaded
    assert all(tensor.dtype == torch.bfloat16 for tensor in loaded.values())


def test_the_shared_tensors_are_still_omitted(tmp_path):
    """The serving path binds the target's own modules when they are absent."""
    from safetensors.torch import load_file

    checkpoint_exporter = CheckpointExporter(
        checkpoint_writer, head_weights, tmp_path / "absent"
    )
    written = checkpoint_exporter.export(
        head_factory.create(make_arch()), tmp_path / "exported", torch.bfloat16
    )
    written_names = set(load_file(written))
    assert not any(
        "lm_head" in name or "embed_tokens" in name for name in written_names
    )


def test_the_destination_is_created(tmp_path):
    checkpoint_exporter = CheckpointExporter(
        checkpoint_writer, head_weights, tmp_path / "absent"
    )
    out_dir = tmp_path / "a" / "b" / "c"
    checkpoint_exporter.export(
        head_factory.create(make_arch()), out_dir, torch.bfloat16
    )
    assert out_dir.is_dir()
