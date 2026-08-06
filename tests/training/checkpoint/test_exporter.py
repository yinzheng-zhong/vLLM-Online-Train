"""Writing a checkpoint directory a restarted engine can serve."""

import json

import torch

from tests.conftest import checkpoint_writer, head_factory, head_weights, make_arch
from vllm_online_train.training.checkpoint.exporter import CheckpointExporter


def test_the_served_config_is_copied_verbatim(tmp_path):
    """Restart-to-adopt cannot be subtly wrong: the exported directory must differ
    from the running head in no field except the weights."""
    source = tmp_path / "served"
    source.mkdir()
    config = {"model_type": "qwen3", "num_hidden_layers": 3, "marker": "original"}
    (source / checkpoint_writer.CONFIG_FILENAME).write_text(json.dumps(config))

    out = tmp_path / "exported"
    exporter = CheckpointExporter(checkpoint_writer, head_weights, source)
    written = exporter.export(
        head_factory.create(make_arch()), out, torch.bfloat16
    )

    assert (out / checkpoint_writer.WEIGHTS_FILENAME).is_file()
    assert written.endswith(checkpoint_writer.WEIGHTS_FILENAME)
    copied = json.loads((out / checkpoint_writer.CONFIG_FILENAME).read_text())
    assert copied == config


def test_a_missing_source_config_still_writes_weights(tmp_path):
    """The operator can copy one in; losing the weights would be worse."""
    exporter = CheckpointExporter(
        checkpoint_writer, head_weights, tmp_path / "absent"
    )
    out = tmp_path / "exported"
    exporter.export(head_factory.create(make_arch()), out, torch.bfloat16)
    assert (out / checkpoint_writer.WEIGHTS_FILENAME).is_file()
    assert not (out / checkpoint_writer.CONFIG_FILENAME).exists()


def test_weights_go_out_at_the_serving_dtype(tmp_path):
    """The trainable head is fp32; a checkpoint at that precision is twice the size
    it needs to be."""
    from safetensors.torch import load_file

    exporter = CheckpointExporter(
        checkpoint_writer, head_weights, tmp_path / "absent"
    )
    written = exporter.export(
        head_factory.create(make_arch()), tmp_path / "exported", torch.bfloat16
    )
    loaded = load_file(written)
    assert loaded
    assert all(tensor.dtype == torch.bfloat16 for tensor in loaded.values())


def test_the_shared_tensors_are_still_omitted(tmp_path):
    """The serving path binds the target's own modules when they are absent."""
    from safetensors.torch import load_file

    exporter = CheckpointExporter(
        checkpoint_writer, head_weights, tmp_path / "absent"
    )
    written = exporter.export(
        head_factory.create(make_arch()), tmp_path / "exported", torch.bfloat16
    )
    names = set(load_file(written))
    assert not any("lm_head" in name or "embed_tokens" in name for name in names)


def test_the_destination_is_created(tmp_path):
    exporter = CheckpointExporter(
        checkpoint_writer, head_weights, tmp_path / "absent"
    )
    out = tmp_path / "a" / "b" / "c"
    exporter.export(head_factory.create(make_arch()), out, torch.bfloat16)
    assert out.is_dir()
