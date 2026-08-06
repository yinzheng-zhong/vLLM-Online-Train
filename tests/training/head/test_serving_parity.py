"""Does what we write load into the real serving module?

This is the load-bearing format test. Everything else in the suite runs against a
32-wide toy head; here a checkpoint from `init_head` is fed to an actual
`DFlashQwen3ForCausalLM` built under a real `VllmConfig`, which exercises
`AutoWeightsLoader`, the `hf_to_vllm_mapper` split-to-fused rewrite, the `skip_substrs`
logic for the omitted shared tensors, and the `+1` into vLLM's aux numbering.

Runs on CPU, but needs the target's config locally, so it skips when `Qwen/Qwen3-4B`
is not cached. Marked `slow`.
"""

import glob
import json

import pytest
import torch

TARGET = "Qwen/Qwen3-4B"
REFERENCE = "z-lab/Qwen3-4B-DFlash-b16"
NUM_LAYERS = 3

pytestmark = pytest.mark.slow


def _reference_snapshot() -> str | None:
    """The published reference checkpoint in the local HF cache, if present."""
    import os

    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub"
    )
    pattern = os.path.join(
        hub,
        f"models--{REFERENCE.replace('/', '--')}",
        "snapshots",
        "*",
        "model.safetensors",
    )
    hits = glob.glob(pattern)
    return hits[0] if hits else None


@pytest.fixture(scope="module")
def target_config():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoConfig.from_pretrained(TARGET, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"{TARGET} config not cached locally: {exc}")


@pytest.fixture(scope="module")
def head_dir(tmp_path_factory, target_config):
    """A real `init_head` checkpoint for the target."""
    from vllm_online_train.cli import main

    out = tmp_path_factory.mktemp("head")
    assert main(
        [
            "--target", TARGET,
            "--out", str(out),
            "--layers", str(NUM_LAYERS),
            "--features", "5",
            "--block-size", "16",
            "--force",
        ]
    ) == 0
    return out


def test_init_head_writes_a_complete_directory(head_dir):
    config = json.loads((head_dir / "config.json").read_text())
    assert config["architectures"] == ["DFlashDraftModel"]
    assert config["num_hidden_layers"] == NUM_LAYERS
    assert config["dflash_config"]["target_layer_ids"] == [1, 9, 17, 25, 33]
    assert config["dflash_config"]["mask_token_id"] == 151669
    assert (head_dir / "model.safetensors").is_file()


def test_matches_the_published_reference_checkpoint(tmp_path, target_config):
    """A 5-layer head must be name-, shape- and dtype-identical to `z-lab`'s.

    That checkpoint is exercised by vLLM's own CI, so structural equality with it is
    the strongest available evidence that this format is the supported one.
    """
    reference = _reference_snapshot()
    if reference is None:
        pytest.skip(f"{REFERENCE} is not cached locally")

    from safetensors import safe_open

    from vllm_online_train.cli import main

    out = tmp_path / "head5"
    assert main(
        ["--target", TARGET, "--out", str(out), "--layers", "5", "--features", "5"]
    ) == 0

    def describe(path):
        with safe_open(str(path), framework="pt") as handle:
            return {
                key: (
                    tuple(handle.get_slice(key).get_shape()),
                    handle.get_slice(key).get_dtype(),
                )
                for key in handle.keys()  # noqa: SIM118 - safe_open is not a mapping
            }

    theirs = describe(reference)
    ours = describe(out / "model.safetensors")
    assert set(ours) == set(theirs), (
        f"only ours: {sorted(set(ours) - set(theirs))}; "
        f"only theirs: {sorted(set(theirs) - set(ours))}"
    )
    for key, expected in theirs.items():
        assert ours[key] == expected, f"{key}: {ours[key]} != {expected}"


def test_checkpoint_loads_into_the_real_serving_module(head_dir):
    """Every trained tensor must land, and only the shared ones must not.

    The perturbed reload is the point. Norm weights start at all-ones in both this
    package's init and vLLM's `RMSNorm`, so comparing against the initial values
    cannot tell "did not load" from "loaded the same value".
    """
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig
    from vllm.config.speculative import SpeculativeConfig
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.models.qwen3_dflash import (
        DFlashQwen3ForCausalLM,
        _get_dflash_fc_input_size,
    )
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    target = ModelConfig(model=TARGET, dtype="bfloat16", max_model_len=2048)
    parallel = ParallelConfig()
    spec = SpeculativeConfig(
        method="dflash",
        model=str(head_dir),
        num_speculative_tokens=15,
        target_model_config=target,
        target_parallel_config=parallel,
    )
    config = VllmConfig(
        model_config=target, parallel_config=parallel, speculative_config=spec
    )

    assert spec.draft_model_config.architectures == ["DFlashDraftModel"]
    # vLLM adds 1 to DFlash's target_layer_ids. Reading the raw config instead of the
    # resolved aux ids captures the wrong layers.
    assert get_eagle3_aux_layers_from_config(spec) == (2, 10, 18, 26, 34)
    assert _get_dflash_fc_input_size(config) == 5 * target.get_hidden_size()

    with set_current_vllm_config(config), torch.device("cpu"):
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:29793",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1)
        except Exception as exc:
            pytest.skip(f"could not initialise a single-rank process group: {exc}")
        model = DFlashQwen3ForCausalLM(vllm_config=config)

    weights = load_file(str(head_dir / "model.safetensors"))
    model.load_weights(list(weights.items()))

    shared = ("lm_head.weight", "model.embed_tokens.weight", "model.mask_embedding")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.load_weights([(n, t * 1.5 + 0.125) for n, t in weights.items()])

    stuck = [
        name
        for name, parameter in model.named_parameters()
        if torch.equal(before[name], parameter.detach())
        and not any(name.endswith(s) for s in shared)
    ]
    assert not stuck, f"tensors not wired to the checkpoint: {stuck}"

    # The serving path binds the target's modules onto the draft, which is the
    # sharing the objective assumes.
    for name in shared:
        parameter = dict(model.named_parameters()).get(name)
        if parameter is not None:
            assert torch.equal(before[name], parameter.detach()), (
                f"{name} was overwritten; it must come from the target"
            )

    # Publishing depends on this rebuild happening inside load_weights.
    inner = model.model
    model.load_weights(list(weights.items()))
    q_size = inner.layers[0].self_attn.q_size
    torch.testing.assert_close(
        inner._fused_kv_weight,
        torch.cat(
            [layer.self_attn.qkv_proj.weight[q_size:] for layer in inner.layers], dim=0
        ),
    )
    assert inner._k_norm_weights.shape[0] == NUM_LAYERS
