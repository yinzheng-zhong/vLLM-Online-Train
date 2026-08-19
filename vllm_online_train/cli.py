"""Materialise a randomly-initialised DFlash checkpoint.

vLLM has no random-init path: `method: "dflash"` resolves through
`speculative_config.model`, which must point at a real directory. This writes one.

```bash
python -m vllm_online_train.cli \
    --target Qwen/Qwen3-4B --out ./heads/qwen3-4b-v0 --layers 3
```
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from vllm_online_train.logger import init_logger
from vllm_online_train.training.checkpoint.draft_config import DraftConfigBuilder
from vllm_online_train.training.checkpoint.init_plan import (
    InitHeadPlanner,
    InitHeadRequest,
)
from vllm_online_train.training.checkpoint.weight_names import WeightNameRewriter
from vllm_online_train.training.checkpoint.writer import CheckpointWriter
from vllm_online_train.training.head.arch import ArchFactory
from vllm_online_train.training.head.block_masks import BlockMaskBuilder
from vllm_online_train.training.head.factory import HeadFactory
from vllm_online_train.training.head.feature_layers import TargetLayerPlanner
from vllm_online_train.training.head.weights import HeadWeights

logger = init_logger(__name__)

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments to parse. `None` reads `sys.argv`.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="online-train-init-head",
        description="Write a randomly-initialised DFlash draft checkpoint.",
    )
    parser.add_argument("--target", required=True, help="Target model id or path.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--layers", type=int, default=5, help="Draft layers (default 5, as DFlash A.1)."
    )
    parser.add_argument(
        "--features",
        type=int,
        default=5,
        help="Target feature layers. Ignored when --target-layer-ids is given. Each "
        "feature costs hidden_size per token in fc width and in capture bytes.",
    )
    parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit DFlash-numbered feature layers, e.g. 1 9 17 25 33.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="K: query slots per block. Serve with --num-speculative-tokens K-1.",
    )
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing checkpoint in --out.",
    )
    return parser.parse_args(argv)


def to_request(parsed_args: argparse.Namespace) -> InitHeadRequest:
    """Turn the parsed namespace into the typed request `head/` consumes.

    Args:
        parsed_args: The parsed command line.

    Returns:
        The request.
    """
    return InitHeadRequest(
        target=parsed_args.target,
        layers=parsed_args.layers,
        features=parsed_args.features,
        block_size=parsed_args.block_size,
        target_layer_ids=parsed_args.target_layer_ids,
        mask_token_id=parsed_args.mask_token_id,
        hidden_size=parsed_args.hidden_size,
        intermediate_size=parsed_args.intermediate_size,
        num_heads=parsed_args.num_heads,
        num_kv_heads=parsed_args.num_kv_heads,
        head_dim=parsed_args.head_dim,
        rope_theta=parsed_args.rope_theta,
    )


def main(argv: list[str] | None = None) -> int:
    """Write the checkpoint and print a summary.

    Args:
        argv: Arguments to parse. `None` reads `sys.argv`.

    Returns:
        The process exit code.

    Raises:
        SystemExit: On an invalid argument, an existing checkpoint without --force,
            or a target the head cannot be built against.
    """
    parsed_args = parse_args(argv)
    if parsed_args.layers < 1:
        raise SystemExit(f"--layers must be >= 1, got {parsed_args.layers}")
    if parsed_args.block_size < 2:
        raise SystemExit(
            f"--block-size must be >= 2 so a block has at least one predicted slot "
            f"beside the anchor, got {parsed_args.block_size}"
        )

    head_weights = HeadWeights()
    checkpoint_writer = CheckpointWriter(WeightNameRewriter())
    head_factory = HeadFactory(BlockMaskBuilder(), head_weights)
    init_head_planner = InitHeadPlanner(TargetLayerPlanner(), ArchFactory())

    out_dir = Path(parsed_args.out)
    weights_path = out_dir / checkpoint_writer.WEIGHTS_FILENAME
    if weights_path.exists() and not parsed_args.force:
        raise SystemExit(
            f"{out_dir} already holds a checkpoint; pass --force to replace."
        )

    try:
        init_head_plan = init_head_planner.plan(to_request(parsed_args))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    logger.debug(
        "resolved plan: %d draft layers, target_layer_ids=%s of %d target layers",
        init_head_plan.head_arch.num_layers,
        init_head_plan.target_layer_ids,
        init_head_plan.num_target_layers,
    )

    head_arch = init_head_plan.head_arch
    target_config = init_head_plan.target_config
    draft_config = DraftConfigBuilder().build(
        head_arch,
        target_layer_ids=init_head_plan.target_layer_ids,
        num_target_layers=init_head_plan.num_target_layers,
        dtype=parsed_args.dtype,
        max_position_embeddings=int(
            getattr(target_config, "max_position_embeddings", None) or 40960
        ),
        bos_token_id=getattr(target_config, "bos_token_id", None),
        eos_token_id=getattr(target_config, "eos_token_id", None),
    )

    generator = torch.Generator().manual_seed(parsed_args.seed)
    draft_head = head_factory.create(
        head_arch, generator=generator, borrowed_weight_device="meta"
    )

    logger.debug(
        "writing checkpoint to %s at dtype=%s", out_dir, parsed_args.dtype
    )
    checkpoint_writer.write(
        out_dir,
        head_weights.export(draft_head),
        head_arch,
        config=draft_config,
        dtype=DTYPES[parsed_args.dtype],
    )

    num_trained_params = sum(
        parameter.numel() for parameter in draft_head.trainable_parameters()
    )
    print(
        f"Wrote {out_dir}\n"
        f"  draft layers      {head_arch.num_layers}\n"
        f"  feature layers    {head_arch.num_features} "
        f"(DFlash ids {init_head_plan.target_layer_ids}, "
        f"aux ids {[i + 1 for i in init_head_plan.target_layer_ids]})\n"
        f"  fc width          {head_arch.feature_width} -> {head_arch.hidden_size}\n"
        f"  block size        {head_arch.block_size} "
        f"(serve with num_speculative_tokens={head_arch.block_size - 1})\n"
        f"  mask_token_id     {head_arch.mask_token_id}\n"
        f"  trained params    {num_trained_params / 1e6:.1f}M\n"
        f"  capture cost      "
        f"{(head_arch.num_features + 1) * head_arch.hidden_size * 2 / 1024:.0f} "
        "KiB/token"
    )
    print(json.dumps(draft_config["dflash_config"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
