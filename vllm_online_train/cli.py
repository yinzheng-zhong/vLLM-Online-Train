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
from vllm.logger import init_logger

from vllm_online_train.checkpoint.config import DraftConfigBuilder
from vllm_online_train.checkpoint.layers import TargetLayerPlanner
from vllm_online_train.checkpoint.naming import WeightNameRewriter
from vllm_online_train.checkpoint.plan import InitHeadPlanner, InitHeadRequest
from vllm_online_train.checkpoint.writer import CheckpointWriter
from vllm_online_train.head.arch import ArchFactory
from vllm_online_train.head.factory import HeadFactory
from vllm_online_train.head.masks import BlockMaskBuilder
from vllm_online_train.head.weights import HeadWeights

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


def to_request(args: argparse.Namespace) -> InitHeadRequest:
    """Turn the parsed namespace into the typed request `head/` consumes.

    Args:
        args: The parsed command line.

    Returns:
        The request.
    """
    return InitHeadRequest(
        target=args.target,
        layers=args.layers,
        features=args.features,
        block_size=args.block_size,
        target_layer_ids=args.target_layer_ids,
        mask_token_id=args.mask_token_id,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        rope_theta=args.rope_theta,
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
    args = parse_args(argv)
    if args.layers < 1:
        raise SystemExit(f"--layers must be >= 1, got {args.layers}")
    if args.block_size < 2:
        raise SystemExit(
            f"--block-size must be >= 2 so a block has at least one predicted slot "
            f"beside the anchor, got {args.block_size}"
        )

    weights = HeadWeights()
    writer = CheckpointWriter(WeightNameRewriter())
    factory = HeadFactory(BlockMaskBuilder(), weights)
    planner = InitHeadPlanner(TargetLayerPlanner(), ArchFactory())

    out = Path(args.out)
    if (out / writer.WEIGHTS_FILENAME).exists() and not args.force:
        raise SystemExit(f"{out} already holds a checkpoint; pass --force to replace.")

    try:
        plan = planner.plan(to_request(args))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    logger.debug(
        "resolved plan: %d draft layers, target_layer_ids=%s of %d target layers",
        plan.arch.num_layers,
        plan.target_layer_ids,
        plan.num_target_layers,
    )

    target = plan.target_config
    config = DraftConfigBuilder().build(
        plan.arch,
        target_layer_ids=plan.target_layer_ids,
        num_target_layers=plan.num_target_layers,
        dtype=args.dtype,
        max_position_embeddings=int(
            getattr(target, "max_position_embeddings", None) or 40960
        ),
        bos_token_id=getattr(target, "bos_token_id", None),
        eos_token_id=getattr(target, "eos_token_id", None),
    )

    generator = torch.Generator().manual_seed(args.seed)
    head = factory.create(plan.arch, generator=generator)

    logger.debug("writing checkpoint to %s at dtype=%s", out, args.dtype)
    writer.write(
        out,
        weights.export(head),
        plan.arch,
        config=config,
        dtype=DTYPES[args.dtype],
    )

    arch = plan.arch
    trained = sum(p.numel() for p in head.trainable_parameters())
    print(
        f"Wrote {out}\n"
        f"  draft layers      {arch.num_layers}\n"
        f"  feature layers    {arch.num_features} (DFlash ids {plan.target_layer_ids}, "
        f"aux ids {[i + 1 for i in plan.target_layer_ids]})\n"
        f"  fc width          {arch.feature_width} -> {arch.hidden_size}\n"
        f"  block size        {arch.block_size} "
        f"(serve with num_speculative_tokens={arch.block_size - 1})\n"
        f"  mask_token_id     {arch.mask_token_id}\n"
        f"  trained params    {trained / 1e6:.1f}M\n"
        f"  capture cost      "
        f"{(arch.num_features + 1) * arch.hidden_size * 2 / 1024:.0f} KiB/token"
    )
    print(json.dumps(config["dflash_config"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
