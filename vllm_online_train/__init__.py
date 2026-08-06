"""Train a speculative draft head from live traffic on a serving vLLM engine.

Two halves, split by what they talk to. `engine/` reads the serving engine; `training/`
knows nothing about it and would be the same for a second head architecture:

    engine/          everything that touches vLLM
      hook/          the foothold: the method patch and the settings it loads
      capture/       engine step -> rollout pool -- head-agnostic
      state/         the borrowed target state, on the engine's device or a second one
    training/        everything that trains the head
      collate/       rollout pool -> anchored replay batches -- head-agnostic
      head/          the DFlash module, its layers and its mask geometry -- typed
      checkpoint/    the on-disk format, export and hot publish -- typed
      loss/, optim/  the objective and the optimizer loop
    config/          the operator's settings, sectioned, plus engine-resolved shapes
    contracts/       the structural interfaces the parts depend on
    step.py          one engine step's activations, as the hook hands them over
    assembler.py     the one wiring site for a training session
    cli.py           `online-train-init-head`

`config/` and `contracts/` are the leaf layer both halves depend on, which is what
keeps `register()` off the training stack.

Enable it with two environment variables:

```bash
VLLM_PLUGINS=online_train ONLINE_TRAIN_CONFIG=./train.json vllm serve ...
```

See `README.md` for the operator guide and `CLAUDE.md` for the design notes.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = ["register"]


def register() -> None:
    """`vllm.general_plugins` entry point. Installs the capture hook.

    vLLM invokes this as a bare `func()` from several processes -- `arg_utils`,
    `engine/core`, `worker_base` and `models/registry` -- so it installs one
    idempotent method wrapper and returns. No CUDA, no model, no config parsing, and
    no import of the training stack.
    """
    try:
        from vllm_online_train.engine.hook import capture_hook

        capture_hook.install()
    except Exception:
        # Swallowed: a failed install leaves the serving path untouched.
        logger.exception(
            "vllm-online-train failed to install its capture hook; serving "
            "continues without online training."
        )
