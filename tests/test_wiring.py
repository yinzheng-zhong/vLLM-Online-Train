"""The wiring sites.

Construction happens in exactly three places -- `assembler.py` for a training
session, `hook/__init__.py` for the plugin install path, and `cli.py` for the offline
tool. What this pins is that the assembler shares the instances that must agree, that
the settings layer stays leaf-level, that the hook is wired once, and that the install
path carries none of the training stack with it.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

from vllm_online_train.assembler import SessionAssembler


@pytest.fixture(scope="module")
def assembler() -> SessionAssembler:
    return SessionAssembler()


def test_the_mask_builder_is_shared_across_packages(assembler):
    """The head's attention geometry and the objective's teacher gather must agree
    on where a block's slots sit."""
    assert assembler.head_factory.masks is assembler.masks
    assert assembler.teacher.positions is assembler.masks


def test_the_checkpoint_writer_and_publisher_share_a_naming_rewriter(assembler):
    """A checkpoint and a hot publish that named tensors differently would make one
    of the two paths untested by the other."""
    assert assembler.writer.naming is assembler.naming
    assert assembler.publisher.naming is assembler.naming


def test_the_head_factory_and_publisher_share_a_weight_helper(assembler):
    """Loading and exporting the borrowed tensors is one job, so a second helper
    would be a second place for the freeze rules to drift."""
    assert assembler.head_factory.weights is assembler.weights
    assert assembler.publisher.weights is assembler.weights


@pytest.mark.parametrize(
    "module", ["vllm_online_train.config", "vllm_online_train.contracts"]
)
def test_the_settings_layer_stays_leaf_level(module):
    """`config/` and `contracts/` are what everything else depends on, so they may
    only reach each other, `step.py` and `logger.py`."""
    allowed = (
        "vllm_online_train.config",
        "vllm_online_train.contracts",
        "vllm_online_train.logger",
        "vllm_online_train.step",
    )
    spec = importlib.util.find_spec(module)
    paths = list(Path(spec.submodule_search_locations[0]).rglob("*.py"))

    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                if name.startswith("vllm_online_train."):
                    assert name.startswith(allowed), (
                        f"{path} imports {name}, which is outside the settings layer"
                    )


def test_the_hook_is_wired_once_and_shared():
    """`register()` is called from several vLLM processes, so the install must go
    through one instance rather than a fresh one per call."""
    import vllm_online_train.engine.hook as hook_package
    from vllm_online_train.engine.hook import capture_hook

    assert hook_package.capture_hook is capture_hook
    assert capture_hook.loader.factory is not None
    assert capture_hook.patcher.guard is not None


def module_scope_imports(tree: ast.Module) -> list[str]:
    """Collect what a module imports at the moment it is first imported.

    Args:
        tree: The parsed module.

    Returns:
        The imported module names, excluding any inside a function body.
    """
    names: list[str] = []
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        pending.extend(ast.iter_child_nodes(node))
    return names


def test_the_install_path_does_not_import_the_training_stack():
    """`register()` imports `hook/` in every vLLM process that loads the plugin. The
    training stack belongs behind the first usable step, not in that import."""
    deferred = (
        "vllm_online_train.assembler",
        "vllm_online_train.engine.capture",
        "vllm_online_train.engine.state.engine",
        "vllm_online_train.training",
    )
    spec = importlib.util.find_spec("vllm_online_train.engine.hook")
    paths = list(Path(spec.submodule_search_locations[0]).rglob("*.py"))

    for path in paths:
        tree = ast.parse(path.read_text())
        for name in module_scope_imports(tree):
            assert not name.startswith(deferred), (
                f"{path} imports {name} at module scope, which `register()` would "
                "pull into every engine process"
            )
