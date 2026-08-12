"""The runner the capture patch cannot reach.

vLLM ships two model runners. The patch wraps `propose_draft_token_ids` on the V1
`GPUModelRunner`; the V2 runner drafts through a `BaseSpeculator` and never calls it.
Without a guard that combination is silent -- the plugin logs that it installed, the
engine serves and speculates, and no rollout is ever captured -- so what these pin is
that a configured run says so.
"""

from types import SimpleNamespace

import pytest

from vllm_online_train.engine.hook.runner_guard import RunnerGuard
from vllm_online_train.engine.hook.settings_loader import SettingsLoader

CONFIG_ENV = SettingsLoader.CONFIG_ENV


class FakeRunnerModule:
    """A stand-in for `vllm.v1.worker.gpu.model_runner`."""

    def __init__(self) -> None:
        self.GPUModelRunner = self._runner_class()

    @staticmethod
    def _runner_class() -> type:
        class GPUModelRunner:
            def __init__(self, vllm_config, device) -> None:
                self.vllm_config = vllm_config
                self.device = device

        return GPUModelRunner


@pytest.fixture
def runner_module(monkeypatch) -> FakeRunnerModule:
    """A guard pointed at a fake V2 runner module rather than the installed vLLM."""
    fake_module = FakeRunnerModule()
    monkeypatch.setattr(
        RunnerGuard,
        "_runner_class",
        classmethod(lambda cls: fake_module.GPUModelRunner),
    )
    return fake_module


@pytest.fixture
def runner_guard() -> RunnerGuard:
    return RunnerGuard()


def test_reports_once_when_training_is_configured(
    runner_module, runner_guard, monkeypatch, caplog
):
    monkeypatch.setenv(CONFIG_ENV, "./train.json")
    runner_guard.install()

    runner_module.GPUModelRunner(SimpleNamespace(), "cuda:0")
    runner_module.GPUModelRunner(SimpleNamespace(), "cuda:0")

    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(error_records) == 1
    assert RunnerGuard.ENV_VAR in error_records[0].getMessage()


def test_stays_quiet_when_training_is_not_configured(
    runner_module, runner_guard, monkeypatch, caplog
):
    """An installed-but-unconfigured plugin is inert, so a V2 engine is not its
    business to complain about."""
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    runner_guard.install()

    runner_module.GPUModelRunner(SimpleNamespace(), "cuda:0")

    assert [r for r in caplog.records if r.levelname == "ERROR"] == []


def test_the_wrapped_constructor_still_builds_the_runner(
    runner_module, runner_guard, monkeypatch
):
    """The guard reports; it never changes what the engine gets."""
    monkeypatch.setenv(CONFIG_ENV, "./train.json")
    runner_guard.install()

    vllm_config = SimpleNamespace()
    model_runner = runner_module.GPUModelRunner(vllm_config, "cuda:0")

    assert model_runner.vllm_config is vllm_config
    assert model_runner.device == "cuda:0"


def test_install_is_idempotent_and_reversible(runner_module, runner_guard):
    original_init = runner_module.GPUModelRunner.__init__

    assert runner_guard.install() is True
    assert runner_guard.install() is True
    assert runner_guard.installed
    assert runner_module.GPUModelRunner.__init__ is not original_init

    runner_guard.uninstall()

    assert not runner_guard.installed
    assert runner_module.GPUModelRunner.__init__ is original_init


def test_install_is_inert_without_a_v2_runner(runner_guard, monkeypatch):
    """The V2 runner is newer than the oldest vLLM this package runs against."""
    monkeypatch.setattr(RunnerGuard, "_runner_class", classmethod(lambda cls: None))

    assert runner_guard.install() is False
    assert not runner_guard.installed
    runner_guard.uninstall()


def test_the_real_module_path_is_the_one_vllm_uses():
    """A renamed module would make the guard silently stop guarding, which is the
    failure it exists to prevent."""
    import importlib

    importlib.import_module(RunnerGuard.MODULE)
    assert RunnerGuard._runner_class() is not None
