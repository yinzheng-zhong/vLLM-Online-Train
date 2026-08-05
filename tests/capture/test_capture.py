"""Which activations capture is allowed to keep, and which requests it opens.

The valid-position rule is the subtlest thing in the subsystem. In a verify step with
`r` of `num_draft` tokens accepted, the activations at block position `i` are
conditioned on `[..., x, d_1..d_i]`. For `i <= r` that prefix is the true sequence;
for `i > r` it contains a rejected token, so those activations describe a state the
model was never really in.

The other rule pinned here is which thread may drive the staging buffer.
`RolloutCapture` itself needs a `FeatureStager`, which needs CUDA, so that ownership is
checked structurally rather than by running two threads at it.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from tests.conftest import make_settings
from vllm_online_train.capture.positions import ValidPositions
from vllm_online_train.capture.sampler import RequestSampler

valid_positions = ValidPositions()

STAGING_PROTOCOL = frozenset({"reserve", "commit", "drain", "_drain"})
"""`FeatureStager`'s driving methods, plus the capture's own wrapper."""


def test_nothing_outside_the_capture_drives_the_staging_buffer():
    """One staging buffer, one set of reservations, and a trainer thread holding the
    same `RolloutCapture`. A drain from that thread can land between a `reserve` and its
    `commit`, which clears the reservations that step made: the tokens never reach the
    buffer, the rollout carries a hole across the gap, and every anchor after it is
    scored against the wrong position. Nothing downstream can detect that."""
    spec = importlib.util.find_spec("vllm_online_train")
    package = Path(spec.submodule_search_locations[0])

    for path in package.rglob("*.py"):
        if path.parent.name == "capture":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            called = getattr(node.func, "attr", None)
            assert called not in STAGING_PROTOCOL, (
                f"{path.relative_to(package)} calls .{called}(); the staging protocol "
                "runs on the engine thread only, inside RolloutCapture.observe"
            )


def test_prefill_chunk_keeps_every_position():
    """No speculation ran, so every scheduled position sits on real tokens."""
    assert valid_positions.count(scheduled=512, num_draft=0, num_valid=0) == 512


def test_plain_decode_keeps_its_single_position():
    assert valid_positions.count(scheduled=1, num_draft=0, num_valid=1) == 1


def test_full_rejection_keeps_only_the_anchor_position():
    """`r = 0` still leaves one usable position: the activations that produced the
    correction token, conditioned on accepted tokens only."""
    assert valid_positions.count(scheduled=16, num_draft=15, num_valid=1) == 1


def test_partial_acceptance_keeps_r_plus_one():
    """`num_valid` is the count of non-(-1) sampled entries, which is `r + 1`."""
    assert valid_positions.count(scheduled=16, num_draft=15, num_valid=8) == 8


def test_never_keeps_more_than_was_scheduled():
    assert valid_positions.count(scheduled=4, num_draft=15, num_valid=16) == 4


def test_draft_counts_default_to_zero_without_metadata():
    assert valid_positions.draft_counts(None, 3) == [0, 0, 0]


def test_draft_counts_are_padded_to_the_batch():
    """A short metadata list must not silently shift counts onto the wrong request."""
    metadata = SimpleNamespace(num_draft_tokens=[4, 4])
    assert valid_positions.draft_counts(metadata, 4) == [4, 4, 0, 0]


def test_full_sample_rate_accepts_everything():
    sampler = RequestSampler(make_settings(capture_sample_rate=1.0).capture, seed=42)
    assert all(sampler.accepts(f"req-{i}") for i in range(50))


def test_the_decision_is_stable_for_one_request():
    """A chunked prefill asks repeatedly and must get the same answer each time,
    or a rollout ends up with a hole in the middle."""
    sampler = RequestSampler(make_settings(capture_sample_rate=0.5).capture, seed=42)
    first = sampler.accepts("req-7")
    assert all(sampler.accepts("req-7") is first for _ in range(10))


def test_the_decision_does_not_depend_on_the_process():
    """Python's `hash` is salted per process, which would make capture depend on
    which worker saw the request."""
    a = RequestSampler(make_settings(capture_sample_rate=0.5).capture, seed=42)
    b = RequestSampler(make_settings(capture_sample_rate=0.5).capture, seed=42)
    ids = [f"req-{i}" for i in range(64)]
    assert [a.accepts(i) for i in ids] == [b.accepts(i) for i in ids]


def test_the_seed_changes_which_requests_are_kept():
    ids = [f"req-{i}" for i in range(200)]
    a = RequestSampler(make_settings(capture_sample_rate=0.5).capture, seed=1)
    b = RequestSampler(make_settings(capture_sample_rate=0.5).capture, seed=2)
    assert [a.accepts(i) for i in ids] != [b.accepts(i) for i in ids]


def test_the_sample_rate_is_approximately_honoured():
    sampler = RequestSampler(make_settings(capture_sample_rate=0.25).capture, seed=42)
    ids = [f"req-{i}" for i in range(2000)]
    kept = sum(sampler.accepts(i) for i in ids)
    assert 0.20 < kept / len(ids) < 0.30


def test_the_hash_lands_in_the_unit_interval():
    values = [RequestSampler.stable_unit_hash(f"r{i}", 3) for i in range(100)]
    assert all(0.0 <= value < 1.0 for value in values)
