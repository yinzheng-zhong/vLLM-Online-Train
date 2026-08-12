from typing import TYPE_CHECKING

from vllm_online_train.config.settings import CaptureSettings
from vllm_online_train.engine.capture.feature_stager import FeatureStager
from vllm_online_train.engine.capture.request_sampler import RequestSampler
from vllm_online_train.engine.capture.rollout_buffer import RolloutBuffer
from vllm_online_train.engine.capture.valid_positions import ValidPositions
from vllm_online_train.logger import init_logger
from vllm_online_train.step import EngineStep

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class RolloutCapture:
    def __init__(
        self,
        capture_settings: CaptureSettings,
        rollout_buffer: RolloutBuffer,
        feature_stager: FeatureStager,
        request_sampler: RequestSampler,
        valid_positions: ValidPositions,
    ) -> None:
        """Tees target activations from the engine step into a `RolloutBuffer`.

        The staging protocol -- drain, reserve, commit -- runs entirely on the engine
        thread inside `observe`, in that order and without interleaving. The trainer
        thread reaches this object only through `as_metrics`.

        Args:
            capture_settings: Which requests are eligible.
            rollout_buffer: Receives the drained chunks.
            feature_stager: Owns the staging buffer and the device-to-host copy.
            request_sampler: Decides per request whether to capture at all.
            valid_positions: Decides how many positions of a request are usable.
        """
        self.capture_settings = capture_settings
        self.rollout_buffer = rollout_buffer
        self.feature_stager = feature_stager
        self.request_sampler = request_sampler
        self.valid_positions = valid_positions

        self._seen_req_ids: set[str] = set()
        self.skipped_staging_full = 0
        self.skipped_sample_rate = 0

    def observe(self, engine_step: EngineStep) -> None:
        """Capture this step's usable activations.

        Args:
            engine_step: The engine step's schedule and activations.

        Raises:
            RuntimeError: If the flat batch does not lay each request's scheduled
                tokens out contiguously in `input_batch` order.
        """
        # Drains first: the previous step's copy has had a whole engine step to
        # complete, and there is a single staging buffer to write into. A request the
        # engine reports finished below was not scheduled this step, so its last chunk
        # is in this drain.
        self._drain()
        scheduler_output = engine_step.scheduler_output
        self._retire_finished(scheduler_output)
        self._register_new(scheduler_output)

        if not self._seen_req_ids:
            return
        req_ids = engine_step.input_batch.req_ids
        num_reqs = len(req_ids)
        if num_reqs == 0:
            return

        draft_counts = self.valid_positions.draft_counts(
            engine_step.spec_decode_metadata, num_reqs
        )
        # One sync for the whole batch: the counts are needed on the host to decide
        # slice lengths at all.
        valid_counts = (engine_step.sampled_token_ids != -1).sum(dim=1).tolist()

        offset = 0
        for req_index, req_id in enumerate(req_ids):
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            start = offset
            offset += num_scheduled_tokens
            if req_id not in self._seen_req_ids:
                continue

            num_usable_positions = self.valid_positions.count(
                num_scheduled_tokens=num_scheduled_tokens,
                num_draft_tokens=draft_counts[req_index],
                num_valid_tokens=(
                    valid_counts[req_index] if req_index < len(valid_counts) else 0
                ),
            )
            if num_usable_positions <= 0:
                continue
            if not self.feature_stager.reserve(req_id, start, num_usable_positions):
                # Capturing part of a chunk would leave a hole in the middle of the
                # rollout, so the rollout goes rather than the chunk.
                self.rollout_buffer.drop(req_id, reason="not_sampled")
                self._seen_req_ids.discard(req_id)
                self.skipped_staging_full += 1

        self._check_layout(offset, scheduler_output, num_reqs)
        self.feature_stager.commit(
            engine_step.input_ids,
            engine_step.aux_hidden_states,
            engine_step.hidden_states,
        )

    @staticmethod
    def _check_layout(
        offset: int, scheduler_output: "SchedulerOutput", num_reqs: int
    ) -> None:
        """Assert the walked token count matches the schedule.

        Raises:
            RuntimeError: On any mismatch. Every reserved offset is derived from this
                assumption.
        """
        if offset != scheduler_output.total_num_scheduled_tokens:
            raise RuntimeError(
                f"flat batch layout mismatch: walked {offset} tokens over "
                f"{num_reqs} requests but the step scheduled "
                f"{scheduler_output.total_num_scheduled_tokens}. Capture assumes "
                "each request's scheduled tokens are contiguous and laid out in "
                "input_batch order."
            )

    def _register_new(self, scheduler_output: "SchedulerOutput") -> None:
        """Open a rollout for each newly scheduled request that is eligible."""
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            if req_id in self._seen_req_ids:
                continue

            if not self.request_sampler.accepts(req_id):
                self.skipped_sample_rate += 1
                logger.debug("request %s skipped: below capture sample rate", req_id)
                continue

            if (
                self.capture_settings.skip_prefix_cache_hits
                and new_req.num_computed_tokens > 0
            ):
                self.rollout_buffer.drop(req_id, reason="prefix_cache_hit")
                logger.debug(
                    "request %s dropped: prefix cache hit (%d computed tokens)",
                    req_id,
                    new_req.num_computed_tokens,
                )
                continue

            prompt_token_ids = new_req.prompt_token_ids
            if prompt_token_ids is None:
                continue

            self._seen_req_ids.add(req_id)
            self.rollout_buffer.begin(req_id, prompt_len=len(prompt_token_ids))

    def _retire_finished(self, scheduler_output: "SchedulerOutput") -> None:
        """Seal the rollouts of requests the engine reports finished."""
        for req_id in scheduler_output.finished_req_ids:
            if req_id in self._seen_req_ids:
                self._seen_req_ids.discard(req_id)
                self.rollout_buffer.finish(req_id)

    def abort(self, req_ids: set[str]) -> None:
        """Discard rollouts for requests that will not finish normally.

        Args:
            req_ids: Ids the engine aborted.
        """
        for req_id in req_ids & self._seen_req_ids:
            self._seen_req_ids.discard(req_id)
            self.rollout_buffer.drop(req_id, reason="aborted")

    def _drain(self) -> None:
        """Hand any in-flight copy to the buffer.

        Valid only between two engine steps, with no reservation open. It clears the
        reservations it drains, so a drain inside a `reserve`/`commit` pair drops that
        step's rows and splices one request's features onto another's rollout.
        """
        for captured_chunk in self.feature_stager.drain():
            self.rollout_buffer.add_chunk(
                captured_chunk.req_id,
                token_ids=captured_chunk.token_ids,
                features=captured_chunk.features,
                final_hidden=captured_chunk.final_hidden,
            )

    def as_metrics(self) -> dict[str, float]:
        """Buffer counters plus this capture's own skip counts."""
        metrics = self.rollout_buffer.as_metrics()
        metrics["capture/active"] = float(len(self._seen_req_ids))
        metrics["capture/skipped_staging_full"] = float(self.skipped_staging_full)
        metrics["capture/skipped_sample_rate"] = float(self.skipped_sample_rate)
        return metrics
