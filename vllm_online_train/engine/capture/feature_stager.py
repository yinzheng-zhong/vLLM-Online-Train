import torch

from vllm_online_train.config.shapes import EngineShapes
from vllm_online_train.engine.capture.records import CapturedChunk
from vllm_online_train.logger import init_logger

logger = init_logger(__name__)


class FeatureStager:
    def __init__(
        self, engine_shapes: EngineShapes, capacity: int, device: torch.device
    ) -> None:
        """Gathers selected positions on the device and copies them to host memory.

        Owns one pinned staging buffer, reused every step, and one side stream the
        copy is issued on. The buffers are allocated outside any inference-mode
        region, because a tensor created under `inference_mode` cannot join an
        autograd graph and `.clone()` inherits the flag: the escape is to `copy_` into
        tensors that already exist.

        Args:
            engine_shapes: Supplies the feature and hidden widths.
            capacity: Positions the staging buffer holds, i.e. `max_rollout_tokens`.
            device: Where the gather runs.
        """
        self.engine_shapes = engine_shapes
        self.capacity = capacity
        self.device = device

        row_width = engine_shapes.feature_width + engine_shapes.hidden_size
        self._staging_activations = torch.empty(
            (capacity, row_width), dtype=engine_shapes.feature_dtype, pin_memory=True
        )
        self._staging_token_ids = torch.empty(
            (capacity,), dtype=torch.int32, pin_memory=True
        )
        self._gather_index = torch.empty(
            (capacity,), dtype=torch.int64, device=device
        )
        self._gather_index_host = torch.empty(
            (capacity,), dtype=torch.int64, pin_memory=True
        )

        self._copy_stream = torch.cuda.Stream(device=device)
        self._copy_done = torch.cuda.Event()
        self._reservations: list[tuple[str, int, int]] = []
        self._staged = 0
        self._in_flight = False
        logger.debug(
            "FeatureStager configured: capacity=%d, row_width=%d, device=%s",
            capacity,
            row_width,
            device,
        )

    @property
    def staged(self) -> int:
        """Positions reserved since the last commit."""
        return self._staged

    def reset(self) -> None:
        """Drop any reservations that were never committed."""
        self._reservations.clear()
        self._staged = 0

    def reserve(self, req_id: str, start: int, length: int) -> bool:
        """Claim staging rows for a contiguous run of flat-batch positions.

        Args:
            req_id: The engine's request id.
            start: First flat-batch index to capture.
            length: Positions to capture.

        Returns:
            False when the buffer cannot hold the run, in which case nothing is
            reserved.
        """
        if self._staged + length > self.capacity:
            return False
        rows = slice(self._staged, self._staged + length)
        self._gather_index_host[rows] = torch.arange(
            start, start + length, dtype=torch.int64
        )
        self._reservations.append((req_id, self._staged, length))
        self._staged += length
        return True

    def commit(
        self,
        input_ids: torch.Tensor,
        aux_hidden_states: list[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> None:
        """Issue the gather and the device-to-host copy for everything reserved.

        Args:
            input_ids: `[>=T]` flat input token ids.
            aux_hidden_states: Per-feature-layer `[>=T, hidden_size]` activations.
            hidden_states: `[>=T, hidden_size]` post-norm final activations.
        """
        staged = self._staged
        if staged == 0:
            return

        gather_index = self._gather_index[:staged]
        gather_index.copy_(self._gather_index_host[:staged], non_blocking=True)

        hidden_size = self.engine_shapes.hidden_size
        # Ordered against the default stream, so the gather cannot read activations
        # before the forward that produced them has retired.
        self._copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._copy_stream):
            for layer_index, aux_layer in enumerate(aux_hidden_states):
                column = slice(
                    layer_index * hidden_size, (layer_index + 1) * hidden_size
                )
                self._staging_activations[:staged, column].copy_(
                    aux_layer.index_select(0, gather_index), non_blocking=True
                )
            feature_width = self.engine_shapes.feature_width
            tail = slice(feature_width, feature_width + hidden_size)
            self._staging_activations[:staged, tail].copy_(
                hidden_states.index_select(0, gather_index), non_blocking=True
            )
            self._staging_token_ids[:staged].copy_(
                input_ids.index_select(0, gather_index), non_blocking=True
            )
            self._copy_done.record(self._copy_stream)
        self._in_flight = True

    def drain(self) -> list[CapturedChunk]:
        """Wait for the in-flight copy and clone the staged rows out.

        Returns:
            One chunk per reservation, empty when no copy was in flight. The rows are
            cloned because the next step overwrites the staging buffer.
        """
        if not self._in_flight:
            self.reset()
            return []
        self._copy_done.synchronize()
        self._in_flight = False

        feature_width = self.engine_shapes.feature_width
        captured_chunks = [
            CapturedChunk(
                req_id=req_id,
                token_ids=self._staging_token_ids[start : start + length].clone(),
                features=self._staging_activations[
                    start : start + length, :feature_width
                ].clone(),
                final_hidden=self._staging_activations[
                    start : start + length, feature_width:
                ].clone(),
            )
            for req_id, start, length in self._reservations
        ]
        self.reset()
        return captured_chunks
