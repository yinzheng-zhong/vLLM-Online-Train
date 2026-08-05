import torch

from vllm_online_train.capture.records import CapturedChunk
from vllm_online_train.config.shapes import EngineShapes


class FeatureStager:
    """Gathers selected positions on the device and copies them to host memory.

    Owns one pinned staging buffer, reused every step, and one side stream the copy is
    issued on. The buffers are allocated outside any inference-mode region, because a
    tensor created under `inference_mode` cannot join an autograd graph and `.clone()`
    inherits the flag: the escape is to `copy_` into tensors that already exist.
    """

    def __init__(
        self, shapes: EngineShapes, capacity: int, device: torch.device
    ) -> None:
        """
        Args:
            shapes: Supplies the feature and hidden widths.
            capacity: Positions the staging buffer holds, i.e. `max_rollout_tokens`.
            device: Where the gather runs.
        """
        self.shapes = shapes
        self.capacity = capacity
        self.device = device

        width = shapes.feature_width + shapes.hidden_size
        self._staging = torch.empty(
            (capacity, width), dtype=shapes.feature_dtype, pin_memory=True
        )
        self._staging_tokens = torch.empty(
            (capacity,), dtype=torch.int32, pin_memory=True
        )
        self._index = torch.empty((capacity,), dtype=torch.int64, device=device)
        self._index_host = torch.empty((capacity,), dtype=torch.int64, pin_memory=True)

        self._copy_stream = torch.cuda.Stream(device=device)
        self._copy_done = torch.cuda.Event()
        self._slices: list[tuple[str, int, int]] = []
        self._staged = 0
        self._in_flight = False

    @property
    def staged(self) -> int:
        """Positions reserved since the last commit."""
        return self._staged

    def reset(self) -> None:
        """Drop any reservations that were never committed."""
        self._slices.clear()
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
        self._index_host[rows] = torch.arange(
            start, start + length, dtype=torch.int64
        )
        self._slices.append((req_id, self._staged, length))
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

        index = self._index[:staged]
        index.copy_(self._index_host[:staged], non_blocking=True)

        hidden_size = self.shapes.hidden_size
        # Ordered against the default stream, so the gather cannot read activations
        # before the forward that produced them has retired.
        self._copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._copy_stream):
            for i, aux in enumerate(aux_hidden_states):
                column = slice(i * hidden_size, (i + 1) * hidden_size)
                self._staging[:staged, column].copy_(
                    aux.index_select(0, index), non_blocking=True
                )
            width = self.shapes.feature_width
            tail = slice(width, width + hidden_size)
            self._staging[:staged, tail].copy_(
                hidden_states.index_select(0, index), non_blocking=True
            )
            self._staging_tokens[:staged].copy_(
                input_ids.index_select(0, index), non_blocking=True
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

        width = self.shapes.feature_width
        chunks = [
            CapturedChunk(
                req_id=req_id,
                token_ids=self._staging_tokens[start : start + length].clone(),
                features=self._staging[start : start + length, :width].clone(),
                final_hidden=self._staging[start : start + length, width:].clone(),
            )
            for req_id, start, length in self._slices
        ]
        self.reset()
        return chunks
