"""Backend-independent decoded-audio contracts."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, eq=False)
class DecodedAudioChunk:
    """Represent one decoded audio chunk on the source-media timeline.

    The sample buffer uses sample-major order and is stored without copying. Consumers must treat
    it as read-only. Integer source PCM is normalized by the PyAV integration before construction;
    floating-point source amplitudes are preserved and may exceed the nominal full-scale range.

    Parameters
    ----------
    start_sample_index
        Zero-based index of the first sample in decode order for the selected audio stream. The
        index remains tied to the complete decoded stream, so the first chunk in an interval may
        start above zero.
    start_timestamp_seconds
        Finite source presentation timestamp in seconds for the first sample in ``samples``.
    sample_rate_hz
        Positive number of samples per second for each channel.
    channel_layout
        Non-empty FFmpeg channel-layout name, such as ``"mono"`` or ``"stereo"``.
    samples
        Contiguous audio samples with shape ``(sample_count, channel_count)`` and dtype
        ``float32``. Integer source formats map their full-scale range approximately to
        ``[-1.0, 1.0)``. Floating-point source values are not clipped.

    Raises
    ------
    TypeError
        If an integer or string field has the wrong type, or ``samples`` is not a NumPy array.
    ValueError
        If an index, timestamp, sample rate, layout, dtype, shape, or sample value violates the
        contract.
    """

    start_sample_index: int
    start_timestamp_seconds: float
    sample_rate_hz: int
    channel_layout: str
    samples: NDArray[np.float32]

    def __post_init__(self) -> None:
        """Validate sample identity, timing, layout, shape, dtype, and finite amplitudes."""
        if isinstance(self.start_sample_index, bool) or not isinstance(
            self.start_sample_index, int
        ):
            raise TypeError(
                "start_sample_index must be an integer, "
                f"got {type(self.start_sample_index).__name__}"
            )
        if self.start_sample_index < 0:
            raise ValueError(
                f"start_sample_index must be non-negative, got {self.start_sample_index}"
            )
        if not math.isfinite(self.start_timestamp_seconds):
            raise ValueError(
                f"start_timestamp_seconds must be finite, got {self.start_timestamp_seconds}"
            )
        if isinstance(self.sample_rate_hz, bool) or not isinstance(self.sample_rate_hz, int):
            raise TypeError(
                f"sample_rate_hz must be an integer, got {type(self.sample_rate_hz).__name__}"
            )
        if self.sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be positive, got {self.sample_rate_hz}")
        if not isinstance(self.channel_layout, str):
            raise TypeError(
                f"channel_layout must be a string, got {type(self.channel_layout).__name__}"
            )
        if not self.channel_layout.strip():
            raise ValueError("channel_layout must not be empty")

        if not isinstance(self.samples, np.ndarray):
            raise TypeError(f"samples must be a NumPy array, got {type(self.samples).__name__}")
        if self.samples.dtype != np.dtype(np.float32):
            raise ValueError(f"samples must have dtype float32, got {self.samples.dtype}")
        if self.samples.ndim != 2:
            raise ValueError(
                f"samples must have shape (sample_count, channel_count), got {self.samples.shape}"
            )
        if self.samples.shape[0] == 0 or self.samples.shape[1] == 0:
            raise ValueError(
                "samples must have positive sample and channel counts, "
                f"got shape {self.samples.shape}"
            )
        if not self.samples.flags.c_contiguous:
            raise ValueError("samples must be C-contiguous in sample-major order")
        if not np.isfinite(self.samples).all():
            raise ValueError("samples must contain only finite values")

    @property
    def sample_count(self) -> int:
        """Return the number of samples per channel in this chunk."""
        return int(self.samples.shape[0])

    @property
    def channel_count(self) -> int:
        """Return the number of interleaved channels in this chunk."""
        return int(self.samples.shape[1])

    @property
    def end_timestamp_seconds(self) -> float:
        """Return the exclusive source timestamp immediately after this chunk."""
        return self.start_timestamp_seconds + self.sample_count / self.sample_rate_hz
