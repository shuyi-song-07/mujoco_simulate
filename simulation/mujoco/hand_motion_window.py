"""Causal fixed-duration wrist labels, independent of robot control."""

from collections import deque

import numpy as np


class FixedWindowHandDisplacement:
    def __init__(self, window_ms: float, max_gap_ms: float):
        if not all(np.isfinite(x) and x > 0 for x in (window_ms, max_gap_ms)):
            raise ValueError("Window and maximum sample gap must be finite and positive")
        self.window_ms = float(window_ms)
        self.max_gap_ms = float(max_gap_ms)
        self.reset()

    def reset(self) -> None:
        self.samples = deque()
        self.last_emitted_end_ms = float("-inf")

    def append(self, captured_ms: float, position_uv: np.ndarray) -> None:
        point = np.asarray(position_uv, dtype=np.float64)
        if (not np.isfinite(captured_ms) or captured_ms <= 0
                or point.shape != (2,) or not np.isfinite(point).all()):
            self.reset()
            return
        if self.samples:
            previous_ms = self.samples[-1][0]
            if captured_ms <= previous_ms:
                return  # Duplicate/out-of-order captures cannot extend a window.
            if captured_ms - previous_ms > self.max_gap_ms:
                self.reset()
        self.samples.append((float(captured_ms), point.copy()))
        # Keep the bracketing sample before the next window's start.
        cutoff = captured_ms - self.window_ms
        while len(self.samples) > 2 and self.samples[1][0] <= cutoff:
            self.samples.popleft()

    def consume(self, frame_ms: float) -> dict:
        result = {"uv": np.zeros(2, dtype=np.float32), "valid": False,
                  "start_ms": -1.0, "end_ms": -1.0}
        if not self.samples or not np.isfinite(frame_ms):
            return result
        end_ms, end_point = self.samples[-1]
        if (end_ms > frame_ms or frame_ms - end_ms > self.max_gap_ms
                or end_ms <= self.last_emitted_end_ms):
            return result
        self.last_emitted_end_ms = end_ms
        start_ms = end_ms - self.window_ms
        if start_ms < self.samples[0][0] or len(self.samples) < 2:
            return result
        for (left_ms, left), (right_ms, right) in zip(self.samples, list(self.samples)[1:]):
            if left_ms <= start_ms <= right_ms:
                alpha = (start_ms - left_ms) / (right_ms - left_ms)
                start_point = left + alpha * (right - left)
                return {"uv": (end_point - start_point).astype(np.float32),
                        "valid": True, "start_ms": start_ms, "end_ms": end_ms}
        return result
