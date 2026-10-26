"""Causal ACT chunk execution on the 30 Hz observation clock.

Every target still passes through PolicyExecutor at 60 Hz.  Chunk offsets are
observation steps, not control steps.  Temporal ensembling combines predictions
for the SAME current step; it never averages adjacent future actions.
"""
from __future__ import annotations

import math
import numpy as np

SCHEDULER_VERSION = "papera-action-chunks-v2"
LEGACY = "first_action_replan_each_observation"
MODES = (LEGACY, "chunked", "temporal_ensemble")


class InvalidActionChunk(ValueError):
    """A model returned an invalid shape or nonfinite action value."""


def _rotation(rpy):
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(roll), np.sin(roll), np.cos(pitch),
                             np.sin(pitch), np.cos(yaw), np.sin(yaw))
    return np.asarray([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                       [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                       [-sp, cp*sr, cp*cr]])


def _rpy(rotation):
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) > 1e-8:
        return np.asarray([math.atan2(rotation[2, 1], rotation[2, 2]), pitch,
                           math.atan2(rotation[1, 0], rotation[0, 0])])
    return np.asarray([0., pitch, math.atan2(-rotation[0, 1], rotation[1, 1])])


def average_actions(actions, weights):
    """Euclidean xyz/width mean and SO(3) chordal rotation mean.

Euler aliases such as +pi/-pi must represent the same rotation. SVD projects
the weighted matrix onto SO(3), correcting reflections. An ambiguous rank-
deficient mean falls back to the newest prediction rather than an arbitrary
orientation. Physical pose/width limits remain the executor's responsibility.
"""
    result = weights @ actions
    for start in (3, 10):
        mean = sum(w * _rotation(a[start:start+3]) for w, a in zip(weights, actions))
        u, singular, vt = np.linalg.svd(mean)
        if singular[1] < 1e-8:
            result[start:start+3] = actions[-1, start:start+3]
            continue
        correction = np.diag([1., 1., np.linalg.det(u @ vt)])
        result[start:start+3] = _rpy(u @ correction @ vt)
    return result


class ActionChunkScheduler:
    """State belongs to one episode; create a fresh instance after reset.

``next_action(t, predict)`` must be called exactly once for t=0,1,... . The
callback is invoked only when a new observation-conditioned chunk is needed.
Explicit origins establish validity: legitimate zero targets are never used
as missing-data sentinels. At most ``chunk_size`` chunks are retained.
"""
    def __init__(self, mode="chunked", *, chunk_size=30, n_action_steps=5,
                 temporal_ensemble_coeff=.01):
        if mode not in MODES:
            raise ValueError(f"Unknown execution schedule: {mode}")
        if type(chunk_size) is not int or not 1 <= chunk_size <= 1000:
            raise ValueError("chunk_size must be an integer in [1, 1000]")
        if type(n_action_steps) is not int or not 1 <= n_action_steps <= chunk_size:
            raise ValueError("n_action_steps must be in [1, chunk_size]")
        if (isinstance(temporal_ensemble_coeff, bool)
                or not isinstance(temporal_ensemble_coeff, (int, float))
                or not math.isfinite(temporal_ensemble_coeff) or temporal_ensemble_coeff < 0):
            raise ValueError("temporal_ensemble_coeff must be finite and nonnegative")
        self.mode, self.chunk_size = mode, chunk_size
        self.n_action_steps = n_action_steps
        self.coeff = float(temporal_ensemble_coeff)
        self._next_frame = 0
        self._chunks = []
        self.query_count = 0

    @classmethod
    def from_settings(cls, settings, *, chunk_size=None):
        # Missing execution is supported only for old programmatic callers;
        # CLI settings require an explicit versioned execution field.
        return cls(settings.get("execution", LEGACY),
                   chunk_size=chunk_size or settings.get("chunk_size", 30),
                   n_action_steps=settings.get("n_action_steps", 5),
                   temporal_ensemble_coeff=settings.get("temporal_ensemble_coeff", .01))

    @property
    def latest_chunk(self):
        if not self._chunks:
            raise RuntimeError("No prediction has been queried")
        return self._chunks[-1][1].copy()

    def next_action(self, frame, predict):
        if type(frame) is not int or frame != self._next_frame:
            raise ValueError("Execution frames must be consecutive integers starting at zero")
        self._chunks = [(origin, chunk) for origin, chunk in self._chunks
                        if origin <= frame < origin + self.chunk_size]
        queried = (self.mode != "chunked" or not self._chunks
                   or frame-self._chunks[-1][0] >= self.n_action_steps)
        if queried:
            # Observation/model infrastructure failures must retain their
            # original exception, not become a failed policy prediction.
            raw_chunk = predict()
            try:
                chunk = np.asarray(raw_chunk, dtype=float)
            except (TypeError, ValueError) as error:
                raise InvalidActionChunk("Model action chunk is not a numeric array") from error
            if chunk.shape != (self.chunk_size, 14) or not np.isfinite(chunk).all():
                raise InvalidActionChunk("Model action chunk has invalid shape or nonfinite values")
            if self.mode != "temporal_ensemble":
                self._chunks.clear()
            self._chunks.append((frame, chunk.copy()))
            self.query_count += 1
        if self.mode == "temporal_ensemble":
            origins = [origin for origin, _ in self._chunks]
            offsets = [frame-origin for origin in origins]
            candidates = np.stack([chunk[offset] for (_, chunk), offset in zip(self._chunks, offsets)])
            # ACT reference implementation weights oldest contributing query
            # most heavily: rank zero has weight 1, then exp(-k*rank).
            weights = np.exp(-self.coeff*np.arange(len(candidates), dtype=float))
            weights /= weights.sum()
            action = average_actions(candidates, weights)
            selected_offset = -1  # weighted target, not one chunk's action
        else:
            origin, chunk = self._chunks[-1]
            selected_offset = frame-origin
            origins, offsets, weights = [origin], [selected_offset], np.asarray([1.])
            action = chunk[selected_offset].copy()
        self._next_frame += 1
        return action, {
            "version": SCHEDULER_VERSION, "execution": self.mode, "frame": frame,
            "queried": queried, "query_index": self.query_count-1,
            "query_origin": self._chunks[-1][0], "selected_offset": selected_offset,
            "source_origins": origins, "source_offsets": offsets,
            "source_weights": weights.tolist(),
        }
