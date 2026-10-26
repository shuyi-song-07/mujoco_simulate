"""Bounded, asynchronous practice/recording diagnostics, separate from datasets."""
from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import threading
import time


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if hasattr(value, "tolist"):
        return safe_json(value.tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class SessionDiagnostics:
    def __init__(self, root, metadata, *, queue_size=2048, chunk_bytes=16*1024*1024, max_chunks=8):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.queue = queue.Queue(maxsize=queue_size)
        self.chunk_bytes, self.max_chunks = chunk_bytes, max_chunks
        self.dropped = self.written = self.bytes = 0
        self.error = None
        self.closed = False
        self.lock = threading.Lock()
        (self.root / "session.json").write_text(json.dumps(safe_json({**metadata,
            "diagnostic_schema": "task3-diagnostics-v1", "state_hz": 10,
            "chunk_bytes": chunk_bytes, "max_chunks": max_chunks,
            "camera_video_saved": False, "full_landmarks_supported": True,
            "landmark_location": "kind=hand_measurement, measurement (request copies may contain measurement_id only)",
            "training_data": False}), ensure_ascii=False, indent=2), encoding="utf-8")
        self.thread = threading.Thread(target=self._write, name="task3-diagnostics", daemon=True)
        self.thread.start()

    def emit(self, kind, **fields):
        # Serialize before handing data to another thread so later mutations
        # cannot change the recorded request or accepted target.
        row = json.dumps(safe_json({"kind": kind, "wall_time_ms": time.time()*1000,
            "monotonic_ns": time.monotonic_ns(), **fields}), ensure_ascii=False, allow_nan=False) + "\n"
        with self.lock:
            if self.closed or self.error:
                self.dropped += 1
                return
            try:
                self.queue.put_nowait(row)
            except queue.Full:
                self.dropped += 1

    def _write(self):
        index = size = 0
        handle = None
        try:
            while True:
                row = self.queue.get()
                try:
                    if row is None:
                        return
                    count = len(row.encode("utf-8"))
                    if handle is None or size + count > self.chunk_bytes:
                        if handle:
                            handle.close()
                        if index >= self.max_chunks:
                            self.error = "diagnostic_capacity_reached"
                            self.dropped += 1
                            # Keep draining the bounded queue; no old logs are deleted.
                            handle = None
                            continue
                        handle = (self.root / f"events-{index:03d}.jsonl").open("w", encoding="utf-8", newline="\n")
                        index += 1
                        size = 0
                    handle.write(row)
                    handle.flush()
                    size += count
                    self.bytes += count
                    self.written += 1
                finally:
                    self.queue.task_done()
        except OSError as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            if handle:
                handle.close()

    def status(self):
        return {"path": str(self.root), "written": self.written, "dropped": self.dropped,
                "bytes": self.bytes, "pending": self.queue.qsize(), "error": self.error,
                "closed": self.closed, "state_hz": 10}

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        if self.thread.is_alive():
            try:
                self.queue.put(None, timeout=5)
                self.thread.join(timeout=10)
            except queue.Full:
                self.error = self.error or "diagnostic_close_queue_timeout"
        if self.thread.is_alive():
            self.error = self.error or "diagnostic_close_timeout"
        (self.root / "summary.json").write_text(json.dumps(self.status(), ensure_ascii=False, indent=2), encoding="utf-8")
