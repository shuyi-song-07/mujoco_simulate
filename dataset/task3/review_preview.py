"""Temporary review-only videos, using the same frame selection as prepare.

These lossy viewing copies are never a training input. Their frame map always
points back to the original images and labels; no human approval is inferred.
"""
from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import tempfile
import threading


def preview_plan(rows, review, count, fps, *, config=None):
    from dataset.task3.prepare_paper_dataset import apply_review
    from dataset.task3.review_server import validate_frame_overrides

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("预览帧率无效")
    if len(rows) != count or any(row.get("raw_frame_index") != n for n, row in enumerate(rows)):
        raise ValueError("原始帧索引缺失、重复或乱序，不能生成对齐预览")
    validate_frame_overrides(rows, review)
    effective = apply_review(rows, review)
    sequence = None
    if config is not None:
        from dataset.task3.sequence_stitching import build_sequence_contract
        from dataset.task3.curation_rules import resegment_rows
        sequence = build_sequence_contract(effective, config)
        effective = sequence.pop('rows')
        resegment_rows(effective)
    selected = [row for row in effective if row["keep"]]
    frames = [row["raw_frame_index"] for row in selected]
    segments = []
    for n, row in enumerate(selected):
        if n == 0 or row["segment_id"] != selected[n - 1]["segment_id"]:
            segments.append({"preview_start": n, "preview_end": n + 1,
                             "raw_start": row["raw_frame_index"], "raw_end": row["raw_frame_index"] + 1})
        else:
            segments[-1].update(preview_end=n + 1, raw_end=row["raw_frame_index"] + 1)
    result = {"frame_map": frames, "segments": segments, "fps": fps,
            "raw_frames": count, "kept_frames": len(frames), "removed_frames": count - len(frames),
            "raw_duration_s": count / fps, "duration_s": len(frames) / fps,
            "review_only": True}
    if sequence is not None:
        result['sequence_validation'] = sequence
    return result


def render_video(source, output, frames, count, fps):
    """Check every episode PTS, then encode retained frames on a new CFR clock."""
    import av

    start, end = float(source["start_s"]), float(source["end_s"])
    if (not all(math.isfinite(t) for t in (start, end)) or start < 0 or
            abs(end - start - count / fps) > .001 or abs(float(source.get("fps", fps)) - fps) > .01):
        raise ValueError("视频片段范围或帧率与原始索引不一致")
    kept = set(frames)
    seen = written = 0
    rate = Fraction(str(fps)).limit_denominator(100000)
    with Path(source["path"]).open("rb") as handle, av.open(handle) as original, av.open(str(output), "w") as preview:
        stream = original.streams.video[0]
        if abs(float(stream.average_rate or 0) - fps) > .01:
            raise ValueError("视频实际帧率与索引不一致")
        encoder = None
        size = None
        for frame in original.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("原视频缺少时间戳")
            t = float(frame.pts * frame.time_base)
            if t < start - .0001:
                continue
            if t >= end - .0001:
                break
            if abs(t - start - seen / fps) > .001:
                raise ValueError("原视频时间戳跳变，不能与标签对齐")
            if size is None:
                size = frame.width, frame.height
            if size != (frame.width, frame.height):
                raise ValueError("原视频尺寸在片段内变化")
            if seen in kept:
                if encoder is None:
                    encoder = preview.add_stream("libx264", rate=rate)
                    encoder.width, encoder.height = size
                    encoder.pix_fmt = "yuv420p"
                    encoder.options = {"crf": "20", "preset": "veryfast"}
                # A fresh frame avoids carrying source PTS or picture type into
                # the preview encoder (especially at non-keyframe cut points).
                image = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
                image.pts, image.time_base = written, 1 / rate
                for packet in encoder.encode(image):
                    preview.mux(packet)
                written += 1
            seen += 1
        if seen != count or written != len(frames):
            raise ValueError(f"视频实际帧数不符：原片段 {seen}/{count}，预览 {written}/{len(frames)}")
        if encoder is not None:
            for packet in encoder.encode():
                preview.mux(packet)


class PreviewCache:
    """One bounded renderer per service; up to eight temporary viewing sets."""

    def __init__(self):
        self.lock = threading.Lock()
        self.entries = OrderedDict()

    def create(self, plan, sources):
        if not plan["frame_map"]:
            return plan | {"videos": []}
        names = [s['camera'] for s in sources]
        if names != ['oblique'] and not (len(names) == 3 and set(names) in
                ({'top','front','oblique'}, {'top','front','side'})):
            raise ValueError("裁剪预览需要 oblique 单视角、top/front/oblique 三视角或历史 top/front/side 三视角")
        signatures = [(str(s["path"]), Path(s["path"]).stat().st_size,
                       Path(s["path"]).stat().st_mtime_ns, s["start_s"], s["end_s"], s["camera"]) for s in sources]
        digest = hashlib.sha256(json.dumps([plan, signatures], sort_keys=True).encode()).hexdigest()
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("正在生成另一份预览，请稍后重试")
        try:
            if digest in self.entries:
                self.entries.move_to_end(digest)
                return self.entries[digest][1]
            temporary = tempfile.TemporaryDirectory(prefix="papera-review-preview-")
            root = Path(temporary.name).resolve()
            try:
                videos = []
                for source in sources:
                    camera = source["camera"]
                    render_video(source, root / f"{camera}.mp4", plan["frame_map"], plan["raw_frames"], plan["fps"])
                    videos.append({"camera": camera, "url": f"/preview-media/{digest}/{camera}",
                                   "start_s": 0, "end_s": plan["duration_s"], "fps": plan["fps"]})
                after = [(str(s["path"]), Path(s["path"]).stat().st_size,
                          Path(s["path"]).stat().st_mtime_ns, s["start_s"], s["end_s"], s["camera"]) for s in sources]
                if after != signatures:
                    raise ValueError("生成期间原视频已变化，请重新审核来源")
                result = plan | {"videos": videos}
                self.entries[digest] = (temporary, result)
            except Exception:
                temporary.cleanup()
                raise
            while len(self.entries) > 8:
                oldest = next(iter(self.entries))
                try:
                    self.entries[oldest][0].cleanup()
                except OSError:
                    break  # An active Windows video stream can hold the file.
                del self.entries[oldest]
            return result
        finally:
            self.lock.release()

    def media(self, digest, camera):
        # Never accept user-supplied paths, even under the temporary directory.
        with self.lock:
            entry = self.entries.get(digest)
            if not entry or camera not in {v['camera'] for v in entry[1]['videos']}:
                raise KeyError("预览已过期，请重新生成")
            self.entries.move_to_end(digest)
            root = Path(entry[0].name).resolve()
            return root / f"{camera}.mp4", root

    def close(self):
        with self.lock:
            for temporary, _ in self.entries.values():
                temporary.cleanup()
            self.entries.clear()
