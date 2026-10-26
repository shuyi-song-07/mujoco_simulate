"""Exact-PTS RGB cache writer shared by formal packs and historical readers."""
from pathlib import Path

import av
import numpy as np


def cache_video(source, selected_indices, output, width, height):
    """Copy exact PTS-matched RGB frames; resize only the derived cache."""
    ids = np.asarray(selected_indices, dtype=np.int64)
    if not len(ids) or np.any(np.diff(ids) <= 0) or ids[0] < 0 or ids[-1] >= source["frame_count"]:
        raise ValueError("Selected video indexes must be increasing valid raw frames")
    if width < 16 or height < 16:
        raise ValueError("Image dimensions must be at least 16 pixels")
    fps = float(source["fps"])
    wanted = float(source["start_s"]) + ids / fps
    output = Path(output)
    temporary = output.with_suffix(".partial.npy")
    memory = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8,
                                     shape=(len(ids), height, width, 3))
    written = 0
    previous_pts = None
    with av.open(str(source["path"])) as container:
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError("Video frame has no presentation timestamp")
            time = float(frame.pts * frame.time_base)
            if previous_pts is not None and time <= previous_pts:
                raise ValueError("Video timestamps are not strictly increasing")
            previous_pts = time
            if time < wanted[written] - .001:
                continue
            if abs(time - wanted[written]) > .001:
                raise ValueError(f"Missing frame at {wanted[written]:.6f}s; next PTS {time:.6f}s")
            memory[written] = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
            written += 1
            if written == len(ids):
                break
    memory.flush()
    del memory
    if written != len(ids):
        raise ValueError(f"Video ended after {written}/{len(ids)} selected frames")
    temporary.replace(output)
    return {"selected_frames": written, "first_pts_s": float(wanted[0]),
            "last_pts_s": float(wanted[-1]), "pts_tolerance_s": .001,
            "source_start_s": source["start_s"], "raw_video_unchanged": True}

