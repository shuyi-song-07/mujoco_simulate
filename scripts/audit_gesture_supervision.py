"""Read-only coverage of the frozen Pack; no training or data mutation.

Uses the same verified reader as ACT/DP. Counts eligible action anchors and
separates each hand, class, missing labels, and valid Unknown skeletons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.task3 import training_pack
from training.papera.pack_data import _close_episode
from training.gesture_auxiliary import CATEGORIES


def audit(pack):
    pack = Path(pack).resolve()
    checked = training_pack.verify(pack, check_hashes=False)
    manifest = checked["manifest"]
    if tuple(manifest["categories"]) != CATEGORIES:
        raise ValueError("Pack category order changed")
    result = {"pack": str(pack), "manifest_sha256": hashlib.sha256(
        (pack / "manifest.json").read_bytes()).hexdigest(), "categories": list(CATEGORIES),
        "label_policy": "recorded_category_no_score_v1",
        "label_time": "observation_measurement_at_current_anchor_not_accepted_control_intent",
        "verification": checked.get("verification"), "splits": {}, "episodes": {}}
    for split in ("train", "val"):
        totals = dict(frames=0, action_anchors=0, valid_by_hand=[0, 0],
                      missing_by_hand=[0, 0], class_counts=[[0, 0, 0], [0, 0, 0]],
                      unknown_with_geometry=[0, 0], unknown_with_task_geometry=[0, 0],
                      both_hands_valid=0, neither_hand_valid=0,
                      episodes=len(manifest["split"][split]))
        for key in manifest["split"][split]:
            episode = training_pack.read_episode(pack, key, checked=checked, include_evidence=False)
            try:
                a = episode["arrays"]
                anchors = a["action_mask"].astype(bool)
                labels, mask = a["category_id"][anchors], a["category_mask"][anchors].astype(bool)
                if np.any(mask & ((labels < 0) | (labels >= 3))):
                    raise ValueError(f"Invalid unmasked category in {key}")
                counts = [[int(((labels[:, side] == category) & mask[:, side]).sum())
                           for category in range(3)] for side in range(2)]
                unknown = labels < 0
                row = dict(frames=len(anchors), action_anchors=int(anchors.sum()),
                           valid_by_hand=mask.sum(0).tolist(), missing_by_hand=(~mask).sum(0).tolist(),
                           class_counts=counts,
                           unknown_with_geometry=(unknown & a["geometry_mask"][anchors].astype(bool)).sum(0).tolist(),
                           unknown_with_task_geometry=(unknown & a["task_geometry_mask"][anchors].astype(bool)).sum(0).tolist(),
                           both_hands_valid=int(mask.all(1).sum()), neither_hand_valid=int((~mask).all(1).sum()))
                result["episodes"][key] = {"split": split, **row}
                for field, value in row.items():
                    totals[field] = ((np.asarray(totals[field]) + value).tolist()
                                     if isinstance(value, list) else totals[field] + value)
            finally:
                _close_episode(episode)
        totals["coverage_by_hand"] = (np.asarray(totals["valid_by_hand"]) / totals["action_anchors"]).tolist()
        totals["class_fractions_by_hand"] = (np.asarray(totals["class_counts"]) /
                                             np.asarray(totals["valid_by_hand"])[:, None]).tolist()
        totals["episodes_with_each_class_by_hand"] = [[sum(
            result["episodes"][key]["class_counts"][side][category] > 0
            for key in manifest["split"][split]) for category in range(3)] for side in range(2)]
        result["splits"][split] = totals
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.pack)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["splits"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
