"""Curation I/O: shared shards, complete decoding, corruption and immutable sources.

The tiny disk fixture isolates the existing hand-timeline auditor with a stub;
that auditor has its own adversarial tests.  One integration test deliberately
runs the real auditor and verifies that its errors cannot be silently dropped.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dataset.task3 import curation_io as io
from simulation.mujoco.dual_arm.action_contract import ACTION_NAMES_V2, STATE_NAMES, V2


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_video(path, count=6):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
        for i in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((480, 640, 3), i*25, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def disk_fixture(parent):
    """Two 3-frame episodes sharing each video and crossing a data-shard boundary."""
    root = parent / "synthetic_success"
    config = {"version": "test-fixture"}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata = {"session": "synthetic", "outcome": "success", "input_kind": "human_teleop",
        "quality_review": "pending", "schema_version": V2, "gripper_trigger_policy": "palm_fist_edge_v2",
        "control_interface_version": "gesture-rate-gripper-v3", "config": config, "config_hash": digest,
        "sampling_policy": {"version": "raw_fixed_phase_v2", "fps": 30, "control_hz": 60}, "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES_V2, "action_units": ["m", "m", "m", "rad", "rad", "rad", "m"]*2,
        "training_views": "opaque", "model_dimensions": {"nq": 25, "nv": 24, "nu": 16, "nbody": 2},
        "object_qpos_adr": 18, "object_dof_adr": 18,
        "joint_mapping": {side: {"joint_names": [f"{side}_joint{i}" for i in range(1,8)],
             "qpos_indices": list(range(offset,offset+7)), "finger_qpos_indices": [offset+7,offset+8]}
             for side, offset in (("left",0),("right",9))}}
    info = {"fps": 30, "total_frames": 6, "total_episodes": 2,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {"action": {"names": ACTION_NAMES_V2}, "observation.state": {"names": STATE_NAMES}}}
    info['features'].update({f'observation.images.{c}': {'dtype': 'video', 'shape': [480, 640, 3]}
                             for c in ('top', 'front', 'side')})
    write_json(root / "meta/info.json", info)
    write_json(root / "meta/task3.json", metadata)
    indexes, manifests, parquet = [], [], []
    for i in range(2):
        archive = root / f"research/episode_{i:06d}_snapshots"
        write_json(archive / "episode_metadata.json", metadata)
        snap = {"time": np.arange(3)/30 + i, "qpos": np.zeros((3,25)), "qvel": np.zeros((3,24)),
            "ctrl": np.zeros((3,16)), "body_pos": np.zeros((3,2,3)), "body_quat": np.zeros((3,2,4)),
            "state": np.zeros((3,16)), "action": np.zeros((3,14))}
        snap["qpos"][:,21] = 1
        snap["body_quat"][:,:,0] = 1
        np.savez(archive / "snapshots_000000.npz", **snap)
        terminal = {k:v[-1] for k,v in snap.items() if k not in ("state","action")}
        np.savez(archive / "terminal_snapshot.npz", **terminal)
        write_json(archive / "terminal.json", {"outcome":"success", "last_raw_frame":2,
            "sample_to_terminal_s":0., "max_qpos_delta":0., "log_dropped":0,
            "status":{"task_state":"DONE", "success":True}})
        rows=[]
        for frame in range(3):
            row={"kind":"control_frame", "frame_index":frame, "sim_time":float(snap['time'][frame]),
                "wall_time_ms":1000+frame*40, "server_monotonic_ns":1000000+frame*40000,
                "state":[0.]*16, "action":[0.]*14, "status":{"paused":True},
                "hand_measurement_mask":{"left":False,"right":False},
                "hand_supervision_mask":{"left":False,"right":False}}
            rows.append(row)
            parquet.append({"episode_index":i,"frame_index":frame,"index":3*i+frame,
                "timestamp":frame/30,"observation.state":[0.]*16,"action":[0.]*14})
        (root / f"research/episode_{i:06d}.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        index={"episode_index":i,"length":3,"dataset_from_index":i*3,"dataset_to_index":i*3+3,
               "data/chunk_index":0,"data/file_index":0}
        for camera in io.CAMERAS:
            key=f"videos/observation.images.{camera}"
            index.update({f"{key}/chunk_index":0,f"{key}/file_index":0,
                          f"{key}/from_timestamp":i/10,f"{key}/to_timestamp":(i+1)/10})
        indexes.append(index)
        manifests.append({"episode_index":i,"session":"synthetic","frames":3,"outcome":"success",
                          "input_kind":"human_teleop","quality_review":"pending","config_hash":digest})
    (root/'research/episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifests),encoding='utf-8')
    for name, rows in (("meta/episodes/chunk-000/file-000.parquet",indexes),
                       ("data/chunk-000/file-000.parquet",parquet[:4]),
                       ("data/chunk-000/file-001.parquet",parquet[4:])):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(rows),path)
    for camera in io.CAMERAS:
        write_video(root/f"videos/observation.images.{camera}/chunk-000/file-000.mp4")
    return root


class CurationIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = disk_fixture(Path(self.temp.name))
        self.ref = io.discover_episodes(self.root)[1]

    def load(self):
        def hand_audit(events, metadata):
            # Consume the actual generator; this tests collection and source
            # hashes rather than skipping its iteration with a plain mock.
            list(events)
            return [], {"hand_freshness_audit":{"uncertain_checks":0}}
        with patch.object(io, "audit_hand_timeline", side_effect=hand_audit):
            return io.load_episode(self.ref)

    def test_shared_video_segment_and_crossing_data_shards_decode_fully(self):
        e = self.load()
        self.assertEqual([r['index'] for r in e['parquet_rows']], [3,4,5])
        self.assertEqual(e['video_sources'][0]['start_s'], .1)
        r = io.audit_episode(e)
        self.assertTrue(r['structural_pass'], r['errors'])
        self.assertEqual([(v['decoded_frames'],v['selected_frames']) for v in r['video_checks']], [(6,3)]*3)
        self.assertEqual(r['quality_review'], 'pending')
        self.assertFalse(r['quality_approved'])
        self.assertTrue(all(len(s['sha256']) == 64 for s in r['source_fingerprints']))

    def test_skip_video_cannot_produce_training_ready_pass(self):
        r = io.audit_episode(self.load(), check_videos=False)
        self.assertTrue(r['structural_data_pass'], r['errors'])
        self.assertFalse(r['video_verified'])
        self.assertFalse(r['complete'])
        self.assertFalse(r['structural_pass'])

    def test_named_joint_mapping_and_nonfinite_state_rejected(self):
        for field in ('mapping','nonfinite','action','reference','terminal'):
            with self.subTest(field=field):
                e=self.load()
                if field=='mapping':e['metadata']['joint_mapping']['right']['qpos_indices'][0]=0
                if field=='nonfinite':e['snapshots']['state'][0,0]=float('nan')
                if field=='action':e['frames'][0]['action'][0]=.1
                if field=='reference':e['frames'][0]['observation_measurement_id']='absent'
                if field=='terminal':e['terminal']['status']['success']=False
                r=io.audit_episode(e,check_videos=False)
                self.assertFalse(r['structural_data_pass'],field)

    def test_cached_source_change_is_not_accepted(self):
        e=self.load();first=io.audit_episode(e)
        self.assertTrue(first['structural_pass'],first['errors'])
        p=e['ref']['research_path'];p.write_bytes(p.read_bytes()+b'\n')
        after=io.audit_episode(e)
        self.assertFalse(after['structural_pass'])
        self.assertTrue(any('changed' in err.lower() for err in after['errors']))

    def test_video_segment_pts_and_truncation_are_rejected(self):
        e=self.load();source=e['video_sources'][0]
        wrong={**source,'start_s':source['start_s']+.01,'end_s':source['end_s']+.01}
        self.assertFalse(io.check_video_source(wrong)['verified'])
        io.check_video_source(source)  # prime decoder cache
        p=source['path'];p.write_bytes(p.read_bytes()[:96])
        self.assertFalse(io.check_video_source(source)['verified'])

    def test_duplicate_indexes_and_dataset_path_escape_fail_closed(self):
        manifest=self.root/'research/episodes.jsonl'
        original=manifest.read_text(encoding='utf-8')
        manifest.write_text(original+original.splitlines()[0]+'\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            io.discover_episodes(self.root)
        manifest.write_text(original,encoding='utf-8')
        path=self.root/'meta/info.json';info=json.loads(path.read_text(encoding='utf-8'))
        info['video_path']='../escape.mp4';write_json(path,info)
        self.assertTrue(any('escapes' in e for e in self.load()['load_errors']))

    def test_real_timeline_auditor_is_called_and_does_not_hide_failures(self):
        e=io.load_episode(self.ref)
        self.assertTrue(e['timeline_errors'] or e['load_errors'])
        r=io.audit_episode(e,check_videos=False)
        self.assertFalse(r['structural_data_pass'])

    def test_only_rebuildable_intervention_errors_are_advisory(self):
        e=self.load()
        e['timeline_errors']=['control label mismatch: intervention_reasons']
        r=io.audit_episode(e)
        self.assertTrue(r['structural_pass'],r['errors'])
        self.assertEqual(r['historical_label_advisories'][0]['original_error'],e['timeline_errors'][0])
        e['timeline_errors'].append('accepted action measurement expired at guard check')
        self.assertFalse(io.audit_episode(e)['structural_pass'])


if __name__ == '__main__':
    unittest.main()
