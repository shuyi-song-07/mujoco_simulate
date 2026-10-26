"""DONE plus a bounded physics tail ends an episode, not the save click."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

from simulation.mujoco.dual_arm.teleop_service import DualArmRecorder
from simulation.mujoco.dual_arm.recording_timeline import terminal_snapshot


class EpisodeCompletionTests(unittest.TestCase):
    def recorder(self, state='DONE'):
        r = DualArmRecorder.__new__(DualArmRecorder)
        r.recording = True
        r.paper_recording = True
        r.sampling_clock = object()
        r.session, r.control_epoch, r.tick = 'test', 4, 90
        r.started_at = 1.
        r.diagnostic = Mock()
        snapshot = {k: np.array([1.]) for k in ('state','action','qpos','qvel','ctrl','body_pos','body_quat')}
        snapshot.update(time=1.5, status={'task_state':state})
        r.controller = SimpleNamespace(get_status=lambda: snapshot['status'], snapshot=lambda: snapshot)
        return r

    def test_done_freezes_once_and_keeps_unsaved_recording(self):
        r = self.recorder()
        with patch('simulation.mujoco.dual_arm.teleop_service.capture_paper') as capture:
            r.capture()
            boundary = dict(r.episode_end)
            r.capture(force=True)
            r.capture(diagnostics_only=True)
        self.assertEqual(capture.call_count, 1)
        self.assertTrue(r.recording)  # Save buttons still have an unsaved episode.
        self.assertEqual(r.episode_end, boundary)
        self.assertEqual(boundary['tick'],90)
        self.assertEqual(boundary['simulation_time'],1.5)
        self.assertEqual(boundary['reason'],'task_done')

    def test_release_without_done_does_not_end_episode(self):
        r=self.recorder('PREGRASP')
        with patch('simulation.mujoco.dual_arm.teleop_service.capture_paper') as capture:
            r.capture();r.capture()
        self.assertEqual(capture.call_count,2)
        self.assertIsNone(getattr(r,'episode_end',None))

    def test_tail_records_until_half_second_with_shared_boundary(self):
        r = self.recorder()
        snapshot = r.controller.snapshot()
        snapshot['status']['success_tail'] = {'seconds':.5, 'end_time':2.}
        with patch('simulation.mujoco.dual_arm.teleop_service.capture_paper') as capture:
            r.capture()
            self.assertIsNone(getattr(r, 'episode_end', None))
            snapshot['time'] = 1.98
            r.capture()
            self.assertIsNone(getattr(r, 'episode_end', None))
            snapshot['time'] = 2.
            r.capture()
            r.capture()
        self.assertEqual(capture.call_count, 3)
        self.assertEqual(r.episode_end['success']['simulation_time'], 1.5)
        self.assertEqual(r.episode_end['simulation_time'], 2.)
        self.assertEqual(r.episode_end['post_success_seconds'], .5)

    def test_saved_terminal_uses_done_snapshot_not_later_physics(self):
        import json
        r=self.recorder()
        with patch('simulation.mujoco.dual_arm.teleop_service.capture_paper'):
            r.capture()
        r.controller.snapshot()['time']=100.
        r.frames=45;r.last_training_snapshot={'time':1.48,'qpos':np.array([1.])}
        r.paper_event_dropped=0;r.metadata=lambda outcome:{'outcome':outcome}
        with tempfile.TemporaryDirectory() as folder:
            r.pending=Path(folder)
            terminal_snapshot(r,'success')
            result=json.loads((r.pending/'terminal.json').read_text())
            self.assertEqual(result['time'],1.5)
            self.assertEqual(result['episode_end'],r.episode_end)

