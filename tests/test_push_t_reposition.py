"""Push-T can change contact points without opening or losing its new plane."""
import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from simulation.mujoco.record_mujoco_panda import MujocoPandaRecorder, parse_args, DEFAULT_MODEL_PATH


class PushRepositionTest(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name)
        args=parse_args(['--headless-test','--task-mode','push_t',
            '--model-path',str(DEFAULT_MODEL_PATH.with_name('push_t_scene.xml')),
            '--dataset-root',str(root/'success'),'--failure-dataset-root',str(root/'failure')])
        with patch.object(MujocoPandaRecorder,'_start_render_workers',lambda _:None):
            self.s=MujocoPandaRecorder(args)
        self.addCleanup(self.s.research.close)

    def settle(self):
        for _ in range(40):
            mujoco.mj_step(self.s.model,self.s.data)

    def test_lift_then_relock_preserves_elevated_plane_and_closed_gripper(self):
        s=self.s;initial=s._commanded_hand_position().copy()
        s.handle_pushing_gesture({'command':'start_vertical'})
        self.assertFalse(s.z_locked)
        for _ in range(12):
            s.handle_pushing_gesture({'command':'up'});self.settle()
        raised=s._commanded_hand_position().copy()
        self.assertGreater(raised[2],initial[2]+.02)
        np.testing.assert_allclose(raised[:2],initial[:2],atol=.006)
        s.handle_pushing_gesture({'command':'start_planar'})
        self.assertTrue(s.z_locked)
        self.assertAlmostEqual(s.locked_z,raised[2],places=6)
        for _ in range(4):
            s.handle_pushing_gesture({'command':'move_xy','dx':.2,'dy':0})
            self.settle();s.maintain_locked_plane()
        self.assertLess(abs(s._commanded_hand_position()[2]-raised[2]),.004)
        for command in ('open','close'):
            s.handle_pushing_gesture({'command':command})
            self.assertEqual(s.data.ctrl[7],0)
            self.assertTrue(s.gripper_closed)

    def test_descent_requests_stop_at_initial_contact_height(self):
        s=self.s
        s.handle_pushing_gesture({'command':'start_vertical'})
        for _ in range(20):
            s.handle_pushing_gesture({'command':'down'});self.settle()
        self.assertGreaterEqual(s._commanded_hand_position()[2],s.push_contact_hand_z-.004)
        self.assertEqual(s.data.ctrl[7],0)

    def test_delayed_planar_command_cannot_cancel_vertical_mode(self):
        s=self.s;s.control_paused=False;s.control_token='test'
        now=time.time()*1000
        for payload in ({'command':'move_xy','dx':.5,'dy':0,'sentAt':now},
                        {'command':'start_vertical','sentAt':now+.01}):
            s.command_queue.put({**payload,'controlToken':'test'})
        s.process_queued_commands();self.assertFalse(s.z_locked)
        s.command_queue.put({'command':'move_xy','dx':.5,'dy':0,'sentAt':now,'controlToken':'test'})
        s.process_queued_commands();self.assertFalse(s.z_locked)
        s.handle_pushing_gesture({'command':'start_planar','sentAt':now+1})
        s.command_queue.put({'command':'down','sentAt':now+.5,'controlToken':'test'})
        s.process_queued_commands();self.assertTrue(s.z_locked)


if __name__=='__main__':unittest.main()
