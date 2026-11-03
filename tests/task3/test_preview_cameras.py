"""Current previews are top/front/oblique; only oblique is recorded for learning."""
import queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from simulation.mujoco.dual_arm.render_workers import MultiCameraRenderer, preview_worker
from simulation.mujoco.dual_arm.training_camera import OPERATOR_CAMERAS


class PreviewCameraTests(unittest.TestCase):
    def test_three_views_render_together_from_one_snapshot(self):
        self.assertEqual(OPERATOR_CAMERAS, ('top', 'front', 'oblique'))
        with patch('simulation.mujoco.dual_arm.render_workers.mujoco.Renderer') as factory:
            factory.return_value.render.return_value = np.zeros((480, 640, 3), np.uint8)
            renderer = MultiCameraRenderer(Mock(), camera_names=OPERATOR_CAMERAS)
            data = Mock()
            images = renderer.render(data)
            calls = factory.return_value.update_scene.call_args_list
            self.assertEqual(tuple(images), OPERATOR_CAMERAS)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(call.args[0] is data for call in calls))
            self.assertEqual(calls[0].kwargs['camera'], 'top')
            self.assertEqual(calls[1].kwargs['camera'], 'front')
            self.assertIs(calls[2].kwargs['camera'], renderer.training_camera)
            renderer.close()

    def test_each_worker_result_contains_only_current_three_views(self):
        snapshots, outputs, ready = queue.Queue(), queue.Queue(), queue.Queue()
        for captured_at, epoch in ((1., 2), (1.1, 2), (1.2, 3)):
            snapshots.put({'_preview': {'captured_at': captured_at, 'control_epoch': epoch, 'session': 'test'}})
        snapshots.put(None)
        controller = SimpleNamespace(model=Mock(), data=Mock())
        with patch('simulation.mujoco.dual_arm.task_controller.DualArmTaskController', return_value=controller), \
             patch('simulation.mujoco.dual_arm.render_workers.apply_snapshot'), \
             patch('simulation.mujoco.dual_arm.render_workers.MultiCameraRenderer') as factory:
            factory.return_value.render.return_value = {
                name: np.zeros((8, 8, 3), np.uint8) for name in OPERATOR_CAMERAS}
            preview_worker({}, snapshots, outputs, ready)
            self.assertEqual(factory.return_value.render.call_count, 3)
        self.assertEqual(ready.get_nowait()['type'], 'ready')
        for captured_at, epoch in ((1., 2), (1.1, 2), (1.2, 3)):
            result = outputs.get_nowait()
            self.assertEqual(tuple(result['images']), OPERATOR_CAMERAS)
            self.assertEqual(result['captured_at'], captured_at)
            self.assertEqual(result['control_epoch'], epoch)

    def test_backend_has_three_routes_and_no_separate_side(self):
        from simulation.mujoco.dual_arm.teleop_service import DualArmRecorder
        recorder = DualArmRecorder.__new__(DualArmRecorder)
        recorder.config = {}
        recorder.args = SimpleNamespace(port=5014)
        recorder.control_epoch = 2
        recorder.preview_frame = {'control_epoch': 2, 'captured_at': 99.9,
                                  'images': {name: name.encode() for name in OPERATOR_CAMERAS}}
        with patch('simulation.mujoco.dual_arm.teleop_service.make_server') as factory, \
             patch('simulation.mujoco.dual_arm.teleop_service.threading.Thread'):
            recorder.start_server()
            app = factory.call_args.args[2]
        client = app.test_client()
        with patch('time.monotonic', return_value=100.):
            for name in OPERATOR_CAMERAS:
                response = client.get(f'/{name}-preview')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.data, name.encode())
                self.assertAlmostEqual(float(response.headers['X-Preview-Age-Ms']), 100.)
            self.assertEqual(client.get('/side-preview').status_code, 404)
        self.assertEqual({f'/{name}-preview' for name in OPERATOR_CAMERAS},
                         {str(rule) for rule in app.url_map.iter_rules() if str(rule).endswith('-preview')})


if __name__ == '__main__':
    unittest.main()
