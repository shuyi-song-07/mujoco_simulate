"""The same frozen training view is shared by replay and future recordings."""
import copy
import unittest
from unittest.mock import Mock, patch

import numpy as np

from simulation.mujoco.dual_arm.config import load_config
from simulation.mujoco.dual_arm.render_workers import dataset_features, MultiCameraRenderer
from simulation.mujoco.dual_arm.training_camera import (OPERATOR_CAMERAS, RECORDING_CAMERAS, LEGACY_CAMERAS,
    training_camera_spec, make_training_camera, recording_camera_names, training_camera_metadata)


class RecordingCameraContractTests(unittest.TestCase):
    def test_paper_recording_declares_only_training_rgb(self):
        from collection.config import recording_config
        from tasks.registry import get
        task = get('tray_transfer')
        config = recording_config(task)
        self.assertEqual(config['task_binding'], task.binding())
        self.assertEqual(task.config()['recording']['cameras'], list(OPERATOR_CAMERAS))
        features = dataset_features(config=config)
        cameras = {key.removeprefix('observation.images.') for key in features
                   if key.startswith('observation.images.')}
        self.assertEqual(cameras, {'oblique'})
        self.assertEqual(features['observation.images.oblique']['shape'], (480, 640, 3))

    def test_historical_four_camera_config_still_loads(self):
        config = load_config('config/collection.json')
        config['recording']['cameras'] = list(RECORDING_CAMERAS)
        self.assertEqual(recording_camera_names(load_config(config)), RECORDING_CAMERAS)

    def test_legacy_full_config_remains_byte_equivalent_and_has_three_views(self):
        config = load_config()
        frozen = copy.deepcopy(config)
        self.assertEqual(recording_camera_names(load_config(config)), LEGACY_CAMERAS)
        self.assertEqual(config, frozen)
        self.assertNotIn('training_camera', load_config(config))

    def test_shared_camera_matches_verified_pilot_and_returns_independent_copies(self):
        import mujoco
        camera = make_training_camera()
        self.assertEqual(camera.type, mujoco.mjtCamera.mjCAMERA_FREE)
        self.assertEqual((camera.distance, camera.azimuth, camera.elevation), (1.4, 150., -35.))
        np.testing.assert_array_equal(camera.lookat, [-.1, -.1, .5])
        altered = training_camera_spec()
        altered['lookat'][0] = 99
        self.assertEqual(training_camera_spec()['lookat'][0], -.1)
        self.assertEqual(len(training_camera_metadata()['spec_sha256']), 64)

    def test_configurable_pose_reaches_collection_and_renderer(self):
        from collection.config import recording_config
        from tasks.registry import get
        config = recording_config(get('tray_to_platform'))
        config['training_camera'] = {**training_camera_spec(),
            'version': 'papera-oblique-config-v1', 'distance': 1.3,
            'azimuth': 145., 'elevation': -40., 'lookat': [0., -.05, .45]}
        frozen = copy.deepcopy(config)
        loaded = load_config(config)
        camera = make_training_camera(loaded)
        self.assertEqual(camera.distance, 1.3)
        self.assertEqual(camera.azimuth, 145.)
        np.testing.assert_array_equal(camera.lookat, [0., -.05, .45])
        self.assertEqual(config, frozen)
        old_sha = training_camera_metadata()['spec_sha256']
        self.assertNotEqual(training_camera_metadata(loaded)['spec_sha256'], old_sha)
        changed = copy.deepcopy(loaded)
        changed['training_camera']['distance'] = 1.2
        self.assertNotEqual(training_camera_metadata(changed)['spec_sha256'],
                            training_camera_metadata(loaded)['spec_sha256'])
        with patch('simulation.mujoco.dual_arm.render_workers.mujoco.Renderer'):
            renderer = MultiCameraRenderer(Mock(), config=loaded, camera_names=('oblique',))
            self.assertEqual(renderer.training_camera.distance, 1.3)
            renderer.close()

    def test_configurable_camera_rejects_invalid_values(self):
        spec = {**training_camera_spec(), 'version': 'papera-oblique-config-v1'}
        for field, value in [('distance', 0), ('distance', -1), ('distance', True),
                ('distance', float('nan')), ('azimuth', float('inf')), ('elevation', -91),
                ('lookat', [0, 0]), ('lookat', [0, False, .5]), ('lookat', [0, 0, float('nan')]),
                ('width', 320), ('fps', 20), ('version', 'typo'), ('extra', 1)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                training_camera_spec({'training_camera': {**spec, field: value}})

    def test_invalid_recording_view_or_nonfrozen_camera_is_rejected(self):
        for field, value in (('cameras', ['top', 'top']), ('cameras', ['top', 'oblique']),
                             ('width', 320), ('fps', 20)):
            with self.subTest(field=field, value=value):
                config = load_config('config/collection.json')
                config['recording'][field] = value
                with self.assertRaises(ValueError):
                    load_config(config)
        config = load_config('config/collection.json')
        config['training_camera'] = training_camera_spec()
        config['training_camera']['lookat'][0] = 0
        with self.assertRaisesRegex(ValueError, 'frozen'):
            load_config(config)
        with self.assertRaisesRegex(ValueError, '640x480'):
            MultiCameraRenderer(Mock(), width=320, height=240, camera_names=('oblique',))

    def test_four_views_share_the_exact_data_and_never_step_physics(self):
        model, data = Mock(), Mock()
        with patch('simulation.mujoco.dual_arm.render_workers.mujoco.Renderer') as factory:
            factory.return_value.render.return_value = np.zeros((480, 640, 3), np.uint8)
            renderer = MultiCameraRenderer(model, camera_names=RECORDING_CAMERAS)
            images = renderer.render(data)
            self.assertEqual(tuple(images), RECORDING_CAMERAS)
            calls = factory.return_value.update_scene.call_args_list
            self.assertEqual(len(calls), 4)
            self.assertTrue(all(call.args[0] is data for call in calls))
            self.assertEqual([call.kwargs['camera'] for call in calls[:3]], list(LEGACY_CAMERAS))
            self.assertIs(calls[3].kwargs['camera'], renderer.training_camera)
            renderer.close()

    def test_current_three_views_render_oblique_with_identical_policy_appearance(self):
        from dataset.task3.curation_io import camera_layout
        import json
        config=load_config('config/collection.json')
        self.assertEqual(recording_camera_names(config),('top','front','oblique'))
        features=json.loads(json.dumps(dataset_features(config=config)))
        self.assertEqual(camera_layout({'features':features}),OPERATOR_CAMERAS)
        with patch('simulation.mujoco.dual_arm.render_workers.mujoco.Renderer') as factory, \
             patch('simulation.mujoco.dual_arm.render_workers.apply_operator_view') as style:
            factory.return_value.render.return_value=np.zeros((480,640,3),np.uint8)
            renderer=MultiCameraRenderer(Mock(),config=config)
            images=renderer.render(Mock(),operator_view=config)
            self.assertEqual(tuple(images),OPERATOR_CAMERAS)
            self.assertEqual([call.args[2] for call in style.call_args_list],['top','front'])
            self.assertIs(factory.return_value.update_scene.call_args.kwargs['camera'],renderer.training_camera)
            renderer.close()

    def test_rollout_all_views_uses_same_three_cameras(self):
        from rollout.rollout import render_camera_names, compare_initial_observations
        self.assertEqual(render_camera_names({'cameras':['oblique']},{'media_mode':'all_views'}),OPERATOR_CAMERAS)
        self.assertEqual(render_camera_names({'cameras':['oblique']},{'media_mode':'policy_only'}),('oblique',))
        self.assertEqual(render_camera_names({'cameras':list(LEGACY_CAMERAS)},{}),LEGACY_CAMERAS)
        reference={name:np.zeros((480,640,3),np.uint8) for name in OPERATOR_CAMERAS}
        reference['state']=np.zeros(16,np.float32)
        self.assertTrue(compare_initial_observations(reference,reference)['compatible'])
        changed={**reference,'oblique':np.full((480,640,3),255,np.uint8)}
        self.assertFalse(compare_initial_observations(reference,changed)['compatible'])

    def test_spooling_frames_does_not_render_before_save(self):
        import queue
        import tempfile
        from pathlib import Path
        from simulation.mujoco.dual_arm.render_workers import recording_worker
        config = load_config('config/collection.json')
        commands, replies = queue.Queue(), queue.Queue()
        with tempfile.TemporaryDirectory() as directory:
            commands.put({'type': 'start', 'pending': directory})
            commands.put({'type': 'frame', 'snapshot': {
                'qpos': [0], 'qvel': [0], 'ctrl': [0], 'body_pos': [[0, 0, 0]],
                'time': 0, 'state': [0]*16, 'action': [0]*14}})
            commands.put({'type': 'flush'})
            commands.put({'type': 'discard'})
            commands.put({'type': 'close'})
            with patch('simulation.mujoco.dual_arm.render_workers.MultiCameraRenderer') as renderer:
                recording_worker(config, commands, replies)
                renderer.return_value.render.assert_not_called()
            messages = []
            while not replies.empty():
                messages.append(replies.get_nowait())
            self.assertNotIn('fatal', [message['type'] for message in messages], messages)
            self.assertEqual([message['frames'] for message in messages if message['type'] == 'flushed'], [1])
            self.assertTrue((Path(directory) / 'snapshots_000000.npz').is_file())


if __name__ == '__main__':
    unittest.main()
