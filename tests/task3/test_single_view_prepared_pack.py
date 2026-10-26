"""Formal single-view data contracts; no ACT models or training are imported."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from dataset.task3 import prepare_paper_dataset as prepared
from dataset.task3.curation_rules import load_config, effective_config
from dataset.task3.review_server import review_binding


def video(path, count=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for i in range(count):
            image = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i*20, np.uint8), format='rgb24')
            for packet in stream.encode(image):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def rows(session):
    output = []
    for i in range(8):
        aux = {'measurement_id': None, 'category': None, 'category_mask': False,
               'geometry_mask': False, 'task_geometry_mask': False}
        output.append({'raw_frame_index': i, 'original_time_s': i/30,
            'source_episode': 'ep', 'session': session, 'source_control_tick': i*2,
            'source_line': i*10,
            'control_epoch': 0, 'task_segment_id': 0, 'contiguous_fragment_id': int(i >= 5),
            'segment_id': int(i >= 5), 'state_recorded': [float(i)]*16,
            'action_recorded': [float(i)]*14, 'next_control_target': {
                'source_tick': i*2+1, 'control_epoch': 0, 'sim_time': i/30+1/60,
                'source_line': i*10+1,
                'alignment': 'next_observed_control_tick_after_post_step_snapshot',
                'action': [float(i)+.01]*14}, 'keep': i not in (3, 4),
            'effective_keep': i not in (3, 4), 'policy_decision': 'remove' if i in (3,4) else 'keep',
            'policy_reason': ['synthetic_test'], 'reason': ['synthetic_test'],
            'cut_reasons': [], 'soft_curation_gap': i in (3,4),
            'gap_before': {'kind': 'automatic_soft_gap'} if i == 5 else None,
            'auxiliary': {'left': dict(aux), 'right': dict(aux)},
            'robot_phase': 'free_motion' if i < 5 else 'releasing',
            'status_summary': {'paused': False},
            'terminal': {'actual_done': i == 7, 'success': i == 7},
            'evidence': {'control_period_s': 1/60, 'observation_period_s': 1/30,
                'source': {'valid': True, 'available': True,
                    'control_period_s': 1/60, 'observation_period_s': 1/30}}})
        output[-1]['evidence']['next_control'] = {
            'tick': i*2+1, 'control_epoch': 0, 'session': session, 'source_line': i*10+1,
            'sim_time': i/30+1/60, 'source': {'valid': True, 'available': True},
            'features': {f'{side}.{key}': [kind, [float(i)+.01]*length]
                for side in ('left','right') for key,kind,length in (
                    ('target_position','position',3), ('target_rpy','rpy',3),
                    ('target_aperture','aperture',1))}}
    return output


def fixture(root):
    run = root/'curation'; run.mkdir()
    raw = root/'raw'; raw.mkdir()
    cfg = load_config()
    episodes = []
    for number, session in enumerate(('train_session', 'val_session')):
        key = f'data_{session}/episode_000000'
        directory = run/'episodes'/key; directory.mkdir(parents=True)
        path = directory/'frames.jsonl'
        prepared.write_rows(path, rows(session))
        media = raw/f'camera_{number}.mp4'; video(media)
        source = {'camera': 'oblique', 'path': str(media), 'fps': 30, 'start_s': 0,
                  'end_s': 8/30, 'frame_count': 8}
        episodes.append({'key': key, 'session': session, 'structural_pass': True,
            'structural_data_pass': True, 'video_verified': True, 'source_fingerprints': [{
                'path': str(media), 'size': media.stat().st_size, 'mtime_ns': media.stat().st_mtime_ns,
                'sha256': hashlib.sha256(media.read_bytes()).hexdigest()}],
            'source_ref': {'key': key}, 'video_sources': [source],
            'review_video_sources': [dict(source, camera=c) for c in ('top','front','side')],
            'index_path': str(path.relative_to(run)),
            'index_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'metrics': {'raw_frames': 8}})
    manifest = {'version': prepared.VERSION, 'pipeline_stage': 'curation',
        'selection_policy': prepared.POLICY, 'source_root': str(raw), 'episodes': episodes,
        'config': cfg, 'config_sha256': prepared.digest(cfg),
        'effective_config_sha256': prepared.digest(effective_config(cfg)),
        'source_fingerprints_sha256': prepared.digest([])}
    from tasks.registry import get
    manifest.update(task_id='tray_transfer',task_binding=get('tray_transfer').binding(),engineering_only=True)
    prepared.write_json(run/'manifest.json', manifest)
    split = root/'split.json'
    prepared.write_json(split, {'train_sessions': ['train_session'], 'val_sessions': ['val_session'],
        'train': [episodes[0]['key']], 'val': [episodes[1]['key']]})
    return run, manifest, split


class SingleViewPreparedPackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run, self.manifest, self.split = fixture(self.root)
        self.loader = patch.object(prepared, 'load_episode', return_value={'load_errors': [], 'measurements': {}})
        self.loader.start(); self.addCleanup(self.loader.stop)

    def review(self, first):
        prepared.write_json(self.run/'reviews.json', {
            'version': 'papera-curation-review-v1', 'revision': 1,
            'binding': review_binding(self.manifest, self.run/'manifest.json'),
            'episodes': {self.manifest['episodes'][0]['key']: first}})

    def test_auto_exports_without_faking_review_and_never_recuts_pack(self):
        from dataset.task3.training_pack import build, verify, read_episode
        output = self.root/'prepared'
        result = prepared.prepare(self.run, output, review_mode='auto')
        self.assertEqual(result['review_mode'], 'auto')
        self.assertFalse(result['human_reviewed'])
        self.assertEqual([e['kept_frames'] for e in result['episodes']], [6, 6])
        self.assertEqual(result['episodes'][0]['review_status'], 'unreviewed')
        self.assertEqual([v['camera'] for v in result['episodes'][0]['video_sources']], ['oblique'])
        pack = self.root/'pack'
        report = build(output, pack, split_file=self.split, width=64, height=48)
        self.assertEqual(report['frame_count'], 12)
        self.assertFalse(report['training_ready'])
        self.assertTrue(verify(pack)['valid'])
        example = read_episode(pack, self.manifest['episodes'][0]['key'])
        np.testing.assert_array_equal(example['arrays']['raw_frame_index'], [0,1,2,5,6,7])
        self.assertEqual(example['images']['oblique'].shape, (6,48,64,3))
        self.assertEqual(example['diagnostics']['gaps'][0]['removed_frames'], 2)
        self.assertEqual(len(example['diagnostics']['contiguous_runs']), 2)
        self.assertGreater(example['arrays']['original_time_s'][3]-example['arrays']['original_time_s'][2], 1/30)

    def test_pending_manual_exclude_survives_auto_but_pending_keep_cannot_restore(self):
        self.review({'quality': 'pending', 'notes': '', 'exclude_intervals': [[1,2]],
                     'keep_intervals': [[3,4]]})
        result = prepared.prepare(self.run, self.root/'p', review_mode='auto')
        with np.load(self.root/'p'/result['episodes'][0]['arrays_path']) as a:
            np.testing.assert_array_equal(a['raw_frame_index'], [0,2,5,6,7])

    def test_reviewed_requires_AB_and_auto_respects_C(self):
        with self.assertRaisesRegex(ValueError, 'No A/B'):
            prepared.prepare(self.run, self.root/'blocked', review_mode='reviewed')
        self.review({'quality': 'C', 'reviewer': 'synthetic', 'watched_full': True,
            'reviewed_at': '2026-10-21T00:00:00+00:00', 'notes': 'synthetic rejection',
            'exclude_intervals': [], 'keep_intervals': []})
        result = prepared.prepare(self.run, self.root/'auto', review_mode='auto')
        self.assertEqual(len(result['episodes']), 1)
        self.assertEqual(result['selection'][0]['reason'], 'human_excluded_quality')

    def test_reviewed_AB_applies_manual_keep_with_real_review_status(self):
        self.review({'quality': 'B', 'reviewer': 'synthetic-only', 'watched_full': True,
            'reviewed_at': '2026-10-21T00:00:00+00:00', 'notes': 'synthetic label',
            'exclude_intervals': [[1,2]], 'keep_intervals': [[3,4]]})
        result = prepared.prepare(self.run,self.root/'p',review_mode='reviewed')
        self.assertEqual(len(result['episodes']),1)
        self.assertTrue(result['human_reviewed'])
        with np.load(self.root/'p'/result['episodes'][0]['arrays_path']) as a:
            np.testing.assert_array_equal(a['raw_frame_index'],[0,2,3,5,6,7])

    def test_auto_skips_explicit_exclusion_and_structurally_invalid(self):
        ep = self.manifest['episodes'][0]
        self.manifest['config']['quality']['excluded_episodes'] = [
            {'episode_key': ep['key'],'reason': 'synthetic explicit user exclusion'}]
        self.manifest['config_sha256'] = prepared.digest(self.manifest['config'])
        self.manifest['effective_config_sha256'] = prepared.digest(effective_config(self.manifest['config']))
        prepared.write_json(self.run/'manifest.json',self.manifest)
        result = prepared.prepare(self.run,self.root/'p',review_mode='auto')
        self.assertEqual(len(result['episodes']),1)
        self.assertIn('explicit_quality_exclusion',result['selection'][0]['reason'])
        self.manifest['episodes'][1]['structural_pass'] = False
        prepared.write_json(self.run/'manifest.json',self.manifest)
        with self.assertRaisesRegex(ValueError,'No structurally valid'):
            prepared.prepare(self.run,self.root/'none',review_mode='auto')
        self.assertFalse((self.root/'none').exists())

    def test_legacy_three_camera_source_cannot_enter_single_view_prepared(self):
        self.manifest['episodes'][0]['video_sources'][0]['camera'] = 'top'
        prepared.write_json(self.run/'manifest.json',self.manifest)
        with self.assertRaisesRegex(ValueError,'oblique'):
            prepared.prepare(self.run,self.root/'blocked',review_mode='auto')
        self.assertFalse((self.root/'blocked').exists())

    def test_failed_episode_isolated_and_resume_reuses_finished_cache(self):
        from dataset.task3 import training_pack as packer
        prepared.prepare(self.run,self.root/'p',review_mode='auto')
        genuine = packer.cache_video
        count = [0]
        def failing_once(source,*args,**kwargs):
            count[0] += 1
            if count[0] == 1:
                raise RuntimeError('synthetic decoder interruption')
            return genuine(source,*args,**kwargs)
        with patch.object(packer,'cache_video',side_effect=failing_once):
            with self.assertRaisesRegex(ValueError,'1 pack episodes failed'):
                packer.build(self.root/'p',self.root/'pack',split_file=self.split,
                    width=64,height=48,continue_on_error=True)
        self.assertFalse((self.root/'pack/manifest.json').exists())
        with patch.object(packer,'cache_video',wraps=genuine) as decoder:
            result = packer.build(self.root/'p',self.root/'pack',split_file=self.split,
                width=64,height=48,resume=True)
            self.assertEqual(decoder.call_count,1)
        self.assertEqual(result['frame_count'],12)

    def test_auto_honors_whole_episode_manual_exclusion(self):
        self.review({'quality': 'pending','notes': '', 'exclude_intervals': [[0,8]],'keep_intervals': []})
        result = prepared.prepare(self.run,self.root/'p',review_mode='auto')
        self.assertEqual(len(result['episodes']),1)
        self.assertEqual(result['selection'][0]['reason'],'no_frames_after_manual_exclusion')

    def test_auxiliary_classes_geometry_and_confidence_stay_frame_aligned(self):
        from dataset.task3.training_pack import build, read_episode
        ep = self.manifest['episodes'][0]
        index = self.run/ep['index_path']
        data = prepared.read_rows(index)
        landmarks = np.arange(63,dtype=float).reshape(21,3)/100
        for row in data:
            row['auxiliary']['left'].update(measurement_id='m',category='ILoveYou',category_score=.01,
                geometry_mask=True,task_geometry_mask=False)
            row['auxiliary']['right'].update(category='Closed_Fist',category_mask=True,category_score=.99)
        prepared.write_rows(index,data)
        ep['index_sha256'] = hashlib.sha256(index.read_bytes()).hexdigest()
        prepared.write_json(self.run/'manifest.json',self.manifest)
        self.loader.stop()
        with patch.object(prepared,'load_episode',return_value={'load_errors': [],'measurements': {'m': {
                'hands': [{'bound_side': 'left','image_landmarks_21': landmarks.tolist(),
                    'hand_center_landmarks_m_21': landmarks.tolist()}]}}}):
            prepared.prepare(self.run,self.root/'p',review_mode='auto')
        build(self.root/'p',self.root/'pack',split_file=self.split,width=64,height=48)
        example = read_episode(self.root/'pack',ep['key'])
        a = example['arrays']
        self.assertTrue((a['category_id'] == [0,2]).all())
        self.assertFalse(a['category_mask'][:,0].any())
        self.assertTrue(a['category_mask'][:,1].all())
        self.assertTrue(a['geometry_mask'][:,0].all())
        self.assertFalse(a['task_geometry_mask'].any())
        self.assertEqual(example['rows'][0]['auxiliary']['left']['category_score'],.01)

    def test_stale_target_and_split_leak_fail_before_output(self):
        result = prepared.prepare(self.run, self.root/'p', review_mode='auto')
        from dataset.task3.training_pack import build
        prepared.write_json(self.split, {'train_sessions': ['train_session'],
            'val_sessions': ['train_session','val_session'], 'train': [], 'val': []})
        with self.assertRaisesRegex(ValueError, 'overlap'):
            build(self.root/'p', self.root/'pack', split_file=self.split, width=64,height=48)
        self.assertFalse((self.root/'pack').exists())
        ep = result['episodes'][0]
        index = self.root/'p'/ep['index_path']
        data = prepared.read_rows(index)
        data[0]['next_control_target']['source_tick'] = 999
        prepared.write_rows(index, data)
        ep['index_sha256'] = hashlib.sha256(index.read_bytes()).hexdigest()
        prepared.write_json(self.root/'p'/'manifest.json', result)
        with self.assertRaisesRegex(ValueError, 'next control|causal|tick|sequence'):
            build(self.root/'p', self.root/'invalid', split_file=self.root/'split_good.json', width=64,height=48)

    def test_resume_detects_tampered_cache(self):
        from dataset.task3.training_pack import build, verify
        prepared.prepare(self.run, self.root/'p', review_mode='auto')
        pack = self.root/'pack'
        result = build(self.root/'p', pack, split_file=self.split, width=64,height=48)
        cache = pack/result['episodes'][0]['image_paths']['oblique']
        with cache.open('ab') as stream:
            stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash|changed'):
            verify(pack)
        with self.assertRaisesRegex(ValueError, 'hash|changed'):
            build(self.root/'p', pack, split_file=self.split, width=64,height=48,resume=True)

    def prepared_inputs(self):
        result = prepared.prepare(self.run,self.root/'p',review_mode='auto')
        entry = result['episodes'][0]
        with np.load(self.root/'p'/entry['arrays_path']) as bundle:
            arrays = {name: bundle[name] for name in bundle.files}
        self.sequence = json.loads((self.root/'p'/entry['sequence_path']).read_text(encoding='utf-8'))
        self.config = result['config']
        return arrays,prepared.read_rows(self.root/'p'/entry['index_path'])

    def test_source_clock_masks_cannot_disagree_with_row_presence(self):
        from dataset.task3.training_pack import derive_arrays
        arrays,index = self.prepared_inputs()
        for name in ('source_control_tick_mask','control_epoch_mask'):
            with self.subTest(name=name):
                changed = copy.deepcopy(arrays)
                changed[name][0] = False
                with self.assertRaisesRegex(ValueError,'mask'):
                    derive_arrays(changed,index,fps=30,sequence=self.sequence,config=self.config)

    def test_causal_target_cannot_precede_observation_in_log_order(self):
        from dataset.task3.training_pack import derive_arrays
        arrays,index = self.prepared_inputs()
        index[0]['source_line'] = 100
        index[0]['next_control_target']['source_line'] = 99
        with self.assertRaises(ValueError):
            derive_arrays(arrays,index,fps=30,sequence=self.sequence,config=self.config)

    def test_fragment_or_task_label_alone_cannot_fabricate_a_time_boundary(self):
        from dataset.task3.training_pack import derive_arrays
        arrays,index = self.prepared_inputs()
        for name in ('contiguous_fragment_id','task_segment_id'):
            with self.subTest(name=name):
                changed,changed_index = copy.deepcopy(arrays),copy.deepcopy(index)
                changed[name][1] = 100
                changed_index[1][name] = 100
                with self.assertRaises(ValueError):
                    derive_arrays(changed,changed_index,fps=30,sequence=self.sequence,config=self.config)

    def test_real_contiguous_mode_transition_remains_contiguous(self):
        from dataset.task3.training_pack import derive_arrays
        arrays,index = self.prepared_inputs()
        index[1]['accepted_left_intent'] = 'Z_OPEN'
        index[1]['robot_phase'] = 'free_stationary'
        _,report = derive_arrays(arrays,index,fps=30,sequence=self.sequence,config=self.config)
        self.assertEqual(len(report['gaps']),1)  # only the real raw2 -> raw5 cut
        transition = next(t for t in report['transitions'] if t['raw_frame'] == 1)
        self.assertFalse(transition['crosses_gap'])

    def test_virtual_arrays_cannot_change_without_revalidating_evidence(self):
        from dataset.task3.training_pack import derive_arrays
        arrays,index = self.prepared_inputs()
        for name in ('training_frame_index','training_time_s','train_sequence_id'):
            changed = copy.deepcopy(arrays)
            changed[name][2] += 1
            with self.subTest(name=name), self.assertRaises(ValueError):
                derive_arrays(changed,index,fps=30,sequence=self.sequence,config=self.config)

    def test_certificates_cannot_silently_be_dropped(self):
        from dataset.task3.training_pack import derive_arrays
        arrays,index = self.prepared_inputs()
        with self.assertRaisesRegex(ValueError, 'certificate'):
            derive_arrays(arrays,index,fps=30)

    def test_prepared_parameter_edit_fails_before_allocating_a_pack(self):
        from dataset.task3.training_pack import build
        result = prepared.prepare(self.run,self.root/'p',review_mode='auto')
        result['config']['physical_motion']['position_m'] *= 1.01
        prepared.write_json(self.root/'p/manifest.json', result)
        with self.assertRaisesRegex(ValueError, 'configuration hash'):
            build(self.root/'p',self.root/'pack',split_file=self.split,width=64,height=48)
        self.assertFalse((self.root/'pack').exists())

    def test_direct_reader_checks_global_frozen_config_binding(self):
        from dataset.task3.training_pack import build, read_episode
        prepared.prepare(self.run,self.root/'p',review_mode='auto')
        result = build(self.root/'p',self.root/'pack',split_file=self.split,width=64,height=48)
        result['curation_config']['physical_motion']['position_m'] *= 1.01
        prepared.write_json(self.root/'pack/manifest.json', result)
        with self.assertRaisesRegex(ValueError, 'config contract'):
            read_episode(self.root/'pack',result['episodes'][0]['key'],verify_files=False)


if __name__ == '__main__':
    unittest.main()
