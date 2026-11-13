"""Single-arm schema uses the existing evidence/curation/sequence rules."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tasks.data_schema import DUAL, SINGLE, SIDES, descriptor, schema_context, schema_for
from dataset.task3.curation_config import load_config
from dataset.task3.curation_rules import group_deltas, select_rows
from dataset.task3.sequence_stitching import build_sequence_contract, verify_sequence_contract
from dataset.task3.prepare_paper_dataset import normalized_geometry
from tests.task3.test_three_frame_curation import wait_rows
from tests.task3.test_single_view_prepared_pack import rows as pack_rows


def single(value):
    """Construct one-arm test facts; no absent right arm is emitted."""
    if isinstance(value, list): return [single(v) for v in value]
    if not isinstance(value, dict): return copy.deepcopy(value)
    result = {}
    for key, item in value.items():
        if key == 'right' or key.startswith('right.') or key.startswith('right_'):
            continue
        if key in ('state_recorded', 'state') and isinstance(item, list) and len(item) == 16:
            result[key] = item[:7] + [0.] + item[7:8]
        elif key in ('action_recorded', 'action') and isinstance(item, list) and len(item) == 14:
            result[key] = item[:7]
        elif key == 'left.joints':
            result[key] = [item[0], item[1] + [0.]]
        elif key == 'ee_speeds_mps': result[key] = item[:1]
        else: result[key] = single(item)
    if 'state_recorded' in result: result['data_schema'] = descriptor(SINGLE)
    return result


class SingleArmSchemaTests(unittest.TestCase):
    def test_schema_is_explicit_and_threads_are_isolated(self):
        def check(layout):
            with schema_context(layout):
                return tuple(SIDES), schema_for().state_dim, schema_for().action_dim
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(check, [DUAL, SINGLE]*20))
        self.assertEqual(values, [(('left', 'right'), 16, 14), (('left',), 9, 7)]*20)
        self.assertEqual(schema_for(), DUAL)
        with self.assertRaises(ValueError): schema_for({**descriptor(SINGLE), 'arm_order': ['right']})
        with self.assertRaises(ValueError): schema_for({'data_schema': descriptor(SINGLE), 'state_dim': 16})

    def test_snapshot_quaternion_and_control_matrix_use_same_rotation(self):
        from dataset.task3.frame_evidence import _rotation_distance
        identity = np.eye(3)
        self.assertEqual(_rotation_distance(np.array([1.,0.,0.,0.]), identity), 0.)
        self.assertEqual(_rotation_distance(identity, np.array([1.,0.,0.,0.])), 0.)
        angle = .03
        q = np.array([np.cos(angle/2), 0., 0., np.sin(angle/2)])
        self.assertAlmostEqual(_rotation_distance(identity, q), angle)

    def test_yaw_is_a_real_joint_not_gripper_width(self):
        with schema_context(SINGLE):
            state = np.zeros(9); past = np.zeros((2, 9)); action = np.zeros(7)
            state[7] = .003
            delta = group_deltas(state, action, past, [action, action])
            self.assertEqual(delta['state_joint_rad'], .003)
            self.assertEqual(delta['state_aperture_m'], 0.)
            state[8] = .001
            self.assertEqual(group_deltas(state, action, past, [action, action])['state_aperture_m'], .001)

    def test_identical_wait_policy_and_verified_seam(self):
        cfg = load_config()
        cfg['waits']['same_intent_frames'] = 0
        dual = select_rows(wait_rows(), cfg)
        one = select_rows(single(wait_rows()), cfg)
        self.assertEqual([r['keep'] for r in one['rows']], [r['keep'] for r in dual['rows']])
        self.assertGreater(one['sequence_contract']['metrics']['safe_stitch_count'], 0)
        verify_sequence_contract(one['rows'], one['sequence_contract'], cfg)
        self.assertTrue(all(len(r['state_recorded']) == 9 and len(r['action_recorded']) == 7 for r in one['rows']))
        self.assertTrue(all('accepted_right_intent' not in r and
                            'right_z_m' not in r['height_evidence'] for r in one['rows']))

    def test_missing_contact_evidence_cannot_be_filled_with_false(self):
        rows = single(wait_rows())
        for r in rows:
            r['evidence']['contact'].pop('left', None)
        out = select_rows(rows)
        self.assertTrue(all(r['keep'] for r in out['rows'][15:65]))

    def test_next_control_identity_remains_required(self):
        rows = single(pack_rows('single-session'))
        result = build_sequence_contract(rows)
        self.assertTrue(result['rows'][0]['causal_target_valid'])
        rows[0]['next_control_target']['source_tick'] += 1
        changed = build_sequence_contract(rows)
        self.assertFalse(changed['rows'][0]['causal_target_valid'])

    def test_prepared_and_pack_arrays_have_one_hand_only(self):
        from dataset.task3.training_pack import derive_arrays
        rows = single(pack_rows('single-session'))
        sequence = build_sequence_contract(rows)
        rows = sequence.pop('rows')
        kept = [r for r in rows if r['keep']]
        arrays = normalized_geometry(kept, {})
        arrays.update(state=np.asarray([r['state_recorded'] for r in kept], np.float32),
            action_recorded=np.asarray([r['action_recorded'] for r in kept], np.float32),
            next_control_target=np.asarray([r['next_control_target']['action'] for r in kept], np.float32),
            next_control_target_mask=np.ones(len(kept), bool))
        for key in ('raw_frame_index', 'source_control_tick', 'control_epoch', 'original_time_s',
                    'task_segment_id', 'contiguous_fragment_id', 'segment_id',
                    'training_frame_index', 'training_time_s', 'train_sequence_id',
                    'sequence_frame_index', 'sequence_time_s', 'causal_target_valid'):
            arrays[key] = np.asarray([r[key] for r in kept])
        arrays['source_control_tick_mask'] = arrays['control_epoch_mask'] = np.ones(len(kept), bool)
        out, _ = derive_arrays(arrays, rows, fps=30, sequence=sequence, config=load_config())
        self.assertEqual(out['state'].shape, (6, 9))
        self.assertEqual(out['action'].shape, (6, 7))
        self.assertEqual(out['category_id'].shape, (6, 1))
        self.assertEqual(out['hand_image'].shape, (6, 1, 21, 3))

    def test_full_prepared_pack_video_index_and_schema(self):
        from dataset.task3 import prepare_paper_dataset as prepared
        from dataset.task3.training_pack import build, verify, read_episode
        from tasks.registry import get
        from tests.task3.test_single_view_prepared_pack import fixture
        temporary_root = Path(__file__).resolve().parents[2]/'.local/checks/single_arm_push_t/tmp'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            run, manifest, split = fixture(root)
            manifest.update(task_id='push_t', task_binding=get('push_t').binding(), data_schema=descriptor(SINGLE))
            for ep in manifest['episodes']:
                path = run/ep['index_path']
                prepared.write_rows(path, single(prepared.read_rows(path)))
                ep['index_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            prepared.write_json(run/'manifest.json', manifest)
            with patch.object(prepared, 'load_episode', return_value={'load_errors': [], 'measurements': {}}):
                result = prepared.prepare(run, root/'prepared', review_mode='auto')
            self.assertEqual((result['state_dim'], result['action_dim'], result['hand_order']), (9,7,['left']))
            build(root/'prepared', root/'pack', split_file=split, width=64, height=48)
            self.assertTrue(verify(root/'pack')['valid'])
            example = read_episode(root/'pack', manifest['episodes'][0]['key'])
            self.assertEqual(example['arrays']['state'].shape, (6,9))
            self.assertEqual(example['images']['oblique'].shape, (6,48,64,3))
            np.testing.assert_array_equal(example['arrays']['raw_frame_index'], [0,1,2,5,6,7])
            example['images']['oblique']._mmap.close()


if __name__ == '__main__': unittest.main()
