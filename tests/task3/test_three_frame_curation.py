"""Current three-frame policy, with real-contract causal controls at each seam."""
import copy
import unittest

import numpy as np

from dataset.task3.curation_config import load_config, validate_config
from dataset.task3.curation_rules import select_rows, _confirm_three_frame_windows
from dataset.task3.sequence_stitching import verify_sequence_contract
from tests.task3.test_sequence_stitching import observations
from tests.task3.test_intent_curation import evidence_rows, active, execution


def wait_rows(count=80, start=15, stop=65, changed=False):
    rows=observations(count)
    for i, r in enumerate(rows):
        f=r['evidence']
        f['events']=[]
        f['accepted_intent']={s:{'mode':'HOLD' if start<=i<stop else 'Z_OPEN' if changed and i>=stop else 'XY_HOLD',
            'reliable':True,'event_id':f'{s}-{i}','measurement_id':f'm{i}'} for s in ('left','right')}
        f['relative_pose']={s:{'available':True,'position':[0.,0.,0.], 'rotation':np.eye(3).tolist()} for s in ('left','right')}
        f['object'].update(linear_speed_mps=0., angular_speed_radps=0.)
        f['task']={'done':False,'success':False}
        f['interval'].update(command_activity_available=True,command_activity=False,
            command_changed_from_previous=False,execution_available=True,execution_pending=False)
        f['motion']['moving']=not start<=i<stop
        r['status_summary']={'paused':False}
    return rows


class ThreeFrameCurationTests(unittest.TestCase):
    def test_current_defaults_and_reject_double_confirmation(self):
        c=load_config()
        self.assertEqual(c['sa']['window_frames'],2)
        self.assertNotIn('minimum_static_frames',c['sa'])
        self.assertEqual(c['waits']['same_intent_frames'],3)
        self.assertEqual(c['waits']['changed_intent_frames'],10)
        for k in ('minimum_middle_s','minimum_recenter_s','minimum_transition_s'):
            self.assertEqual(c['waits'][k],0)
        c['sa']['minimum_static_frames']=3
        with self.assertRaises(ValueError): validate_config(c)

    def test_one_window_backfills_three_frames_without_ab_veto(self):
        rows=evidence_rows(3)
        for r,v in zip(rows,[-.0015,.0015,0.]): r['state_recorded'][0]=v
        selected=select_rows(rows)['rows']
        self.assertTrue(all(r['sa_stagnant'] for r in selected))
        self.assertEqual(selected[0]['sa_stationarity']['run_length'],3)
        self.assertEqual(selected[2]['sa_window']['history_count'],2)

    def test_equality_is_candidate_and_two_frames_are_not_enough(self):
        rows=evidence_rows(3)
        rows[2]['state_recorded'][0]=load_config()['sa']['thresholds']['state_joint_rad']
        self.assertTrue(all(r['sa_stagnant'] for r in select_rows(rows)['rows']))
        self.assertFalse(any(r['sa_stagnant'] for r in select_rows(rows[:2])['rows']))

    def test_each_new_frame_checks_two_predecessors_and_stops_on_exceedance(self):
        rows=evidence_rows(6)
        for r,v in zip(rows,[0,0,0,.0015,.003,.0045]): r['state_recorded'][0]=v
        out=select_rows(rows)['rows']
        self.assertTrue(all(r['sa_stagnant'] for r in out[:4]))
        self.assertFalse(any(r['sa_stagnant'] for r in out[4:]))
        self.assertEqual(out[0]['sa_stationarity']['end_raw_frame'],3)

    def test_new_run_does_not_overlap_old_run(self):
        rows=evidence_rows(7)
        for r,v in zip(rows,[0,0,0,.01,.01,.01,.01]): r['state_recorded'][0]=v
        out=select_rows(rows)['rows']
        self.assertEqual(out[2]['sa_stationarity']['end_raw_frame'],2)
        self.assertEqual(out[3]['sa_stationarity']['start_raw_frame'],3)

    def test_invalid_or_reset_cannot_complete_window(self):
        for invalid in (False,True):
            rows=evidence_rows(4)
            if invalid: rows[2]['state_recorded'][0]=float('nan')
            else: rows[2]['evidence']['source']['hard_boundary']=True
            self.assertFalse(any(r['sa_stagnant'] for r in select_rows(rows)['rows']))

    def test_zero_dwell_only_retains_required_safe_seam_anchors(self):
        rows=wait_rows(); before=copy.deepcopy(rows)
        cfg=load_config(); cfg['waits']['same_intent_frames']=0
        out=select_rows(rows,cfg)
        self.assertEqual(rows,before)
        self.assertEqual(sum(r['keep'] for r in out['rows'][15:65]),2)
        self.assertTrue(out['rows'][15]['keep']); self.assertTrue(out['rows'][64]['keep'])
        self.assertTrue(all(not r['keep'] for r in out['rows'][16:64]))
        self.assertEqual(out['sequence_contract']['metrics']['safe_stitch_count'],1)
        self.assertEqual(out['sequence_contract']['metrics']['hard_gap_count'],0)
        verify_sequence_contract(out['rows'],out['sequence_contract'],cfg)
        for a,b in zip(rows,out['rows']): self.assertEqual(a['next_control_target'],b['next_control_target'])

    def test_same_intent_keeps_first_one_and_last_two_original_frames(self):
        rows=wait_rows()
        out=select_rows(rows)
        self.assertEqual([r['raw_frame_index'] for r in out['rows'][15:65] if r['keep']], [15,63,64])
        self.assertEqual(out['sequence_contract']['metrics']['safe_stitch_count'],1)
        self.assertEqual(out['sequence_contract']['metrics']['hard_gap_count'],0)
        verify_sequence_contract(out['rows'],out['sequence_contract'],load_config())
        for a,b in zip(rows,out['rows']): self.assertEqual(a['next_control_target'],b['next_control_target'])
        short=select_rows(wait_rows(stop=18))['rows']
        self.assertTrue(all(r['keep'] for r in short[15:18]))

    def test_fast_four_frame_wait_has_no_point_two_second_gate(self):
        out=select_rows(wait_rows(stop=19))
        self.assertFalse(out['rows'][16]['keep'])
        self.assertTrue(all(out['rows'][i]['keep'] for i in (15,17,18)))
        self.assertEqual(out['sequence_contract']['metrics']['safe_stitch_count'],1)

    def test_neighboring_cores_cannot_remove_each_others_seam_anchors(self):
        rows=wait_rows()
        # Slow subthreshold drift partitions the fixed-anchor physical cores.
        # Each certified local seam must retain its anchors after unioning cuts.
        for i,r in enumerate(rows):
            value=max(0,min(i-15,49))*.00004
            r['state_recorded'][0]=value
            for f in [r['evidence'],*r['evidence']['interval']['controls'],r['evidence']['next_control']]:
                f['features']['left.joints'][1][0]=value
        cfg=load_config(); cfg['waits']['same_intent_frames']=0
        out=select_rows(rows,cfg)
        verify_sequence_contract(out['rows'],out['sequence_contract'],cfg)
        self.assertEqual(out['sequence_contract']['metrics']['hard_gap_count'],0)
        self.assertGreater(out['sequence_contract']['metrics']['safe_stitch_count'],0)
        self.assertLess(sum(r['keep'] for r in out['rows'][15:65]),50)

    def test_transition_keeps_five_original_frames_at_each_end(self):
        out=select_rows(wait_rows(changed=True))['rows']
        self.assertEqual([r['raw_frame_index'] for r in out[15:65] if r['keep']],list(range(15,20))+list(range(60,65)))
        short=select_rows(wait_rows(stop=22,changed=True))['rows']
        self.assertTrue(all(r['keep'] for r in short[15:22]))

    def test_gripper_and_contact_events_are_not_dwell(self):
        for kind in ('close','contact'):
            rows=wait_rows()
            if kind=='close': execution(rows,35)
            else: rows[35]['evidence']['events']=[{'kind':'contact_changed','side':'left'}]
            out=select_rows(rows)['rows']
            self.assertTrue(all(r['keep'] for r in out[29:42]))

    def test_object_motion_and_other_hand_motion_stay(self):
        rows=wait_rows(); rows[40]['evidence']['motion']['moving']=True
        out=select_rows(rows)['rows']
        self.assertTrue(out[40]['keep'])
        rows=wait_rows()
        rows[40]['evidence']['interval']['controls'][0]['features']['right.target_position'][1][0]=.01
        out=select_rows(rows)
        self.assertEqual(out['sequence_contract']['metrics']['safe_stitch_count'],0)
        self.assertTrue(out['rows'][40]['keep'] or out['sequence_contract']['metrics']['hard_gap_count']>0)

    def test_original_hard_boundary_remains_separate(self):
        rows=wait_rows(); rows[40]['control_epoch']=2
        out=select_rows(rows)
        self.assertGreater(out['metrics']['task_segment_count'],1)

if __name__=='__main__': unittest.main()
