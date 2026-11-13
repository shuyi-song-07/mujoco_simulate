"""Single-camera causality and controller hold checks with synthetic evidence."""
import copy
import unittest
import numpy as np

from tasks.registry import get
from simulation.mujoco.single_arm.task_controller import SingleArmTaskController
from simulation.mujoco.dual_arm.hand_measurement import HandInputGuard, RATE_PROTOCOL
from tests.task3.gesture_fixture import hand


def packet(i=0,mode='XY_HOLD'):
    stamp=100000+i*50
    shape={'XY_HOLD':'ily','Z_OPEN':'palm','Z_CLOSE':'fist','HOLD':'unknown'}[mode]
    measurement={'schema_version':'task3-hand-v1','measurement_id':f'synthetic-push:{i}',
        'source_session':'session','source_control_epoch':0,'capture_epoch_ms':stamp,
        'capture_browser_monotonic_ms':i*50,'accepted_recognition':True,'width':640,'height':480,
        'hands':[hand(shape,'left')]}
    return {'command':'continuous_input','control_protocol':RATE_PROTOCOL,'interaction_modes':[mode],
        'mode':'XY' if mode=='XY_HOLD' else 'Z','release_intent':[mode=='Z_OPEN'],
        'delta_task_m':{'left':[.001,0.,0.] if mode=='XY_HOLD' else [0.,0.,0.]},
        'widths_m':[0.],'measurement_id':measurement['measurement_id'],'sentAt':stamp,
        'eventId':f'synthetic-push-event:{i}','telemetry':{'calibrated':True,
            'hands':{'left':{'visible':True}},'control_mode':'XY' if mode=='XY_HOLD' else 'Z',
            'aperture_calibration':{'phase':'NOT_REQUIRED','reason':'categorical_rate_control'},
            'calibration':{'palm_binding_guard':False},'recognition_result':{'measurement':measurement}}}


class SinglePushGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.config=get('push_t').config()

    def check(self,guard,p,**kw):
        return guard.check(p,session=kw.get('session','session'),epoch=kw.get('epoch',0),
            now_ms=p['sentAt']+kw.get('age',10),control_protocol=RATE_PROTOCOL)

    def ready_guard(self):
        guard=HandInputGuard(('left',))
        for i in range(5):
            self.assertEqual(self.check(guard,packet(i)),'tracking_recovery' if i<4 else None)
        return guard

    def test_single_trusted_hand_recovers_then_moves_locked_gripper(self):
        guard=self.ready_guard();c=SingleArmTaskController(self.config);c.pause('hand_lost')
        p=packet(5);self.assertIsNone(self.check(guard,p))
        before=c.action_vector_m().copy();result=c.apply_command(p,now_ms=p['sentAt']+10)
        self.assertTrue(result['accepted']);self.assertFalse(c.paused)
        self.assertGreater(c.action_vector_m()[0],before[0]);self.assertEqual(c.action_vector_m()[-1],0.)

    def test_faults_invalidate_recovery_without_issuing_motion(self):
        faults=('stale','future','wrong_session','wrong_epoch','identity','missing','duplicate_hand',
                'wrong_side','wrong_plane','near_edge','nan','binding','wrong_intent','wrong_gesture')
        for fault in faults:
            with self.subTest(fault=fault):
                guard=self.ready_guard();p=packet(5);measurement=p['telemetry']['recognition_result']['measurement']
                h=measurement['hands'][0];options={}
                if fault=='stale':options['age']=151
                if fault=='future':options['age']=-1
                if fault=='wrong_session':measurement['source_session']='other'
                if fault=='wrong_epoch':measurement['source_control_epoch']=1
                if fault=='identity':h['identity_score']=.79
                if fault=='missing':measurement['hands']=[]
                if fault=='duplicate_hand':measurement['hands'].append(copy.deepcopy(h))
                if fault=='wrong_side':h['bound_side']='right'
                if fault=='wrong_plane':p['mode']='Z'
                if fault=='near_edge':h['image_landmarks_21'][0][0]=.05
                if fault=='nan':h['image_landmarks_21'][0][0]=float('nan')
                if fault=='binding':p['telemetry']['calibration']['palm_binding_guard']=True
                if fault=='wrong_intent':p['release_intent']=[True]
                if fault=='wrong_gesture':h['gesture']='Open_Palm'
                self.assertIsNotNone(self.check(guard,p,**options));self.assertEqual(guard.samples,0)

    def test_measurement_and_event_dedup_are_independent(self):
        guard=self.ready_guard();p=packet(5);self.assertIsNone(self.check(guard,p))
        self.assertEqual(self.check(guard,p),'duplicate_measurement')
        c=SingleArmTaskController(self.config);self.assertTrue(c.apply_command(p,now_ms=p['sentAt']+10)['accepted'])
        before=c.action_vector_m();next_packet=packet(6);next_packet['eventId']=p['eventId']
        self.assertEqual(c.apply_command(next_packet,now_ms=next_packet['sentAt']+10)['reason'],'duplicate_event')
        np.testing.assert_array_equal(before,c.action_vector_m())

    def test_controller_direct_rejects_stale_order_and_wrong_plane(self):
        c=SingleArmTaskController(self.config);p=packet(5)
        self.assertTrue(c.apply_command(p,now_ms=p['sentAt']+10)['accepted']);before=c.action_vector_m()
        for bad,now,reason in ((packet(4),100260,'out_of_order_command'),
                              (packet(6),100451,'stale_command')):
            self.assertEqual(c.apply_command(bad,now_ms=now)['reason'],reason)
        p=packet(6);p['mode']='Z'
        self.assertEqual(c.apply_command(p,now_ms=p['sentAt']+10)['reason'],'interaction_plane_mismatch')
        np.testing.assert_array_equal(before,c.action_vector_m())

    def test_telemetry_loss_pauses_and_valid_heartbeat_never_resumes(self):
        c=SingleArmTaskController(self.config);p=packet(0);c.apply_command(p,now_ms=100010)
        c.update_telemetry({'calibrated':True,'hands':{'left':{'visible':False}},'control_mode':'XY'})
        self.assertTrue(c.paused);self.assertEqual(c.pause_reason,'hand_lost');self.assertIsNone(c.mode_event_id)
        held=c.data.ctrl.copy()
        c.update_telemetry(p['telemetry'])
        self.assertTrue(c.paused)
        for _ in range(10):c.step(1/60)
        np.testing.assert_array_equal(held,c.data.ctrl)
        p=packet(1);p['telemetry']['calibrated']=False
        self.assertEqual(c.apply_command(p,now_ms=p['sentAt']+10)['reason'],'calibration_required')
        p=packet(2);self.assertTrue(c.apply_command(p,now_ms=p['sentAt']+10)['accepted']);self.assertFalse(c.paused)

    def test_record_start_revokes_practice_origin_without_changing_state(self):
        c=SingleArmTaskController(self.config);p=packet(0);c.apply_command(p,now_ms=100010)
        action=c.action_vector_m();qpos=c.data.qpos.copy();ctrl=c.data.ctrl.copy()
        c.begin_recording();snapshot=c.snapshot()
        self.assertTrue(c.paused);self.assertEqual(c.interaction_mode,'HOLD');self.assertIsNone(c.mode_event_id)
        self.assertIsNone(snapshot['status']['physical_contact']['interaction_mode_event_id']['left'])
        np.testing.assert_array_equal(action,c.action_vector_m());np.testing.assert_array_equal(qpos,c.data.qpos)
        np.testing.assert_array_equal(ctrl,c.data.ctrl)
        p=packet(1);p['eventId']='synthetic-push-event:0'
        self.assertEqual(c.apply_command(p,now_ms=p['sentAt']+10)['reason'],'duplicate_event')
        p=packet(2);self.assertTrue(c.apply_command(p,now_ms=p['sentAt']+10)['accepted'])
        self.assertEqual(c.mode_event_id,p['eventId'])

    def test_unknown_hold_cannot_continue_ik_target_advancement(self):
        c=SingleArmTaskController(self.config);p=packet(0);c.apply_command(p,now_ms=100010)
        p=packet(1,'HOLD');self.assertTrue(c.apply_command(p,now_ms=p['sentAt']+10)['accepted'])
        self.assertTrue(c.paused);held=c.data.ctrl.copy()
        for _ in range(10):c.step(1/60)
        np.testing.assert_array_equal(c.data.ctrl,held)

    def test_shared_mailbox_interruption_hook_stops_motion(self):
        c=SingleArmTaskController(self.config);p=packet(0);c.apply_command(p,now_ms=100010)
        c.contact_control.cancel_release('operator_pause')
        self.assertTrue(c.paused);self.assertIsNone(c.mode_event_id)
        held=c.data.ctrl.copy()
        for _ in range(10):c.step(1/60)
        np.testing.assert_array_equal(c.data.ctrl,held)


if __name__=='__main__':unittest.main()
