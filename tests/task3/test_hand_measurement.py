import json
import tempfile
from pathlib import Path
import unittest
from simulation.mujoco.dual_arm.hand_measurement import inspect_measurement
from simulation.mujoco.dual_arm.hand_measurement import HandInputGuard
from simulation.mujoco.dual_arm.session_diagnostics import SessionDiagnostics


def measurement():
    result = {"schema_version":"task3-hand-v1", "measurement_id":"camera:1", "source_session":"session", "source_control_epoch":0,
        "capture_epoch_ms":100000, "capture_browser_monotonic_ms":20, "width":640,"height":480,
        "hands":[{"bound_side":s,"raw_label":s,"valid":True,"identity_score":.99,
                  "image_landmarks_21":[[.3,.5,0] for _ in range(21)], "hand_center_landmarks_m_21":[[.01,.02,.03] for _ in range(21)]} for s in ("left","right")]}
    for i, hand in enumerate(result["hands"]):
        for point in hand["image_landmarks_21"]:
            point[0] += .4*i
        hand["image_landmarks_21"][5][0] -= .05
        hand["image_landmarks_21"][17][0] += .05
    return result


class MeasurementTests(unittest.TestCase):
    def test_release_intent_must_match_raw_geometry_and_ready_calibration(self):
        def packet(i, intent=(True, False), ratio=1.2):
            m=measurement();m.update(measurement_id=f'release:{i}',capture_epoch_ms=100000+i*50,accepted_recognition=True)
            for h in m['hands']:
                cx=h['image_landmarks_21'][0][0]
                h['image_landmarks_21'][4][0]=cx-ratio*.05
                h['image_landmarks_21'][8][0]=cx+ratio*.05
                h['aperture_ratio_image']=999  # Cached frontend scalar is not proof.
            return {'sentAt':m['capture_epoch_ms'],'measurement_id':m['measurement_id'],'release_intent':list(intent),
                    'telemetry':{'calibrated':True,'aperture_calibration':{'phase':'READY',
                        'endpoints':{s:{'closed':.15,'open':1.2} for s in ('left','right')}},'recognition_result':{'measurement':m}}}
        good=HandInputGuard()
        for i in range(5):
            p=packet(i);result=good.check(p,session='session',epoch=0,now_ms=p['sentAt']+10)
        self.assertIsNone(result)
        bad=packet(5,ratio=.25)
        self.assertEqual(good.check(bad,session='session',epoch=0,now_ms=bad['sentAt']+10),'release_intent_geometry_mismatch')
        self.assertEqual(good.status()['samples'],0)
        for intent in ([1,False],[True],['true',False],None):
            p=packet(0);p['release_intent']=intent
            self.assertEqual(HandInputGuard().check(p,session='session',epoch=0,now_ms=p['sentAt']+10),'invalid_release_intent')
        for endpoint in (None,{}, {'closed':.15,'open':float('nan')},{'closed':.5,'open':.7}):
            p=packet(0);p['telemetry']['aperture_calibration']['endpoints']['left']=endpoint
            self.assertEqual(HandInputGuard().check(p,session='session',epoch=0,now_ms=p['sentAt']+10),'release_calibration_required')

    def test_backend_rejects_edge_points_even_if_client_claims_trusted(self):
        def packet(i):
            m=measurement();m.update(measurement_id=f'edge:{i}',capture_epoch_ms=100000+i*50,accepted_recognition=True)
            return {'sentAt':m['capture_epoch_ms'],'measurement_id':m['measurement_id'],
                    'telemetry':{'calibrated':True,'aperture_calibration':{'phase':'READY'},'recognition_result':{'measurement':m}}}
        for axis,value in ((0,.059),(0,.941),(1,.059),(1,.941)):
            with self.subTest(axis=axis,value=value):
                guard=HandInputGuard()
                for i in range(5):
                    p=packet(i);result=guard.check(p,session='session',epoch=0,now_ms=p['sentAt']+10)
                self.assertIsNone(result)
                p=packet(5);h=p['telemetry']['recognition_result']['measurement']['hands'][0]
                h['near_edge']=False;h['image_landmarks_21'][20][axis]=value
                self.assertEqual(guard.check(p,session='session',epoch=0,now_ms=p['sentAt']+10),'hand_near_frame_edge')
                self.assertEqual(guard.status()['samples'],0)
                for i in range(6,11):
                    p=packet(i);result=guard.check(p,session='session',epoch=0,now_ms=p['sentAt']+10)
                    self.assertEqual(result,None if i==10 else 'tracking_recovery')

    def test_rejected_duplicate_or_reordered_measurement_requires_full_recovery(self):
        def packet(i, key=None):
            m = measurement()
            m.update(measurement_id=key or f'frame:{i}', capture_epoch_ms=100000+i*50, accepted_recognition=True)
            return {'sentAt':m['capture_epoch_ms'], 'measurement_id':m['measurement_id'],
                    'telemetry':{'calibrated':True, 'aperture_calibration':{'phase':'READY'},
                                 'recognition_result':{'measurement':m}}}
        for cause in ('duplicate_measurement', 'out_of_order_measurement'):
            with self.subTest(cause=cause):
                guard = HandInputGuard()
                for i in range(5):
                    p = packet(i)
                    result = guard.check(p, session='session', epoch=0, now_ms=p['sentAt']+10)
                self.assertIsNone(result)
                bad = packet(4) if cause == 'duplicate_measurement' else packet(3, 'late-new-id')
                self.assertEqual(guard.check(bad, session='session', epoch=0, now_ms=100210), cause)
                self.assertEqual(guard.status()['samples'], 0)
                for i in range(5, 10):
                    p = packet(i)
                    result = guard.check(p, session='session', epoch=0, now_ms=p['sentAt']+10)
                    self.assertEqual(result, None if i == 9 else 'tracking_recovery')

    def test_backend_recovery_rejects_stale_single_identity_and_duplicate_measurement(self):
        guard=HandInputGuard()
        def packet(i):
            m=measurement();m.update(measurement_id=f"frame:{i}",capture_epoch_ms=100000+i*50,accepted_recognition=True)
            return {"sentAt":m["capture_epoch_ms"],"measurement_id":m["measurement_id"],
                    "telemetry":{"calibrated":True,"aperture_calibration":{"phase":"READY"},"recognition_result":{"measurement":m}}}
        for i in range(5):
            p=packet(i);reason=guard.check(p,session="session",epoch=0,now_ms=p["sentAt"]+10)
            self.assertEqual(reason,None if i==4 else "tracking_recovery")
        self.assertEqual(guard.check(p,session="session",epoch=0,now_ms=p["sentAt"]+10),"duplicate_measurement")
        p=packet(6);p["telemetry"]["recognition_result"]["measurement"]["hands"].pop()
        self.assertEqual(guard.check(p,session="session",epoch=0,now_ms=p["sentAt"]+10),"untrusted_hand_measurement")
        p=packet(7)
        self.assertEqual(guard.check(p,session="session",epoch=0,now_ms=p["sentAt"]+200),"untrusted_hand_measurement")
        p=packet(8);p["telemetry"]["recognition_result"]["measurement"]["hands"][0]["identity_score"]=.7
        self.assertEqual(guard.check(p,session="session",epoch=0,now_ms=p["sentAt"]+10),"invalid_identity_score")
    def test_coordinates_reach_log_and_missing_is_not_zero_filled(self):
        m=measurement(); self.assertIsNone(inspect_measurement(m,session="session",epoch=0))
        with tempfile.TemporaryDirectory() as tmp:
            log=SessionDiagnostics(Path(tmp)/"session",{})
            log.emit("hand_measurement",measurement=m)
            log.close()
            row=json.loads(next(log.root.glob("events-*.jsonl")).read_text())
            self.assertEqual(row["measurement"]["hands"][0]["image_landmarks_21"],m["hands"][0]["image_landmarks_21"])
            self.assertEqual(log.dropped,0)
        m["hands"][0].update(valid=False,image_landmarks_21=None)
        self.assertIsNone(inspect_measurement(m,session="session",epoch=0))
        self.assertIsNone(m["hands"][0]["image_landmarks_21"])

    def test_malformed_or_wrong_context_cannot_claim_valid(self):
        m=measurement();self.assertEqual(inspect_measurement(m,session="new",epoch=0),"stale_measurement_context")
        m["hands"][0]["image_landmarks_21"].pop()
        self.assertEqual(inspect_measurement(m,session="session",epoch=0),"invalid_landmark_shape")
        m=measurement();m["hands"][0]["identity_score"]=.7
        self.assertEqual(inspect_measurement(m,session="session",epoch=0),"invalid_identity_score")
