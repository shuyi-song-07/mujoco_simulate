import unittest
from gesture_fixture import packet, hand, PROTOCOL
from simulation.mujoco.dual_arm.hand_measurement import finger_geometry, inspect_gesture_input, HandInputGuard


class GestureHandGuardTests(unittest.TestCase):
    def check(self, guard, p, age=10):
        return guard.check(p,session="session",epoch=0,now_ms=p["sentAt"]+age,control_protocol=PROTOCOL)

    def test_pinch_geometry_ignores_thumb_index_and_classifier_none(self):
        p=packet(shapes=("pinch","pinch"),modes=("PINCH","PINCH"))
        self.assertIsNone(inspect_gesture_input(p))
        h=p["telemetry"]["recognition_result"]["measurement"]["hands"][0]
        h["image_landmarks_21"][4]=h["image_landmarks_21"][8][:]
        self.assertIsNone(inspect_gesture_input(p))
        h["image_landmarks_21"][12]=h["image_landmarks_21"][9][:]
        h["finger_geometry"]={"idle_extended":True,"full_palm":True}
        self.assertEqual(inspect_gesture_input(p),"interaction_geometry_mismatch")

    def test_release_is_complete_palm_not_cached_aperture_or_geometry_flags(self):
        p=packet(modes=("RELEASE","RELEASE"))
        self.assertIsNone(inspect_gesture_input(p))
        h=p["telemetry"]["recognition_result"]["measurement"]["hands"][0]
        h["aperture_ratio_image"]=0.
        self.assertIsNone(inspect_gesture_input(p),"aperture scalar is not the palm command")
        h["gesture_score"]=.59
        self.assertEqual(inspect_gesture_input(p),"interaction_geometry_mismatch")
        h["gesture_score"]=.99;h["image_landmarks_21"][16]=h["image_landmarks_21"][13][:]
        h["finger_geometry"]={"full_palm":True}
        self.assertEqual(inspect_gesture_input(p),"interaction_geometry_mismatch")

    def test_actual_fist_ily_and_mode_planes(self):
        for shape,mode in (("fist","Z_HOLD"),("ily","XY_HOLD")):
            p=packet(shapes=(shape,shape),modes=(mode,mode))
            self.assertIsNone(inspect_gesture_input(p))
            p["mode"]="Z" if mode=="XY_HOLD" else "XY"
            self.assertEqual(inspect_gesture_input(p),"interaction_plane_mismatch")
        p=packet(shapes=("fist","pinch"),modes=("Z_HOLD","PINCH"))
        self.assertIsNone(inspect_gesture_input(p),"independent aperture with compatible Z plane")

    def test_mismatched_intents_old_protocol_motion_and_forged_modes_rejected(self):
        p=packet();p["control_protocol"]="grasp-hold-release-v1"
        self.assertEqual(inspect_gesture_input(p),"gesture_control_protocol_required")
        p=packet();p["interaction_modes"]=["PINCH"]
        self.assertEqual(inspect_gesture_input(p),"invalid_interaction_modes")
        p=packet();p["release_intent"]=[True,False]
        self.assertEqual(inspect_gesture_input(p),"release_intent_mode_mismatch")
        for mode in ("HOLD","PINCH","RELEASE","OPENING"):
            p=packet(modes=(mode,mode));p["delta_task_m"]["left"][2]=.001
            self.assertEqual(inspect_gesture_input(p),"interaction_motion_forbidden")
        p=packet(shapes=("fist","palm"),modes=("OPENING","OPENING"))
        self.assertEqual(inspect_gesture_input(p),"interaction_geometry_mismatch")

    def test_temporal_recovery_stillness_and_accumulated_slow_motion(self):
        guard=HandInputGuard()
        for i in range(5):
            reason=self.check(guard,packet(i))
            self.assertEqual(reason,None if i==4 else "tracking_recovery")
        self.assertIsNone(self.check(guard,packet(5,modes=("PINCH","PINCH"))))
        for i in range(6,11):
            p=packet(i,modes=("PINCH","PINCH"),dy=(i-5)*.002)
            reason=self.check(guard,p)
        self.assertEqual(reason,"gesture_wrist_not_still","small steps must accumulate against a fixed wrist reference")
        for i in range(11,15):
            reason=self.check(guard,packet(i,modes=("PINCH","PINCH"),dy=.01))
        self.assertIsNone(reason,"valid stillness waiting cannot reset recovery forever")
        self.assertEqual(self.check(guard,packet(15,modes=("OPENING","OPENING"),dy=.03)),"opening_wrist_motion")

    def test_opening_requires_valid_geometry_but_not_continued_palm_shape(self):
        g=HandInputGuard()
        for i in range(5):self.check(g,packet(i))
        self.assertIsNone(self.check(g,packet(5,shapes=('pinch','pinch'),modes=('OPENING','OPENING'))))
        for i in range(6,11):
            p=packet(i,modes=('OPENING','OPENING'))
            h=p['telemetry']['recognition_result']['measurement']['hands'][0]
            h['image_landmarks_21'][11]=h['image_landmarks_21'][10][:]
            self.assertEqual(self.check(g,p),'interaction_geometry_mismatch')

    def test_single_hand_identity_stale_duplicate_and_nan_do_not_enable(self):
        for fault in ("single","identity","stale","nan","duplicate"):
            g=HandInputGuard()
            for i in range(5): self.check(g,packet(i))
            p=packet(5,modes=("RELEASE","RELEASE"));m=p["telemetry"]["recognition_result"]["measurement"]
            if fault=="single":m["hands"].pop()
            if fault=="identity":m["hands"][0]["identity_score"]=.79
            if fault=="nan":m["hands"][0]["image_landmarks_21"][8][0]=float("nan")
            if fault=="duplicate":p=packet(4)
            self.assertIsNotNone(self.check(g,p,age=200 if fault=="stale" else 10),fault)

    def test_image_rotation_and_scale_do_not_invert_finger_roles(self):
        import math
        h=hand();base=finger_geometry(h,640,480)
        for point in h["image_landmarks_21"]:
            x,y=(point[0]-.3)*640,(point[1]-.5)*480
            point[:2]=[.3+(x*math.cos(.6)-y*math.sin(.6))*1.2/640,
                       .5+(x*math.sin(.6)+y*math.cos(.6))*1.2/480]
        changed=finger_geometry(h,640,480)
        self.assertEqual(changed["idle_extended"],base["idle_extended"])
        self.assertEqual(changed["full_palm"],base["full_palm"])
        for f in base["fingers"]:
            self.assertAlmostEqual(changed["fingers"][f]["straightness"],base["fingers"][f]["straightness"])

    def test_bounded_wrist_jitter_allows_pinch_but_real_motion_and_slow_drift_stop_it(self):
        g = HandInputGuard()
        for i in range(8):
            self.check(g, packet(i, dy=.003 if i % 2 else -.003))
        for i in range(8, 20):
            self.assertIsNone(self.check(g, packet(i, modes=("PINCH", "PINCH"),
                                                   dy=.003 if i % 2 else -.003)))
        self.assertEqual(self.check(g, packet(20, modes=("PINCH", "PINCH"), dy=.03)),
                         "gesture_wrist_not_still")
        self.assertEqual(self.check(g, packet(21, modes=("OPENING", "OPENING"), dy=.06)),
                         "opening_wrist_motion")
        g = HandInputGuard()
        for i in range(5): self.check(g, packet(i))
        reasons = [self.check(g, packet(i, modes=("PINCH", "PINCH"), dy=(i-4)*.001))
                   for i in range(5, 30)]
        self.assertIn("gesture_wrist_not_still", reasons,
                      "bounded jitter allowance must not turn into unlimited drift")


if __name__ == "__main__":
    unittest.main()
