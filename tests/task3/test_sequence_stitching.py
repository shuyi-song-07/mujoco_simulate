"""Negative safety cases for virtual Dataset time, independent of ACT sampling."""
import copy
import unittest

import numpy as np

from dataset.task3.curation_config import load_config
from dataset.task3.sequence_stitching import build_sequence_contract, verify_sequence_contract


def observations(count=80):
    identity = np.eye(3).tolist()
    action = [0., -.2, .45, 0., 0., 0., .08, 0., .2, .45, 0., 0., 0., .08]
    state = [0.]*16
    state[7] = state[15] = .08
    controls = []
    for tick in range(count*2):
        features = {"object.position": ["position", [0., 0., .4]],
                    "object.rotation": ["rotation", identity]}
        for side, y in (("left", -.2), ("right", .2)):
            features.update({f"{side}.{key}": [kind, value] for key, kind, value in (
                ("joints", "joint", [0.]*7), ("actual_aperture", "aperture", [.08]),
                ("actual_position", "position", [0., y, .45]), ("actual_rotation", "rotation", identity),
                ("target_position", "position", [0., y, .45]), ("target_rpy", "rpy", [0.]*3),
                ("target_aperture", "aperture", [.08]))})
        execution = {"available": True, "strategy": "palm_fist_edge_v2",
            "sides": {s: {"available": True, "pending_authorization": False,
                "release_confirmation_pending": False, "pending": False} for s in ("left", "right")}}
        controls.append(copy.deepcopy({"tick": tick, "source_line": tick*10+10,
            "sim_time": tick/60, "session": "s", "control_epoch": 1,
            "source": {"valid": True, "available": True, "hard_boundary": False, "reasons": [],
                       "observation_period_s": 1/30, "control_period_s": 1/60},
            "features": features, "invalid": False,
            "motion": {"available": True, "moving": False, "ee_speeds_mps": [0., 0.]},
            "object": {"position": [0., 0., .4], "rotation": identity,
                       "linear_velocity": [0.]*3, "angular_velocity": [0.]*3},
            "execution": execution,
            "grippers": {s: {"actual_width_m": .08, "command_width_m": .08, "goal_width_m": .08,
                "executed_open": False, "executed_close": False, "aborted": False,
                "authorized_mode": None, "edge_event_id": None, "edge_count": 0,
                "baseline_mode": "Z_OPEN", "baseline_source": "initial_open_state"} for s in ("left", "right")},
            "contact": {s: {"fingers": [False, False], "any_finger": False, "double_finger": False,
                            "qualified": False, "stable": False} for s in ("left", "right")},
            "paused": False, "pause_reason": None}))
    rows = []
    for raw in range(count):
        current, next_control = copy.deepcopy(controls[raw*2]), copy.deepcopy(controls[raw*2+1])
        current["interval"] = {"controls": copy.deepcopy(controls[max(0,raw*2-1):raw*2+1]), "events": []}
        current["next_control"] = next_control
        rows.append({"source_episode": "s/0", "session": "s", "episode_index": 0,
            "raw_frame_index": raw, "source_control_tick": raw*2, "control_epoch": 1,
            "source_line": raw*20+10, "original_time_s": raw/30,
            "state_recorded": state[:], "action_recorded": action[:],
            "task_segment_id": 0, "hard_boundary": raw == 0, "frame_role": "SAME_INTENT_IDLE",
            "keep": True, "effective_keep": True, "evidence": current,
            "next_control_target": {"action": action[:], "source_tick": raw*2+1,
                "source_line": raw*20+20, "control_epoch": 1, "sim_time": (raw*2+1)/60,
                "alignment": "next_observed_control_tick_after_post_step_snapshot"}})
    return rows


def cut(rows, start=10, stop=40, manual=False):
    for row in rows[start:stop]:
        row.update(keep=False, effective_keep=False)
        if manual:
            row["human_decision"] = "exclude"
    return rows


class SequenceStitchingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def build(self, rows, restore=False):
        return build_sequence_contract(rows, self.config, restore_boundaries=restore)

    def test_verified_gap_keeps_one_sequence_and_distinct_raw_virtual_clocks(self):
        rows = cut(observations())
        original = copy.deepcopy(rows)
        out = self.build(rows)
        self.assertEqual(rows, original)
        self.assertEqual(out["metrics"]["safe_stitch_count"], 1)
        self.assertEqual(out["metrics"]["sequence_count"], 1)
        self.assertEqual(out["metrics"]["full_valid_action_windows_30"], 21)
        seam = out["rows"][40]
        self.assertEqual(seam["training_frame_index"], 10)
        self.assertEqual(seam["training_time_s"], 10/30)
        self.assertEqual(seam["original_time_s"], 40/30)
        self.assertEqual(seam["next_control_target"]["source_tick"], 81)
        self.assertEqual(out["gaps"][0]["deleted_frame_count"], 30)
        verify_sequence_contract(out["rows"], out, self.config)

    def test_hidden_command_excursion_cannot_pass_equal_endpoints(self):
        rows = cut(observations())
        rows[20]["evidence"]["interval"]["controls"][0]["features"]["right.target_position"][1][0] = .01
        out = self.build(rows)
        self.assertEqual(out["metrics"]["safe_stitch_count"], 0)
        self.assertIn("cumulative_change:right.target_position", out["gaps"][0]["checks"]["reasons"])

    def test_slow_cumulative_object_motion_and_forged_motion_flags_rejected(self):
        self.config["physical_motion"]["position_m"] = .0001  # Frozen strict tolerance.
        rows = cut(observations())
        for i in range(10, 41):
            for facts in [rows[i]["evidence"], *rows[i]["evidence"]["interval"]["controls"]]:
                facts["features"]["object.position"][1][0] = (i-9)*.00001
        out = self.build(rows)
        self.assertIn("cumulative_change:object.position", out["gaps"][0]["checks"]["reasons"])

    def test_direct_velocity_values_checked_without_trusting_boolean(self):
        for field in ("linear_velocity", "angular_velocity"):
            with self.subTest(field=field):
                rows = cut(observations())
                rows[20]["evidence"]["interval"]["controls"][0]["object"][field][0] = .1
                out = self.build(rows)
                self.assertEqual(out["metrics"]["safe_stitch_count"], 0)

    def test_endpoint_only_speed_failure_preserves_edge_and_certifies_static_middle(self):
        rows = cut(observations())
        for row in rows[10:]:
            for f in [row['evidence'], *row['evidence']['interval']['controls']]:
                f['features']['object.position'][1][0] = .0003
        out = self.build(rows, restore=True)
        self.assertTrue(out['rows'][10]['keep'])
        self.assertFalse(out['rows'][20]['keep'])
        self.assertEqual(out['metrics']['safe_stitch_count'], 1)
        self.assertTrue(out['gaps'][0]['checks']['passed'])
        verify_sequence_contract(out['rows'], out, self.config)

    def test_endpoint_speed_failure_without_safe_trim_restores_whole_gap(self):
        rows = cut(observations())
        for i in range(10, 41):
            for f in [rows[i]['evidence'], *rows[i]['evidence']['interval']['controls']]:
                f['features']['object.position'][1][0] = (i-9)*.000015
        out = self.build(rows, restore=True)
        self.assertTrue(all(row['keep'] for row in out['rows'][10:40]))
        self.assertEqual(out['metrics']['safe_stitch_count'], 0)

    def test_slow_joint_motion_below_sa_threshold_restored_not_only_hard_cut(self):
        rows = cut(observations())
        for i in range(10, 41):
            displacement = (i-9)*.00005
            self.assertLess(displacement, self.config["sa"]["thresholds"]["state_joint_rad"])
            rows[i]["state_recorded"][0] = displacement
            for facts in [rows[i]["evidence"], *rows[i]["evidence"]["interval"]["controls"]]:
                facts["features"]["left.joints"][1][0] = displacement
        out = self.build(rows, restore=True)
        self.assertEqual(out["metrics"]["physical_motion_restored_frames"], 30)
        self.assertTrue(all(row["keep"] for row in out["rows"][10:40]))
        self.assertEqual(out["gaps"], [])
        verify_sequence_contract(out["rows"], out, self.config)

    def test_hidden_physical_excursion_is_restored_even_when_endpoints_equal(self):
        rows = cut(observations())
        rows[25]["evidence"]["interval"]["controls"][0]["features"]["object.position"][1][0] = .001
        out = self.build(rows, restore=True)
        self.assertEqual(out["metrics"]["physical_motion_restored_frames"], 30)
        self.assertEqual(out["gaps"], [])

    def test_manual_physical_exclusion_stays_hard_while_automatic_frames_restore(self):
        rows = cut(observations())
        rows[25]["human_decision"] = "exclude"
        rows[25]["evidence"]["interval"]["controls"][0]["object"]["linear_velocity"][0] = .1
        out = self.build(rows, restore=True)
        self.assertFalse(out["rows"][25]["keep"])
        self.assertEqual(out["metrics"]["physical_motion_restored_frames"], 29)
        self.assertEqual(out["gaps"][0]["kind"], "hard_boundary")
        self.assertEqual(out["gaps"][0]["deleted_frame_count"], 1)

    def test_trusted_hold_pending_authorization_is_not_idle(self):
        rows = cut(observations())
        facts = rows[20]["evidence"]["interval"]["controls"][0]
        facts["execution"]["sides"]["left"].update(pending_authorization=True, pending=True)
        out = self.build(rows)
        self.assertIn("pending_gripper_authorization:left", out["gaps"][0]["checks"]["reasons"])

    def test_request_authorization_between_two_controls_protected(self):
        rows = cut(observations())
        execution = copy.deepcopy(rows[20]["evidence"]["execution"])
        execution["sides"]["right"]["pending"] = True
        rows[20]["evidence"]["interval"]["events"] = [{"kind": "request_accepted", "source_line": 399,
            "command": "continuous_input", "execution": execution, "session": "s", "control_epoch": 1}]
        out = self.build(rows)
        self.assertIn("accepted_request_pending_execution", out["gaps"][0]["checks"]["reasons"])

    def test_missing_controls_or_contact_cannot_certify(self):
        for kind in ("controls", "contact", "source", "rotation"):
            with self.subTest(kind=kind):
                rows = cut(observations())
                if kind == "controls": rows[20]["evidence"]["interval"].pop("controls")
                if kind == "contact": rows[20]["evidence"]["contact"]["left"]["stable"] = None
                if kind == "source": rows[20]["evidence"]["source"]["valid"] = False
                if kind == "rotation": rows[20]["evidence"]["features"].pop("object.rotation")
                out = self.build(rows, restore=True)
                self.assertEqual(out["metrics"]["safe_stitch_count"], 0)

    def test_epoch_rejected_request_and_rebase_cannot_certify(self):
        for kind in ("epoch", "rejected", "rebase"):
            with self.subTest(kind=kind):
                rows = cut(observations())
                if kind == "epoch": rows[20]["control_epoch"] = 2
                if kind == "rejected":
                    rows[20]["evidence"]["interval"]["events"] = [{"kind": "request_rejected", "source_line": 399,
                        "reason": "workspace_limit"}]
                if kind == "rebase":
                    rows[20]["evidence"]["interval"]["controls"][0]["grippers"]["left"]["baseline_mode"] = None
                out = self.build(rows)
                self.assertEqual(out["metrics"]["safe_stitch_count"], 0)

    def test_causal_targets_use_next_raw_tick_and_exact_action(self):
        for key, value in (("source_tick", 80), ("source_line", 800), ("control_epoch", 2)):
            with self.subTest(key=key):
                rows = cut(observations())
                rows[40]["next_control_target"][key] = value
                out = self.build(rows)
                self.assertFalse(out["rows"][40]["causal_target_valid"])
                self.assertEqual(out["metrics"]["safe_stitch_count"], 0)

    def test_recorded_state_and_action_must_match_physical_evidence(self):
        for key in ("state_recorded", "action_recorded"):
            rows = cut(observations())
            rows[40][key][0] += .01
            out = self.build(rows)
            self.assertIn("recorded_array_evidence_mismatch:" + key, out["gaps"][0]["checks"]["reasons"])

    def test_next_target_source_ids_are_preserved(self):
        rows = cut(observations())
        rows[40]["next_control_target"]["action_event_id"] = "fabricated"
        out = self.build(rows)
        self.assertFalse(out["rows"][40]["causal_target_valid"])

    def test_manual_exclusion_is_reverified_and_never_restored(self):
        safe = self.build(cut(observations(), manual=True), restore=True)
        self.assertEqual(safe["gaps"][0]["kind"], "safe_stitch")
        rows = cut(observations(), manual=True)
        rows[10]["evidence"]["interval"]["events"] = [{"kind": "contact_changed", "source_line": 209}]
        unsafe = self.build(rows, restore=True)
        self.assertEqual(unsafe["gaps"][0]["kind"], "hard_boundary")
        self.assertEqual(unsafe["metrics"]["boundary_restored_frames"], 0)

    def test_automatic_boundary_can_restore_original_event_context(self):
        rows = cut(observations())
        rows[10]["evidence"]["interval"]["events"] = [{"kind": "contact_changed", "source_line": 209,
                                                        "session": "s", "control_epoch": 1}]
        rows[10]["evidence"]["grippers"]["left"]["executed_close"] = True
        out = self.build(rows, restore=True)
        self.assertEqual(out["metrics"]["safe_stitch_count"], 1)
        self.assertGreater(out["metrics"]["boundary_restored_frames"], 0)
        self.assertTrue(out["rows"][10]["keep"])
        self.assertTrue(all(row["keep"] for row in out["rows"][10:17]))
        verify_sequence_contract(out["rows"], out, self.config)

    def test_accepted_request_excursion_between_static_control_ticks_rejected(self):
        rows = cut(observations())
        accepted = rows[20]["action_recorded"][:]
        accepted[7] += .01
        rows[20]["evidence"]["interval"]["events"] = [{"kind": "request_accepted", "source_line": 399,
            "command": "continuous_input", "execution": copy.deepcopy(rows[20]["evidence"]["execution"]),
            "session": "s", "control_epoch": 1, "accepted_target": accepted}]
        result = self.build(rows)
        self.assertIn("accepted_request_target_excursion:right.action_position_m", result["gaps"][0]["checks"]["reasons"])

    def test_accepted_request_hidden_baseline_rebuild_and_event_order_rejected(self):
        for change in ("baseline", "line", "epoch"):
            rows = cut(observations())
            event = {"kind": "request_accepted", "source_line": 399,
                "command": "continuous_input", "execution": copy.deepcopy(rows[20]["evidence"]["execution"]),
                "session": "s", "control_epoch": 1, "accepted_target": rows[20]["action_recorded"][:],
                "gripper_edges": {s: {"baseline_mode": "Z_OPEN", "baseline_source": "initial_open_state",
                    "authorized_mode": None, "edge_count": 0, "event_id": None} for s in ("left", "right")}}
            if change == "baseline": event["gripper_edges"]["left"]["baseline_mode"] = None
            if change == "line": event["source_line"] = 9000
            if change == "epoch": event["control_epoch"] = 2
            rows[20]["evidence"]["interval"]["events"] = [event]
            self.assertEqual(self.build(rows)["metrics"]["safe_stitch_count"], 0)

    def test_native_moving_frames_remain_contiguous(self):
        rows = observations()
        for row in rows:
            row["evidence"]["motion"]["moving"] = True
        out = self.build(rows)
        self.assertEqual(out["metrics"]["sequence_count"], 1)
        self.assertEqual(out["metrics"]["full_valid_action_windows_30"], 51)

    def test_manual_keep_changes_stitch_and_invalidates_old_metadata(self):
        out = self.build(cut(observations()))
        out["rows"][20].update(keep=True, effective_keep=True, human_decision="keep")
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            verify_sequence_contract(out["rows"], out, self.config)
        updated = self.build(out["rows"])
        self.assertEqual(updated["metrics"]["safe_stitch_count"], 2)

    def test_verifier_detects_clock_and_hidden_control_tampering(self):
        for kind in ("clock", "control", "certificate"):
            with self.subTest(kind=kind):
                out = self.build(cut(observations()))
                if kind == "clock": out["rows"][40]["training_time_s"] += 1
                if kind == "control": out["rows"][20]["evidence"]["interval"]["controls"][0]["grippers"]["right"]["goal_width_m"] = 0.
                if kind == "certificate": out["gaps"][0]["checks"]["maximum_change_from_left"] = {}
                with self.assertRaises(ValueError): verify_sequence_contract(out["rows"], out, self.config)

    def test_unknown_semantics_do_not_invent_motion_or_confidence_threshold(self):
        rows = cut(observations())
        for row in rows[10:40]:
            row["evidence"]["accepted_intent"] = {"left": {"mode": "HOLD", "reliable": False},
                                                       "right": {"mode": None, "reliable": False}}
        self.assertEqual(self.build(rows)["metrics"]["safe_stitch_count"], 1)

    def test_prefix_tail_have_no_fictitious_internal_stitch(self):
        rows = observations()
        cut(rows, 0, 10)
        cut(rows, 70, 80)
        out = self.build(rows)
        self.assertEqual(out["gaps"], [])
        self.assertEqual(out["rows"][10]["training_frame_index"], 0)
        self.assertEqual(out["rows"][69]["training_frame_index"], 59)

    def test_missing_frozen_clock_is_retained_reviewable_not_invented(self):
        rows = observations(5)
        rows[2]["evidence"]["source"]["observation_period_s"] = None
        out = self.build(rows)
        self.assertFalse(out["rows"][2]["virtual_clock_valid"])
        self.assertIsNone(out["rows"][2]["training_time_s"])
        self.assertIsNone(out["rows"][2]["sequence_time_s"])
        self.assertNotEqual(out["rows"][1]["train_sequence_id"], out["rows"][2]["train_sequence_id"])
        self.assertNotEqual(out["rows"][2]["train_sequence_id"], out["rows"][3]["train_sequence_id"])
        self.assertEqual(out["metrics"]["invalid_virtual_clock_frames"], 1)
        verify_sequence_contract(out["rows"], out, self.config)


if __name__ == "__main__":
    unittest.main()
