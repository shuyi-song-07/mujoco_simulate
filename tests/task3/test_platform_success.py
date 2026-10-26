"""Synthetic status fixtures for platform success NEGATIVES, not physical demos."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from simulation.mujoco.dual_arm.task_states import TaskState, ControlMode
from tasks.tray_to_platform.success import PlatformSuccessMonitor, collection_success, CRITERION, OPENING_CRITERION


def configuration():
    return {"scene": {"table_top_z": .35, "placement_platform": {"height_m": .10},
        "platform_task": {"minimum_lift_m": .03, "lift_stable_seconds": .12, "minimum_open_fraction": .5}},
        "cooperation": {"minimum_contact_force_n": .05, "grasp_stable_seconds": .12},
        "evaluation": {"target_position_tolerance_m": .045, "target_height_tolerance_m": .005,
            "target_orientation_tolerance_rad": .06, "stable_linear_speed_mps": .01,
            "stable_angular_speed_radps": .05, "stable_success_seconds": .3}}


class SyntheticController:
    def __init__(self):
        self.config = configuration()
        self.initial_state = {"object_position": [0, 0, .3842]}
        self.task_state, self.control_mode = TaskState.PREGRASP, ControlMode.IDLE
        self.task_elapsed_seconds = 0.
        self.metrics = SimpleNamespace(completion_time_seconds=None)
        self._success_stable_since = None
        self.contact_control = SimpleNamespace()
        self.status = {"simulation_time": 0., "object_position": [0., 0., .3842], "target_position": [.15, 0., .484],
            "target_orientation_error_rad": 0., "object_linear_velocity_mps": [0., 0., 0.],
            "object_angular_velocity_radps": [0., 0., 0.], "collision_count": 0,
            "contacts": {f"{s}_fingers": [False, False] for s in ("left", "right")},
            "grippers": {s: {"max_width_m": .08, "actual_width_m": .08, "command_width_m": .08} for s in ("left", "right")},
            "physical_contact": {"support_ready": True, "support_surface": "table", "bottom_height_m": 0.,
                "support_forces_n": {"table": 3.43, "placement_platform": 0.},
                "support_requirements": {"minimum_table_force_n": .343}, "placement": {"contained": False}}}

    def get_status(self):
        return {**copy.deepcopy(self.status), "task_state": str(self.task_state)}

    def tick(self, monitor, t):
        self.status["simulation_time"] = t
        self.task_elapsed_seconds = t
        return monitor.update(self)

    def grip(self, both=True):
        for s in ("left", "right"):
            grasp = s == "left" or both
            self.status["contacts"].update({s: grasp, f"{s}_fingers": [grasp, grasp], f"{s}_grip_forces_n": [.3, .3] if grasp else [0., 0.]})
            if grasp:
                self.status["grippers"][s].update(command_width_m=.03, actual_width_m=.036)

    def lift(self):
        self.status["object_position"][2] = .53
        self.status["physical_contact"].update(support_ready=False, bottom_height_m=.146,
            support_forces_n={"table": 0., "placement_platform": 0.})

    def place(self, opening=.04):
        self.status["object_position"] = list(self.status["target_position"])
        for s in ("left", "right"):
            self.status["contacts"].update({s: False, f"{s}_fingers": [False, False], f"{s}_grip_forces_n": [0., 0.]})
            self.status["grippers"][s].update(command_width_m=opening, actual_width_m=opening)
        self.status["physical_contact"].update(support_ready=True, support_surface="placement_platform", bottom_height_m=0.,
            support_forces_n={"table": 0., "placement_platform": 3.43}, placement={"contained": True})


def successful_history(c, m, both=True, lift=True):
    c.grip(both)
    c.tick(m, 0.)
    c.tick(m, .13)
    if lift:
        c.lift()
    c.tick(m, .14)
    c.tick(m, .27)
    c.place()


class PlatformSuccessTests(unittest.TestCase):
    def monitor(self, c):
        return PlatformSuccessMonitor(c.config, c.initial_state)

    def test_synthetic_half_open_requires_done_and_stability(self):
        c = SyntheticController(); m = self.monitor(c)
        successful_history(c, m)
        self.assertTrue(m.ever_dual and m.lift_verified)
        self.assertFalse(c.tick(m, .3)["physical_success"])
        result = c.tick(m, .61)
        self.assertTrue(result["physical_success"])
        self.assertFalse(result["task_success"])
        c.task_state = TaskState.DONE
        result = c.tick(m, .62)
        self.assertTrue(result["task_success"])
        self.assertEqual(result["minimum_open_widths_m"], {"left": .04, "right": .04})

    def test_no_lift_single_grasp_and_fake_opposed_contacts_rejected(self):
        for case in ("no_lift", "one_side", "wrong_force_axis", "transient_lift"):
            with self.subTest(case=case):
                c = SyntheticController(); m = self.monitor(c)
                if case == "wrong_force_axis":
                    c.grip()
                    c.status["contacts"]["right_grip_forces_n"] = [.3, -.3]
                    c.tick(m, 0.); c.tick(m, .13); c.lift(); c.tick(m, .14); c.tick(m, .27); c.place()
                elif case == "transient_lift":
                    c.grip(); c.tick(m, 0.); c.tick(m, .13); c.lift(); c.tick(m, .14); c.tick(m, .24); c.place()
                else:
                    successful_history(c, m, both=case != "one_side", lift=case != "no_lift")
                c.tick(m, .3); c.task_state = TaskState.DONE
                self.assertFalse(c.tick(m, .7)["task_success"])

    def test_bottom_must_lift_not_just_body_center(self):
        c = SyntheticController(); m = self.monitor(c)
        c.grip(); c.tick(m, 0.); c.tick(m, .13); c.lift()
        c.status["physical_contact"]["bottom_height_m"] = .001  # A tilted synthetic tray still touches table.
        c.tick(m, .14); c.tick(m, .4)
        self.assertFalse(m.lift_verified)

    def test_each_final_physical_negative_blocks_success(self):
        changes = {
            "airborne": lambda s: s["physical_contact"].update(support_ready=False),
            "old_table": lambda s: s["physical_contact"].update(support_surface="table"),
            "no_real_load": lambda s: s["physical_contact"]["support_forces_n"].update(placement_platform=0.),
            "missing_load": lambda s: s["physical_contact"].pop("support_forces_n"),
            "not_contained": lambda s: s["physical_contact"]["placement"].update(contained=False),
            "wrong_xy": lambda s: s.update(object_position=[.30, 0., .484]),
            "wrong_height": lambda s: s.update(object_position=[.15, 0., .384]),
            "linear_motion": lambda s: s.update(object_linear_velocity_mps=[.1, 0., 0.]),
            "angular_motion": lambda s: s.update(object_angular_velocity_radps=[.2, 0., 0.]),
            "still_touching": lambda s: s["contacts"].update(right_fingers=[False, True]),
            "command_only_open": lambda s: s["grippers"]["right"].update(actual_width_m=.039),
            "not_commanded_open": lambda s: s["grippers"]["right"].update(command_width_m=.039),
            "collision": lambda s: s.update(collision_count=1),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                c = SyntheticController(); m = self.monitor(c); successful_history(c, m)
                change(c.status); c.task_state = TaskState.DONE
                c.tick(m, .3)
                self.assertFalse(c.tick(m, .8)["task_success"])
        c = SyntheticController(); m = self.monitor(c); successful_history(c, m)
        c.task_state = TaskState.FAIL; c.tick(m, .3)
        self.assertFalse(c.tick(m, .8)["task_success"])

    def test_collection_hook_uses_same_history_and_actual_width(self):
        c = SyntheticController()
        m = c.contact_control.platform_task_monitor = self.monitor(c)
        successful_history(c, m)
        for t in (.3, .61):
            c.status["simulation_time"] = t; c.task_elapsed_seconds = t
            collection_success(c.contact_control, c, t, None, None, False)
        self.assertEqual(c.task_state, TaskState.DONE)
        self.assertTrue(m.last_result["task_success"])
        self.assertEqual(c.metrics.completion_time_seconds, .61)

    def test_any_orientation_succeeds_only_if_contained_and_supported(self):
        for angle in (.06618, .5, 1.57, 3.14):
            with self.subTest(angle=angle):
                c = SyntheticController();m=c.contact_control.platform_task_monitor=self.monitor(c)
                successful_history(c,m)
                c.status['target_orientation_error_rad']=angle
                for t in (.3,.61):
                    c.status['simulation_time']=t
                    collection_success(c.contact_control,c,t,None,None,False)
                self.assertEqual(c.task_state,TaskState.DONE)
                self.assertTrue(c.tick(m,.62)['task_success'])  # Same deployment monitor.
                self.assertNotIn('orientation',m.last_result['final_conditions'])
                c.status['physical_contact']['placement']['contained']=False
                self.assertFalse(c.tick(m,.63)['task_success'])

    def test_reset_and_criterion_mismatch_rejected(self):
        c = SyntheticController(); m = self.monitor(c)
        c.tick(m, 1.)
        with self.assertRaises(ValueError): c.tick(m, 0.)
        for kwargs in ({"min_lift_m": .02}, {"minimum_open_fraction": 1.}, {"min_lift_m": float("nan")}):
            with self.assertRaises(ValueError): PlatformSuccessMonitor(c.config, c.initial_state, **kwargs)

    def test_task_criterion_is_distinct_from_legacy_opening_rule(self):
        from rollout.rollout import validate_settings
        settings = json.loads((Path(__file__).resolve().parents[2]/'tasks/tray_to_platform/evaluation.json').read_text(encoding='utf-8'))
        validate_settings(settings)
        self.assertEqual(settings['task_success_criterion'], CRITERION)
        self.assertEqual(settings['success_criterion'], OPENING_CRITERION)
        c = SyntheticController()
        m = PlatformSuccessMonitor(c.config, c.initial_state, success_criterion=settings['success_criterion'])
        result = c.tick(m, 0.)
        self.assertEqual(result['task_success_criterion'], CRITERION)
        self.assertEqual(result['success_criterion'], CRITERION)
        self.assertEqual(result['opening_criterion'], OPENING_CRITERION)
        for rejected in ('papera-physical-full-open-v1', 'unknown'):
            with self.assertRaises(ValueError):
                PlatformSuccessMonitor(c.config, c.initial_state, success_criterion=rejected)

    def test_platform_preflight_extra_failures_reach_summary(self):
        from tasks.tray_to_platform.scene import preflight
        config = configuration()
        config['scene']['object_half_size'] = [.12, .16, .034]
        config['scene']['platform_task']['transport_clearance_m'] = .025
        controller = SimpleNamespace(initial_state={'object_position': [0., 0., .3842]},
            target_position=[.15, 0., .484], target_yaw=.12, reset=lambda seed: None,
            get_status=lambda: {'object_yaw_rad': .08})
        base_checks = {'all_waypoints_in_workspace': True, 'all_waypoints_static_ik': True,
                       'all_object_waypoints_in_all_frusta': False}
        baseline = {'cases': [{'seed': 7, 'checks': base_checks, 'flags': ['all_object_waypoints_in_all_frusta'],
                              'static_ik': [], 'retained': True}],
            'summary': {'case_count': 1, 'retained_count': 1, 'filtered_count': 0,
                        'cases_with_flags': 1, 'flag_counts': {k: int(not v) for k, v in base_checks.items()}}}
        requested = []

        def waypoint(c, position, yaw, label):
            requested.append((position.copy(), yaw, label))
            return [{'side': side, 'waypoint': label, 'inside_workspace': label != 'clear_platform',
                     'static_ik_pass': label != 'above_platform'} for side in ('left', 'right')]

        with patch('rollout.protocol.preflight_cases', return_value=baseline) as basic, \
             patch('rollout.protocol._waypoint_checks', side_effect=waypoint), \
             patch('simulation.mujoco.dual_arm.task_controller.DualArmTaskController', return_value=controller):
            report = preflight(config, iter([7]))
        self.assertEqual(basic.call_args.args[1], [7])
        self.assertEqual([r[1] for r in requested], [.08, .12])
        self.assertAlmostEqual(requested[0][0][2], .509)
        row = report['cases'][0]
        self.assertEqual(len(row['static_ik']), 4)
        for name in ('all_waypoints_in_workspace', 'all_waypoints_static_ik',
                     'platform_clearance_waypoints_in_workspace', 'platform_clearance_waypoints_static_ik'):
            self.assertIn(name, row['flags'])
            self.assertEqual(report['summary']['flag_counts'][name], 1)
        self.assertIn('all_object_waypoints_in_all_frusta', row['flags'])
        self.assertEqual(report['summary']['retained_count'], 1)
        self.assertEqual(report['summary']['filtered_count'], 0)

    def test_registered_platform_preflight_real_controllers_four_seeds(self):
        from tasks.registry import get, hook
        config = get('tray_to_platform').config()
        scene = config['scene']
        expected_clearance_z = (scene['table_top_z'] + scene['placement_platform']['height_m']
                                + scene['object_half_size'][2] + scene['platform_task']['transport_clearance_m'])
        seeds = [0, 1, 7, 17]
        # No mocked controllers or preflight functions: this covers the actual
        # initial_state/status schema, XML factory, reset and both-arm static IK.
        report = hook(config, 'preflight')(config, seeds)
        self.assertEqual(report['seeds'], seeds)
        self.assertEqual(report['summary']['case_count'], 4)
        self.assertEqual(report['summary']['retained_count'], 4)
        self.assertEqual(report['summary']['filtered_count'], 0)
        self.assertEqual(len(report['high_platform_checks']), 4)
        for row in report['cases']:
            for check in ('finite_reset', 'no_initial_robot_collision',
                          'all_waypoints_in_workspace', 'all_waypoints_static_ik',
                          'platform_clearance_waypoints_in_workspace', 'platform_clearance_waypoints_static_ik'):
                self.assertTrue(row['checks'][check], (row['seed'], check, row['static_ik']))
            self.assertEqual(len(row['static_ik']), 12)
            actual = next(w for w in row['static_ik'] if w['waypoint'] == 'clear_platform' and w['side'] == 'left')
            self.assertAlmostEqual(actual['target_world_m'][2], expected_clearance_z)


if __name__ == "__main__":
    unittest.main()
