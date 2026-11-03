"""Browser-compatible health responses for Panda recorder runtime states."""

import json
import math
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simulation.mujoco.record_mujoco_panda import (
    DEFAULT_MODEL_PATH,
    GestureServer,
    MujocoPandaRecorder,
    parse_args,
)


def reject_nonfinite_json(constant):
    raise ValueError(f"Non-finite JSON number: {constant}")


class RuntimeStatusTest(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.dataset_directory = Path(temporary_directory.name)

    def recorder(self, task_mode):
        model_path = DEFAULT_MODEL_PATH
        if task_mode == "push_t":
            model_path = model_path.with_name("push_t_scene.xml")
        args = parse_args([
            "--headless-test",
            "--task-mode", task_mode,
            "--model-path", str(model_path),
            "--random-seed", "1",
            "--dataset-root", str(self.dataset_directory / "success"),
            "--failure-dataset-root", str(self.dataset_directory / "failure"),
        ])
        with patch.object(MujocoPandaRecorder, "_start_render_workers", lambda self: None):
            state = MujocoPandaRecorder(args)
        server = GestureServer(
            queue.Queue(), 0, lambda: None, lambda: None, lambda: None,
            state.get_runtime_status,
        )
        self.addCleanup(server.server.server_close)
        return state, server.server.app.test_client()

    def strict_health(self, client):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        status = json.loads(response.get_data(as_text=True), parse_constant=reject_nonfinite_json)
        self.assertTrue(status["ok"])
        self.assertEqual(status["service"], "mujoco-panda-recorder")
        self.assertTrue(math.isfinite(status["grasp_close_elapsed_s"]))
        return status

    def test_startup_health_is_strict_json_for_each_task(self):
        for task_mode in ("push_t", "pick_place"):
            with self.subTest(task_mode=task_mode):
                _, client = self.recorder(task_mode)
                status = self.strict_health(client)
                self.assertEqual(status["task_mode"], task_mode)
                self.assertEqual(status["gripper_closed"], task_mode == "push_t")
                self.assertEqual(status["grasp_close_elapsed_s"], 0.0)

    def test_reset_health_is_strict_json_after_a_closed_grasp(self):
        for task_mode in ("push_t", "pick_place"):
            with self.subTest(task_mode=task_mode):
                state, client = self.recorder(task_mode)
                state.gripper_closed = True
                state.data.time = 2.0
                state.grasp_close_started_at = 1.5
                self.assertEqual(self.strict_health(client)["grasp_close_elapsed_s"], 0.5)

                state.reset_simulation()

                status = self.strict_health(client)
                self.assertEqual(status["task_mode"], task_mode)
                self.assertEqual(status["gripper_closed"], task_mode == "push_t")
                self.assertEqual(status["grasp_close_elapsed_s"], 0.0)

    def test_elapsed_time_preserves_closed_grasps_and_stays_nonnegative(self):
        state, client = self.recorder("pick_place")
        state.data.time = 2.0
        for closed, started_at, expected in (
            (True, 1.5, 0.5),
            (True, 3.0, 0.0),
            (True, float("-inf"), 0.0),
            (False, 1.5, 0.0),
        ):
            with self.subTest(closed=closed, started_at=started_at):
                state.gripper_closed = closed
                state.grasp_close_started_at = started_at
                status = self.strict_health(client)
                self.assertEqual(status["grasp_close_elapsed_s"], expected)


if __name__ == "__main__":
    unittest.main()
