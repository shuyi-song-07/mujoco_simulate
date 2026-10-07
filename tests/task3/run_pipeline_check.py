"""Generate physics-only pilot fixtures and audit the complete Task3 recorder.

These scripted trajectories are engineering tests, not human demonstrations.
Outputs default to .local/task3-check and are never mixed into datasets/.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulation.mujoco.dual_arm.record_mujoco_dual_arm import DualArmRecorder
from simulation.mujoco.dual_arm.dual_arm_task_controller import scripted_demo
from dataset.task3.validate_task3_dataset import validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/task3-check/dual_recording")
    parser.add_argument("--success-episodes", type=int, default=2)
    args = parser.parse_args()
    settings = SimpleNamespace(config=ROOT / "config/task3_dual_arm.json", seed=0, dataset_root=args.output,
                               no_previews=True, operator_id="scripted_pipeline_test", port=5002, headless=True)
    recorder = DualArmRecorder(settings)
    pilot_status = []
    try:
        recorder.start_recording()
        discarded_pending = recorder.pending
        recorder.discard_recording("scripted_discard_check")
        assert recorder.controller.seed == 1
        assert not discarded_pending.exists()
        assert list(args.output.with_name(args.output.name + "_rejected").glob("*/attempt_000000/task3.json"))
        for _ in range(args.success_episodes):
            recorder.start_recording()
            # Raw confirmation rows are retained, with no redundant training images.
            recorder.telemetry = {"control_mode": "GRASP_CONFIRM", "training_recordable": False}
            before = recorder.frames
            for _ in range(42):
                recorder.capture()
            assert recorder.frames == before
            recorder.telemetry = {}

            def capture(controller):
                recorder.tick += 1
                if controller.gripper_event:
                    recorder.pending_events.append({"type": "gripper", **controller.last_gripper_event})
                recorder.capture(force=True)

            status = scripted_demo(recorder.controller, on_step=capture, reset=False)
            assert status["task_state"] == "DONE", status
            pilot_status.append(status)
            recorder.save_recording("success")
            # A save is reviewable/readable while the application is still
            # alive; the next save must append without overwriting this one.
            durable = validate_dataset(recorder.roots["success"])
            assert durable["valid"], durable
            assert durable["episode_count"] == len(pilot_status), durable
            snapshots = recorder.roots["success"] / f"research/episode_{len(pilot_status)-1:06d}_snapshots"
            assert list(snapshots.glob("*.npz")), "Recovery snapshots must survive a successful save"
        recorder.start_recording()
        recorder.handle({"command": "record_save", "eventId": "premature-success"})
        assert not recorder.last_command_result["accepted"]
        recorder.handle({"command": "record_stop", "eventId": "unsaved-stop"})
        assert not recorder.last_command_result["accepted"] and not recorder.stop_requested
        for _ in range(8):
            recorder.controller.step(1/30)
            recorder.tick += 1
            recorder.capture(force=True)
        recorder.save_recording("failure")
    finally:
        recorder.close()
    success = validate_dataset(recorder.roots["success"])
    failure = validate_dataset(recorder.roots["failure"], allow_failures=True)
    assert success["valid"], success
    assert failure["valid"], failure
    assert not validate_dataset(recorder.roots["failure"])["valid"], "Training must reject failure data"
    result = {"success_dataset": str(recorder.roots["success"]), "failure_dataset": str(recorder.roots["failure"]),
              "success": success, "failure": failure, "physics_pilots": pilot_status,
              "premature_success_rejected": True, "unsaved_stop_rejected": True, "fixture_kind": "scripted_not_human"}
    (args.output.parent / "pipeline-report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"valid": True, "success_episodes": success["episode_count"], "success_frames": success["frame_count"],
                      "failure_frames": failure["frame_count"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    mp.freeze_support()
    main()
