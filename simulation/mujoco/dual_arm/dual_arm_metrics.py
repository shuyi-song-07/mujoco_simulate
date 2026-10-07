"""Structured, simulation-time Task 3 safety and trajectory metrics."""
from dataclasses import asdict, dataclass


@dataclass
class DualArmMetrics:
    collision_count: int = 0
    contact_loss_count: int = 0
    stale_command_count: int = 0
    rejected_command_count: int = 0
    pause_count: int = 0
    gripper_event_count: int = 0
    grasp_verified_count: int = 0
    workspace_clamp_count: int = 0
    max_relative_pose_error_m: float = 0.0
    max_relative_orientation_error_rad: float = 0.0
    object_path_length_m: float = 0.0
    max_input_disagreement_m: float = 0.0
    completion_time_seconds: float = 0.0
    peak_object_height_m: float = 0.0

    def as_dict(self):
        return asdict(self)
