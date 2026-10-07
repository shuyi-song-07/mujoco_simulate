"""Task progress, user control mode and gripper latch are orthogonal."""
from enum import StrEnum


class TaskState(StrEnum):
    PREGRASP = "PREGRASP"
    GRASP_VERIFY = "GRASP_VERIFY"
    DUAL_GRASPED = "DUAL_GRASPED"
    RELEASE_VERIFY = "RELEASE_VERIFY"
    DONE = "DONE"
    FAIL = "FAIL"


class ControlMode(StrEnum):
    IDLE = "IDLE"
    XY = "XY"
    Z = "Z"
    GRASP_CONFIRM = "GRASP_CONFIRM"
    RELEASE_CONFIRM = "RELEASE_CONFIRM"
    PAUSE = "PAUSE"


class GripperLatch(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
