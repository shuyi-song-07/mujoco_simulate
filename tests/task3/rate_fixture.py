"""Synthetic classifier observations for categorical rate-control tests only."""
from gesture_fixture import packet as legacy_packet

PROTOCOL = "gesture-rate-gripper-v3"
CALIBRATION = {"phase": "NOT_REQUIRED", "reason": "categorical_rate_control"}


def packet(i=0, modes=("HOLD", "HOLD"), shapes=None, *, dx=0., dy=0.):
    if shapes is None:
        shapes = tuple({"XY_HOLD": "ily", "Z_OPEN": "palm", "Z_CLOSE": "fist"}.get(m, "palm") for m in modes)
    p = legacy_packet(i, shapes, modes, dx=dx, dy=dy)
    p.update(control_protocol=PROTOCOL, release_intent=[m == "Z_OPEN" for m in modes],
             widths_m=[0. if m == "Z_CLOSE" else .08 for m in modes])
    p['telemetry']['aperture_calibration'] = dict(CALIBRATION)
    p['telemetry']['calibration'] = {'palm_binding_guard': False, 'source': 'dual_open_palm'}
    return p
