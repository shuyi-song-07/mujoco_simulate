"""Synthetic classifier observations for categorical rate-control tests only."""
from gesture_fixture import packet as legacy_packet
from simulation.mujoco.dual_arm.action_contract import V2

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


def send(c, modes=("HOLD", "HOLD"), *, delta=None, widths=None, protocol=PROTOCOL):
    """Submit an explicit scripted rate command through the real controller."""
    action = c.action_vector_m()
    if widths is not None:
        action[[6, 13]] = widths
    if delta is not None:
        for offset, change in zip((0, 7), delta):
            action[offset:offset+3] += change
    c._rate_serial = getattr(c, "_rate_serial", 0)+1
    return c.apply_action(action, schema_version=V2, source="scripted_test",
        event_id=f"rate-{c._rate_serial}", mode="XY" if "XY_HOLD" in modes else "Z",
        interaction_modes=list(modes), control_protocol=protocol,
        release_intent=[m == "Z_OPEN" for m in modes])


def tick(c, modes=("HOLD", "HOLD"), **kw):
    result = send(c, modes, **kw)
    c.step(.02)
    return result


def pump(c, modes, steps=50, **kw):
    for _ in range(steps):
        result = tick(c, modes, **kw)
        if not result["accepted"]:
            raise AssertionError(result)
    return result


def settle(c, seconds):
    for _ in range(round(seconds/.02)):
        c._last_command_sim_time = c.data.time
        c.step(.02)
