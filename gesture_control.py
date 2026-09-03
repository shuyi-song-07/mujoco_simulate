import time
import numpy as np
import mujoco
import mujoco.viewer
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
from queue import Queue

command_queue = Queue()

app = Flask(__name__)
CORS(app)


@app.post("/control")
def control():
    payload = request.get_json(silent=True) or {}
    command = payload.get("command")

    if command:
        command_queue.put(payload)

    return jsonify({"ok": True})


def run_server():
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        use_reloader=False
    )

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Panda 初始姿态
data.qpos[:7] = [
    0.0,
    -0.5,
    0.0,
    -2.0,
    0.0,
    1.5,
    0.8
]

data.ctrl[:7] = data.qpos[:7]

mujoco.mj_forward(model, data)

# 找到 hand
hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
target_plate_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "target_plate"
)
overview_camera_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, "overview_camera"
)
finger_joint_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
]

print("hand id =", hand_id)

jacp = np.zeros((3, model.nv))

STEP = 0.015
VERTICAL_STEP = 0.008
MAX_PLANAR_STEP = 0.015
z_locked = False
locked_z = None
cube_over_target = False
gripper_closed = False
z_lock_used_for_current_grasp = False
vertical_anchor_xy = None
last_gripper_command_at = float("-inf")
TARGET_RADIUS = 0.075
TABLE_TOP_Z = 0.4
TARGET_TOP_Z = 0.416
CUBE_HALF_SIZE = 0.035
PLACEMENT_CLEARANCE = 0.002


def move_hand(dx, dy, dz):
    global locked_z

    # 当前 hand 的位置
    current_pos = data.xpos[hand_id].copy()

    if vertical_anchor_xy is not None:
        # Keep the end effector on one vertical line during up/down motion.
        xy_correction = np.clip(vertical_anchor_xy - current_pos[:2], -0.01, 0.01)
        dx, dy = float(xy_correction[0]), float(xy_correction[1])

    # 希望移动的距离
    if z_locked:
        # Correct any small Z drift while accepting only planar input.
        dz = float(locked_z - current_pos[2])
    elif gripper_closed and dz < 0:
        # Do not let the arm force a held cube through its support surface.
        cube_z = float(data.xpos[cube_id][2])
        support_z = TARGET_TOP_Z if cube_over_target else TABLE_TOP_Z
        minimum_cube_z = support_z + CUBE_HALF_SIZE + PLACEMENT_CLEARANCE
        if cube_z <= minimum_cube_z:
            dz = 0.0
            print("Placement surface reached: downward motion stopped")

    error = np.array([dx, dy, dz], dtype=float)

    # 计算 hand 的 Jacobian
    mujoco.mj_jacBody(
        model,
        data,
        jacp,
        None,
        hand_id
    )

    # Panda 7 个机械臂关节
    J = jacp[:, :7]

    # 末端位移 -> 关节角变化
    dq = np.linalg.pinv(J) @ error

    # 更新关节目标
    data.ctrl[:7] += dq

    # 限制关节范围
    for i in range(7):
        low, high = model.actuator_ctrlrange[i]
        data.ctrl[i] = np.clip(data.ctrl[i], low, high)

    print(
        "hand:",
        np.round(current_pos, 3),
        "→ ctrl:",
        np.round(data.ctrl[:7], 2)
    )


def set_z_lock(locked):
    global z_locked, locked_z
    z_locked = locked
    locked_z = float(data.xpos[hand_id][2]) if locked else None
    print(f"Z {'locked' if locked else 'unlocked'}", locked_z or "")


def start_vertical_motion():
    global vertical_anchor_xy
    vertical_anchor_xy = data.xpos[hand_id][:2].copy()
    print("Vertical path fixed at XY:", np.round(vertical_anchor_xy, 3))


def open_gripper_immediately():
    """Release the object before any simultaneous arm motion is applied."""
    data.ctrl[7] = 255
    for joint_id in finger_joint_ids:
        data.qpos[model.jnt_qposadr[joint_id]] = 0.04
        data.qvel[model.jnt_dofadr[joint_id]] = 0.0
    mujoco.mj_forward(model, data)


def stabilize_cube_on_target():
    """Remove release overlap/velocity without attaching the cube to the plate."""
    if not cube_over_target:
        return

    resting_cube_z = TARGET_TOP_Z + CUBE_HALF_SIZE + PLACEMENT_CLEARANCE
    cube_joint_id = model.body_jntadr[cube_id]
    cube_qpos_adr = model.jnt_qposadr[cube_joint_id]
    cube_dof_adr = model.jnt_dofadr[cube_joint_id]

    if data.xpos[cube_id][2] <= resting_cube_z + 0.03:
        data.qpos[cube_qpos_adr + 2] = max(
            data.qpos[cube_qpos_adr + 2], resting_cube_z
        )
        data.qvel[cube_dof_adr:cube_dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)
        print("Cube released and stabilized on target plate")


def update_target_state():
    global cube_over_target
    cube_xy = data.xpos[cube_id][:2]
    target_xy = data.xpos[target_plate_id][:2]
    is_over_target = np.linalg.norm(cube_xy - target_xy) <= TARGET_RADIUS

    if is_over_target and not cube_over_target:
        cube_over_target = True
        if z_locked:
            set_z_lock(False)
        print("Cube reached white plate: Z automatically unlocked")
    elif not is_over_target:
        cube_over_target = False


def key_callback(keycode):

    # ---------- 前后左右 ----------

    # ↑ 前
    if keycode == 265:
        move_hand(STEP, 0, 0)
        print("前")

    # ↓ 后
    elif keycode == 264:
        move_hand(-STEP, 0, 0)
        print("后")

    # ← 左
    elif keycode == 263:
        move_hand(0, STEP, 0)
        print("左")

    # → 右
    elif keycode == 262:
        move_hand(0, -STEP, 0)
        print("右")

    # ---------- 上下 ----------

    # I：上
    elif keycode == ord('I') or keycode == ord('i'):
        move_hand(0, 0, VERTICAL_STEP)
        print("上")

    # K：下
    elif keycode == ord('K') or keycode == ord('k'):
        move_hand(0, 0, -VERTICAL_STEP)
        print("下")

    # ---------- 夹爪 ----------

    # J：张开
    elif keycode == ord('J') or keycode == ord('j'):
        open_gripper_immediately()
        print("夹爪张开")

    # L：闭合
    elif keycode == ord('L') or keycode == ord('l'):
        data.ctrl[7] = 0
        print("夹爪闭合")

threading.Thread(
    target=run_server,
    daemon=True
).start()     

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = overview_camera_id
    while viewer.is_running():

        while not command_queue.empty():
            payload = command_queue.get_nowait()
            command = payload.get("command")

            if command == "close":
                sent_at = float(payload.get("sentAt", 0.0))
                if sent_at < last_gripper_command_at:
                    continue
                last_gripper_command_at = sent_at
                if not gripper_closed:
                    z_lock_used_for_current_grasp = False
                data.ctrl[7] = 0
                gripper_closed = True

            elif command == "open":
                sent_at = float(payload.get("sentAt", 0.0))
                if sent_at < last_gripper_command_at:
                    continue
                last_gripper_command_at = sent_at
                open_gripper_immediately()
                gripper_closed = False
                set_z_lock(False)
                stabilize_cube_on_target()

            elif command == "left":
                move_hand(0, STEP, 0)

            elif command == "right":
                move_hand(0, -STEP, 0)   

            elif command == "forward":
                move_hand(STEP, 0, 0)
            elif command == "backward":
                move_hand(-STEP, 0, 0) 
            elif command == "up":
                move_hand(0, 0, VERTICAL_STEP)

            elif command == "down":
                move_hand(0, 0, -VERTICAL_STEP)

            elif command == "unlock_z":
                set_z_lock(False)
                z_lock_used_for_current_grasp = True

            elif command == "start_vertical":
                start_vertical_motion()

            elif command == "start_planar":
                vertical_anchor_xy = None
                if (
                    gripper_closed
                    and not z_locked
                    and not z_lock_used_for_current_grasp
                    and not cube_over_target
                ):
                    set_z_lock(True)
                    z_lock_used_for_current_grasp = True

            elif command == "move_xy":
                # Browser values are normalized hand displacement per frame.
                vertical_anchor_xy = None
                if (
                    gripper_closed
                    and not z_locked
                    and not z_lock_used_for_current_grasp
                    and not cube_over_target
                ):
                    set_z_lock(True)
                    z_lock_used_for_current_grasp = True
                dx = np.clip(float(payload.get("dx", 0.0)), -1.0, 1.0)
                dy = np.clip(float(payload.get("dy", 0.0)), -1.0, 1.0)
                move_hand(dx * MAX_PLANAR_STEP, dy * MAX_PLANAR_STEP, 0.0)

        mujoco.mj_step(model, data)
        update_target_state()
        viewer.sync()
        time.sleep(model.opt.timestep)
