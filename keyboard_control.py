import time
import numpy as np
import mujoco
import mujoco.viewer

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

print("hand id =", hand_id)

jacp = np.zeros((3, model.nv))

STEP = 0.02


def move_hand(dx, dy, dz):

    # 当前 hand 的位置
    current_pos = data.xpos[hand_id].copy()

    # 希望移动的距离
    error = np.array([dx, dy, dz])

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


def key_callback(keycode):

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

    # Page Up：上
    elif keycode == 266:
        move_hand(0, 0, STEP)
        print("上")

    # Page Down：下
    elif keycode == 267:
        move_hand(0, 0, -STEP)
        print("下")

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    print("↑ ↓ ← → = 平面移动")
    print("Page Up = 上")
    print("Page Down = 下")

    while viewer.is_running():

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(model.opt.timestep)